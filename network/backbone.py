# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------

import torch
import importlib
import torch.nn.functional as F

from tqdm import tqdm
from copy import deepcopy
from util.common_utils import *
from network.unet2 import UNet
from network.loss import Loss_backbone_train
from torch.optim.lr_scheduler import MultiStepLR
from util.dist_util import get_dist_info, master_only
from torch.nn.parallel import DataParallel, DistributedDataParallel

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device("cpu")


class Model():
    """Base Deblur model for single image deblur."""

    def __init__(self, config, log, writer):
        self.config = config
        self.log = log
        self.writer = writer
        network_config = config['network']
        # define network
        if network_config['mask_generator'] == 1:
            self.net = UNet().to(device)
        elif network_config['mask_generator'] == 2:
            self.net = UNet(out_channels=5).to(device)
        else:
            raise NotImplementedError

        # load pretrained models
        if config['path']['pretrain_path'] is not None:
            self.load_network(self.net, config['path']['pretrain_path'], config['path']['strict_load'], param_key='net')

        # load pretrained models
        if config['path']['resume_state'] is not None:
            self.load_network(self.net, config['path']['resume_state'], config['path']['strict_load'], param_key='net')

        if config['is_train']:
            self.rank = config['rank']
            self.init_training_settings()
        else:
            self.metrics = {}

    def load_network(self, net, load_path, strict=True, param_key='params'):
        """Load network.

        Args:
            load_path (str): The path of networks to be loaded.
            net (nn.Module): Network.
            strict (bool): Whether strictly loaded.
            param_key (str): The parameter key of loaded network. If set to
                None, use the root 'path'.
                Default: 'params'.
        """
        net = self.get_bare_model(net)
        self.log.info(f'Loading model from {load_path}.')
        load_net = torch.load(load_path, map_location=lambda storage, loc: storage)
        if param_key is not None:
            load_net = load_net[param_key]
        # print(' load net keys', load_net.keys)
        # remove unnecessary 'module.'
        for k, v in deepcopy(load_net).items():
            if k.startswith('module.'):
                load_net[k[7:]] = v
                load_net.pop(k)
        net.load_state_dict(load_net, strict=strict)

    def get_bare_model(self, net):
        """Get bare model, especially under wrapping with
        DistributedDataParallel or DataParallel.
        """
        if isinstance(net, (DataParallel, DistributedDataParallel)):
            net = net.module
        return net

    def init_training_settings(self):
        self.net.train()

        # define losses
        self.loss = Loss_backbone_train(self.config['network'], device)

        # set up optimizers and schedulers
        self.optimizer = torch.optim.AdamW([{'params': self.net.parameters()}], lr=self.config['train']['lr'])
        total_parameters = sum([param.nelement() for param in self.net.parameters()])
        if self.rank == 0:
            self.log.info("Total parameters: %.2fM" % (total_parameters / 1e6))

        # using multi-step as the learning rate change strategy
        # self.scheduler = MultiStepLR(self.optimizer, milestones=[20, 40, 80, 160], gamma=0.5)  # learning rates
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                                    self.config['train']['num_iters'])  # learning rates

    def resume_training(self, resume_state):
        """Reload the optimizers and schedulers for resumed training.

        Args:
            resume_state (dict): Resume state.
        """
        self.optimizer.load_state_dict(resume_state['optimizer'])
        self.scheduler.load_state_dict(resume_state['scheduler'])

    def optimize_parameters(self, train_data, current_iter, use_grad_clip=False):
        self.optimizer.zero_grad()
        x1, x2, img_name = train_data['x1'].to(device), train_data['x2'].to(device), train_data['img_name']
        gt_x, gt_m = train_data['gt_x'].to(device), train_data['gt_m'].to(device)
        score_map = torch.sign(torch.abs(blur_2th(x1)) - torch.min(torch.abs(blur_2th(x1)),
                                                                   torch.abs(blur_2th(x2))))

        # get the network output
        out_x, out_m1, out_m2 = self.net(x1, x2)

        self.losses = self.loss(out_x=out_x, out_m=[out_m1, out_m2], gt_x=gt_x, gt_m=[gt_m, 1 - gt_m])
        self.losses['total_loss'].backward()

        if use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.01)

        self.optimizer.step()
        # change the learning rate
        self.scheduler.step()

        # write to tensorboard
        if self.rank == 0 and current_iter % self.config['val']['save_freq'] == 0:
            self.writer.add_image("train/gt_x", gt_x[0], current_iter, dataformats='CHW')
            self.writer.add_image("train/gt_m1", gt_m[0].squeeze(), current_iter, dataformats='HW')
            self.writer.add_image("train/out_x", out_x[0], current_iter, dataformats='CHW')
            self.writer.add_image("train/out_m1", out_m1[0].squeeze(), current_iter, dataformats='HW')

    def save(self, epoch, current_iter):
        if self.rank == 0:
            state = {
                'epoch': epoch,
                'iter': current_iter,
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'net': self.net.state_dict(),
            }
            save_filename = f'{current_iter}.pth'
            save_path = os.path.join(self.config['path']['save_path'], self.config['name'], save_filename)
            torch.save(state, save_path)

    def validation(self, val_loader, current_iter):
        """Validation function.

        Args:
            dataloader (torch.utils.data.DataLoader): Validation dataloader.
            current_iter (int): Current iteration.
            tb_logger (tensorboard logger): Tensorboard logger.
            save_img (bool): Whether to save images. Default: False.
            rgb2bgr (bool): Whether to save images using rgb2bgr. Default: True
            use_image (bool): Whether to use saved images to compute metrics (PSNR, SSIM), if not, then use data directly from network' output. Default: True
        """
        self.net.eval()
        rank, world_size = get_dist_info()
        if rank == 0:
            pbar = tqdm(total=len(val_loader), unit='image')

        for idx, val_data in enumerate(val_loader):
            if idx % world_size != rank:
                continue
            x1, x2, img_name = val_data['x1'].to(device), val_data['x2'].to(device), val_data['img_name']
            img_name = img_name[0]

            with torch.no_grad():
                out_x, out_m1, out_m2 = self.net(x1, x2)
                # calculate metrics
                self.losses = self.loss(out_x=out_x, out_m=[out_m1, out_m2])

            if self.config['val']['save_img'] and current_iter % self.config['val']['save_freq'] == 0:
                with torch.no_grad():
                    if rank == 0:
                        self.writer.add_image("val/x1", x1[0], current_iter, dataformats='CHW')
                        self.writer.add_image("val/x2", x2[0], current_iter, dataformats='CHW')
                        self.writer.add_image("val/out_x", out_x[0], current_iter, dataformats='CHW')
                        self.writer.add_image("val/out_m1", out_m1[0].squeeze(), current_iter, dataformats='HW')


            if rank == 0:
                for _ in range(world_size):
                    pbar.update(1)
                    pbar.set_description(f'Test {img_name}')
        if rank == 0:
            pbar.close()

        self.net.train()

        # tentative for out of GPU memory
        torch.cuda.empty_cache()

    def test(self, test_loader):

        self.net.eval()
        pbar = tqdm(total=len(test_loader), unit='image')

        for idx, test_data in enumerate(test_loader):
            x1, x2, img_name = test_data['x1'], test_data['x2'], test_data['img_name']
            img_name = img_name[0]

            with torch.no_grad():
                out_x, out_m1, out_m2 = self.net(x1, x2)

                if self.config['test']['save_img']:
                    save_path_x = os.path.join(self.config['path']['save_path']+self.config['name'],
                                               '%s_x.png' % (img_name))
                    save_path_m1 = os.path.join(self.config['path']['save_path']+self.config['name'],
                                                '%s_m0.png' % (img_name))
                    save_path_m2 = os.path.join(self.config['path']['save_path']+self.config['name'],
                                                '%s_m1.png' % (img_name))

                    cv2.imwrite(save_path_x, torch_to_np(out_x, data_type='uint8'))
                    cv2.imwrite(save_path_m1, torch_to_np(out_m1, data_type='uint8'))
                    if self.config['network']['mask_generator'] == 2:
                        cv2.imwrite(save_path_m2, torch_to_np(out_m2, data_type='uint8'))

            pbar.update(1)
            pbar.set_description(f'Test {img_name}')
        pbar.close()
