import os
import time
import yaml
import torch
import warnings
import argparse
import datetime
import numpy as np

from tqdm import tqdm
from util.logger import Logger
from util.common_utils import *
from network.lst_skip import lst_skip
from network.unet2 import UNet
from network.loss import Loss_inpa_test
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from util.dataset import Lytro, Real_MFF, MFI_WHU
from torch.nn.parallel import DataParallel, DistributedDataParallel

warnings.filterwarnings("ignore")

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device("cpu")


def get_dataset(dataset_config, is_train=False):
    if dataset_config.get('name') == 'Lytro':
        return Lytro(dataset_config.get('data_path'), dataset_config.get('img_size'))
    elif dataset_config.get('name') == 'Real-MFF':
        return Real_MFF(dataset_config.get('data_path'), dataset_config.get('img_size'))
    elif dataset_config.get('name') == 'MFI-WHU':
        return MFI_WHU(dataset_config.get('data_path'), dataset_config.get('img_size'))
    else:
        raise NotImplementedError


def get_bare_model(net):
    """Get bare model, especially under wrapping with
    DistributedDataParallel or DataParallel.
    """
    if isinstance(net, (DataParallel, DistributedDataParallel)):
        net = net.module
    return net


def load_network(net, load_path, strict=True, param_key='params'):
    """Load network.

    Args:
        load_path (str): The path of networks to be loaded.
        net (nn.Module): Network.
        strict (bool): Whether strictly loaded.
        param_key (str): The parameter key of loaded network. If set to
            None, use the root 'path'.
            Default: 'params'.
    """
    net = get_bare_model(net)
    log.info(f'Loading model from {load_path}.')
    load_net = torch.load(load_path, map_location=lambda storage, loc: storage)
    if param_key is not None:
        load_net = load_net[param_key]
    net.load_state_dict(load_net, strict=strict)


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./config/test_inpa.yaml')
    args = parser.parse_args()

    return args


def print_config(config, depth=0):
    config_str = ""
    for k, v in config.items():
        config_str += '\n' + '  ' * (depth + 1)
        if isinstance(v, dict):
            config_str += '{:} '.format(k + ':') + print_config(v, depth + 1)
        else:
            config_str += '{:} {}'.format(k + ':', str(v))
    return config_str


def main(config, log=None, writer=None):
    ##################################################### dataset ######################################################

    test_set = get_dataset(config['dataset'].get('test'), False)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    log.info(
        'Testing statistics:'
        f'\n\tNumber of test images: {len(test_set)}')

    # pretrained network
    if config['network']['mask_generator'] == 1:
        net_pre = UNet().to(device)
    elif config['network']['mask_generator'] == 2:
        net_pre = UNet(out_channels=5).to(device)
    else:
        raise NotImplementedError

    # load pretrained models
    if config['path']['pretrain_path'] is not None:
        load_network(net_pre, config['path']['pretrain_path'], config['path']['strict_load'],
                     param_key='net')
    # fix
    for param in net_pre.parameters():
        param.requires_grad = False

    # define losses
    loss = Loss_inpa_test(config['network'], device)
    alpha_list = []
    beta_list = []

    start_time = time.time()
    for idx, test_data in enumerate(test_loader):
        y1, y2 = test_data['y1'].to(device), test_data['y2'].to(device)
        x1, x2 = test_data['x1'].to(device), test_data['x2'].to(device)
        if test_data.get('gt_x') is not None:
            gt_x, gt_m = test_data['gt_x'].to(device), test_data['gt_m'].to(device)

        img_name = test_data['img_name'][0]
        log.info(img_name)

        n, c, h, w = y1.shape
        config['dataset']['test']['img_size'] = [h, w]
        alpha_list.append([])
        beta_list.append([])

        current_time = time.time()
        if current_time - start_time > 0.5:
            log.info('data loading finished in {:.4} seconds'.format(current_time - start_time))
        start_time = current_time

        ################################################## network #####################################################

        if config['network']['input'] == 'source_img':
            net_input = torch.cat([y1, y2], dim=1)
        else:
            net_input_ = torch.zeros([1, config['network']['input_channel'],
                                     config['dataset']['test']['img_size'][0], config['dataset']['test']['img_size'][1]])
            net_input = get_noise(net_input_, spatial_size=config['dataset']['test']['img_size'],
                                  input_channel=config['network']['input_channel'],
                                  input_type=config['network']['input']).to(device)
            del net_input_

        net = lst_skip(config['network']['input_channel'],
                       channels=[64, 128, 256],
                       channels_skip=16,
                       upsample_mode='bilinear',
                       attention_mode=config['network']['attention'],
                       need_bias=False, pad=config['network']['pad'],
                       act_fun='LeakyReLU', scales=config['network']['multi_scale'],
                       fuse_type=config['network']['fuse_type'], scales_k=config['network']['scales_k']).to(device)

        current_time = time.time()
        if current_time - start_time > 0.5:
            log.info('input and network prepared in {:.4} seconds'.format(current_time - start_time))
        start_time = current_time

        ############################################### optimizer ######################################################
        # set up optimizers and schedulers
        optimizer = torch.optim.Adam([{'params': net.parameters()}], lr=config['test']['lr'])
        total_parameters = sum([param.nelement() for param in net.parameters()])
        log.info("Total parameters: %.2fM" % (total_parameters / 1e6))

        # using multi-step as the learning rate change strategy
        scheduler = MultiStepLR(optimizer, milestones=[200, 400, 800], gamma=0.5)  # learning rates

        net_input_saved = net_input.detach().clone()
        noise = net_input.detach().clone()

        pre_feats_enc, pre_feats_dec = net_pre.get_feats(x1, x2)
        score_map = get_score_map(y1, y2, config['network']['score_map'])

        # load tail
        net.load_tail(config['path']['pretrain_path'], config['path']['strict_load'], param_key='net')

        ################################################ start iteration ###############################################
        start_time_ = time.time()
        for step in tqdm(range(config['test']['num_iters'] + 1)):
            optimizer.zero_grad()

            # add_noise
            if config['network']['reg_noise_std'] > 0:
                net_input = net_input_saved + (noise.normal_() * config['network']['reg_noise_std'])

            # get the network output
            if config['network']['fuse_type'] in ['weighted', 'weighted1']:
                out_x, out_m1, out_m2, alphas, betas = net(net_input, pre_feats_enc, pre_feats_dec)
            else:
                out_x, out_m1, out_m2 = net(net_input, pre_feats_enc, pre_feats_dec)
            if config['network']['fdb_map'] == 'optimize':
                fdb_mask = 1. - (2. * out_m2 - 1.) ** 2
                pred_x = fdb_mask * out_x + (1 - fdb_mask) * (out_m1 * x2 + (1 - out_m1) * x1)
            else:
                pred_x = out_m1 * x2 + (1 - out_m1) * x1
            losses = loss(out_x=out_x, pred_x=out_m1 * x2 + out_m2 * x1, out_m=[out_m2, out_m1],
                          score_m=[score_map, 1 - score_map])

            pred_x0 = out_m1 * x2 + (1 - out_m1) * x1

            losses['total_loss'].backward()
            optimizer.step()

            # change the learning rate
            scheduler.step()

            current_time = time.time()
            if current_time - start_time > 0.5:
                log.info('backprop finished in {:.4} seconds'.format(current_time - start_time))
            start_time = current_time

            ############################################### saving #####################################################
            with torch.no_grad():
                # write to tensorboard
                if writer is not None:
                    for key, value in losses.items():
                        writer.add_scalar(img_name + "/" + key, value.detach().item(), step)
                    if config['network']['fuse_type'] in ['weighted', 'weighted1']:
                        for i, alpha_ in enumerate(alphas):
                            if step % config['test']['num_iters'] == 0 and step != 0:
                                alpha_list[idx].append(torch.abs(alpha_).detach().item())
                            writer.add_scalar(img_name + "/alpha" + str(i), alpha_.detach().item(), step)
                        for j, beta_ in enumerate(betas):
                            if step % config['test']['num_iters'] == 0 and step != 0:
                                beta_list[idx].append(torch.abs(beta_).detach().item())
                            writer.add_scalar(img_name + "/beta" + str(j), beta_.detach().item(), step)
                if step % config['test']['save_freq'] == 0:
                    out_x_np = torch_to_np(out_x, data_type='uint8')
                    pred_x_np = torch_to_np(pred_x, data_type='uint8')
                    pred_x0_np = torch_to_np(pred_x0, data_type='uint8')
                    if test_data.get('gt_x') is not None:
                        gt_x_np = torch_to_np(gt_x, data_type='uint8')

                    # write to tensorboard
                    if writer is not None:
                        if fdb_mask is not None:
                            writer.add_image(img_name + "/fdb_m", fdb_mask.detach().squeeze(), step, dataformats='HW')
                        writer.add_image(img_name + "/out_x", out_x[0].detach(), step, dataformats='CHW')
                        writer.add_image(img_name + "/pred_x", pred_x[0].detach(), step, dataformats='CHW')
                        writer.add_image(img_name + "/out_m1", out_m1.detach().squeeze(), step, dataformats='HW')
                        # writer.add_image(img_name + "/out_m2", out_m2.squeeze(), step, dataformats='HW')

                    if config['test']['save_img']:
                        save_path_x = os.path.join(config['path']['save_path'], config['name'],
                                                   '%s_%d_x.png' % (img_name, step))
                        save_path_x1 = os.path.join(config['path']['save_path'], config['name'],
                                                   '%s_%d_x1.png' % (img_name, step))
                        save_path_x0 = os.path.join(config['path']['save_path'], config['name'],
                                                   '%s_%d_x0.png' % (img_name, step))
                        save_path_m1 = os.path.join(config['path']['save_path'], config['name'],
                                                    '%s_%d_m0.png' % (img_name, step))
                        save_path_m2 = os.path.join(config['path']['save_path'], config['name'],
                                                    '%s_%d_m1.png' % (img_name, step))
                        save_path_m3 = os.path.join(config['path']['save_path'], config['name'],
                                                    '%s_%d_m3.png' % (img_name, step))
                        save_path_s = os.path.join(config['path']['save_path'], config['name'],
                                                    '%s_s.png' % (img_name))
                        # out_x_np = torch_to_np(out_x, data_type='uint8')
                        # out_x_np = cv2.merge([out_x_np, cr[0].squeeze().numpy(), cb[0].squeeze().numpy()])
                        # out_x_np = cv2.cvtColor(out_x_np, cv2.COLOR_YCrCb2BGR)
                        cv2.imwrite(save_path_x, out_x_np)
                        cv2.imwrite(save_path_x1, pred_x_np)
                        cv2.imwrite(save_path_x0, pred_x0_np)
                        cv2.imwrite(save_path_m1, torch_to_np(out_m1, data_type='uint8'))
                        # cv2.imwrite(save_path_s, torch_to_np(score_map, data_type='uint8'))
                        if fdb_mask is not None:
                            cv2.imwrite(save_path_m3, torch_to_np(fdb_mask, data_type='uint8'))
                        if config['network']['mask_generator'] == 2:
                            cv2.imwrite(save_path_m2, torch_to_np(out_m2, data_type='uint8'))

                        # torch.save(net, os.path.join(opt.save_path, "%s_xnet.pth" % imgname))

            current_time = time.time()
            if current_time - start_time > 0.5:
                log.info('saving finished in {:.4} seconds'.format(current_time - start_time))
            start_time = current_time
            # torch.cuda.empty_cache()

        # log
        total_time = time.time() - start_time_
        time_sec_avg = total_time
        eta_sec = time_sec_avg * (len(test_loader) - idx - 1)
        eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
        log_str = f'[{config["name"]}] ' \
                  f'[iter:{step:8,d}, ' \
                  f'lr:({optimizer.param_groups[0]["lr"]:.3e},)] ' \
                  f'[eta: {eta_str}'
        for metric, value in losses.items():
            log_str += f'\t # {metric}: {value:.4f}'
        log.info(log_str)

        log.info(log_str)


if __name__ == '__main__':
    ################################################### preperation ####################################################
    args = parse()

    with open(args.config, mode='r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    set_random_seed(config['rand_seed'])

    os.makedirs(os.path.join(config['path'].get('save_path'), config['name']), exist_ok=True)
    os.makedirs(os.path.join(config['path'].get('writer_path'), config['name']), exist_ok=True)
    writer = SummaryWriter(os.path.join(config['path'].get('writer_path'), config['name'])) if config['use_writer'] else None
    log = Logger(filename=os.path.join(config['path'].get('save_path'), config['name'] + '.log'), level='debug')
    log.info(print_config(config))

    main(config, log, writer)

    if writer is not None:
        writer.close()
    log.info("all done at %s" % time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
