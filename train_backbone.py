# from __future__ import print_function

import os
import time
import yaml
import torch
import warnings
import argparse
import datetime
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp

from util.logger import Logger
from util.common_utils import *
from network.backbone import Model
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from util.dist_util import get_dist_info, init_dist
from torch.utils.data.distributed import DistributedSampler
from util.dataset import Lytro, Real_MFF, MFI_WHU, NYUDF

warnings.filterwarnings("ignore")

torch.backends.cudnn.benchmark = False

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device("cpu")


def get_dataset(dataset_config, is_train=True):
    if dataset_config.get('name') == 'Lytro':
        return Lytro(dataset_config.get('data_path'), dataset_config.get('img_size'))
    elif dataset_config.get('name') == 'Real-MFF':
        return Real_MFF(dataset_config.get('data_path'), dataset_config.get('img_size'))
    elif dataset_config.get('name') == 'MFI-WHU':
        return MFI_WHU(dataset_config.get('data_path'), dataset_config.get('img_size'))
    elif dataset_config.get('name') == 'NYUDF':
        return NYUDF(dataset_config.get('data_path'), dataset_config.get('img_size'), dataset_config.get('use_hflip'),
                     dataset_config.get('use_vflip'), dataset_config.get('use_cflip'), dataset_config.get('use_rot'),
                     is_train)
    else:
        raise NotImplementedError


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./config/train_backbone.yaml')
    parser.add_argument('--local_rank', type=int, default=0)
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

    train_set = get_dataset(config['dataset'].get('train'), True)
    val_set = get_dataset(config['dataset'].get('val'), False)

    train_sampler = DistributedSampler(train_set)

    train_loader = DataLoader(train_set, batch_size=config['train'].get('batch_size'), shuffle=False,
                              num_workers=8, sampler=train_sampler)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=True)

    if config['rank'] == 0:
        log.info(
            'Training statistics:'
            f'\n\tNumber of train images: {len(train_set)}'
            f'\n\tBatch size: {config["train"]["batch_size"]}'
            f'\n\tWorld size (gpu number): {config["world_size"]}'
            f'\n\tRequire iter number per epoch: {config["train"]["num_iters"]}.')

    ################################################## network #####################################################

    # load resume states if necessary
    if config['path'].get('resume_state') is not None:
        device_id = torch.cuda.current_device()
        resume_state = torch.load(
            config['path'].get('resume_state'),
            map_location=lambda storage, loc: storage.cuda(device_id))
    else:
        resume_state = None

    # create model
    if resume_state:  # resume training
        model = Model(config, log, writer)
        model.resume_training(resume_state)  # handle optimizers and schedulers
        print(f"Resuming training from epoch: {resume_state['epoch']}, iter: {resume_state['iter']}.")
        start_epoch = resume_state['epoch']
        start_iter = resume_state['iter']
    else:
        model = Model(config, log, writer)
        start_epoch = 0
        start_iter = 0

    ##################################################### training #####################################################

    start_time = time.time()
    iter_time = time.time()
    epoch = start_epoch
    current_iter = start_iter
    while current_iter < config['train']['num_iters']:
        train_sampler.set_epoch(epoch)

        ################################################ start iteration ###############################################

        for idx, train_data in enumerate(train_loader):

            model.optimize_parameters(train_data, current_iter)

            # write to tensorboard
            if config['rank'] == 0:
                writer.add_scalar("loss_pixl_x", model.losses['pixel_loss_x'], global_step=current_iter)
                writer.add_scalar("loss_pixl_m", model.losses['pixel_loss_m'], global_step=current_iter)
                writer.add_scalar("loss_recon", model.losses['recon_loss'], global_step=current_iter)
                writer.add_scalar("loss_prior", model.losses['prior_loss'], global_step=current_iter)
                writer.add_scalar("loss_total", model.losses['total_loss'], global_step=current_iter)

            iter_time = time.time() - iter_time
            # log
            if current_iter % config['train']['print_freq'] == 0 and config['rank'] == 0:
                total_time = time.time() - start_time
                time_sec_avg = total_time / (current_iter - start_iter + 1)
                eta_sec = time_sec_avg * (config['train']['num_iters'] - current_iter - 1)
                eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
                log_str = f'[{config["name"]}] ' \
                          f'[epoch:{epoch:3d}, iter:{current_iter:8,d}, ' \
                          f'lr:({model.optimizer.param_groups[0]["lr"]:.3e},)] ' \
                          f'[eta: {eta_str}, time (data): {iter_time:.3f}'
                for metric, value in model.losses.items():
                    log_str += f'\t # {metric}: {value:.4f}'
                log.info(log_str)

            ############################################### saving #####################################################

            if current_iter % config['val']['save_freq'] == 0:
                if config['rank'] == 0:
                    # save models and training states
                    log.info('Saving models and training states.')
                    model.save(epoch, current_iter)

                    # validation
                    log.info(
                        'Validating statistics:'
                        f'\n\tNumber of val images: {len(val_set)}')

                model.validation(val_loader, current_iter)

                if config['rank'] == 0:
                    log_str = f'Validation, \t'
                    for metric, value in model.losses.items():
                        log_str += f'\t # {metric}: {value:.4f}'
                    log.info(log_str)

            current_iter += 1
        epoch += 1

    if config['rank'] == 0:
        # save models and training states
        log.info('Saving models and training states.')
        # model.save(epoch, current_iter)

        # validation
        log.info(
            'Validating statistics:'
            f'\n\tNumber of val images: {len(val_set)}')

    model.validation(val_loader, current_iter)

    if config['rank'] == 0:
        log_str = f'Validation, \t'
        for metric, value in model.losses.items():
            log_str += f'\t # {metric}: {value:.4f}'
        log.info(log_str)


if __name__ == '__main__':

    ################################################### preperation ####################################################

    args = parse()

    with open(args.config, mode='r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # DDP initialization
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')

    config['rank'] = args.local_rank
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(config['rank'] % num_gpus)
    dist.init_process_group(backend=config['dist']['dist_backend'])
    _, config['world_size'] = get_dist_info()

    set_random_seed(config['rand_seed'] + config['rank'])

    if config['rank'] == 0:
        os.makedirs(os.path.join(config['path'].get('save_path'), config['name']), exist_ok=True)
        os.makedirs(os.path.join(config['path'].get('writer_path'), config['name']), exist_ok=True)
        writer = SummaryWriter(os.path.join(config['path'].get('writer_path'), config['name']))
        log = Logger(filename=os.path.join(config['path'].get('save_path'), config['name'] + '.log'), level='debug')
        log.info(print_config(config))
    else:
        writer = None
        log = None

    config['is_train'] = True

    main(config, log, writer)

    if config['rank'] == 0:
        writer.close()
        log.info("all done at %s" % time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
