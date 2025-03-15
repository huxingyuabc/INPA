import torch.nn as nn

from network.SSIM import SSIM
from torch.nn import MSELoss, L1Loss

EPS = 1e-7


class Loss_backbone_train(nn.Module):
    def __init__(self, config, device):
        super(Loss_backbone_train, self).__init__()
        self.mse_loss = MSELoss().to(device)
        self.bce_loss = nn.BCELoss().to(device)
        self.l1_loss = L1Loss().to(device)
        self.ssim_loss = SSIM().to(device)
        self.alpha = config['loss']['alpha']
        self.beta = config['loss']['beta']
        self.rate = config['loss']['rate']
        self.scales = config['multi_scale']
        self.mask_generator = config['mask_generator']

    def forward(self, out_x, out_m, gt_x=None, gt_m=None):
        loss_pixl_x = 0
        loss_pixl_m = 0
        loss_recon = 0
        loss_prior = 0

        if gt_x is not None:
            # pixel loss
            loss_pixl_x = self.l1_loss(out_x, gt_x)
            for i in range(self.mask_generator):
                assert gt_m[i].min() >= 0 and gt_m[i].max() <= 1
                loss_pixl_m += self.bce_loss(out_m[i], gt_m[i])

            total_loss = loss_pixl_x + 1 * loss_pixl_m

            losses = {"total_loss": total_loss, "recon_loss": loss_recon, "prior_loss": loss_prior,
                      'pixel_loss_x': loss_pixl_x, 'pixel_loss_m': loss_pixl_m}
        else:
            losses = {"recon_loss": loss_recon}
        return losses


class Loss_inpa_test(nn.Module):
    def __init__(self, config, device):
        super(Loss_inpa_test, self).__init__()
        self.mse_loss = MSELoss().to(device)
        self.l1_loss = L1Loss().to(device)
        self.ssim_loss = SSIM().to(device)
        self.alpha = config['loss']['alpha']
        self.beta = config['loss']['beta']
        self.gamma = config['loss']['gamma']
        self.rate = config['loss']['rate']
        self.scales = config['multi_scale']
        self.mask_generator = config['mask_generator']
        self.thresh = config['loss']['thresh']
        self.num_iters = config['num_iters']

    def forward(self, out_x, pred_x, out_m, score_m=None):

        loss_recon = self.l1_loss(out_x, pred_x)
        loss_prior = self.l1_loss(out_m[0], score_m[0])
        loss_total = self.alpha * loss_recon + self.beta * loss_prior

        losses = {"total_loss": loss_total, "recon_loss": loss_recon, 'prior_loss': loss_prior}
        return losses
