import numpy as np
import h5py
import os
import imageio
import torch
import cv2

from tqdm import tqdm

# modify the script to your NYUDv2 dataset path
file = h5py.File("/data/nyu_depth_v2_labeled.mat")
images = file["images"]
depths = file["depths"]
images = torch.from_numpy(np.array(images) / 255.).cuda()
depths = torch.from_numpy(np.array(depths)).cuda()
images = images.permute((0, 1, 3, 2))
depths = depths.permute((0, 2, 1))

# modify the script to where you want to place the synthetic dataset
path_converted_img = '/data/MFIF-SYNDoF/images/'
path_converted_depth = '/data/MFIF-SYNDoF/depths/'
path_converted_sigma_map = '/data/MFIF-SYNDoF/sigma_maps/'
path_converted_defocus_img = '/data/MFIF-SYNDoF/defocus_images/'
path_converted_gt_mask = '/data/MFIF-SYNDoF/focus_maps/'
if not os.path.isdir(path_converted_img):
    os.makedirs(path_converted_img)
if not os.path.isdir(path_converted_depth):
    os.makedirs(path_converted_depth)
if not os.path.isdir(path_converted_sigma_map):
    os.makedirs(path_converted_sigma_map)
if not os.path.isdir(path_converted_defocus_img):
    os.makedirs(path_converted_defocus_img)
if not os.path.isdir(path_converted_gt_mask):
    os.makedirs(path_converted_gt_mask)

######################## camera parameters #######################

f = 0.5
k = 0.48
F = 1.8
c1 = 0.1
c2 = 0.9
k_size = 21
alpha = 1000
depth_scale = 10
kernel_type = 'guassian'
max_coc = 29
max_depth_layers = 350
min_depth_layers = 200
max_count = 20
count = 0
epsilon = 0.0001

n, c, h, w = images.shape
kernels1 = torch.zeros((h, w, k_size, k_size), device='cuda')
kernels2 = torch.zeros((h, w, k_size, k_size), device='cuda')
I1_ = torch.zeros((c, h, w), device='cuda')
I2_ = torch.zeros((c, h, w), device='cuda')

######################## blur kernel calculation #################

def get_gaussian(ki, kj, s):
    return 1. / (2 * np.pi * s * s) * torch.exp(-(ki * ki + kj * kj) / (2 * s * s))

def get_blur_kernel(s):
    kernelx = cv2.getGaussianKernel(k_size, s)
    kernel = torch.from_numpy(kernelx * kernelx.T).to('cuda')
    return kernel

########################### sythesizing ##########################

for i in tqdm(range(images.shape[0])):
    img = images[i, :, :, :]
    depth = depths[i, :, :]
    ######################## focal plane #########################
    dmin = depth.min()
    dmax = depth.max()
    d1 = dmin + c1 * (dmax - dmin)
    d2 = dmin + c2 * (dmax - dmin)
    ##################### focus map generation ###################
    # standard deviation of the PSF that can be used to measure the amount of defocus blur
    sigma1 = (torch.abs(depth - d1) * k * f * f) / (depth * (d1 - f) * F)
    sigma2 = (torch.abs(depth - d2) * k * f * f) / (depth * (d2 - f) * F)
    ################## defocus image generation #################
    # spatially varying non-uniform blur kernels generation
    for hi in range(h):
        for hj in range(w):
            kernels1[hi][hj] = get_blur_kernel(alpha * sigma1[hi][hj].item())
            kernels2[hi][hj] = get_blur_kernel(alpha * sigma2[hi][hj].item())

    # pixel-wise convolution
    padding = (21 - 1) / 2
    slide_win = torch.nn.functional.unfold(img.unsqueeze(0), k_size, padding=int(padding))
    slide_win = slide_win.view(c, k_size * k_size, h, w).permute(0, 2, 3, 1)
    blur_kernel1 = kernels1.view(h, w, k_size * k_size)
    blur_kernel2 = kernels2.view(h, w, k_size * k_size)
    for hk in range(c):
        for hi in range(h):
            for hj in range(w):
                I1_[hk, hi, hj] = (slide_win[hk, hi, hj] @ blur_kernel1[hi, hj].T.double())
                I2_[hk, hi, hj] = (slide_win[hk, hi, hj] @ blur_kernel2[hi, hj].T.double())
    mask = (torch.sign(sigma1 - sigma2) + 1) / 2.
    I1 = I1_ * (1. - mask) + img * mask
    I2 = I2_ * mask + img * (1. - mask)

    ############################ saving ###########################
    iconpath_img = path_converted_img + str(i) + '.png'
    # iconpath_depth_npy = path_converted_depth + str(i) + '.npy'
    iconpath_depth_img = path_converted_depth + str(i) + '.png'
    iconpath_sigma1 = path_converted_sigma_map + str(i) + '_1.png'
    iconpath_sigma2 = path_converted_sigma_map + str(i) + '_2.png'
    iconpath_I1 = path_converted_defocus_img + str(i) + '_1.png'
    iconpath_I2 = path_converted_defocus_img + str(i) + '_2.png'
    iconpath_mask1 = path_converted_gt_mask + str(i) + '.png'

    imageio.imwrite(iconpath_img, np.uint8(img.permute(1, 2, 0).cpu().numpy() * 255.))
    # np.save(iconpath_depth_npy, depth.cpu().numpy())

    imageio.imwrite(iconpath_depth_img, np.uint8(depth.cpu().numpy() / 10. * 255.))
    smin = sigma1.min()
    smax = sigma1.max()
    sigma1 = smin + sigma1 / (smax - smin)
    smin = sigma2.min()
    smax = sigma2.max()
    sigma2 = smin + sigma2 / (smax - smin)
    mask = (torch.sign(sigma1 - sigma2) + 1) / 2.
    imageio.imwrite(iconpath_sigma1, np.uint8(sigma1.cpu().numpy() * 255.))
    imageio.imwrite(iconpath_sigma2, np.uint8(sigma2.cpu().numpy() * 255.))
    imageio.imwrite(iconpath_I1, np.uint8(I1.permute(1, 2, 0).cpu().numpy() * 255.))
    imageio.imwrite(iconpath_I2, np.uint8(I2.permute(1, 2, 0).cpu().numpy() * 255.))
    imageio.imwrite(iconpath_mask1, np.uint8(mask.cpu().numpy() * 255.))