import os
import glob
import torch
import numpy as np

from util.common_utils import *
from torchvision import transforms
from torch.utils.data import Dataset


def group_crop(img_list, crop_size=None):
    """Paired random crop.

    It crops lists of lq and gt images with corresponding locations.

    Args:
        img_gts (list[ndarray] | ndarray): GT images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        img_lqs (list[ndarray] | ndarray): LQ images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        gt_patch_size (int): GT patch size.
        scale (int): Scale factor.
        gt_path (str): Path to ground-truth.

    Returns:
        list[ndarray] | ndarray: GT images and LQ images. If returned results
            only have one element, just return ndarray.
    """

    if not isinstance(img_list, list):
        img_list = [img_list]

    h, w, _ = img_list[0].shape
    if crop_size is not None:
        crop_h, crop_w = crop_size
    else:
        crop_h, crop_w = h, w

    def _crop(img):
        if h < crop_h or w < crop_w:
            return cv2.resize(img, [crop_w, crop_h])

        # randomly choose top and left coordinates for lq patch
        top = random.randint(0, h - crop_h)
        left = random.randint(0, w - crop_w)

        # crop lq patch
        return img[top:top + crop_h, left:left + crop_w, ...]

    img_list = [
        _crop(img) for img in img_list
    ]

    if len(img_list) == 1:
        img_list = img_list[0]

    return img_list


def group_augment(img_list, hflip=True, vflip=False, cflip=False, rot=False):
    """Augment: horizontal flips OR rotate (0, 90, 180, 270 degrees).

    We use vertical flip and transpose for rotation implementation.
    All the images in the list use the same augmentation.

    Args:
        imgs (list[ndarray] | ndarray): Images to be augmented. If the input
            is an ndarray, it will be transformed to a list.
        hflip (bool): Horizontal flip. Default: True.
        rotation (bool): Ratotation. Default: True.
        flows (list[ndarray]: Flows to be augmented. If the input is an
            ndarray, it will be transformed to a list.
            Dimension is (h, w, 2). Default: None.
        return_status (bool): Return the status of flip and rotation.
            Default: False.

    Returns:
        list[ndarray] | ndarray: Augmented images and flows. If returned
            results only have one element, just return ndarray.

    """
    hflip = hflip and random.random() < 0.5
    if vflip or rot:
        vflip = random.random() < 0.5
    rot90 = rot and random.random() < 0.5
    idx = np.random.permutation(3)

    def _augment(img):
        if img.ndim == 2:
            h, w = img.shape
            img = img.reshape(h, w, 1)
        if hflip:  # horizontal
            # cv2.flip(img, 1, img)
            img = img[::-1, :, :].copy()
        if vflip:  # vertical
            # cv2.flip(img, 0, img)
            img = img[:, ::-1, :].copy()
        if cflip and img.shape[2] == 3:  # flip RGB
            img = img[:, :, idx]
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    if not isinstance(img_list, list):
        img_list = [img_list]
    img_list = [_augment(img) for img in img_list]
    if len(img_list) == 1:
        img_list = img_list[0]

    return img_list


class Lytro(Dataset):
    def __init__(self, root_dir, img_size):
        self.images1 = glob.glob(os.path.join(root_dir, '*-A.jpg'))
        self.images1.sort()
        self.images2 = glob.glob(os.path.join(root_dir, '*-B.jpg'))
        self.images2.sort()
        self.transform = transforms.ToTensor()
        self.img_size = img_size

    def __len__(self):
        return len(self.images1)

    def __getitem__(self, idx):
        x1 = cv2.imread(self.images1[idx])
        x2 = cv2.imread(self.images2[idx])
        y1, cr, cb = cv2.split(cv2.cvtColor(x1, cv2.COLOR_BGR2YCrCb))
        y2, _, _ = cv2.split(cv2.cvtColor(x2, cv2.COLOR_BGR2YCrCb))

        x1, x2, y1, y2, cr, cb = group_crop([x1, x2, y1, y2, cr, cb], crop_size=self.img_size)

        y1 = self.transform(y1)
        y2 = self.transform(y2)
        x1 = self.transform(x1)
        x2 = self.transform(x2)

        return {'x1': x1, 'x2': x2, 'img_name': os.path.basename(self.images1[idx]).split('.')[0][:-2],
                'y1': y1, 'y2': y2, 'cr': cr, 'cb': cb}


class Real_MFF(Dataset):
    def __init__(self, root_dir, img_size=None):
        if not os.path.exists(os.path.join(root_dir, 'list.txt')):
            files = os.listdir(root_dir)
        else:
            with open(os.path.join(root_dir, 'list.txt'), 'r') as f:
                files = f.read().rstrip().split('\n')

        self.images1 = [os.path.join(root_dir, file, file + '_1.png') for file in files]
        self.images2 = [os.path.join(root_dir, file, file + '_2.png') for file in files]
        self.gts = [os.path.join(root_dir, file, file + '_0.png') for file in files]
        self.transform = transforms.ToTensor()
        self.img_size = img_size

    def __len__(self):
        return len(self.images1)

    def __getitem__(self, idx):
        x1 = cv2.imread(self.images1[idx])
        x2 = cv2.imread(self.images2[idx])
        gt_x = cv2.imread(self.gts[idx])
        y1, cr, cb = cv2.split(cv2.cvtColor(x1, cv2.COLOR_BGR2YCrCb))
        y2, _, _ = cv2.split(cv2.cvtColor(x2, cv2.COLOR_BGR2YCrCb))
        gt_y, _, _ = cv2.split(cv2.cvtColor(gt_x, cv2.COLOR_BGR2YCrCb))

        x1, x2, gt_x, y1, y2, gt_y, cr, cb = group_crop([x1, x2, gt_x, y1, y2, gt_y, cr, cb], crop_size=self.img_size)

        x1 = self.transform(x1)
        x2 = self.transform(x2)
        y1 = self.transform(y1)
        y2 = self.transform(y2)
        gt_x = self.transform(gt_x)

        return {'x1': x1, 'x2': x2, 'img_name': os.path.basename(self.images1[idx]).split('.')[0][:-2],
                'gt_x': gt_x, 'y1': y1, 'y2': y2, 'gt_y': gt_y, 'cr': cr, 'cb': cb}


class MFI_WHU(Dataset):
    def __init__(self, root_dir, img_size=None):
        self.images1 = [os.path.join(root_dir, 'source_1', str(i) + '.jpg') for i in range(1, 121)]
        self.images2 = [os.path.join(root_dir, 'source_2', str(i) + '.jpg') for i in range(1, 121)]
        self.gts = [os.path.join(root_dir, 'full_clear', str(i) + '.jpg') for i in range(1, 121)]
        self.transform = transforms.ToTensor()
        self.img_size = img_size

    def __len__(self):
        return len(self.images1)

    def __getitem__(self, idx):
        x1 = cv2.imread(self.images1[idx])
        x2 = cv2.imread(self.images2[idx])
        gt_x = cv2.imread(self.gts[idx])
        y1, cr, cb = cv2.split(cv2.cvtColor(x1, cv2.COLOR_BGR2YCrCb))
        y2, _, _ = cv2.split(cv2.cvtColor(x2, cv2.COLOR_BGR2YCrCb))
        gt_y, _, _ = cv2.split(cv2.cvtColor(gt_x, cv2.COLOR_BGR2YCrCb))

        x1, x2, gt_x, y1, y2, gt_y, cr, cb = group_crop([x1, x2, gt_x, y1, y2, gt_y, cr, cb], crop_size=self.img_size)

        y1 = self.transform(y1)
        y2 = self.transform(y2)
        gt_y = self.transform(gt_y)
        x1 = self.transform(x1)
        x2 = self.transform(x2)
        gt_x = self.transform(gt_x)

        return {'x1': x1, 'x2': x2, 'img_name': os.path.basename(self.images1[idx]).split('.')[0],
                'gt_x': gt_x, 'y1': y1, 'y2': y2, 'gt_y': gt_y, 'cr': cr, 'cb': cb}


class NYUDF(Dataset):
    def __init__(self, root_dir, img_size=None, use_hflip=False, use_vflip=False, use_cflip=False, use_rot=False,
                 is_train=True):
        self.root_dir = root_dir
        self.img_size = img_size
        self.use_hflip = use_hflip
        self.use_vflip = use_vflip
        self.use_cflip = use_cflip
        self.use_rot = use_rot
        self.is_train = is_train
        self.transform = transforms.ToTensor()
        self.num_images = len(os.listdir(os.path.join(self.root_dir, 'defocus_images'))) // 2
        self.num_train_images = int(0.75 * self.num_images)

    def __len__(self):
        return self.num_train_images if self.is_train else self.num_images - self.num_train_images

    def __getitem__(self, idx):
        idx = idx if self.is_train else idx + self.num_train_images
        x1 = cv2.imread(os.path.join(self.root_dir, 'defocus_images', str(idx) + '_1.png'))
        x2 = cv2.imread(os.path.join(self.root_dir, 'defocus_images', str(idx) + '_2.png'))
        gt_x = cv2.imread(os.path.join(self.root_dir, 'images', str(idx) + '.png'))
        sigma1 = cv2.imread(os.path.join(self.root_dir, 'sigma_maps', str(idx) + '_1.png'), 0)
        sigma2 = cv2.imread(os.path.join(self.root_dir, 'sigma_maps', str(idx) + '_2.png'), 0)
        gt_d = cv2.imread(os.path.join(self.root_dir, 'depths', str(idx) + '.png'), 0)
        y1, cr, cb = cv2.split(cv2.cvtColor(x1, cv2.COLOR_BGR2YCrCb))
        y2, _, _ = cv2.split(cv2.cvtColor(x2, cv2.COLOR_BGR2YCrCb))

        temp = group_crop([x1, x2, y1, y2, cr, cb, gt_x, sigma1, sigma2, gt_d], crop_size=self.img_size)
        x1, x2, y1, y2, cr, cb, gt_x, sigma1, sigma2, gt_d = temp

        if self.is_train:
            # flip, rotation
            temp = group_augment([x1, x2, gt_x, y1, y2, cr, cb, sigma1, sigma2, gt_d],
                                 hflip=self.use_hflip,
                                 vflip=self.use_vflip,
                                 cflip=self.use_cflip,
                                 rot=self.use_rot)
            x1, x2, gt_x, y1, y2, cr, cb, sigma1, sigma2, gt_d = temp

        x1 = self.transform(x1)
        x2 = self.transform(x2)
        y1 = self.transform(y1)
        y2 = self.transform(y2)
        gt_x = self.transform(gt_x)
        sigma1 = self.transform(sigma1)
        sigma2 = self.transform(sigma2)
        gt_d = self.transform(gt_d)

        gt_m1 = (torch.sign(1000. * sigma1 - 1000. * sigma2) + 1) / 2.
        # gt_m2 = 1. - gt_m1

        # normalize
        # TODO

        return {'x1': x1, 'x2': x2, 'img_name': str(idx),
                'gt_x': gt_x, 'gt_m': gt_m1, 'gt_d': gt_d, 'y1': y1, 'y2': y2, 'cr': cr, 'cb': cb}
