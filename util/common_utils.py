import os
import PIL
import sys
import cv2
import math
import torch
import random
import numpy as np
import torch.nn.functional as F

from PIL import Image
from util.guided_filter import guided_filter


def get_score_map(y1, y2, mode='blur2th'):
    score_map_ = torch.sign(torch.abs(blur_2th(y1)) - torch.min(torch.abs(blur_2th(y1)), torch.abs(blur_2th(y2))))
    if mode == 'blur2th':
        score_map = score_map_
    elif mode == 'max_select':
        _, score_map = torch.max(torch.cat([y1, y2], dim=1), 1)
        score_map = score_map.float()
    elif mode == 'gradient':
        score_map = torch.sign(torch.abs(gradient(y1)) - torch.min(torch.abs(gradient(y1)), torch.abs(gradient(y2))))
    elif mode == 'guassian':
        score_map = guassian(score_map_)
    elif mode == 'guided_filter':
        score_map = torch.from_numpy(
            guided_filter(I=y1.squeeze().cpu().numpy(), p=score_map_.squeeze().cpu().numpy(), r=8, eps=0.05)).unsqueeze(
            0).unsqueeze(0)
    elif mode == 'LBP':
        y1_np = torch_to_np(y1, data_type='uint8')
        y2_np = torch_to_np(y2, data_type='uint8')
        lbp1 = torch.from_numpy(lbpSharpness(y1_np, 21, 0.016)).unsqueeze(0).unsqueeze(0).to(y1.device)
        lbp2 = torch.from_numpy(lbpSharpness(y2_np, 21, 0.016)).unsqueeze(0).unsqueeze(0).to(y1.device)
        score_map = torch.sign(lbp1 - torch.min(lbp1, lbp2))
    else:
        raise NotImplementedError
    return score_map.cuda()


def blur_2th(img, times=2):
    filtr = torch.tensor([[0.0947, 0.1183, 0.0947], [0.1183, 0.1478, 0.1183], [0.0947, 0.1183, 0.0947]],
                         device=img.device)
    assert img.ndim == 4 and (img.shape[1] == 1 or img.shape[1] == 3)
    filtr = filtr.expand(img.shape[1], img.shape[1], 3, 3)
    blur = F.conv2d(img, filtr, bias=None, stride=1, padding=1)
    for i in range(times - 1):
        blur = F.conv2d(blur, filtr, bias=None, stride=1, padding=1)
    diff = torch.abs(img - blur)
    return diff


def guassian(input1):
    filtr = torch.tensor([[0.0947, 0.1183, 0.0947], [0.1183, 0.1478, 0.1183], [0.0947, 0.1183, 0.0947]]).type(
        torch.cuda.FloatTensor)
    filtr = filtr.expand(input1.shape[1], input1.shape[1], 3, 3)
    blur = F.conv2d(input1, filtr, bias=None, stride=1, padding=1)
    return blur


def gradient(input1):
    n, c, w, h = input1.shape
    filter1 = torch.reshape(torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]).type(torch.cuda.FloatTensor),
                            [1, 1, 3, 3])
    filter1 = filter1.repeat_interleave(c, dim=1)
    d = torch.nn.functional.conv2d(input1, filter1, bias=None, stride=1, padding=1)
    return d


def im2double(im):
    min_val = np.min(im.ravel())
    max_val = np.max(im.ravel())
    out = (im.astype('float') - min_val) / (max_val - min_val)
    return out


def s(x):
    temp = x > 0
    return temp.astype(float)


def lbpCode(im_gray, threshold):
    interpOff = math.sqrt(2) / 2
    I = im2double(im_gray)
    pt = cv2.copyMakeBorder(I, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
    right = pt[1:-1, 2:]
    left = pt[1:-1, :-2]
    above = pt[:-2, 1:-1]
    below = pt[2:, 1:-1]
    aboveRight = pt[:-2, 2:]
    aboveLeft = pt[:-2, :-2]
    belowRight = pt[2:, 2:]
    belowLeft = pt[2:, :-2]
    interp0 = right
    interp1 = (1 - interpOff) * ((1 - interpOff) * I + interpOff * right) + interpOff * (
            (1 - interpOff) * above + interpOff * aboveRight)

    interp2 = above
    interp3 = (1 - interpOff) * ((1 - interpOff) * I + interpOff * left) + interpOff * (
            (1 - interpOff) * above + interpOff * aboveLeft)

    interp4 = left
    interp5 = (1 - interpOff) * ((1 - interpOff) * I + interpOff * left) + interpOff * (
            (1 - interpOff) * below + interpOff * belowLeft)

    interp6 = below
    interp7 = (1 - interpOff) * ((1 - interpOff) * I + interpOff * right) + interpOff * (
            (1 - interpOff) * below + interpOff * belowRight)

    s0 = s(interp0 - I - threshold)
    s1 = s(interp1 - I - threshold)
    s2 = s(interp2 - I - threshold)
    s3 = s(interp3 - I - threshold)
    s4 = s(interp4 - I - threshold)
    s5 = s(interp5 - I - threshold)
    s6 = s(interp6 - I - threshold)
    s7 = s(interp7 - I - threshold)
    LBP81 = s0 * 1 + s1 * 2 + s2 * 4 + s3 * 8 + s4 * 16 + s5 * 32 + s6 * 64 + s7 * 128
    LBP81.astype(int)

    U = np.abs(s0 - s7) + np.abs(s1 - s0) + np.abs(s2 - s1) + np.abs(s3 - s2) + np.abs(s4 - s3) + np.abs(
        s5 - s4) + np.abs(s6 - s5) + np.abs(s7 - s6)
    LBP81riu2 = s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7
    LBP81riu2[U > 2] = 9

    return LBP81riu2


def lbpSharpness(im_gray, s, threshold):
    lbpmap = lbpCode(im_gray, threshold)
    window_r = (s - 1) // 2
    h, w = im_gray.shape[:2]
    lbpmap_pad = cv2.copyMakeBorder(lbpmap, window_r, window_r, window_r, window_r, cv2.BORDER_REPLICATE)

    lbpmap_sum = (lbpmap_pad == 6).astype(float) + (lbpmap_pad == 7).astype(float) + (lbpmap_pad == 8).astype(
        float) + (lbpmap_pad == 9).astype(float)
    integral = cv2.integral(lbpmap_sum)
    integral = integral.astype(float)

    map = (integral[s - 1:-1, s - 1:-1] - integral[0:h, s - 1:-1] - integral[s - 1:-1, 0:w] + integral[0:h, 0:w]) / \
          math.pow(s, 2)

    return map


def crop_image(img, d=32):
    '''Make dimensions divisible by `d`'''
    imgsize = img.shape

    new_size = (imgsize[0] - imgsize[0] % d,
                imgsize[1] - imgsize[1] % d)

    bbox = [
        int((imgsize[0] - new_size[0]) / 2),
        int((imgsize[1] - new_size[1]) / 2),
        int((imgsize[0] + new_size[0]) / 2),
        int((imgsize[1] + new_size[1]) / 2),
    ]

    img_cropped = img[0:new_size[0], 0:new_size[1], :]
    return img_cropped


def get_params(opt_over, net, net_input, downsampler=None):
    '''Returns parameters that we want to optimize over.

    Args:
        opt_over: comma separated list, e.g. "net,input" or "net"
        net: network
        net_input: torch.Tensor that stores input `z`
    '''
    opt_over_list = opt_over.split(',')
    params = []

    for opt in opt_over_list:

        if opt == 'net':
            params += [x for x in net.parameters()]
        elif opt == 'down':
            assert downsampler is not None
            params += [x for x in downsampler.parameters()]
        elif opt == 'input':
            net_input.requires_grad = True
            params += [net_input]
        else:
            assert False, 'what is it?'

    return params


def load(path, channel):
    """Load PIL image."""
    if channel == 1:
        img = Image.open(path).convert('L')
    else:
        img = Image.open(path)

    return img


def get_image(path, imsize=-1, channel=1):
    """Load an image and resize to a cpecific size.

    Args:
        path: path to image
        imsize: tuple or scalar with dimensions; -1 for `no resize`
    """
    img = load(path, channel)

    if isinstance(imsize, int):
        imsize = (imsize, imsize)

    if imsize[0] != -1:
        img = img.resize(imsize)

    img_np = pil_to_np(img)
    img = np_to_torch(img_np)

    return img


def fill_noise(x, noise_type):
    """Fills tensor `x` with noise of type `noise_type`."""
    torch.manual_seed(0)
    if noise_type == 'u':
        x.uniform_()
    elif noise_type == 'n':
        x.normal_()
    else:
        assert False


def get_noise(net_input, spatial_size, input_channel, input_type='noise', noise_type='u', var=1. / 10,
              fourier_base=2 ** (8 / (8 - 1)), n_freqs=8):
    """Returns a pytorch.Tensor of size (1 x `input_depth` x `spatial_size[0]` x `spatial_size[1]`)
    initialized in a specific way.
    Args:
        input_depth: number of channels in the tensor
        method: `noise` for fillting tensor with noise; `meshgrid` for np.meshgrid
        spatial_size: spatial size of the tensor to initialize
        noise_type: 'u' for uniform; 'n' for normal
        var: a factor, a noise will be multiplicated by. Basically it is standard deviation scaler.
    """
    if isinstance(spatial_size, int):
        spatial_size = (spatial_size, spatial_size)
    if input_type == 'noise':
        shape = [1, input_channel, spatial_size[0], spatial_size[1]]
        # net_input = torch.zeros(shape)

        fill_noise(net_input, noise_type)
        net_input *= var
    elif input_type == 'meshgrid':
        assert input_channel == 2
        meshgrid = get_meshgrid(spatial_size)
        net_input = np_to_torch(meshgrid).type(torch.cuda.FloatTensor)
    elif input_type == 'fourier':
        freqs = fourier_base ** torch.linspace(0., n_freqs - 1, steps=n_freqs)
        net_input = generate_fourier_feature_maps(freqs, spatial_size, only_cosine=False)
        del freqs
    elif input_type == 'infer_freqs':
        net_input = fourier_base ** torch.linspace(0., n_freqs - 1, steps=n_freqs)
    else:
        assert False

    return net_input


def generate_fourier_feature_maps(net_input, spatial_size, only_cosine=False):
    X, Y = torch.meshgrid(torch.arange(0, spatial_size[0]) / float(spatial_size[0] - 1),
                          torch.arange(0, spatial_size[1]) / float(spatial_size[1] - 1))
    meshgrid = torch.stack([X, Y]).permute(1, 2, 0).unsqueeze(0).to(net_input.device)
    vp = net_input * torch.unsqueeze(meshgrid, -1)
    if only_cosine:
        vp_cat = torch.cat((torch.cos(vp),), dim=-1)
    else:
        vp_cat = torch.cat((torch.cos(vp), torch.sin(vp)), dim=-1)
    del X, Y
    return vp_cat.flatten(-2, -1).permute(0, 3, 1, 2)


def get_meshgrid(spatial_size):
    X, Y = np.meshgrid(np.arange(0, spatial_size[1]) / float(spatial_size[1] - 1),
                       np.arange(0, spatial_size[0]) / float(spatial_size[0] - 1))
    meshgrid = np.concatenate([X[None, :], Y[None, :]])
    return meshgrid


def pil_to_np(img_PIL):
    '''Converts image in PIL format to np.array.

    From W x H x C [0...255] to C x W x H [0..1]
    '''
    ar = np.array(img_PIL)

    if len(ar.shape) == 3:
        ar = ar.transpose(2, 0, 1)
    else:
        ar = ar[None, ...]

    return ar.astype(np.float32) / 255.


def np_to_pil(img_np):
    '''Converts image in np.array format to PIL image.

    From C x W x H [0..1] to  W x H x C [0...255]
    '''
    ar = np.clip(img_np * 255, 0, 255).astype(np.uint8)

    if img_np.shape[0] == 1:
        ar = ar[0]
    else:
        ar = ar.transpose(1, 2, 0)

    return Image.fromarray(ar)


def np_to_torch(img_np):
    '''Converts image in numpy.array to torch.Tensor.

    From C x W x H [0..1] to  C x W x H [0..1]
    '''
    if len(img_np.shape) == 3:
        return torch.from_numpy(img_np)[None, :]
    else:
        return torch.from_numpy(img_np)[None, None, :]


def torch_to_np(img_var, data_type='float'):
    '''Converts an image in torch.Tensor format to np.array.

    From 1 x C x W x H [0..1] to  C x W x H [0..1]
    '''
    img_var = img_var.detach().cpu().numpy()[0]
    if data_type == 'uint8' and img_var.shape[0] == 1:
        img_var = np.uint8(255 * img_var.squeeze())
    elif data_type == 'uint8' and img_var.shape[0] == 3:
        img_var = np.uint8(255 * img_var.transpose(1, 2, 0))
    return img_var


def read_img(path_to_image):
    img = cv2.imread(path_to_image)
    x = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(x)

    y = np_to_torch((y / 255.).astype(np.float32)).cuda()
    img = np_to_torch((img.transpose(2, 0, 1) / 255.).astype(np.float32)).cuda()

    return img, y, cb, cr


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


