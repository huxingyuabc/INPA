import torch
import torch.nn as nn

from .common import act, bn, conv, Concat
from .non_local_dot_product import NONLocalBlock2D
import torch.nn.functional as F


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class PositionAttention(nn.Module):
    """ Position attention module"""

    def __init__(self, in_channels, **kwargs):
        super(PositionAttention, self).__init__()
        self.conv_b = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv_c = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv_d = nn.Conv2d(in_channels, in_channels, 1)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        feat_b = self.conv_b(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        feat_c = self.conv_c(x).view(batch_size, -1, height * width)
        attention_s = self.softmax(torch.bmm(feat_b, feat_c))
        feat_d = self.conv_d(x).view(batch_size, -1, height * width)
        feat_e = torch.bmm(feat_d, attention_s.permute(0, 2, 1)).view(batch_size, -1, height, width)
        out = self.alpha * feat_e + x

        return out


class ChannelAttention(nn.Module):
    """Channel attention module"""

    def __init__(self, **kwargs):
        super(ChannelAttention, self).__init__()
        self.beta = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        feat_a = x.view(batch_size, -1, height * width)
        feat_a_transpose = x.view(batch_size, -1, height * width).permute(0, 2, 1)
        attention = torch.bmm(feat_a, feat_a_transpose)
        attention_new = torch.max(attention, dim=-1, keepdim=True)[0].expand_as(attention) - attention
        attention = self.softmax(attention_new)

        feat_e = torch.bmm(attention, feat_a).view(batch_size, -1, height, width)
        out = self.beta * feat_e + x

        return out


class DualAttention(nn.Module):
    def __init__(self, in_channels, norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
        super(DualAttention, self).__init__()
        inter_channels = in_channels
        # inter_channels = in_channels // 4
        self.pam = PositionAttention(inter_channels, **kwargs)
        self.cam = ChannelAttention(**kwargs)
        self.conv_p1 = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.conv_c1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.out = SELayer(inter_channels)

    def forward(self, x):
        feat_p = self.pam(x)
        feat_p = self.conv_p1(feat_p)

        feat_c = self.cam(x)
        feat_c = self.conv_c1(feat_c)

        feat_fusion = feat_p + feat_c

        fusion_out = self.out(feat_fusion)
        outputs = fusion_out

        return outputs


class lst_skip(nn.Module):
    def __init__(self, input_channels=3,
                 channels=[16, 32, 64, 128, 128],
                 channels_skip=4,
                 filter_size=3,
                 filter_skip_size=1,
                 need_bias=True,
                 pad='zeros',
                 upsample_mode='nearest',
                 downsample_mode='stride',
                 attention_mode=None,
                 fuse_type='weighted',
                 act_fun='LeakyReLU',
                 scales=1,
                 scales_k=4):

        """Assembles encoder-decoder with skip connections.

        Arguments:
            act_fun: Either string 'LeakyReLU|Swish|ELU|none' or module (e.g. nn.ReLU)
            pad (string): zero|reflection (default: 'zero')
            upsample_mode (string): 'nearest|bilinear' (default: 'nearest')
            downsample_mode (string): 'stride|avg|max|lanczos2' (default: 'stride')

        """
        super(lst_skip, self).__init__()
        self.len = len(channels)

        self.down_conv1 = nn.ModuleList()
        self.down_bn = nn.ModuleList()
        self.down_conv2 = nn.ModuleList()
        self.att = nn.ModuleList()

        self.skip_conv = nn.ModuleList()
        self.up_bn1 = nn.ModuleList()
        self.up_conv = nn.ModuleList()
        self.up_bn2 = nn.ModuleList()

        self.act = act(act_fun)
        self.skip_bn = bn(channels_skip)
        self.upsample = nn.Upsample(scale_factor=2, mode=upsample_mode)
        self.upsample_mode = upsample_mode
        self.sigmoid = nn.Sigmoid()
        self.scales = scales
        self.fuse_type = fuse_type
        self.scales_k = scales_k
        if self.fuse_type in ['weighted', 'weighted1']:
            self.alpha = nn.ParameterList()
            self.beta = nn.ParameterList()

        self.tail1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1)
        )
        self.tail2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1)
        )

        for i in range(self.len):
            if i == 0:
                self.down_conv1.append(conv(input_channels, channels[i], filter_size, stride=2, bias=need_bias,
                                            pad=pad, downsample_mode=downsample_mode))
            else:
                self.down_conv1.append(conv(channels[i - 1], channels[i], filter_size, stride=2, bias=need_bias,
                                            pad=pad, downsample_mode=downsample_mode))
            self.down_bn.append(bn(channels[i]))
            if i > 1 and attention_mode == 'non-local':
                self.att.append(NONLocalBlock2D(channels[i]))
            elif i > 1 and attention_mode == 'dual-attention':
                self.att.append(DualAttention(channels[i]))
            self.down_conv2.append(conv(channels[i], channels[i], filter_size, bias=need_bias, pad=pad))
            self.skip_conv.append(conv(channels[i], channels_skip, filter_skip_size, bias=need_bias, pad=pad))
            if self.fuse_type == 'weighted':
                self.alpha.append(nn.Parameter(torch.tensor(0, dtype=torch.float32), requires_grad=True))
            if self.fuse_type == 'weighted1':
                self.alpha.append(nn.Parameter(torch.tensor(1, dtype=torch.float32), requires_grad=True))

            if i == self.len - 1:
                self.up_bn1.append(bn(channels_skip + channels[i]))
                self.up_conv.append(conv(channels_skip + channels[i], channels[i], filter_size, bias=need_bias,
                                         pad=pad))
            else:
                self.up_bn1.append(bn(channels_skip + channels[i + 1]))
                self.up_conv.append(conv(channels_skip + channels[i + 1], channels[i], filter_size, bias=need_bias,
                                         pad=pad))
            self.up_bn2.append(bn(channels[i]))
            if self.fuse_type == 'weighted':
                self.beta.append(nn.Parameter(torch.tensor(0, dtype=torch.float32), requires_grad=True))
            if self.fuse_type == 'weighted1':
                self.beta.append(nn.Parameter(torch.tensor(1, dtype=torch.float32), requires_grad=True))

    def load_tail(self, load_path, strict=True, param_key='net'):
        load_net = torch.load(load_path, map_location=lambda storage, loc: storage)
        if param_key is not None:
            load_net = load_net[param_key]
        # remove unnecessary 'module.'
        state_dict = {k[6:]: v for k, v in load_net.items() if 'tail1' in k.split('.')}
        self.tail1.load_state_dict(state_dict, strict=strict)
        state_dict = {k[6:]: v for k, v in load_net.items() if 'tail2' in k.split('.')}
        self.tail2.load_state_dict(state_dict, strict=strict)
        # fix
        for param in self.tail1.parameters():
            param.requires_grad = False
        for param in self.tail2.parameters():
            param.requires_grad = False

    def concat(self, inputs, dim):

        inputs_shapes2 = [x.shape[2] for x in inputs]
        inputs_shapes3 = [x.shape[3] for x in inputs]

        target_shape2 = min(inputs_shapes2)
        target_shape3 = min(inputs_shapes3)

        inputs_ = []
        for inp in inputs:
            diff2 = (inp.size(2) - target_shape2) // 2
            diff3 = (inp.size(3) - target_shape3) // 2
            inputs_.append(inp[:, :, diff2: diff2 + target_shape2, diff3:diff3 + target_shape3])

        return torch.cat(inputs_, dim=dim)

    def forward(self, inputs, pre_feats_enc, pre_feats_dec):
        out_down = [0, 0, 0, 0, 0]
        out_skip = [0, 0, 0, 0, 0]
        out_up = [0, 0, 0, 0, 0]
        for i in range(self.len):
            if i == 0:
                out_down[i] = self.down_conv1[i](inputs)
            else:
                out_down[i] = self.down_conv1[i](out_down[i - 1])
            out_down[i] = self.down_bn[i](out_down[i])
            out_down[i] = self.act(out_down[i])
            if i > 1 and len(self.att) > 0:
                out_down[i] = self.att[i-2](out_down[i])
            out_down[i] = self.down_conv2[i](out_down[i])
            out_down[i] = self.down_bn[i](out_down[i])
            out_down[i] = self.act(out_down[i])
            if self.fuse_type == 'weighted':
                out_down[i] = self.alpha[i] * out_down[i] + pre_feats_enc[i] if i < self.scales_k//2 else out_down[i]
            if self.fuse_type == 'weighted1':
                out_down[i] = out_down[i] + self.alpha[i] * pre_feats_enc[i] if i < self.scales_k//2 else out_down[i]
            else:
                out_down[i] = out_down[i] + pre_feats_enc[i]

            out_skip[i] = self.skip_conv[i](out_down[i])
            out_skip[i] = self.skip_bn(out_skip[i])
            out_skip[i] = self.act(out_skip[i])

        for i in range(self.len - 1, -1, -1):
            if i == self.len - 1:
                out_up[i] = self.up_bn1[i](self.concat([out_skip[i], out_down[i]], dim=1))
            else:
                out_up[i] = self.up_bn1[i](self.concat([out_skip[i], out_up[i + 1]], dim=1))
            if i == 0:
                out_up[i] = F.interpolate(out_up[i], inputs.shape[2:], mode=self.upsample_mode)
            else:
                out_up[i] = F.interpolate(out_up[i], out_down[i-1].shape[2:], mode=self.upsample_mode)
            out_up[i] = self.up_conv[i](out_up[i])
            out_up[i] = self.up_bn2[i](out_up[i])
            out_up[i] = self.act(out_up[i])
            if self.fuse_type == 'weighted':
                out_up[i] = self.beta[i] * out_up[i] + pre_feats_dec[i] if i < self.scales_k//2 else out_up[i]
            elif self.fuse_type == 'weighted1':
                out_up[i] = out_up[i] + self.beta[i] * pre_feats_dec[i] if i < self.scales_k//2 else out_up[i]
            else:
                out_up[i] = out_up[i] + pre_feats_dec[i]

        out_x = self.sigmoid(self.tail1(out_up[0]))
        out_m1 = self.sigmoid(self.tail2(out_up[0]))

        if self.fuse_type in ['weighted', 'weighted1']:
            return out_x, out_m1, 1 - out_m1, self.alpha, self.beta
        else:
            return out_x, out_m1, 1 - out_m1
