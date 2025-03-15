import torch
from torch import nn
import torch.nn.functional as F


class DoubleConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def forward(self, inputs):
        return self.layers(inputs)


class UNet(nn.Module):
    def __init__(self, in_channels=6, out_channels=4):
        super(UNet, self).__init__()
        self.drop_out4 = nn.Dropout2d(p=0.4)
        self.drop_out5 = nn.Dropout2d(p=0.4)

        self.conv1 = DoubleConv(in_channels=in_channels, out_channels=64)
        self.conv2 = DoubleConv(in_channels=64, out_channels=128)
        self.conv3 = DoubleConv(in_channels=128, out_channels=256)
        self.conv4 = DoubleConv(in_channels=256, out_channels=512)

        self.conv5 = DoubleConv(in_channels=512, out_channels=1024)

        self.conv6 = DoubleConv(in_channels=512 + 1024, out_channels=512)
        self.conv7 = DoubleConv(in_channels=256 + 512, out_channels=256)
        self.conv8 = DoubleConv(in_channels=128 + 256, out_channels=128)
        self.conv9 = DoubleConv(in_channels=64 + 128, out_channels=64)

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
        if out_channels == 5:
            self.tail3 = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1)
            )
        else:
            self.tail3 = None

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

    def add(self, inputs):
        inputs_shapes2 = [x.shape[2] for x in inputs]
        inputs_shapes3 = [x.shape[3] for x in inputs]

        target_shape2 = min(inputs_shapes2)
        target_shape3 = min(inputs_shapes3)

        inputs_ = []
        for inp in inputs:
            diff2 = (inp.size(2) - target_shape2) // 2
            diff3 = (inp.size(3) - target_shape3) // 2
            inputs_.append(inp[:, :, diff2: diff2 + target_shape2, diff3:diff3 + target_shape3])

        res = 0
        for x in inputs_:
            res += x
        return res

    def forward(self, input1, input2, features_dec=None, features_enc=None):
        input = torch.cat((input1, input2), dim=1)
        input = input + features_dec[0] if features_enc is None and features_dec is not None else input
        conv1 = self.conv1(input)

        conv2 = F.max_pool2d(conv1, kernel_size=2, stride=2, padding=(conv1.shape[2] % 2, conv1.shape[3] % 2))
        if features_enc is not None:
            conv2 = conv2 + features_enc[0]
        elif features_dec is not None:
            conv2 = conv2 + features_dec[1]
        conv2 = self.conv2(conv2)

        conv3 = F.max_pool2d(conv2, kernel_size=2, stride=2, padding=(conv2.shape[2] % 2, conv2.shape[3] % 2))
        if features_enc is not None:
            conv3 = conv3 + features_enc[1]
        elif features_dec is not None:
            conv3 = conv3 + features_dec[2]
        conv3 = self.conv3(conv3)

        conv4 = F.max_pool2d(conv3, kernel_size=2, stride=2, padding=(conv3.shape[2] % 2, conv3.shape[3] % 2))
        if features_enc is not None:
            conv4 = conv4 + features_enc[2]
        elif features_dec is not None:
            conv4 = conv4 + features_dec[3]
        conv4 = self.conv4(conv4)
        drop_4 = self.drop_out4(conv4)

        conv5 = F.max_pool2d(drop_4, kernel_size=2, stride=2, padding=(drop_4.shape[2] % 2, drop_4.shape[3] % 2))
        conv5 = self.conv5(conv5)

        conv6 = self.conv6(self.concat((F.interpolate(self.drop_out5(conv5), scale_factor=2), drop_4), dim=1))
        conv7 = self.conv7(self.concat((F.interpolate(conv6, scale_factor=2), conv3), dim=1))
        if features_enc is not None:
            conv7 = conv7 + features_dec[2]
        conv8 = self.conv8(self.concat((F.interpolate(conv7, scale_factor=2), conv2), dim=1))
        if features_enc is not None:
            conv8 = conv8 + features_dec[1]
        conv9 = self.conv9(self.concat((F.interpolate(conv8, scale_factor=2), conv1), dim=1))
        if features_enc is not None:
            conv9 = conv9 + features_dec[0]

        out_x = torch.sigmoid(self.tail1(conv9))
        out_m1 = torch.sigmoid(self.tail2(conv9))
        out_m2 = torch.sigmoid(self.tail3(conv9)) if self.tail3 is not None else 1-out_m1

        return out_x, out_m1, out_m2


    def get_feats(self, input1, input2):
        feats_enc = []
        input = torch.cat((input1, input2), dim=1)
        conv1 = self.conv1(input)

        conv2 = F.max_pool2d(conv1, kernel_size=2, stride=2, padding=(conv1.shape[2] % 2, conv1.shape[3] % 2))
        feats_enc.append(conv2)
        conv2 = self.conv2(conv2)

        conv3 = F.max_pool2d(conv2, kernel_size=2, stride=2, padding=(conv2.shape[2] % 2, conv2.shape[3] % 2))
        feats_enc.append(conv3)
        conv3 = self.conv3(conv3)

        conv4 = F.max_pool2d(conv3, kernel_size=2, stride=2, padding=(conv3.shape[2] % 2, conv3.shape[3] % 2))
        feats_enc.append(conv4)
        conv4 = self.conv4(conv4)
        drop_4 = self.drop_out4(conv4)

        conv5 = F.max_pool2d(drop_4, kernel_size=2, stride=2, padding=(drop_4.shape[2] % 2, drop_4.shape[3] % 2))
        feats_enc.append(conv5)
        conv5 = self.conv5(conv5)

        conv6 = self.conv6(self.concat((F.interpolate(self.drop_out5(conv5), scale_factor=2), drop_4), dim=1))
        conv7 = self.conv7(self.concat((F.interpolate(conv6, scale_factor=2), conv3), dim=1))
        conv8 = self.conv8(self.concat((F.interpolate(conv7, scale_factor=2), conv2), dim=1))
        conv9 = self.conv9(self.concat((F.interpolate(conv8, scale_factor=2), conv1), dim=1))
        return feats_enc, [conv9, conv8, conv7, conv6]
