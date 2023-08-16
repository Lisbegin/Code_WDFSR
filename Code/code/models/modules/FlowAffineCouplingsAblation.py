# Copyright (c) 2020 Huawei Technologies Co., Ltd.
# Licensed under CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike 4.0 International) (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#
# The code is released for academic research use only. For commercial use, please contact Huawei Technologies Co., Ltd.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file contains content licensed by https://github.com/chaiyujin/glow-pytorch/blob/master/LICENSE

import torch
from torch import nn as nn

from models.modules import thops
from models.modules.flow import Conv2d, Conv2dZeros
from utils.util import opt_get
from models.modules.CBAM import *
class Spatial_Attention_Module(nn.Module):
    def __init__(self, k: int):
        super(Spatial_Attention_Module, self).__init__()
        self.avg_pooling = torch.mean
        self.max_pooling = torch.max
        # In order to keep the size of the front and rear images consistent
        # with calculate, k = 1 + 2p, k denote kernel_size, and p denote padding number
        # so, when p = 1 -> k = 3; p = 2 -> k = 5; p = 3 -> k = 7, it works. when p = 4 -> k = 9, it is too big to use in network
        assert k in [3, 5, 7], "kernel size = 1 + 2 * padding, so kernel size must be 3, 5, 7"
        self.conv = nn.Conv2d(2, 1, kernel_size = (k, k), stride = (1, 1), padding = ((k - 1) // 2, (k - 1) // 2),
                              bias = False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # compress the C channel to 1 and keep the dimensions
        avg_x = self.avg_pooling(x, dim = 1, keepdim = True)
        max_x, _ = self.max_pooling(x, dim = 1, keepdim = True)
        v = self.conv(torch.cat((max_x, avg_x), dim = 1))
        v = self.sigmoid(v)
        return x * v

class Channel_Attention_Module_Conv(nn.Module):
    def __init__(self, channels, gamma = 2, b = 1):
        super(Channel_Attention_Module_Conv, self).__init__()
        kernel_size = int(abs((math.log(channels, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.max_pooling = nn.AdaptiveMaxPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size = kernel_size, padding = (kernel_size - 1) // 2, bias = False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_x = self.avg_pooling(x)
        max_x = self.max_pooling(x)
        avg_out = self.conv(avg_x.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        max_out = self.conv(max_x.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        v = self.sigmoid(avg_out + max_out)
        return x * v
class CondAffineSeparatedAndCond(nn.Module):
    def __init__(self, in_channels, opt):
        super().__init__()
        self.need_features = True
        self.in_channels = in_channels
        self.in_channels_rrdb = 320
        self.kernel_hidden = 1
        self.version=opt_get(opt, ['network_G', 'flow', 'version'],  1)
        self.affine_eps = 0.0001
        self.n_hidden_layers = 1
        hidden_channels = opt_get(opt, ['network_G', 'flow', 'CondAffineSeparatedAndCond', 'hidden_channels'])
        self.hidden_channels = 64 if hidden_channels is None else hidden_channels

        self.affine_eps = opt_get(opt, ['network_G', 'flow', 'CondAffineSeparatedAndCond', 'eps'],  0.0001)

        self.channels_for_nn = self.in_channels // 2
        self.channels_for_co = self.in_channels - self.channels_for_nn
        if self.channels_for_nn is None:
            self.channels_for_nn = self.in_channels // 2
        if self.version==1:
            self.model1=CBAMBlock("Conv", 3, channels = self.channels_for_co * 2, gamma = 2, b = 1)
            self.model2=CBAMBlock("Conv", 3, channels = self.hidden_channels, gamma = 2, b = 1)
        elif self.version==3:
            self.model1=nn.Sequential(CBAMBlock("Conv", 3, channels = self.channels_for_co * 2, gamma = 2, b = 1),CBAMBlock("Conv", 3, channels = self.channels_for_co * 2, gamma = 2, b = 1),CBAMBlock("Conv", 3, channels = self.channels_for_co * 2, gamma = 2, b = 1))
            self.model2=nn.Sequential(CBAMBlock("Conv", 3, channels = self.hidden_channels, gamma = 2, b = 1),CBAMBlock("Conv", 3, channels = self.hidden_channels, gamma = 2, b = 1),CBAMBlock("Conv", 3, channels = self.hidden_channels, gamma = 2, b = 1))
        elif self.version==2:
            #self.model3=CBAMBlock("Conv", 3, channels = 320, gamma = 2, b = 1)
            self.spatial1=Spatial_Attention_Module(k = 3)
            self.spatial2=Spatial_Attention_Module(k = 3)
            self.channel1=Channel_Attention_Module_Conv(channels = self.in_channels_rrdb, gamma = 2, b = 1)
            self.channel2=Channel_Attention_Module_Conv(channels = self.in_channels_rrdb, gamma = 2, b = 1)
            self.model1=None
            self.model2=None
        else:
            self.model1=None
            self.model2=None
            #self.model3=None
        if self.channels_for_nn is None:
            self.channels_for_nn = self.in_channels // 2
        
        
        self.fAffine = self.F(in_channels=self.channels_for_nn + self.in_channels_rrdb,
                              out_channels=self.channels_for_co * 2,
                              hidden_channels=self.hidden_channels,
                              kernel_hidden=self.kernel_hidden,
                              n_hidden_layers=self.n_hidden_layers,model=self.model1)

        self.fFeatures = self.F(in_channels=self.in_channels_rrdb,
                                out_channels=self.in_channels * 2,
                                hidden_channels=self.hidden_channels,
                                kernel_hidden=self.kernel_hidden,
                                n_hidden_layers=self.n_hidden_layers,model=self.model2)

    def forward(self, input: torch.Tensor, logdet=None, reverse=False, ft=None):
        if(self.version==2):
            #ft=self.model3(ft)
            ft2=ft
            ft1_after_channel=self.channel1(ft)
            ft1=self.spatial1(ft1_after_channel)
            ft2=self.channel2(ft2)+ft1_after_channel*0.1
            ft2=self.spatial2(ft2)+ft1*0.1
        if not reverse:
            z = input
            assert z.shape[1] == self.in_channels, (z.shape[1], self.in_channels)

            # Feature Conditional
            if self.version==1:
                scaleFt, shiftFt = self.feature_extract(ft, self.fFeatures)
            else:
                scaleFt, shiftFt = self.feature_extract(ft1, self.fFeatures)
            z = z + shiftFt
            z = z * scaleFt
            logdet = logdet + self.get_logdet(scaleFt)

            # Self Conditional
            z1, z2 = self.split(z)
            if self.version==1:
                scale, shift = self.feature_extract_aff(z1, ft, self.fAffine)
            else:
                scale, shift = self.feature_extract_aff(z1, ft2, self.fAffine)
            self.asserts(scale, shift, z1, z2)
            z2 = z2 + shift
            z2 = z2 * scale

            logdet = logdet + self.get_logdet(scale)
            z = thops.cat_feature(z1, z2)
            output = z
        else:
            z = input
           
            # Self Conditional
            z1, z2 = self.split(z)
            if self.version==1:
                scale, shift = self.feature_extract_aff(z1, ft, self.fAffine)
            else:
                scale, shift = self.feature_extract_aff(z1, ft2, self.fAffine)
            self.asserts(scale, shift, z1, z2)
            z2 = z2 / scale
            z2 = z2 - shift
            z = thops.cat_feature(z1, z2)
            logdet = logdet - self.get_logdet(scale)

            # Feature Conditional
            if self.version==1:
                scaleFt, shiftFt = self.feature_extract(ft, self.fFeatures)
            else:
                scaleFt, shiftFt = self.feature_extract(ft1, self.fFeatures)
            z = z / scaleFt
            z = z - shiftFt
            logdet = logdet - self.get_logdet(scaleFt)

            output = z
        return output, logdet

    def asserts(self, scale, shift, z1, z2):
        assert z1.shape[1] == self.channels_for_nn, (z1.shape[1], self.channels_for_nn)
        assert z2.shape[1] == self.channels_for_co, (z2.shape[1], self.channels_for_co)
        assert scale.shape[1] == shift.shape[1], (scale.shape[1], shift.shape[1])
        assert scale.shape[1] == z2.shape[1], (scale.shape[1], z1.shape[1], z2.shape[1])

    def get_logdet(self, scale):
        return thops.sum(torch.log(scale), dim=[1, 2, 3])

    def feature_extract(self, z, f):
        h = f(z)
        shift, scale = thops.split_feature(h, "cross")
        scale = (torch.sigmoid(scale + 2.) + self.affine_eps)
        return scale, shift

    def feature_extract_aff(self, z1, ft, f):
        z = torch.cat([z1, ft], dim=1)
        
        h = f(z)
        shift, scale = thops.split_feature(h, "cross")
        scale = (torch.sigmoid(scale + 2.) + self.affine_eps)
        return scale, shift

    def split(self, z):
        z1 = z[:, :self.channels_for_nn]
        z2 = z[:, self.channels_for_nn:]
        assert z1.shape[1] + z2.shape[1] == z.shape[1], (z1.shape[1], z2.shape[1], z.shape[1])
        return z1, z2

    def F(self, in_channels, out_channels, hidden_channels, kernel_hidden=1, n_hidden_layers=1,model=None):
        layers = [Conv2d(in_channels, hidden_channels), nn.ReLU(inplace=False)]

        for _ in range(n_hidden_layers):
            layers.append(Conv2d(hidden_channels, hidden_channels, kernel_size=[kernel_hidden, kernel_hidden]))
            layers.append(nn.ReLU(inplace=False))
        layers.append(Conv2dZeros(hidden_channels, out_channels))
        if(model!=None):
            layers.append(model)
        
        return nn.Sequential(*layers)
