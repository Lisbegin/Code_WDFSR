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

import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F
import os
from models.modules import thops

#qr
class InvertibleConv1x1(nn.Module):
    def __init__(self, num_channels, LU_decomposed=False):
        super().__init__()
        self.w_shape = [num_channels, num_channels]

        self.num_channels=num_channels
        np_s = np.ones(num_channels, dtype='float32')
        np_u = np.zeros((num_channels, num_channels), dtype='float32')
        np_u = np.triu(np_u, k=1).astype('float32')
        
        self.register_parameter("S", nn.Parameter(torch.Tensor(np_s).cuda()))
        self.register_parameter("U", nn.Parameter(torch.Tensor(np_u).cuda()))
        v_np = np.random.randn(num_channels , num_channels, 1).astype('float32')
        self.register_parameter("v", nn.Parameter(torch.Tensor(v_np).cuda()))

        # self.register_parameter("S", nn.Parameter(torch.Tensor(np_s)))
        # self.register_parameter("U", nn.Parameter(torch.Tensor(np_u)))
        # v_np = np.random.randn(num_channels , num_channels, 1).astype('float32')
        # self.register_parameter("v", nn.Parameter(torch.Tensor(v_np)))

    def forward(self, z, logdet=None, reverse=False):
        log_s = torch.log(torch.abs(self.S))
        u_mask = np.triu(np.ones(self.w_shape, dtype='float32'), 1)
        r = self.U * torch.Tensor(u_mask).cuda() + torch.diag(self.S)
        I = torch.eye(self.num_channels).cuda()
        # r = self.U * torch.Tensor(u_mask)+ torch.diag(self.S)
        # I = torch.eye(self.num_channels)
        q = I
        for i in range(self.num_channels):
            v=self.v[i]
            vT = torch.transpose(v,0,1)
            q_i = I - 2 * torch.matmul(v, vT) / torch.matmul(vT, v)
            q = torch.matmul(q, q_i)
        H,W=z.shape[2],z.shape[3]
        w = torch.matmul(q, r)
        if not reverse:
            w = w.view(self.num_channels,self.num_channels,1,1)
            
            z = F.conv2d(z, w)
            logdet += torch.sum(log_s) * (H*W)
            
            return z, logdet
        else:
            q_inv = torch.transpose(q,0,1)
            r_inv = torch.inverse(r)
            w_inv = torch.matmul(r_inv, q_inv)

            w_inv = w_inv.view(self.num_channels,self.num_channels,1,1)
            z = F.conv2d(z, w_inv)
            logdet -= torch.sum(log_s) * (H*W)

            return z, logdet
# class InvertibleConv1x1(nn.Module):
#     def __init__(self, num_channels, LU_decomposed=False):
#         super().__init__()
#         self.w_shape = [num_channels, num_channels]

#         self.num_channels=num_channels
#         np_s = np.ones(num_channels, dtype='float32')
#         np_u = np.zeros((num_channels, num_channels), dtype='float32')
#         np_u = np.triu(np_u, k=1).astype('float32')
        
#         self.register_parameter("S", nn.Parameter(torch.Tensor(np_s)))
#         self.register_parameter("U", nn.Parameter(torch.Tensor(np_u)))
#         v_np = np.random.randn(num_channels , num_channels, 1).astype('float32')
#         self.register_parameter("v", nn.Parameter(torch.Tensor(v_np)))

        
        

#     def forward(self, z, logdet=None, reverse=False):
#         log_s = torch.log(torch.abs(self.S))
#         u_mask = np.triu(np.ones(self.w_shape, dtype='float32'), 1)
#         r = self.U * torch.Tensor(u_mask) + torch.diag(self.S)
        
#         # Householder transformations
#         I = torch.eye(self.num_channels)
#         q = I
#         for i in range(self.num_channels):
#             v=self.v[i]
#             vT = torch.transpose(v,0,1)
#             q_i = I - 2 * torch.matmul(v, vT) / torch.matmul(vT, v)
#             q = torch.matmul(q, q_i)
#         H,W=z.shape[2],z.shape[3]
#         w = torch.matmul(q, r)
#         if not reverse:
#             w = w.view(self.num_channels,self.num_channels,1,1)
            
#             z = F.conv2d(z, w)
#             logdet += torch.sum(log_s) * (H*W)
            
#             return z, logdet
#         else:
#             q_inv = torch.transpose(q,0,1)
#             r_inv = torch.inverse(r)
#             w_inv = torch.matmul(r_inv, q_inv)

#             w_inv = w_inv.view(self.num_channels,self.num_channels,1,1)
#             z = F.conv2d(z, w_inv)
#             logdet -= torch.sum(log_s) * (H*W)

#             return z, logdet
