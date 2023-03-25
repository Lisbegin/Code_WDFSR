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
# import pycuda.gpuarray as gpuarray
# import pycuda.autoinit
# import skcuda.linalg as sklin

from torch.linalg import det
#QR
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
        

    def forward(self, z, logdet=None, reverse=False):
        log_s = torch.log(torch.abs(self.S))
        u_mask = np.triu(np.ones(self.w_shape, dtype='float32'), 1)
        r = self.U * torch.Tensor(u_mask).cuda() + torch.diag(self.S)
        
        # Householder transformations
        I = torch.eye(self.num_channels).cuda()
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



# #1XX1---------------------------
# class InvertibleConv1x1(nn.Module):
#     def __init__(self, num_channels, LU_decomposed=False):
#         super().__init__()
#         w_shape = [num_channels, num_channels]
#         w_init = np.linalg.qr(np.random.randn(*w_shape))[0].astype(np.float32)
#         if not LU_decomposed:
#             # Sample a random orthogonal matrix:
#             self.register_parameter("weight", nn.Parameter(torch.Tensor(w_init)))
#         else:
#             # W = PL(U+diag(s))
#             np_p, np_l, np_u = scipy.linalg.lu(w_init)
#             np_s = np.diag(np_u)
#             np_sign_s = np.sign(np_s)
#             np_log_s = np.log(np.abs(np_s))
#             np_u = np.triu(np_u, k=1)
#             l_mask = np.tril(np.ones(w_shape, dtype=np.float32), -1)
#             eye = np.eye(*w_shape, dtype=np.float32)

#             self.register_buffer('p', torch.Tensor(np_p.astype(np.float32))) # remains fixed
#             self.register_buffer('sign_s', torch.Tensor(np_sign_s.astype(np.float32))) # the sign is fixed
#             self.l = nn.Parameter(torch.Tensor(np_l.astype(np.float32))) # optimized except diagonal 1
#             self.log_s = nn.Parameter(torch.Tensor(np_log_s.astype(np.float32)))
#             self.u = nn.Parameter(torch.Tensor(np_u.astype(np.float32))) # optimized
#             self.l_mask = torch.Tensor(l_mask)
#             self.eye = torch.Tensor(eye)
#         self.w_shape = w_shape
#         self.LU = LU_decomposed

#     def get_weight(self, input, reverse):
#         # The difference in computational cost will become significant for large c, although for the networks in
#         # our experiments we did not measure a large difference in wallclock computation time.
#         if not self.LU:
#             if not reverse:
#                 # pixels = thops.pixels(input)
#                 # GPU version
#                 # dlogdet = torch.slogdet(self.weight)[1] * pixels
#                 # CPU version is 2x faster, https://github.com/didriknielsen/survae_flows/issues/5.
#                 dlogdet = (torch.slogdet(self.weight.to('cpu'))[1] * thops.pixels(input)).to(self.weight.device)
#                 weight = self.weight.view(self.w_shape[0], self.w_shape[1], 1, 1)
#             else:
#                 dlogdet = 0
#                 weight = torch.inverse(self.weight.double()).float().view(self.w_shape[0], self.w_shape[1], 1, 1)


#             return weight, dlogdet
#         else:
#             self.p = self.p.to(input.device)
#             self.sign_s = self.sign_s.to(input.device)
#             self.l_mask = self.l_mask.to(input.device)
#             self.eye = self.eye.to(input.device)
#             l = self.l * self.l_mask + self.eye
#             u = self.u * self.l_mask.transpose(0, 1).contiguous() + torch.diag(self.sign_s * torch.exp(self.log_s))
#             dlogdet = thops.sum(self.log_s) * thops.pixels(input)
#             if not reverse:
#                 w = torch.matmul(self.p, torch.matmul(l, u))
#             else:
#                 l = torch.inverse(l.double()).float()
#                 u = torch.inverse(u.double()).float()
#                 w = torch.matmul(u, torch.matmul(l, self.p.inverse()))
#             return w.view(self.w_shape[0], self.w_shape[1], 1, 1), dlogdet

#     def forward(self, input, logdet=None, reverse=False):
#         """
#         log-det = log|abs(|W|)| * pixels
#         """
#         weight, dlogdet = self.get_weight(input, reverse)
#         if not reverse:
#             z = F.conv2d(input, weight) # fc layer, ie, permute channel
#             if logdet is not None:
#                 logdet = logdet + dlogdet
#             return z, logdet
#         else:
#             z = F.conv2d(input, weight)
#             if logdet is not None:
#                 logdet = logdet - dlogdet
#             return z, logdet
