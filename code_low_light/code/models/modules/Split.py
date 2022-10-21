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
import math
from models.modules import thops
from models.modules.FlowStep import FlowStep
from models.modules.flow import Conv2dZeros, GaussianDiag
from utils.util import opt_get
import numpy as np
import models.modules.thops as thops
import scipy.special
import torch

class Split2d(nn.Module):
    def __init__(self, num_channels, logs_eps=0, cond_channels=0, position=None, consume_ratio=0.5, opt=None):
        super().__init__()

        self.num_channels_consume = int(math.floor(num_channels * consume_ratio))
        self.num_channels_pass = num_channels - self.num_channels_consume

        self.conv = Conv2dZeros(in_channels=self.num_channels_pass + cond_channels,
                                out_channels=self.num_channels_consume * 2)
        self.logs_eps = logs_eps
        self.position = position
        self.studentZ=StudentT(20)
        self.opt = opt
        self.heat=opt['heat']
        self.V=float(opt['V'])
    def split2d_prior(self, z, ft):
        if ft is not None:
            z = torch.cat([z, ft], dim=1)
        h = self.conv(z)
        return thops.split_feature(h, "cross")

    def exp_eps(self, logs):
        return torch.exp(logs) + self.logs_eps

    def forward(self, input, logdet=0., reverse=False, eps_std=None, eps=None, ft=None, y_onehot=None):
        if not reverse:
            # self.input = input
            z1, z2 = self.split_ratio(input)
            mean, logs = self.split2d_prior(z1, ft)
            
            eps = (z2 - mean) / self.exp_eps(logs)

            #logdet = logdet + self.get_logdet(logs, mean, z2)
            logdet = logdet + self.studentZ.logp(z2)
            # print(logs.shape, mean.shape, z2.shape)
            # self.eps = eps
            # print('split, enc eps:', eps)
            return z1, logdet, eps
        else:
            z1 = input
            mean, logs = self.split2d_prior(z1, ft)

            if eps is None:
                #print("WARNING: eps is None, generating eps untested functionality!")
                eps = GaussianDiag.sample_eps(mean.shape, eps_std)

            eps = eps.to(mean.device)
            z2 = mean + self.exp_eps(logs) * eps
            #z2=self.studentZ.sample(z2.shape,0.3,device=z1.device)
            #这里是一个改进的点
            logdet = logdet - self.studentZ.logp(z2)
            z2 = torch.normal(mean=0, std=self.heat, size=z2.shape).to(z1.device)
            z2=z2+self.V
            z = thops.cat_feature(z1, z2)
            #logdet = logdet - self.get_logdet(logs, mean, z2)


            return z, logdet
            # return z, logdet, eps

    def get_logdet(self, logs, mean, z2):
        logdet_diff = GaussianDiag.logp(mean, logs, z2)
        # print("Split2D: logdet diff", logdet_diff.item())
        return logdet_diff

    def split_ratio(self, input):
        z1, z2 = input[:, :self.num_channels_pass, ...], input[:, self.num_channels_pass:, ...]
        return z1, z2

class StudentT:
    def __init__(self, df):
        self.df=df
    def logp(self,x):
        '''
        Multivariate t-student density:
        output:
            the sum density of the given element
        '''
        #df=100
        #d=x.shape[1]
        #norm_const = scipy.special.loggamma(0.5*(df+d))-scipy.special.loggamma(0.5*df)-0.5*d*np.log(np.pi*df)
        #import pdb; pdb.set_trace()
        d=x.shape[1]
        norm_const = scipy.special.loggamma(0.5*(self.df+d))-scipy.special.loggamma(0.5*self.df)-0.5*d*np.log(np.pi*self.df)       
        x_norms = thops.sum(((x) ** 2), dim=[1])
        likelihood = norm_const-0.5*(self.df+d)*torch.log(1+(1/self.df)*x_norms)
        
        return thops.sum(likelihood, dim=[1,2])

    def sample(self,z_shape, eps_std=None, device=None):
        '''generate random variables of multivariate t distribution
        Parameters
        ----------
        m : array_like
            mean of random variable, length determines dimension of random variable
        S : array_like
            square array of covariance  matrix
        df : int or float
            degrees of freedom
        n : int
            number of observations, return random array will be (n, len(m))
        Returns
        -------
        rvs : ndarray, (n, len(m))
            each row is an independent draw of a multivariate t distributed
            random variable
        '''
        #df=100
        #import pdb; pdb.set_trace()
        x_shape = torch.Size((z_shape[0], 1, z_shape[2],z_shape[3]))
        x = np.random.chisquare(self.df, z_shape)/self.df  
        #x = np.tile(x, (1,z_shape[1],1,1))
        x = torch.Tensor(x.astype(np.float32))
        z = torch.normal(mean=torch.zeros(z_shape),std=torch.ones(z_shape) * eps_std)
        

        return (z/torch.sqrt(x)).to(device)