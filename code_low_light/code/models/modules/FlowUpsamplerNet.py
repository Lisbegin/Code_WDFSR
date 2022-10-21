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

from audioop import reverse
import numpy as np
import torch
from torch import nn as nn

import models.modules.Split
from models.modules import flow, thops
from models.modules.Split import Split2d
from models.modules.glow_arch import f_conv2d_bias
from models.modules.FlowStep import FlowStep
from utils.util import opt_get


class FlowUpsamplerNet(nn.Module):
    def __init__(self, image_shape, hidden_channels, K, L=None,
                 actnorm_scale=1.0,
                 flow_permutation=None,
                 flow_coupling="affine",
                 LU_decomposed=False, opt=None):

        super().__init__()

        self.layers1 = nn.ModuleList()
        self.layers2 = nn.ModuleList()
        self.layers3 = nn.ModuleList()
        self.layers4 = nn.ModuleList()
        self.L = opt_get(opt, ['network_G', 'flow', 'L'])
        self.K1 = opt_get(opt, ['network_G', 'flow', 'K1'])
        self.K2 = opt_get(opt, ['network_G', 'flow', 'K2'])
        self.K3 = opt_get(opt, ['network_G', 'flow', 'K3'])
        self.K4 = opt_get(opt, ['network_G', 'flow', 'K4'])
        if isinstance(self.K1, int):
            self.K1 = [K1 for K1 in [self.K1, ] * (self.L + 1)]
        
        if isinstance(self.K2, int):
            self.K2 = [K2 for K2 in [self.K2, ] * (self.L + 1)]
        if isinstance(self.K3, int):
            self.K3 = [K3 for K3 in [self.K3, ] * (self.L + 1)]
        if isinstance(self.K4, int):
            self.K4 = [K4 for K4 in [self.K4, ] * (self.L + 1)]
        self.opt = opt
        H, W, self.C = image_shape
        self.check_image_shape()

        if opt['scale'] == 16:
            self.levelToName = {
                0: 'fea_up16',
                1: 'fea_up8',
                2: 'fea_up4',
                3: 'fea_up2',
                4: 'fea_up1',
            }

        if opt['scale'] == 8:
            self.levelToName = {
                0: 'fea_up8',
                1: 'fea_up4',
                2: 'fea_up2',
                3: 'fea_up1',
                4: 'fea_up0'
            }

        elif opt['scale'] == 4:
            self.levelToName = {
                0: 'fea_up1',
                1: 'fea_up2',
                2: 'fea_up4',
                3: 'fea_up8',
                4: 'fea_up0'
            }

        affineInCh = self.get_affineInCh(opt_get)
        flow_permutation = self.get_flow_permutation(flow_permutation, opt)

        normOpt = opt_get(opt, ['network_G', 'flow', 'norm'])

        conditional_channels = {}
        n_rrdb = self.get_n_rrdb_channels(opt, opt_get)
        n_bypass_channels = opt_get(opt, ['network_G', 'flow', 'levelConditional', 'n_channels'])
        conditional_channels[0] = n_rrdb
        for level in range(1, self.L + 1):
            n_bypass = 0 if n_bypass_channels is None else (self.L - level) * n_bypass_channels
            conditional_channels[level] = n_rrdb + n_bypass

        # Upsampler
        self.output_shapes1=[]
        self.output_shapes2=[]
        self.output_shapes3=[]
        self.output_shapes4=[]
       
        self.Haar=flow.HaarDownsampling(self.C)
        H=H//2
        W=W//2
        self.temp=3
        self.temp_H=H
        self.temp_W=W
        #Haar变换之后 第一个
        #                  (2,4)
        for level in range(1, self.L+1):
            # 1. Squeeze 
            if level!=1:
                H, W = self.arch_squeeze(H, W,type=1)
            # 2. K FlowStep
            self.arch_additionalFlowAffine(H, LU_decomposed, W, actnorm_scale, hidden_channels, opt,type=1)
            self.arch_FlowStep(H, self.K1[level], LU_decomposed, W, actnorm_scale, affineInCh, flow_coupling,
                               flow_permutation,
                               hidden_channels, normOpt, opt, opt_get,
                               n_conditinal_channels=conditional_channels[level],type=1)
            # Split
            self.arch_split(H, W, level, self.L, opt, opt_get,type=1)

        self.C=self.temp
        H=self.temp_H
        W=self.temp_W
        #Haar变换之后 后三个
        for level in range(1, self.L+1):
            # 1. Squeeze  
            if level!=1:
                H, W = self.arch_squeeze(H, W,type=2)

            # 2. K FlowStep
            self.arch_additionalFlowAffine(H, LU_decomposed, W, actnorm_scale, hidden_channels, opt,type=2)
            self.arch_FlowStep(H, self.K2[level], LU_decomposed, W, actnorm_scale, affineInCh, flow_coupling,
                               flow_permutation,
                               hidden_channels, normOpt, opt, opt_get,
                               n_conditinal_channels=conditional_channels[level],type=2)
            # Split
            self.arch_split(H, W, level, self.L, opt, opt_get,type=2)
        self.C=self.temp
        H=self.temp_H
        W=self.temp_W
        for level in range(1, self.L+1):
            # 1. Squeeze  
            if level!=1:
                H, W = self.arch_squeeze(H, W,type=3)

            # 2. K FlowStep
            self.arch_additionalFlowAffine(H, LU_decomposed, W, actnorm_scale, hidden_channels, opt,type=3)
            self.arch_FlowStep(H, self.K2[level], LU_decomposed, W, actnorm_scale, affineInCh, flow_coupling,
                               flow_permutation,
                               hidden_channels, normOpt, opt, opt_get,
                               n_conditinal_channels=conditional_channels[level],type=3)
            # Split
            self.arch_split(H, W, level, self.L, opt, opt_get,type=3)
        self.C=self.temp
        H=self.temp_H
        W=self.temp_W
        for level in range(1, self.L+1):
            # 1. Squeeze  
            if level!=1:
                H, W = self.arch_squeeze(H, W,type=4)

            # 2. K FlowStep
            self.arch_additionalFlowAffine(H, LU_decomposed, W, actnorm_scale, hidden_channels, opt,type=4)
            self.arch_FlowStep(H, self.K2[level], LU_decomposed, W, actnorm_scale, affineInCh, flow_coupling,
                               flow_permutation,
                               hidden_channels, normOpt, opt, opt_get,
                               n_conditinal_channels=conditional_channels[level],type=4)
            # Split
            self.arch_split(H, W, level, self.L, opt, opt_get,type=4)






        if opt_get(opt, ['network_G', 'flow', 'split', 'enable']):
            self.f = f_conv2d_bias(affineInCh, 2 * 3 * 64 // 2 // 2)
        else:
            self.f = f_conv2d_bias(affineInCh, 2 * 3 * 64)

        self.H = H
        self.W = W
        self.scaleH = 160 / H
        self.scaleW = 160 / W

    def get_n_rrdb_channels(self, opt, opt_get):
        blocks = opt_get(opt, ['network_G', 'flow', 'stackRRDB', 'blocks'])
        n_rrdb = 64 if blocks is None else (len(blocks) + 1) * 64
        return n_rrdb

    def arch_FlowStep(self, H, K, LU_decomposed, W, actnorm_scale, affineInCh, flow_coupling, flow_permutation,
                      hidden_channels, normOpt, opt, opt_get, n_conditinal_channels=None,type=None):
        condAff = self.get_condAffSetting(opt, opt_get)
        if condAff is not None:
            condAff['in_channels_rrdb'] = n_conditinal_channels

        for k in range(K):
            position_name = get_position_name(H, self.opt['scale'])
            if normOpt: normOpt['position'] = position_name

            
            if type==1:
                self.layers1.append(
                FlowStep(in_channels=self.C,
                         hidden_channels=hidden_channels,
                         actnorm_scale=actnorm_scale,
                         flow_permutation=flow_permutation,
                         flow_coupling=flow_coupling,
                         acOpt=condAff,
                         position=position_name,
                         LU_decomposed=LU_decomposed, opt=opt, idx=k, normOpt=normOpt,in_shape=(self.C, H, W)))
               
                self.output_shapes1.append([-1, self.C, H, W])
            elif type==2:
                self.layers2.append(
                FlowStep(in_channels=self.C,
                         hidden_channels=hidden_channels,
                         actnorm_scale=actnorm_scale,
                         flow_permutation=flow_permutation,
                         flow_coupling=flow_coupling,
                         acOpt=condAff,
                         position=position_name,
                         LU_decomposed=LU_decomposed, opt=opt, idx=k, normOpt=normOpt,in_shape=(self.C, H, W)))
                self.output_shapes2.append([-1, self.C, H, W])
            elif type==3:
                self.layers3.append(
                FlowStep(in_channels=self.C,
                         hidden_channels=hidden_channels,
                         actnorm_scale=actnorm_scale,
                         flow_permutation=flow_permutation,
                         flow_coupling=flow_coupling,
                         acOpt=condAff,
                         position=position_name,
                         LU_decomposed=LU_decomposed, opt=opt, idx=k, normOpt=normOpt,in_shape=(self.C, H, W)))
                self.output_shapes3.append([-1, self.C, H, W])
            else:    
                self.layers4.append(
                FlowStep(in_channels=self.C,
                         hidden_channels=hidden_channels,
                         actnorm_scale=actnorm_scale,
                         flow_permutation=flow_permutation,
                         flow_coupling=flow_coupling,
                         acOpt=condAff,
                         position=position_name,
                         LU_decomposed=LU_decomposed, opt=opt, idx=k, normOpt=normOpt,in_shape=(self.C, H, W)))
                self.output_shapes4.append([-1, self.C, H, W])

    def get_condAffSetting(self, opt, opt_get):
        condAff = opt_get(opt, ['network_G', 'flow', 'condAff']) or None
        condAff = opt_get(opt, ['network_G', 'flow', 'condFtAffine']) or condAff
        return condAff

    def arch_split(self, H, W, L, levels, opt, opt_get,type):
        correct_splits = opt_get(opt, ['network_G', 'flow', 'split', 'correct_splits'], False)
        #correction = 0 if correct_splits else 1
        correction = 0
        if opt_get(opt, ['network_G', 'flow', 'split', 'enable']) and L < levels - correction:
            logs_eps = opt_get(opt, ['network_G', 'flow', 'split', 'logs_eps']) or 0
            consume_ratio = opt_get(opt, ['network_G', 'flow', 'split', 'consume_ratio']) or 0.5
            position_name = get_position_name(H, self.opt['scale'])
            position = position_name if opt_get(opt, ['network_G', 'flow', 'split', 'conditional']) else None
            cond_channels = opt_get(opt, ['network_G', 'flow', 'split', 'cond_channels'])
            cond_channels = 0 if cond_channels is None else cond_channels

            t = opt_get(opt, ['network_G', 'flow', 'split', 'type'], 'Split2d')

            if t == 'Split2d':
                split = models.modules.Split.Split2d(num_channels=self.C, logs_eps=logs_eps, position=position,
                                                     cond_channels=cond_channels, consume_ratio=consume_ratio, opt=opt)
            
            if type==1:
                self.layers1.append(split)
                self.output_shapes1.append([-1, self.C, H, W])
            elif type==2:
                self.layers2.append(split)
                self.output_shapes2.append([-1, self.C, H, W])
            elif type==3:
                self.layers3.append(split)
                self.output_shapes3.append([-1, self.C, H, W])
            elif type==4:
                self.layers4.append(split)
                self.output_shapes4.append([-1, self.C, H, W])
            self.C = split.num_channels_pass

    def arch_additionalFlowAffine(self, H, LU_decomposed, W, actnorm_scale, hidden_channels, opt,type):
        if 'additionalFlowNoAffine' in opt['network_G']['flow']:
            n_additionalFlowNoAffine = int(opt['network_G']['flow']['additionalFlowNoAffine'])
            for _ in range(n_additionalFlowNoAffine):
                
                if type==1:
                    self.layers1.append(
                    FlowStep(in_channels=self.C,
                             hidden_channels=hidden_channels,
                             actnorm_scale=actnorm_scale,
                             flow_permutation='invconv',
                             flow_coupling='noCoupling',
                             LU_decomposed=LU_decomposed, opt=opt,in_shape=(self.C, H, W)))
                    self.output_shapes1.append([-1, self.C, H, W])
                elif type==2:
                    self.layers2.append(
                    FlowStep(in_channels=self.C,
                             hidden_channels=hidden_channels,
                             actnorm_scale=actnorm_scale,
                             flow_permutation='invconv',
                             flow_coupling='noCoupling',
                             LU_decomposed=LU_decomposed, opt=opt,in_shape=(self.C, H, W)))
                    self.output_shapes2.append([-1, self.C, H, W])
                elif type==3:
                    self.layers3.append(
                    FlowStep(in_channels=self.C,
                             hidden_channels=hidden_channels,
                             actnorm_scale=actnorm_scale,
                             flow_permutation='invconv',
                             flow_coupling='noCoupling',
                             LU_decomposed=LU_decomposed, opt=opt,in_shape=(self.C, H, W)))
                    self.output_shapes3.append([-1, self.C, H, W])
                elif type==4:
                    self.layers4.append(
                    FlowStep(in_channels=self.C,
                             hidden_channels=hidden_channels,
                             actnorm_scale=actnorm_scale,
                             flow_permutation='invconv',
                             flow_coupling='noCoupling',
                             LU_decomposed=LU_decomposed, opt=opt,in_shape=(self.C, H, W)))
                    self.output_shapes4.append([-1, self.C, H, W])

    def arch_squeeze(self, H, W,type):
        self.C, H, W = self.C * 4, H // 2, W // 2
        if type==1:
            self.layers1.append(flow.SqueezeLayer(factor=2))
            self.output_shapes1.append([-1, self.C, H, W])
        elif type==2:
            self.layers2.append(flow.SqueezeLayer(factor=2))
            self.output_shapes2.append([-1, self.C, H, W])
        elif type==3:
            self.layers3.append(flow.SqueezeLayer(factor=2))
            self.output_shapes3.append([-1, self.C, H, W])
        elif type==4:
            self.layers4.append(flow.SqueezeLayer(factor=2))
            self.output_shapes4.append([-1, self.C, H, W])
        return H, W

    def get_flow_permutation(self, flow_permutation, opt):
        flow_permutation = opt['network_G']['flow'].get('flow_permutation', 'invconv')
        return flow_permutation

    def get_affineInCh(self, opt_get):
        affineInCh = opt_get(self.opt, ['network_G', 'flow', 'stackRRDB', 'blocks']) or []
        affineInCh = (len(affineInCh) + 1) * 64
        return affineInCh

    def check_image_shape(self):
        assert self.C == 1 or self.C == 3, ("image_shape should be HWC, like (64, 64, 3)"
                                            "self.C == 1 or self.C == 3")

    def forward(self, gt=None, rrdbResults=None, z=None, epses=None, logdet=0., reverse=False, eps_std=None,
                y_onehot=None):

        if reverse:
            epses_copy = [eps for eps in epses] if isinstance(epses, list) else epses

            sr, logdet = self.decode(rrdbResults, z, eps_std, epses=epses_copy, logdet=logdet, y_onehot=y_onehot)
            return sr, logdet
        else:
            assert gt is not None
            assert rrdbResults is not None
            z, logdet = self.encode(gt, rrdbResults, logdet=logdet, epses=epses, y_onehot=y_onehot)

            return z, logdet

    def encode(self, gt, rrdbResults, logdet=0.0, epses=None, y_onehot=None):
        
        reverse = False
        level_conditionals = {}
        bypasses = {}
        
        L = opt_get(self.opt, ['network_G', 'flow', 'L'])

        for level in range(1, L + 1):
            bypasses[level] = torch.nn.functional.interpolate(gt, scale_factor=2 ** -level, mode='bilinear', align_corners=False)
        
       
        gt, logdet = self.Haar(gt, logdet, reverse=reverse)
        fl_fea1,fl_fea2,fl_fea3,fl_fea4 = gt[:,:3,:,:],gt[:,3:6,:,:],gt[:,6:9,:,:],gt[:,9:,:,:]
        
        #Harr1第一个
        for layer, shape in zip(self.layers1, self.output_shapes1):
            size = shape[2]
            level = int(np.log(160 / size) / np.log(2))
            
            if level > 0 and level not in level_conditionals.keys():
                level_conditionals[level] = rrdbResults[self.levelToName[level]]

            level_conditionals[level] = rrdbResults[self.levelToName[level]]

            if isinstance(layer, FlowStep):
                fl_fea1, logdet = layer(fl_fea1, logdet, reverse=reverse, rrdbResults=level_conditionals[level])
            elif isinstance(layer, Split2d):
                fl_fea1, logdet = self.forward_split2d(epses, fl_fea1, layer, logdet, reverse, level_conditionals[level],
                                                      y_onehot=y_onehot)
            else:
                fl_fea1, logdet = layer(fl_fea1, logdet, reverse=reverse)
        z1=fl_fea1
        #Harr后三个
        for layer, shape in zip(self.layers2, self.output_shapes2):
            size = shape[2]
            level = int(np.log(160 / size) / np.log(2))

            if level > 0 and level not in level_conditionals.keys():
                level_conditionals[level] = rrdbResults[self.levelToName[level]]

            level_conditionals[level] = rrdbResults[self.levelToName[level]]

            if isinstance(layer, FlowStep):
                fl_fea2, logdet = layer(fl_fea2, logdet, reverse=reverse, rrdbResults=level_conditionals[level])
            elif isinstance(layer, Split2d):
                fl_fea2, logdet = self.forward_split2d(epses, fl_fea2, layer, logdet, reverse, level_conditionals[level],
                                                      y_onehot=y_onehot)
            else:
                fl_fea2, logdet = layer(fl_fea2, logdet, reverse=reverse)

        z2=fl_fea2

        for layer, shape in zip(self.layers3, self.output_shapes3):
            size = shape[2]
            level = int(np.log(160 / size) / np.log(2))

            if level > 0 and level not in level_conditionals.keys():
                level_conditionals[level] = rrdbResults[self.levelToName[level]]

            level_conditionals[level] = rrdbResults[self.levelToName[level]]

            if isinstance(layer, FlowStep):
                fl_fea3, logdet = layer(fl_fea3, logdet, reverse=reverse, rrdbResults=level_conditionals[level])
            elif isinstance(layer, Split2d):
                fl_fea3, logdet = self.forward_split2d(epses, fl_fea3, layer, logdet, reverse, level_conditionals[level],
                                                      y_onehot=y_onehot)
            else:
                fl_fea3, logdet = layer(fl_fea3, logdet, reverse=reverse)

        z3=fl_fea3
        for layer, shape in zip(self.layers4, self.output_shapes4):
            size = shape[2]
            level = int(np.log(160 / size) / np.log(2))

            if level > 0 and level not in level_conditionals.keys():
                level_conditionals[level] = rrdbResults[self.levelToName[level]]

            level_conditionals[level] = rrdbResults[self.levelToName[level]]

            if isinstance(layer, FlowStep):
                fl_fea4, logdet = layer(fl_fea4, logdet, reverse=reverse, rrdbResults=level_conditionals[level])
            elif isinstance(layer, Split2d):
                fl_fea4, logdet = self.forward_split2d(epses, fl_fea4, layer, logdet, reverse, level_conditionals[level],
                                                      y_onehot=y_onehot)
            else:
                fl_fea4, logdet = layer(fl_fea4, logdet, reverse=reverse)

        z4=fl_fea4
        #看看z1234是什么
        ##
     
        z = torch.cat((torch.cat((torch.cat((z1,z2),dim=1),z3),dim=1),z4),dim=1)
        
        if not isinstance(epses, list):
            return z, logdet

        epses.append(z)
        return epses, logdet

    def forward_preFlow(self, fl_fea, logdet, reverse):
        if hasattr(self, 'preFlow'):
            for l in self.preFlow:
                fl_fea, logdet = l(fl_fea, logdet, reverse=reverse)
        return fl_fea, logdet

    def forward_split2d(self, epses, fl_fea, layer, logdet, reverse, rrdbResults, y_onehot=None):
        ft = None if layer.position is None else rrdbResults[layer.position]
        fl_fea, logdet, eps = layer(fl_fea, logdet, reverse=reverse, eps=epses, ft=ft, y_onehot=y_onehot)

        if isinstance(epses, list):
            epses.append(eps)
        return fl_fea, logdet

    def decode(self, rrdbResults, z, eps_std=None, epses=None, logdet=0.0, y_onehot=None):
        z = epses.pop() if isinstance(epses, list) else z

        #fl_fea1,fl_fea2 ,fl_fea3,fl_fea4= z[:,:16,:,:],z[:,16:32,:,:],z[:,32:48,:,:],z[:,48:,:,:]
        # debug.imwrite("fl_fea", fl_fea)
        # fl_fea1,fl_fea2 ,fl_fea3,fl_fea4= z[:,:48,:,:],z[:,48:96,:,:],z[:,96:144,:,:],z[:,144:,:,:]
        # debug.imwrite("fl_fea", fl_fea)
        fl_fea1,fl_fea2 ,fl_fea3,fl_fea4= z[:,:16,:,:],z[:,16:32,:,:],z[:,32:48,:,:],z[:,48:,:,:]
        bypasses = {}
        level_conditionals = {}
        if not opt_get(self.opt, ['network_G', 'flow', 'levelConditional', 'conditional']) == True:
            for level in range(self.L + 1):
                level_conditionals[level] = rrdbResults[self.levelToName[level]]
                

         #Harr1第一个
        #print("1--------------------------------")
        for layer, shape in zip(reversed(self.layers1), reversed(self.output_shapes1)):
            size = shape[2]
            level = int(np.log(160 / size) / np.log(2))
            # size = fl_fea.shape[2]
            # level = int(np.log(160 / size) / np.log(2))
           # print(fl_fea1.shape,size,level,rrdbResults[self.levelToName[level]].shape)
            if isinstance(layer, Split2d):
                fl_fea1, logdet = self.forward_split2d_reverse(eps_std, epses, fl_fea1, layer,
                                                              rrdbResults[self.levelToName[level]], logdet=logdet,
                                                              y_onehot=y_onehot)
            elif isinstance(layer, FlowStep):
               
                fl_fea1, logdet = layer(fl_fea1, logdet=logdet, reverse=True, rrdbResults=level_conditionals[level])
            else:
                fl_fea1, logdet = layer(fl_fea1, logdet=logdet, reverse=True)
        #print("2")
        z1=fl_fea1

        #Harr后三个
        for layer, shape in zip(reversed(self.layers2), reversed(self.output_shapes2)):
            size = shape[2]
            level = int(np.log(160 / size) / np.log(2))
            # size = fl_fea.shape[2]
            # level = int(np.log(160 / size) / np.log(2))

            if isinstance(layer, Split2d):
                fl_fea2, logdet = self.forward_split2d_reverse(eps_std, epses, fl_fea2, layer,
                                                              rrdbResults[self.levelToName[level]], logdet=logdet,
                                                              y_onehot=y_onehot)
            elif isinstance(layer, FlowStep):
                fl_fea2, logdet = layer(fl_fea2, logdet=logdet, reverse=True, rrdbResults=level_conditionals[level])
            else:
                fl_fea2, logdet = layer(fl_fea2, logdet=logdet, reverse=True)
        z2=fl_fea2
        for layer, shape in zip(reversed(self.layers3), reversed(self.output_shapes3)):
            size = shape[2]
            level = int(np.log(160 / size) / np.log(2))
            # size = fl_fea.shape[2]
            # level = int(np.log(160 / size) / np.log(2))

            if isinstance(layer, Split2d):
                fl_fea3, logdet = self.forward_split2d_reverse(eps_std, epses, fl_fea3, layer,
                                                              rrdbResults[self.levelToName[level]], logdet=logdet,
                                                              y_onehot=y_onehot)
            elif isinstance(layer, FlowStep):
                fl_fea3, logdet = layer(fl_fea3, logdet=logdet, reverse=True, rrdbResults=level_conditionals[level])
            else:
                fl_fea3, logdet = layer(fl_fea3, logdet=logdet, reverse=True)
        z3=fl_fea3

        for layer, shape in zip(reversed(self.layers4), reversed(self.output_shapes4)):
            size = shape[2]
            level = int(np.log(160 / size) / np.log(2))
            # size = fl_fea.shape[2]
            # level = int(np.log(160 / size) / np.log(2))

            if isinstance(layer, Split2d):
                fl_fea4, logdet = self.forward_split2d_reverse(eps_std, epses, fl_fea4, layer,
                                                              rrdbResults[self.levelToName[level]], logdet=logdet,
                                                              y_onehot=y_onehot)
            elif isinstance(layer, FlowStep):
                fl_fea4, logdet = layer(fl_fea4, logdet=logdet, reverse=True, rrdbResults=level_conditionals[level])
            else:
                fl_fea4, logdet = layer(fl_fea4, logdet=logdet, reverse=True)
        z4=fl_fea4


        z = torch.cat((torch.cat((torch.cat((z1,z2),dim=1),z3),dim=1),z4),dim=1)
        fl_fea,_=self.Haar(z,logdet,reverse=True)
        
        sr = fl_fea

        assert sr.shape[1] == 3
        return sr, logdet

    def forward_split2d_reverse(self, eps_std, epses, fl_fea, layer, rrdbResults, logdet, y_onehot=None):
        ft = None if layer.position is None else rrdbResults[layer.position]
        fl_fea, logdet = layer(fl_fea, logdet=logdet, reverse=True,
                               eps=epses.pop() if isinstance(epses, list) else None,
                               eps_std=eps_std, ft=ft, y_onehot=y_onehot)
        return fl_fea, logdet


def get_position_name(H, scale):
    downscale_factor = 160 // H
    position_name = 'fea_up{}'.format(scale / downscale_factor)
    return position_name
