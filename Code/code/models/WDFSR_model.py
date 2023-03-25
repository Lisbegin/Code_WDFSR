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
# This file contains content licensed by https://github.com/xinntao/BasicSR/blob/master/LICENSE/LICENSE

import logging
from collections import OrderedDict
from utils.util import get_resume_paths, opt_get
from models.modules.loss import GANLoss
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.networks as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from models.data_parallel_my_v2 import BalancedDataParallel
logger = logging.getLogger('base')
import numpy as np
import models.modules.thops as thops
import scipy.special
import torch

class WDFSRModel(BaseModel):
    def __init__(self, opt, step):
        super(WDFSRModel, self).__init__(opt)
        self.opt = opt
        
        self.heats = opt['val']['heats']
        self.n_sample = opt['val']['n_sample']
        self.hr_size = opt_get(opt, ['datasets', 'train', 'center_crop_hr_size'])
        self.hr_size = 160 if self.hr_size is None else self.hr_size
        self.lr_size = self.hr_size // opt['scale']
        self.studentZ=StudentT(20)
        self.l2=nn.MSELoss().to(self.device)
        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt['train']

        # define network and load pretrained models
        self.netG = networks.define_Flow(opt, step).to(self.device)
        if opt['dist']:
            self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()])
        else:
            #self.netG = BalancedDataParallel(3, self.netG, dim=0)
            self.netG = DataParallel(self.netG)
        # print network
        #self.print_network()

        if opt_get(opt, ['path', 'resume_state'], 1) is not None:
            self.load()
            print("loading pretrained_models")
        else:
            print("WARNING: skipping initial loading, due to resume_state None")
        self.eps_std_reverse = train_opt['eps_std_reverse']
        if self.is_train:
            self.netG.train()
            if train_opt['weight_fl'] > 0:
                loss_type = train_opt['pixel_criterion_hr']
                if loss_type == 'l1':
                    self.cri_pix_hr = nn.L1Loss().to(self.device)
                elif loss_type == 'l2':
                    self.cri_pix_hr = nn.MSELoss().to(self.device)
                else:
                    raise NotImplementedError('Loss type [{:s}] is not recognized.'.format(loss_type))
                self.l_pix_w_hr = train_opt['weight_fl']
            else:
                logger.info('Remove HR pixel loss.')
                self.cri_pix_hr = None


            if train_opt['feature_weight'] > 0:
                l_fea_type = train_opt['feature_criterion']
                if l_fea_type == 'l1':
                    self.cri_fea = nn.L1Loss().to(self.device)
                elif l_fea_type == 'l2':
                    self.cri_fea = nn.MSELoss().to(self.device)
                else:
                    raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_fea_type))
                self.l_fea_w = train_opt['feature_weight']
                # load VGG perceptual loss
                self.netF = networks.define_F(opt, use_bn=False).to(self.device)
                if opt['dist']:
                    self.netF = DistributedDataParallel(self.netF, device_ids=[torch.cuda.current_device()])
                else:
                    self.netF = DataParallel(self.netF)
            else:
                logger.info('Remove feature loss.')
                self.cri_fea = None

        # HR GAN loss
            # put here to be compatible with PSNR version
            self.D_update_ratio = train_opt['D_update_ratio'] if train_opt['D_update_ratio'] else 1
            self.D_init_iters = train_opt['D_init_iters'] if train_opt['D_init_iters'] else 0
            if train_opt['gan_weight'] > 0:
                self.cri_gan = GANLoss(train_opt['gan_type'], 1.0, 0.0).to(self.device)
                self.l_gan_w = train_opt['gan_weight']

                # define GAN Discriminator
                self.netD = networks.define_D(opt).to(self.device)
                if opt['dist']:
                    self.netD = DistributedDataParallel(self.netD, device_ids=[torch.cuda.current_device()])
                else:
                    self.netD = DataParallel(self.netD)
                self.netD.train()
            else:
                logger.info('Remove GAN loss.')
                self.cri_gan = None

            self.init_optimizer_and_scheduler(train_opt)
            self.log_dict = OrderedDict()

    def to(self, device):
        self.device = device
        self.netG.to(device)

    def init_optimizer_and_scheduler(self, train_opt):
        # optimizers
        self.optimizers = []
        wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
        optim_params_RRDB = []
        optim_params_other = []
        for k, v in self.netG.named_parameters():  # can optimize for a part of the model
            #print(k, v.requires_grad)
            if v.requires_grad:
                if '.RRDB.' in k:
                    optim_params_RRDB.append(v)
                    print('opt', k)
                else:
                    optim_params_other.append(v)
                if self.rank <= 0:
                    logger.warning('Params [{:s}] will not optimize.'.format(k))

        print('rrdb params', len(optim_params_RRDB))

        self.optimizer_G = torch.optim.Adam(
            [
                {"params": optim_params_other, "lr": train_opt['lr_G'], 'beta1': train_opt['beta1'],
                 'beta2': train_opt['beta2'], 'weight_decay': wd_G},
                {"params": optim_params_RRDB, "lr": train_opt.get('lr_RRDB', train_opt['lr_G']),
                 'beta1': train_opt['beta1'],
                 'beta2': train_opt['beta2'], 'weight_decay': wd_G}
            ],
        )
        self.optimizers.append(self.optimizer_G)

        # if self.cri_gan:
        #         wd_D = train_opt['weight_decay_D'] if train_opt['weight_decay_D'] else 0
        #         self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=train_opt['lr_D'],
        #                                             weight_decay=wd_D,
        #                                             betas=(train_opt['beta1_D'], train_opt['beta2_D']))
        #         self.optimizers.append(self.optimizer_D)



        # schedulers
        if train_opt['lr_scheme'] == 'MultiStepLR':
            for optimizer in self.optimizers:
                self.schedulers.append(
                    lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['lr_steps'],
                                                     restarts=train_opt['restarts'],
                                                     weights=train_opt['restart_weights'],
                                                     gamma=train_opt['lr_gamma'],
                                                     clear_state=train_opt['clear_state'],
                                                     lr_steps_invese=train_opt.get('lr_steps_inverse', [])))
        elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
            for optimizer in self.optimizers:
                self.schedulers.append(
                    lr_scheduler.CosineAnnealingLR_Restart(
                        optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
                        restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
        else:
            raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

    def add_optimizer_and_scheduler_RRDB(self, train_opt):
        # optimizers
        assert len(self.optimizers) == 1, self.optimizers
        assert len(self.optimizer_G.param_groups[1]['params']) == 0, self.optimizer_G.param_groups[1]
        for k, v in self.netG.named_parameters():  # can optimize for a part of the model
            if v.requires_grad:
                if '.RRDB.' in k:
                    self.optimizer_G.param_groups[1]['params'].append(v)
        print("add into it succsseful",len(self.optimizer_G.param_groups[1]['params']))
        assert len(self.optimizer_G.param_groups[1]['params']) > 0

    def feed_data(self, data, need_GT=True):
        self.var_L = data['LQ'].to(self.device)  # LQ
        if need_GT:
            self.real_H = data['GT'].to(self.device)  # GT
 
    def optimize_parameters(self, step):

        train_RRDB_delay = opt_get(self.opt, ['network_G', 'train_RRDB_delay'])
        if train_RRDB_delay is not None and step > int(train_RRDB_delay * self.opt['train']['niter']) \
                and not self.netG.module.RRDB_training:
            if self.netG.module.set_rrdb_training(True):
                self.add_optimizer_and_scheduler_RRDB(self.opt['train'])

        # self.print_rrdb_state()
        # total = sum([param.nelement() for param in self.netG.parameters()]) 
        # print("Number of parameter: %.2fM" % (total/1e6))
        self.netG.train()
        self.log_dict = OrderedDict()
        self.optimizer_G.zero_grad()

        losses = {}
        l_g_total = 0
        weight_fl = opt_get(self.opt, ['train', 'weight_fl'])
        weight_fl = 1 if weight_fl is None else weight_fl
        if weight_fl > 0:
            z, nll, y_logits = self.netG(gt=self.real_H, lr=self.var_L, reverse=False)
            nll_loss = torch.mean(nll)
            losses['nll_loss'] = nll_loss * weight_fl
            l_g_total=nll_loss * weight_fl
        if l_g_total != 0:
            l_g_total.backward()
            self.optimizer_G.step()
            self.optimizer_G.zero_grad()


        #pixel_loss
        weight_l1 = opt_get(self.opt, ['train', 'weight_l1']) or 0
        l_g_total = 0
    
        if weight_l1 > 0:
            z = self.get_z(heat=0, seed=None, batch_size=self.var_L.shape[0], lr_shape=self.var_L.shape)
            sr, logdet = self.netG(lr=self.var_L, z=z, eps_std=self.eps_std_reverse, reverse=True, reverse_with_grad=True)
            l_loss=self.cri_pix_hr(sr , self.real_H)
            l_g_total = l_loss * weight_l1
        if l_g_total != 0:
            l_g_total.backward()
            self.optimizer_G.step()
            self.optimizer_G.zero_grad()
        
        ########################

        if self.cri_gan or self.cri_fea:
            self.optimizer_G.zero_grad()
            z = self.get_z(heat=self.eps_std_reverse, seed=None, batch_size=self.var_L.shape[0], lr_shape=self.var_L.shape)
            fake_H, logdet = self.netG(lr=self.var_L, z=z, eps_std=0, reverse=True, reverse_with_grad=True)
            l_g_fea_gan = 0

            # feature loss
            if self.cri_fea:
                real_fea = self.netF(self.real_H).detach()
                fake_fea = self.netF(fake_H)
                l_g_fea = self.l_fea_w * self.cri_fea(fake_fea, real_fea)
                l_g_fea_gan += l_g_fea

            if not torch.isnan(l_g_fea_gan):
                l_g_fea_gan.backward()
                self.optimizer_G.step()
        total_loss = sum(losses.values())
        mean = total_loss.item()
        
        return mean

    def print_rrdb_state(self):
        for name, param in self.netG.module.named_parameters():
            if "RRDB.conv_first.weight" in name:
                print(name, param.requires_grad, param.data.abs().sum())
        print('params', [len(p['params']) for p in self.optimizer_G.param_groups])

    def test(self):
        self.netG.eval()
        self.fake_H = {}
        for heat in self.heats:
            for i in range(self.n_sample):
                z = self.get_z(heat, seed=None, batch_size=self.var_L.shape[0], lr_shape=self.var_L.shape)
                #z=self.studentZ.sample(self.var_L.shape,heat,self.device,self.netG)
                with torch.no_grad():
                    self.fake_H[(heat, i)], logdet = self.netG(lr=self.var_L, z=z, eps_std=heat, reverse=True)
        with torch.no_grad():
            _, nll, _ = self.netG(gt=self.real_H, lr=self.var_L, reverse=False)
        self.netG.train()
        return nll.mean().item()

    def get_encode_nll(self, lq, gt):
        self.netG.eval()
        with torch.no_grad():
            _, nll, _ = self.netG(gt=gt, lr=lq, reverse=False)
        self.netG.train()
        return nll.mean().item()

    def get_sr(self, lq, heat=None, seed=None, z=None, epses=None):
        return self.get_sr_with_z(lq, heat, seed, z, epses)[0]

    def get_encode_z(self, lq, gt, epses=None, add_gt_noise=True):
        self.netG.eval()
        with torch.no_grad():
            z, _, _ = self.netG(gt=gt, lr=lq, reverse=False, epses=epses, add_gt_noise=add_gt_noise)
        self.netG.train()
        return z

    def get_encode_z_and_nll(self, lq, gt, epses=None, add_gt_noise=True):
        self.netG.eval()
        with torch.no_grad():
            z, nll, _ = self.netG(gt=gt, lr=lq, reverse=False, epses=epses, add_gt_noise=add_gt_noise)
        self.netG.train()
        return z, nll

    def get_sr_with_z(self, lq, heat=None, seed=None, z=None, epses=None):
        self.netG.eval()

        z = self.get_z(heat, seed, batch_size=lq.shape[0], lr_shape=lq.shape) if z is None and epses is None else z
        #z=self.studentZ.sample(lq.shape,heat,self.device,self.netG)
        with torch.no_grad():
            sr, logdet = self.netG(lr=lq, z=z, eps_std=heat, reverse=True, epses=epses)
        self.netG.train()
        return sr, z

    def get_z(self, heat, seed=None, batch_size=1, lr_shape=None):
        if seed: torch.manual_seed(seed)
        if opt_get(self.opt, ['network_G', 'flow', 'split', 'enable']):
            C = self.netG.module.flowUpsamplerNet.C
            H = int(self.opt['scale'] * lr_shape[2] // self.netG.module.flowUpsamplerNet.scaleH)
            W = int(self.opt['scale'] * lr_shape[3] // self.netG.module.flowUpsamplerNet.scaleW)
            z = torch.normal(mean=0, std=heat, size=(batch_size, 64, H, W)) if heat > 0 else torch.zeros(
                (batch_size, 64, H, W))
        else:
            L = opt_get(self.opt, ['network_G', 'flow', 'L']) or 3
            fac = 2 ** (L - 3)
            z_size = int(self.lr_size // (2 ** (L - 3)))
            z = torch.normal(mean=0, std=heat, size=(batch_size, 3 * 8 * 8 * fac * fac, z_size, z_size))
        return z

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict['LQ'] = self.var_L.detach()[0].float().cpu()
        for heat in self.heats:
            for i in range(self.n_sample):
                out_dict[('SR', heat, i)] = self.fake_H[(heat, i)].detach()[0].float().cpu()
        if need_GT:
            out_dict['GT'] = self.real_H.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel) or isinstance(self.netG, DistributedDataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

    def load(self):
        _, get_resume_model_path = get_resume_paths(self.opt)
        if get_resume_model_path is not None:
            self.load_network(get_resume_model_path, self.netG, strict=True, submodule=None)
            return
        print("-----------dont't load RRDB---------------------")
        load_path_G = self.opt['path']['pretrain_model_G']
        load_submodule = self.opt['path']['load_submodule'] if 'load_submodule' in self.opt['path'].keys() else 'RRDB'
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, self.opt['path'].get('strict_load', True),
                              submodule=load_submodule)

    def save(self, iter_label):
        self.save_network(self.netG, 'G', iter_label)
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

    def sample(self,z_shape, eps_std=None, device=None,netG=None):
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
        C = 64
        H = int(4 * z_shape[2] // 8)
        W = int(4 * z_shape[3] // 8)
        x_shape = torch.Size((z_shape[0], 1, H,W))
        z_shape=torch.Size((z_shape[0], C, H,W))
        x = np.random.chisquare(self.df, z_shape)/self.df  
        #x = np.tile(x, (1,C,1,1))
        x = torch.Tensor(x.astype(np.float32))
        z = torch.normal(mean=torch.zeros(z_shape),std=torch.ones(z_shape) * eps_std)
        return (z/torch.sqrt(x)).to(device)