
# from audioop import reverse
# from urllib import request
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# from Cython.Build import cythonize
# import sys
# sys.path.append('./models/modules')
# import pyximport
# pyximport.install(
#     inplace=True,   
#     )
# import inverse_op_cython as inverse_op_cython
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
# class invertible_conv2D_emerging(nn.Module):
#     def __init__(self, num_channels, ksize=3, dilation=1):
#         super().__init__()
#         self.kcent = (ksize - 1) // 2
#         self.ksize=ksize
#         self.dilation=dilation
#         self.num_channels=num_channels
#         filter_shape = [num_channels,num_channels,ksize,ksize]
#         self.w1_np = get_conv_weight_np(filter_shape)
#         self.w2_np = get_conv_weight_np(filter_shape)
#         self.register_parameter("W1", nn.Parameter(torch.Tensor(self.w1_np).cuda()))
#         self.register_parameter("W2", nn.Parameter(torch.Tensor(self.w2_np).cuda()))
#         self.register_parameter("b", nn.Parameter(torch.zeros((1,num_channels,1,1)).cuda()))
#         #mask
#         mask_np = get_conv_square_ar_mask(ksize, ksize,num_channels,num_channels )
#         mask_upsidedown_np = mask_np[::-1, ::-1, ::-1, ::-1].copy()
#         self.mask = torch.Tensor(mask_np).permute(2,3,0,1).cuda()
#         self.mask_upsidedown = torch.Tensor(mask_upsidedown_np).permute(2,3,0,1).cuda()

        
#         self.inver1=Inverse(is_upper=1, dilation=dilation)
#         self.inver2=Inverse(is_upper=0, dilation=dilation)
#     def forward(self,z,logdet,reverse=False):
#         self.w1 = self.W1 * self.mask
#         self.w2 = self.W2 * self.mask_upsidedown
#         w1_s = self.w1[:,:,self.kcent::, self.kcent:]
#         w2_s = self.w2[:,:,:-self.kcent, :-self.kcent]
#         pad = self.kcent * self.dilation
#         pd = (0,1,0,1)
#         W,H=z.shape[2],z.shape[3]
#         if not reverse:
#             pd =(0,pad,0,pad)
#             z = F.pad(z, pd, 'constant')
#             z = F.conv2d(z, w1_s) # fc layer, ie, permute channel
#             if logdet is not None:
#                     logdet = logdet + torch.sum(self.log_abs_diagonal(self.w1))*W*H
#             pd =(0,pad,0,pad)
#             z = F.pad(z, pd, 'constant')
#             z = F.conv2d(z, w2_s) # fc layer, ie, permute channel
#             if logdet is not None:
#                     logdet = logdet + torch.sum(self.log_abs_diagonal(self.w2))*W*H
#             z=z+self.b
#             return z, logdet
#         else:
#             z=z-self.b
#             pd =(0,pad,0,pad)
#             logdet =0#逆的logdet没用就不计算了
#             w2_s_inver=torch.inverse(w2_s)
#             w1_s_inver=torch.inverse(w2_s)
#             z = F.pad(z, pd, 'constant')
#             z = F.conv2d(z, w2_s_inver)

#             z = F.pad(z, pd, 'constant')
#             z = F.conv2d(z, w1_s_inver)
                    
#             return z, logdet
#     def log_abs_diagonal(self,w):
#         return torch.log(torch.abs(torch.diag(w[:,:,self.kcent, self.kcent])))   
# def get_conv_square_ar_mask(h, w, n_in, n_out, zerodiagonal=False):
#     """
#     Function to get autoregressive convolution with square shape.
#     """
#     l = (h - 1) // 2
#     m = (w - 1) // 2
#     mask = np.ones([h, w, n_in, n_out], dtype=np.float32)
#     mask[:l, :, :, :] = 0
#     mask[:, :m, :, :] = 0
#     mask[l, m, :, :] = get_linear_ar_mask(n_in, n_out, zerodiagonal)
#     return mask
# def get_linear_ar_mask(n_in, n_out, zerodiagonal=False):
#     assert n_in % n_out == 0 or n_out % n_in == 0, "%d - %d" % (n_in, n_out)

#     mask = np.ones([n_in, n_out], dtype=np.float32)
#     if n_out >= n_in:
#         k = n_out // n_in
#         for i in range(n_in):
#             mask[i + 1:, i * k:(i + 1) * k] = 0
#             if zerodiagonal:
#                 mask[i:i + 1, i * k:(i + 1) * k] = 0
#     else:
#         k = n_in // n_out
#         for i in range(n_out):
#             mask[(i + 1) * k:, i:i + 1] = 0
#             if zerodiagonal:
#                 mask[i * k:(i + 1) * k:, i:i + 1] = 0
#     return mask
# def get_conv_weight_np(filter_shape, stable_init=True, unit_testing=False):
#     weight_np = np.random.randn(*filter_shape) * 0.02
#     kcent = (filter_shape[3] - 1) // 2
#     if stable_init or unit_testing:
#         weight_np[ :, :,kcent, kcent] += 1. * np.eye(filter_shape[0])
        
#     weight_np = weight_np.astype('float32')
#     return weight_np
# class Inverse():
#     def __init__(self, is_upper, dilation):
#         self.is_upper = is_upper
#         self.dilation = dilation

#     def __call__(self, z, w, b):
#         z=z.cpu()
#         w=w.cpu()
#         b=b.cpu()
#         # start = time.time()
#         z = z - b
#         #原本[C,C,K,K]   [B,C,H,W]
#         #这里变成[K,K,C,C]  [B,H,W,C]
#         w=w.permute(2,3,0,1).detach()
#         z=z.permute(0,2,3,1).detach()
#         z_np = np.array(z, dtype='float64')
#         w_np = np.array(w, dtype='float64')
#         ksize = w_np.shape[0]
#         kcent = (ksize - 1) // 2
#         diagonal = np.diag(w_np[kcent, kcent, :, :])
        
#         alpha = 1. / np.min(np.abs(diagonal))
#         alpha = max(1., alpha)

#         w_np *= alpha

#         x_np = inverse_op_cython.inverse_conv(
#             z_np, w_np, int(self.is_upper), self.dilation)

#         x_np *= alpha

#         # print('Inverse \t alpha {} \t compute time: {:.2f} seconds'.format(
#         #                                         alpha, time.time() - start))
#         #原本[B,H,W,C]  
#         #  [B,C,H,W]
        
#         return torch.tensor(x_np).permute(0,3,1,2)
# pd =(0,1,0,1)
# z=torch.randn((1,3,6,6))
# out = F.pad(z, pd, 'constant')
# weight=torch.randn((3,3,2,2))
# print("input:",z.shape)
# print("padding out:",out.shape)
# print("weight:",weight.shape)
# z = F.conv2d(out, weight)
# print(z.shape)
# w1=torch.randn((6,6,3,3))
# def log_abs_diagonal(w):
#         return torch.log(torch.abs(torch.diag(w[:,:,1, 1])))
# print("-----test",torch.sum(log_abs_diagonal(w1)))
# print(w1[:,:,:1,:1].shape,w1.shape)  
# print("*********************Emerging************************")
# emerging=invertible_conv2D_emerging(12)
# input=torch.randn((1,12,6,6))
# out=emerging(input,0)
# print("out shape:",out[0].shape)
# print("*********************reverse************************")
# out=emerging(input,0,reverse=True)
# print("out shape:",out[0].shape)
# print("********************changing*************************")
# pd =(0,1,0,1)
# z=torch.randn((1,3,6,6))
# out = F.pad(z, pd, 'constant')
# print(" pad out shape",out.shape)
# weight=torch.randn((2,2,3,3))
# out=out.permute(0,2,3,1)
# print("permute shape ",out.shape)
# emger=invertible_conv2D_emerging(12)
class invertible_conv2D_emerging():
    def __init__(self, num_channels, ksize=3, dilation=1):
        super().__init__()
        self.layer1=[]
        eval("self.layer"+str(1)).append([123,1])
        
        print(eval("self.layer"+str(1)))
invertible_conv2D_emerging(12)