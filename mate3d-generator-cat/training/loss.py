# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Loss functions."""

import numpy as np
import torch
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import upfirdn2d
from training.dual_discriminator import filtered_resizing
from hparams import hparams as hp
import functools
import torch.nn.functional as F
#----------------------------------------------------------------------------

class Loss:
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg): # to be overridden by subclass
        raise NotImplementedError()

#----------------------------------------------------------------------------

def eikonal_loss(eikonal_term):
    if eikonal_term == None:
        eikonal_loss = 0
    else:
        eikonal_loss = ((eikonal_term.norm(dim=-1) - 1) ** 2).mean()



    return eikonal_loss


def surface_loss(sdf, beta=100):

    minimal_surface_loss = torch.exp(-beta * torch.abs(sdf)).mean()

    return minimal_surface_loss

class InitSDFLoss(Loss):
    # def __init__(self, device, G, D, D_seg, D_rgbseg, augment_pipe=None, r1_gamma=10, style_mixing_prob=0, pl_weight=0, pl_batch_shrink=2, pl_decay=0.01, pl_no_weight_grad=False, blur_init_sigma=0, blur_fade_kimg=0, r1_gamma_init=0, r1_gamma_fade_kimg=0, neural_rendering_resolution_initial=64, neural_rendering_resolution_final=None, neural_rendering_resolution_fade_kimg=0, gpc_reg_fade_kimg=1000, gpc_reg_prob=None, dual_discrimination=False, filter_mode='antialiased'):
    def __init__(self, device, G, D, augment_pipe=None, r1_gamma=10, style_mixing_prob=0, pl_weight=0, pl_batch_shrink=2, pl_decay=0.01, pl_no_weight_grad=False, blur_init_sigma=0, blur_fade_kimg=0, r1_gamma_init=0, r1_gamma_fade_kimg=0, neural_rendering_resolution_initial=64, neural_rendering_resolution_final=None, neural_rendering_resolution_fade_kimg=0, gpc_reg_fade_kimg=1000, gpc_reg_prob=None, dual_discrimination=False, filter_mode='antialiased'):
        super().__init__()
        self.device             = device
        self.G                  = G
        self.D                  = D
        self.augment_pipe       = augment_pipe
        self.r1_gamma           = r1_gamma
        self.style_mixing_prob  = style_mixing_prob
        self.pl_weight          = pl_weight
        self.pl_batch_shrink    = pl_batch_shrink
        self.pl_decay           = pl_decay
        self.pl_no_weight_grad  = pl_no_weight_grad
        self.pl_mean            = torch.zeros([], device=device)
        self.blur_init_sigma    = blur_init_sigma
        self.blur_fade_kimg     = blur_fade_kimg
        self.r1_gamma_init      = r1_gamma_init
        self.r1_gamma_fade_kimg = r1_gamma_fade_kimg
        self.neural_rendering_resolution_initial = neural_rendering_resolution_initial
        self.neural_rendering_resolution_final = neural_rendering_resolution_final
        self.neural_rendering_resolution_fade_kimg = neural_rendering_resolution_fade_kimg
        self.gpc_reg_fade_kimg = gpc_reg_fade_kimg
        self.gpc_reg_prob = gpc_reg_prob
        self.dual_discrimination = dual_discrimination
        self.filter_mode = filter_mode
        self.resample_filter = upfirdn2d.setup_filter([1,3,3,1], device=device)
        self.blur_raw_target = True
        assert self.gpc_reg_prob is None or (0 <= self.gpc_reg_prob <= 1)

    def run_G(self, z, c, swapping_prob, neural_rendering_resolution, update_emas=False):
        if swapping_prob is not None:
            c_swapped = torch.roll(c.clone(), 1, 0)
            c_gen_conditioning = torch.where(torch.rand((c.shape[0], 1), device=c.device) < swapping_prob, c_swapped, c)
        else:
            c_gen_conditioning = torch.zeros_like(c)

        ws = self.G.mapping(z, c_gen_conditioning, update_emas=update_emas)
        if self.style_mixing_prob > 0:
            with torch.autograd.profiler.record_function('style_mixing'):
                cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                ws[:, cutoff:] = self.G.mapping(torch.randn_like(z), c, update_emas=False)[:, cutoff:]
        gen_output = self.G.synthesis(ws, c, neural_rendering_resolution=neural_rendering_resolution, update_emas=update_emas)
        return gen_output, ws

    def accumulate_gradients(self, phase, real_img, real_seg, real_c, gen_z, gen_c, gain, cur_nimg):
        # assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth', 'D_segmain', 'D_segreg', 'D_segboth', 'D_rgbsegmain', 'D_rgbsegreg', 'D_rgbsegboth']
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        if self.G.rendering_kwargs.get('density_reg', 0) == 0:
            phase = {'Greg': 'none', 'Gboth': 'Gmain'}.get(phase, phase)
        if self.r1_gamma == 0:
            phase = {'Dreg': 'none', 'Dboth': 'Dmain'}.get(phase, phase)
            # phase = {'D_segreg': 'none', 'D_segboth': 'Dmain'}.get(phase, phase)
            # phase = {'D_rgbsegreg': 'none', 'D_rgbsegboth': 'Dmain'}.get(phase, phase)
        blur_sigma = max(1 - cur_nimg / (self.blur_fade_kimg * 1e3), 0) * self.blur_init_sigma if self.blur_fade_kimg > 0 else 0
        r1_gamma = self.r1_gamma

        alpha = min(cur_nimg / (self.gpc_reg_fade_kimg * 1e3), 1) if self.gpc_reg_fade_kimg > 0 else 1
        swapping_prob = (1 - alpha) * 1 + alpha * self.gpc_reg_prob if self.gpc_reg_prob is not None else None

        if self.neural_rendering_resolution_final is not None:
            alpha = min(cur_nimg / (self.neural_rendering_resolution_fade_kimg * 1e3), 1)
            neural_rendering_resolution = int(np.rint(self.neural_rendering_resolution_initial * (1 - alpha) + self.neural_rendering_resolution_final * alpha))
        else:
            neural_rendering_resolution = self.neural_rendering_resolution_initial

        real_img_raw = filtered_resizing(real_img, size=neural_rendering_resolution, f=self.resample_filter, filter_mode=self.filter_mode)

        if self.blur_raw_target:
            blur_size = np.floor(blur_sigma * 3)
            if blur_size > 0:
                f = torch.arange(-blur_size, blur_size + 1, device=real_img_raw.device).div(blur_sigma).square().neg().exp2()
                real_img_raw = upfirdn2d.filter2d(real_img_raw, f / f.sum())

        real_img = {'image': real_img, 'image_raw': real_img_raw}

        if phase in ['Gmain', 'Gboth']:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, swapping_prob=swapping_prob, neural_rendering_resolution=neural_rendering_resolution)
                target_value = gen_img['pts'].detach().norm(dim=-1)-((gen_img['ray_end']-gen_img['ray_start'])/4)

                initloss = torch.nn.L1Loss()(gen_img['all_sdf'].reshape(gen_img['all_sdf'].shape[0],-1),target_value.reshape(gen_img['all_sdf'].shape[0],-1))
                
                for i in gen_img['all_sdf_aux']:
                    initloss += torch.nn.L1Loss()(i.reshape(gen_img['all_sdf'].shape[0],-1),target_value.reshape(i.shape[0],-1))
                # initloss = torch.nn.L1Loss()(gen_img['all_sdf'],target_value)
                
                loss = initloss



                training_stats.report('Loss/G_init/initloss', initloss)
                training_stats.report('Loss/beta', self.G.decoder.sigmoid_beta)
            with torch.autograd.profiler.record_function('Gmain_backward'):
                # loss_Gmain.mean().mul(gain).backward()
                (loss).mul(gain).backward()
                # (loss_Gmain.mean()+rgb_consist_loss+sigma_consist_loss+sdf_consist_loss+g_eikonal1+g_minimal_surface).mul(gain).backward()
                # (loss_Gmain.mean()+rgb_consist_loss+sigma_consist_loss+sdf_consist_loss+g_minimal_surface).mul(gain).backward()

        # Density Regularization
        if phase in ['Greg', 'Gboth'] and self.G.rendering_kwargs.get('density_reg', 0) > 0 and self.G.rendering_kwargs['reg_type'] == 'l1':
            pass

        # Alternative density regularization
        if phase in ['Greg', 'Gboth'] and self.G.rendering_kwargs.get('density_reg', 0) > 0 and self.G.rendering_kwargs['reg_type'] == 'monotonic-detach':
            pass

        # Alternative density regularization
        if phase in ['Greg', 'Gboth'] and self.G.rendering_kwargs.get('density_reg', 0) > 0 and self.G.rendering_kwargs['reg_type'] == 'monotonic-fixed':
            pass

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if phase in ['Dmain', 'Dboth']:
            pass

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if phase in ['Dmain', 'Dreg', 'Dboth']:
            pass
        
        
class StyleGAN2Loss(Loss):
    def __init__(self, device, G, D, D_semantic, augment_pipe=None, r1_gamma=10, style_mixing_prob=0, pl_weight=0, pl_batch_shrink=2, pl_decay=0.01, pl_no_weight_grad=False, blur_init_sigma=0, blur_fade_kimg=0, r1_gamma_init=0, r1_gamma_fade_kimg=0, neural_rendering_resolution_initial=64, neural_rendering_resolution_final=None, neural_rendering_resolution_fade_kimg=0, gpc_reg_fade_kimg=1000, gpc_reg_prob=None, dual_discrimination=False, filter_mode='antialiased'):
        super().__init__()
        self.device             = device
        self.G                  = G
        self.D                  = D
        self.D_semantic         = D_semantic
        self.augment_pipe       = augment_pipe
        self.r1_gamma           = r1_gamma
        self.style_mixing_prob  = style_mixing_prob
        self.pl_weight          = pl_weight
        self.pl_batch_shrink    = pl_batch_shrink
        self.pl_decay           = pl_decay
        self.pl_no_weight_grad  = pl_no_weight_grad
        self.pl_mean            = torch.zeros([], device=device)
        self.blur_init_sigma    = blur_init_sigma
        self.blur_fade_kimg     = blur_fade_kimg
        self.r1_gamma_init      = r1_gamma_init
        self.r1_gamma_fade_kimg = r1_gamma_fade_kimg
        self.neural_rendering_resolution_initial = neural_rendering_resolution_initial
        self.neural_rendering_resolution_final = neural_rendering_resolution_final
        self.neural_rendering_resolution_fade_kimg = neural_rendering_resolution_fade_kimg
        self.gpc_reg_fade_kimg = gpc_reg_fade_kimg
        self.gpc_reg_prob = gpc_reg_prob
        self.dual_discrimination = dual_discrimination
        self.filter_mode = filter_mode
        self.resample_filter = upfirdn2d.setup_filter([1,3,3,1], device=device)
        self.blur_raw_target = True
        assert self.gpc_reg_prob is None or (0 <= self.gpc_reg_prob <= 1)

    def run_G(self, z, c, eik, swapping_prob, neural_rendering_resolution, update_emas=False):
        if swapping_prob is not None:
            c_swapped = torch.roll(c.clone(), 1, 0)
            c_gen_conditioning = torch.where(torch.rand((c.shape[0], 1), device=c.device) < swapping_prob, c_swapped, c)
        else:
            c_gen_conditioning = torch.zeros_like(c)

        ws = self.G.mapping(z, c_gen_conditioning, update_emas=update_emas)
        if self.style_mixing_prob > 0:
            with torch.autograd.profiler.record_function('style_mixing'):
                cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                ws[:, cutoff:] = self.G.mapping(torch.randn_like(z), c, update_emas=False)[:, cutoff:]
        gen_output = self.G.synthesis(ws, c, eik, neural_rendering_resolution=neural_rendering_resolution, update_emas=update_emas)
        return gen_output, ws

    def run_D(self, img, c, blur_sigma=0, blur_sigma_raw=0, update_emas=False):
        blur_size = np.floor(blur_sigma * 3)
        if blur_size > 0:
            with torch.autograd.profiler.record_function('blur'):
                f = torch.arange(-blur_size, blur_size + 1, device=img['image'].device).div(blur_sigma).square().neg().exp2()
                img['image'] = upfirdn2d.filter2d(img['image'], f / f.sum())

        if self.augment_pipe is not None:
            augmented_pair = self.augment_pipe(torch.cat([img['image'],
                                                    torch.nn.functional.interpolate(img['image_raw'], size=img['image'].shape[2:], mode='bilinear', antialias=True)],
                                                    dim=1))
            img['image'] = augmented_pair[:, :img['image'].shape[1]]
            img['image_raw'] = torch.nn.functional.interpolate(augmented_pair[:, img['image'].shape[1]:], size=img['image_raw'].shape[2:], mode='bilinear', antialias=True)

        logits = self.D(img, c, update_emas=update_emas)
        return logits

    def run_D_semantic(self, img, c, blur_sigma=0, blur_sigma_raw=0, update_emas=False):
        blur_size = np.floor(blur_sigma * 3)
        if blur_size > 0:
            with torch.autograd.profiler.record_function('blur'):
                f = torch.arange(-blur_size, blur_size + 1, device=img['image'].device).div(blur_sigma).square().neg().exp2()
                img['image'] = upfirdn2d.filter2d(img['image'], f / f.sum())

        if self.augment_pipe is not None:
            augmented_pair = self.augment_pipe(torch.cat([img['image'],
                                                    torch.nn.functional.interpolate(img['image_raw'], size=img['image'].shape[2:], mode='bilinear', antialias=True)],
                                                    dim=1))
            img['image'] = augmented_pair[:, :img['image'].shape[1]]
            img['image_raw'] = torch.nn.functional.interpolate(augmented_pair[:, img['image'].shape[1]:], size=img['image_raw'].shape[2:], mode='bilinear', antialias=True)

        logits = self.D_semantic(img, c, update_emas=update_emas)
        return logits
    
    def accumulate_gradients(self, phase, real_img, real_seg, real_c, gen_z, gen_c, gain, cur_nimg):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth','D_semanticmain', 'D_semanticreg', 'D_semanticboth']
        if self.G.rendering_kwargs.get('density_reg', 0) == 0:
            phase = {'Greg': 'none', 'Gboth': 'Gmain'}.get(phase, phase)
        if self.r1_gamma == 0:
            phase = {'Dreg': 'none', 'Dboth': 'Dmain'}.get(phase, phase)
        blur_sigma = max(1 - cur_nimg / (self.blur_fade_kimg * 1e3), 0) * self.blur_init_sigma if self.blur_fade_kimg > 0 else 0
        r1_gamma = self.r1_gamma

        alpha = min(cur_nimg / (self.gpc_reg_fade_kimg * 1e3), 1) if self.gpc_reg_fade_kimg > 0 else 1
        swapping_prob = (1 - alpha) * 1 + alpha * self.gpc_reg_prob if self.gpc_reg_prob is not None else None

        if self.neural_rendering_resolution_final is not None:
            alpha = min(cur_nimg / (self.neural_rendering_resolution_fade_kimg * 1e3), 1)
            neural_rendering_resolution = int(np.rint(self.neural_rendering_resolution_initial * (1 - alpha) + self.neural_rendering_resolution_final * alpha))
        else:
            neural_rendering_resolution = self.neural_rendering_resolution_initial

        real_img_raw = filtered_resizing(real_img, size=neural_rendering_resolution, f=self.resample_filter, filter_mode=self.filter_mode)
        real_seg_raw = filtered_resizing(real_seg, size=neural_rendering_resolution, f=self.resample_filter, filter_mode=self.filter_mode)

        if self.blur_raw_target:
            blur_size = np.floor(blur_sigma * 3)
            if blur_size > 0:
                f = torch.arange(-blur_size, blur_size + 1, device=real_img_raw.device).div(blur_sigma).square().neg().exp2()
                real_img_raw = upfirdn2d.filter2d(real_img_raw, f / f.sum())

        real_img = {'image': real_img, 'image_raw': real_img_raw, 'seg': real_seg, 'seg_raw': real_seg_raw}

        # Gmain: Maximize logits for generated images.
        
        if phase in ['Gmain', 'Gboth']:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, eik=True, swapping_prob=swapping_prob, neural_rendering_resolution=neural_rendering_resolution)
                gen_logits = self.run_D(gen_img, gen_c, blur_sigma=blur_sigma)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Gmain = torch.nn.functional.softplus(-gen_logits)
                
                
                if hp.softmax:
                    gen_img_semantic = torch.nn.functional.softmax(gen_img['semantic'],1) *2-1
                else:
                    gen_img_semantic = gen_img['semantic']
                input_img = {}
                input_img['image'] = torch.cat(
                    [gen_img['image'].detach(), gen_img_semantic], dim=1)
                
                
                gen_img_semantic_raw = gen_img['semantic_raw']
                    
                if hp.segrgb:
                    if hp.with_back:
                        input_img['image_raw'] = torch.cat(
                            [gen_img['image_raw'].detach(), gen_img_semantic_raw]+[i for i in gen_img['image_raw_seg_list']], dim=1)
                    else:
                        input_img['image_raw'] = torch.cat(
                            [gen_img['image_raw'].detach(), gen_img_semantic_raw]+[i for i in [gen_img['image_raw_seg_list'][0]] + gen_img['image_raw_seg_list'][2:]], dim=1)     
                            # [gen_img['image_raw'].detach(), gen_img_semantic_raw]+[i for i in gen_img['image_raw_seg_list'][1:], dim=1)     
                else:
                    input_img['image_raw'] = torch.cat(
                            [gen_img['image_raw'].detach(), gen_img_semantic_raw], dim=1)
                    
                    
                gen_logits_semantic = self.run_D_semantic(
                    input_img, gen_c, blur_sigma=blur_sigma)
                training_stats.report(
                    'Loss/scores/fake_semantic', gen_logits_semantic)
                training_stats.report(
                    'Loss/signs/fake_semantic', gen_logits_semantic.sign())
                loss_Gmain += torch.nn.functional.softplus(
                    -gen_logits_semantic)  *0.1
                

                training_stats.report('Loss/G/loss', loss_Gmain)


                if hp.semantic_density_detach:
                    rgb_consist_loss = torch.nn.MSELoss()((gen_img['image_raw'].detach()+1)/2,(sum(gen_img['image_raw_seg_list'])+hp.number)/2)
                else:
                    rgb_consist_loss = torch.nn.MSELoss()((gen_img['image_raw']+1)/2,(sum(gen_img['image_raw_seg_list'])+hp.number)/2)

                onehot_semantic_raw = torch.zeros_like(gen_img['semantic_raw']).scatter_(1, torch.argmax(gen_img['semantic_raw'],1).unsqueeze(1), 1.)
                onehot_semantic = torch.zeros_like(gen_img['semantic']).scatter_(1, torch.argmax(gen_img['semantic'],1).unsqueeze(1), 1.)


                semantic_rgb_seg = []
                semantic_rgb_seg_raw = []
                for i in range(hp.number):
                    seg_temp = onehot_semantic[:,i:i+1,...].repeat(1,3,1,1)
                    image_seg_temp = torch.clamp(gen_img['image'],-1,1).clone()

                    image_seg_temp = (image_seg_temp+1)*seg_temp-1
                    # image_seg_temp[seg_temp==0] = -1
                    semantic_rgb_seg.append(image_seg_temp)

                    seg_temp = onehot_semantic_raw[:,i:i+1,...].repeat(1,3,1,1)
                    image_seg_temp = gen_img['image_raw'].clone()
                    image_seg_temp = (image_seg_temp+1)*seg_temp-1
                    # image_seg_temp[seg_temp==0] = -1
                    semantic_rgb_seg_raw.append(image_seg_temp)

                if hp.with_back:
                    rgb_semantic_seg_loss_raw = torch.nn.MSELoss()(torch.cat(semantic_rgb_seg_raw).detach(),torch.cat(gen_img['image_raw_seg_list']))
                else:
                    rgb_semantic_seg_loss_raw = torch.nn.MSELoss()(torch.cat([semantic_rgb_seg_raw[0]] + semantic_rgb_seg_raw[2:]).detach(),torch.cat([gen_img['image_raw_seg_list'][0]]+gen_img['image_raw_seg_list'][2:]))
                    # rgb_semantic_seg_loss_raw = torch.nn.MSELoss()(torch.cat(semantic_rgb_seg_raw[1:]),torch.cat(gen_img['image_raw_seg_list'][1:]))


                sigma_div_concat = sum(gen_img['all_densities_aux'])
                if hp.semantic_density_detach:
                    sigma_consist_loss = torch.nn.MSELoss()(gen_img['all_densities'].detach(), sigma_div_concat)
                else:
                    sigma_consist_loss = torch.nn.MSELoss()(gen_img['all_densities'], sigma_div_concat)


                sdf_out_list = []
                for index, i in enumerate(gen_img['all_sdf_aux']):
                    # if index == 0: # remove background
                    #     continue

                    i = torch.max(i,torch.zeros_like(i))
                    sdf_out_list.append(i)
                sdf_sub = functools.reduce(torch.min, sdf_out_list)
                sdf_global_out = torch.max(gen_img['all_sdf'],torch.zeros_like(i))

                if hp.semantic_density_detach:
                    sdf_consist_loss = torch.nn.MSELoss()(sdf_global_out.detach(), sdf_sub)
                else:
                    sdf_consist_loss = torch.nn.MSELoss()(sdf_global_out, sdf_sub)

                sdf_in_list = []
                for i in gen_img['all_sdf_aux']:
                    i = torch.min(i,torch.zeros_like(i))
                    sdf_in_list.append(i)
                sdf_in_list_1 = torch.split(torch.cat(sdf_in_list,-1).clone(),1,-1)
                sdf_in_list_2 = torch.split(torch.cat(sdf_in_list,-1).clone(),1,-1)
                for i, i_ele in enumerate(sdf_in_list_1):
                    for j, j_ele in enumerate(sdf_in_list_2):
                        if i == j:
                            continue
                        # sdf_consist_loss += torch.nn.MSELoss()(torch.max(i_ele,j_ele),torch.zeros_like(i_ele))                        

                g_eikonal = eikonal_loss(torch.cat(gen_img['eikonal'],-2))
                if hp.eik_seg:
                    g_eikonal += eikonal_loss(torch.cat(gen_img['eikonal_aux'],-2))

                g_minimal_surface = surface_loss(sdf=gen_img['all_sdf'].reshape(gen_img['all_sdf'].shape[0],-1,1) if 0.05 > 0 else None,
                                                                beta=100)
                
                seg_consistency = torch.nn.MSELoss()(onehot_semantic_raw, F.interpolate(onehot_semantic,(128,128)))
                if hp.eik_seg:
                    for ii in range(hp.number):
                        if ii ==0:
                            continue
                        g_minimal_surface += surface_loss(sdf=gen_img['all_sdf_aux'][ii].reshape(gen_img['all_sdf_aux'][ii].shape[0],-1,1) if 0.05 > 0 else None,
                                                                    beta=100)
                
                training_stats.report('Loss/G_eikonal/loss', g_eikonal)
                training_stats.report('Loss/G_g_minimal_surface/loss', g_minimal_surface)
                training_stats.report('Loss/G_rgb_consist_loss/loss', rgb_consist_loss)
                training_stats.report('Loss/G_rgb_semantic_seg_loss_raw/loss', rgb_semantic_seg_loss_raw)
                training_stats.report('Loss/G_sigma_consist_loss/loss', sigma_consist_loss)
                training_stats.report('Loss/G_sdf_consist_loss/loss', sdf_consist_loss)
                training_stats.report('Loss/beta', self.G.decoder.sigmoid_beta)

            with torch.autograd.profiler.record_function('Gmain_backward'):
                (loss_Gmain+0.1*g_eikonal+0.05*g_minimal_surface+0.0*rgb_semantic_seg_loss_raw+0.1*sdf_consist_loss+0.0000*sigma_consist_loss).mean().mul(gain).backward()
                # (loss_Gmain+0.1*g_eikonal+0.05*g_minimal_surface+0.1*rgb_semantic_seg_loss_raw+0.1*sdf_consist_loss+0.0001*sigma_consist_loss).mean().mul(gain).backward()

        # Density Regularization
        if phase in ['Greg', 'Gboth'] and self.G.rendering_kwargs.get('density_reg', 0) > 0 and self.G.rendering_kwargs['reg_type'] == 'l1':
            if swapping_prob is not None:
                c_swapped = torch.roll(gen_c.clone(), 1, 0)
                c_gen_conditioning = torch.where(torch.rand([], device=gen_c.device) < swapping_prob, c_swapped, gen_c)
            else:
                c_gen_conditioning = torch.zeros_like(gen_c)

            ws = self.G.mapping(gen_z, c_gen_conditioning, update_emas=False)
            if self.style_mixing_prob > 0:
                with torch.autograd.profiler.record_function('style_mixing'):
                    cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                    cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                    ws[:, cutoff:]  = self.G.mapping(torch.randn_like(z), c, update_emas=False)[:, cutoff:]
            initial_coordinates = torch.rand((ws.shape[0], 1000, 3), device=ws.device) * 2 - 1
            perturbed_coordinates = initial_coordinates + torch.randn_like(initial_coordinates) * self.G.rendering_kwargs['density_reg_p_dist']
            all_coordinates = torch.cat([initial_coordinates, perturbed_coordinates], dim=1)
            temp = self.G.sample_mixed(all_coordinates, torch.randn_like(all_coordinates), ws, update_emas=False)
            sigma = temp['sigma']
            sdf = temp['sdf']
            sigma_initial = sigma[:, :sigma.shape[1]//2]
            sigma_perturbed = sigma[:, sigma.shape[1]//2:]
            sdf_initial = sdf[:, :sdf.shape[1]//2]
            sdf_perturbed = sdf[:, sdf.shape[1]//2:]
            TVloss = torch.nn.functional.l1_loss(sigma_initial, sigma_perturbed) * self.G.rendering_kwargs['density_reg'] #+ torch.nn.functional.l1_loss(sdf_initial, sdf_perturbed) * self.G.rendering_kwargs['density_reg']
            TVloss.mul(gain).backward()

        # Alternative density regularization
        if phase in ['Greg', 'Gboth'] and self.G.rendering_kwargs.get('density_reg', 0) > 0 and self.G.rendering_kwargs['reg_type'] == 'monotonic-detach':
            if swapping_prob is not None:
                c_swapped = torch.roll(gen_c.clone(), 1, 0)
                c_gen_conditioning = torch.where(torch.rand([], device=gen_c.device) < swapping_prob, c_swapped, gen_c)
            else:
                c_gen_conditioning = torch.zeros_like(gen_c)

            ws = self.G.mapping(gen_z, c_gen_conditioning, update_emas=False)

            initial_coordinates = torch.rand((ws.shape[0], 2000, 3), device=ws.device) * 2 - 1 # Front

            perturbed_coordinates = initial_coordinates + torch.tensor([0, 0, -1], device=ws.device) * (1/256) * self.G.rendering_kwargs['box_warp'] # Behind
            all_coordinates = torch.cat([initial_coordinates, perturbed_coordinates], dim=1)
            sigma = self.G.sample_mixed(all_coordinates, torch.randn_like(all_coordinates), ws, update_emas=False)['sigma']
            sigma_initial = sigma[:, :sigma.shape[1]//2]
            sigma_perturbed = sigma[:, sigma.shape[1]//2:]

            monotonic_loss = torch.relu(sigma_initial.detach() - sigma_perturbed).mean() * 10
            monotonic_loss.mul(gain).backward()


            if swapping_prob is not None:
                c_swapped = torch.roll(gen_c.clone(), 1, 0)
                c_gen_conditioning = torch.where(torch.rand([], device=gen_c.device) < swapping_prob, c_swapped, gen_c)
            else:
                c_gen_conditioning = torch.zeros_like(gen_c)

            ws = self.G.mapping(gen_z, c_gen_conditioning, update_emas=False)
            if self.style_mixing_prob > 0:
                with torch.autograd.profiler.record_function('style_mixing'):
                    cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                    cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                    ws[:, cutoff:] = self.G.mapping(torch.randn_like(z), c, update_emas=False)[:, cutoff:]
            initial_coordinates = torch.rand((ws.shape[0], 1000, 3), device=ws.device) * 2 - 1
            perturbed_coordinates = initial_coordinates + torch.randn_like(initial_coordinates) * (1/256) * self.G.rendering_kwargs['box_warp']
            all_coordinates = torch.cat([initial_coordinates, perturbed_coordinates], dim=1)
            sigma = self.G.sample_mixed(all_coordinates, torch.randn_like(all_coordinates), ws, update_emas=False)['sigma']
            sigma_initial = sigma[:, :sigma.shape[1]//2]
            sigma_perturbed = sigma[:, sigma.shape[1]//2:]

            TVloss = torch.nn.functional.l1_loss(sigma_initial, sigma_perturbed) * self.G.rendering_kwargs['density_reg']
            TVloss.mul(gain).backward()

        # Alternative density regularization
        if phase in ['Greg', 'Gboth'] and self.G.rendering_kwargs.get('density_reg', 0) > 0 and self.G.rendering_kwargs['reg_type'] == 'monotonic-fixed':
            if swapping_prob is not None:
                c_swapped = torch.roll(gen_c.clone(), 1, 0)
                c_gen_conditioning = torch.where(torch.rand([], device=gen_c.device) < swapping_prob, c_swapped, gen_c)
            else:
                c_gen_conditioning = torch.zeros_like(gen_c)

            ws_tex, ws_shp = self.G.mapping(gen_z, c_gen_conditioning, update_emas=False)

            initial_coordinates = torch.rand((ws_tex.shape[0], 2000, 3), device=ws_tex.device) * 2 - 1 # Front

            perturbed_coordinates = initial_coordinates + torch.tensor([0, 0, -1], device=ws_tex.device) * (1/256) * self.G.rendering_kwargs['box_warp'] # Behind
            all_coordinates = torch.cat([initial_coordinates, perturbed_coordinates], dim=1)
            sigma = self.G.sample_mixed(all_coordinates, torch.randn_like(all_coordinates), ws_tex, ws_shp, update_emas=False)['sigma']
            sigma_initial = sigma[:, :sigma.shape[1]//2]
            sigma_perturbed = sigma[:, sigma.shape[1]//2:]

            monotonic_loss = torch.relu(sigma_initial - sigma_perturbed).mean() * 10
            monotonic_loss.mul(gain).backward()


            if swapping_prob is not None:
                c_swapped = torch.roll(gen_c.clone(), 1, 0)
                c_gen_conditioning = torch.where(torch.rand([], device=gen_c.device) < swapping_prob, c_swapped, gen_c)
            else:
                c_gen_conditioning = torch.zeros_like(gen_c)

            ws_tex, ws_shp = self.G.mapping(gen_z, c_gen_conditioning, update_emas=False)
            if self.style_mixing_prob > 0:
                with torch.autograd.profiler.record_function('style_mixing'):
                    cutoff = torch.empty([], dtype=torch.int64, device=ws_tex.device).random_(1, ws.shape[1])
                    cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                    ws_tex[:, cutoff:], ws_shp[:, cutoff:] = self.G.mapping(torch.randn_like(z), c, update_emas=False)[:, cutoff:]
            initial_coordinates = torch.rand((ws_tex.shape[0], 1000, 3), device=ws.device) * 2 - 1
            perturbed_coordinates = initial_coordinates + torch.randn_like(initial_coordinates) * (1/256) * self.G.rendering_kwargs['box_warp']
            all_coordinates = torch.cat([initial_coordinates, perturbed_coordinates], dim=1)
            sigma = self.G.sample_mixed(all_coordinates, torch.randn_like(all_coordinates), ws, update_emas=False)['sigma']
            sigma_initial = sigma[:, :sigma.shape[1]//2]
            sigma_perturbed = sigma[:, sigma.shape[1]//2:]

            TVloss = torch.nn.functional.l1_loss(sigma_initial, sigma_perturbed) * self.G.rendering_kwargs['density_reg']
            TVloss.mul(gain).backward()

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if phase in ['Dmain', 'Dboth']:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_img, _gen_ws  = self.run_G(gen_z, gen_c, eik=False, swapping_prob=swapping_prob, neural_rendering_resolution=neural_rendering_resolution, update_emas=True)
                gen_logits = self.run_D(gen_img, gen_c, blur_sigma=blur_sigma, update_emas=True)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Dgen = torch.nn.functional.softplus(gen_logits)
            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.mean().mul(gain).backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if phase in ['Dmain', 'Dreg', 'Dboth']:
            name = 'Dreal' if phase == 'Dmain' else 'Dr1' if phase == 'Dreg' else 'Dreal_Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp_image = real_img['image'].detach(
                ).requires_grad_(phase in ['Dreg', 'Dboth'])
                real_img_tmp_image_raw = real_img['image_raw'].detach(
                ).requires_grad_(phase in ['Dreg', 'Dboth'])
                real_img_tmp = {'image': real_img_tmp_image,
                                'image_raw': real_img_tmp_image_raw}

                real_logits = self.run_D(
                    real_img_tmp, real_c, blur_sigma=blur_sigma)
                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())

                loss_Dreal = 0
                if phase in ['Dmain', 'Dboth']:
                    loss_Dreal = torch.nn.functional.softplus(-real_logits)
                    training_stats.report(
                        'Loss/D/loss', loss_Dgen + loss_Dreal)

                loss_Dr1 = 0
                if phase in ['Dreg', 'Dboth']:
                    if self.dual_discrimination:
                        with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                            r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[
                                                           real_img_tmp['image'], real_img_tmp['image_raw']], create_graph=True, only_inputs=True)
                            r1_grads_image = r1_grads[0]
                            r1_grads_image_raw = r1_grads[1]
                        r1_penalty = r1_grads_image.square().sum(
                            [1, 2, 3]) + r1_grads_image_raw.square().sum([1, 2, 3])
                    else:  # single discrimination
                        with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                            r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[
                                                           real_img_tmp['image']], create_graph=True, only_inputs=True)
                            r1_grads_image = r1_grads[0]
                        r1_penalty = r1_grads_image.square().sum([1, 2, 3])
                    loss_Dr1 = r1_penalty * (r1_gamma / 2)
                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/D/reg', loss_Dr1)

            with torch.autograd.profiler.record_function(name + '_backward'):
                (loss_Dreal + loss_Dr1).mean().mul(gain).backward()

                
                
                
                
        # D_semanticmain: Minimize logits for generated images and masks.
        loss_Dgen_semantic = 0
        if phase in ['D_semanticmain', 'D_semanticboth']:
            with torch.autograd.profiler.record_function('Dgen_semantic_forward'):
                gen_img, _gen_ws  = self.run_G(gen_z, gen_c, eik=False, swapping_prob=swapping_prob, neural_rendering_resolution=neural_rendering_resolution, update_emas=True)
                input_img = {}

                
                if hp.softmax:
                    gen_img_semantic = torch.nn.functional.softmax(gen_img['semantic'],1)  *2-1
                else:
                    gen_img_semantic = gen_img['semantic']
                    
                input_img['image'] = torch.cat(
                    [gen_img['image'], gen_img_semantic], dim=1)
                
                
                gen_img_semantic_raw = gen_img['semantic_raw']
                
                    
                if hp.segrgb:
                    if hp.with_back:
                        input_img['image_raw'] = torch.cat(
                            [gen_img['image_raw'], gen_img_semantic_raw]+gen_img['image_raw_seg_list'], dim=1)
                    else:
                        input_img['image_raw'] = torch.cat(
                            [gen_img['image_raw'], gen_img_semantic_raw]+[gen_img['image_raw_seg_list'][0]]+gen_img['image_raw_seg_list'][2:], dim=1)
                            # [gen_img['image_raw'], gen_img_semantic_raw]+gen_img['image_raw_seg_list'][1:], dim=1)
                else:
                    input_img['image_raw'] = torch.cat(
                            [gen_img['image_raw'], gen_img_semantic_raw], dim=1)
                    
                    
                gen_logits_semantic = self.run_D_semantic(
                    input_img, gen_c, blur_sigma=blur_sigma)

                training_stats.report(
                    'Loss/scores/fake_semantic', gen_logits_semantic)
                training_stats.report(
                    'Loss/signs/fake_semantic', gen_logits_semantic.sign())
                loss_Dgen_semantic = torch.nn.functional.softplus(
                    gen_logits_semantic)
            with torch.autograd.profiler.record_function('Dgen_semantic_backward'):
                loss_Dgen_semantic.mean().mul(gain).backward()

        # D_semanticmain: Maximize logits for real images and masks.
        # Dr1: Apply R1 regularization.
        if phase in ['D_semanticmain', 'D_semanticreg', 'D_semanticboth']:
            name = 'Dreal_semantic' if phase == 'D_semanticmain' else 'Dr1_semantic' if phase == 'D_semanticreg' else 'Dreal_Dr1_semantic'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp_image = real_img['image'].detach().requires_grad_(
                    phase in ['D_semanticreg', 'D_semanticboth'])
                real_img_tmp_image_raw = real_img['image_raw'].detach().requires_grad_(
                    phase in ['D_semanticreg', 'D_semanticboth'])


                real_semantic_tmp_image = real_img['seg']*2-1
                real_mask_tmp_image = (real_semantic_tmp_image).detach().requires_grad_(phase in ['D_semanticreg', 'D_semanticboth'])

                real_semantic_tmp_image_raw = real_img['seg_raw']*2-1
                real_mask_tmp_image_raw = (real_semantic_tmp_image_raw).detach().requires_grad_(phase in ['D_semanticreg', 'D_semanticboth'])                    
                

                if hp.segrgb:
                    image_raw_seg_list = []
                    image_seg_list = []
                    for i in range(hp.number):
                        seg_temp = real_semantic_tmp_image[:,i:i+1,...].repeat(1,3,1,1)
                        seg_temp_raw = real_semantic_tmp_image_raw[:,i:i+1,...].repeat(1,3,1,1)
                        image_raw_seg_temp = real_img_tmp_image_raw.clone()
                        image_raw_seg_temp = (image_raw_seg_temp+1)*((seg_temp_raw+1)/2)-1
                        image_seg_temp = real_img_tmp_image.clone()
                        image_seg_temp = (image_seg_temp+1)*((seg_temp+1)/2)-1

                        image_raw_seg_temp = image_raw_seg_temp.detach().requires_grad_(phase in ['D_semanticreg', 'D_semanticboth'])
                        image_seg_temp = image_seg_temp.detach().requires_grad_(phase in ['D_semanticreg', 'D_semanticboth'])

                        image_raw_seg_list.append(image_raw_seg_temp)
                        image_seg_list.append(image_seg_temp)
                    
                
                if hp.segrgb:
                    if hp.with_back:
                        real_img_tmp = {'image': torch.cat([real_img_tmp_image, real_mask_tmp_image], dim=1), 'image_raw': torch.cat([
                            real_img_tmp_image_raw, real_mask_tmp_image_raw]+image_raw_seg_list, dim=1)}
                    else:
                        real_img_tmp = {'image': torch.cat([real_img_tmp_image, real_mask_tmp_image], dim=1), 'image_raw': torch.cat([
                            real_img_tmp_image_raw, real_mask_tmp_image_raw]+[image_raw_seg_list[0]]+image_raw_seg_list[2:], dim=1)}
                            # real_img_tmp_image_raw, real_mask_tmp_image_raw]+image_raw_seg_list[1:], dim=1)}
                else:
                    real_img_tmp = {'image': torch.cat([real_img_tmp_image, real_mask_tmp_image], dim=1), 'image_raw': torch.cat([
                            real_img_tmp_image_raw, real_mask_tmp_image_raw], dim=1)}

                real_logits_semantic = self.run_D_semantic(
                    real_img_tmp, real_c, blur_sigma=blur_sigma)
                training_stats.report(
                    'Loss/scores/real_semantic', real_logits_semantic)
                training_stats.report(
                    'Loss/signs/real_semantic', real_logits_semantic.sign())

                loss_Dreal_semantic = 0
                if phase in ['D_semanticmain', 'D_semanticboth']:
                    loss_Dreal_semantic = torch.nn.functional.softplus(
                        -real_logits_semantic)
                    training_stats.report(
                        'Loss/D/loss_semantic', loss_Dgen_semantic + loss_Dreal_semantic)

                loss_Dr1_semantic = 0
                if phase in ['D_semanticreg', 'D_semanticboth']:
                    if self.dual_discrimination:
                        with torch.autograd.profiler.record_function('r1_grads_semantic'), conv2d_gradfix.no_weight_gradients():
                            r1_grads_semantic = torch.autograd.grad(outputs=[real_logits_semantic.sum()], inputs=[
                                                                    real_img_tmp['image'], real_img_tmp['image_raw']], create_graph=True, only_inputs=True)
                            r1_grads_image_semantic = r1_grads_semantic[0]
                            r1_grads_image_raw_semantic = r1_grads_semantic[1]
                        r1_penalty_semantic = r1_grads_image_semantic.square().sum(
                            [1, 2, 3]) + r1_grads_image_raw_semantic.square().sum([1, 2, 3])
                    else:
                        with torch.autograd.profiler.record_function('r1_grads_semantic'):
                            r1_grads_semantic = torch.autograd.grad(outputs=[real_logits_semantic.sum(
                            )], inputs=[real_img_tmp['image']], create_graph=True, only_inputs=True)
                            r1_grads_image_semantic = r1_grads_semantic[0]
                        r1_penalty_semantic = r1_grads_image_semantic.square().sum([
                            1, 2, 3])
                    loss_Dr1_semantic = r1_penalty_semantic * r1_gamma *10* 0.5
                    # loss_Dr1_semantic = r1_penalty_semantic * r1_gamma * 0.5
                    training_stats.report(
                        'Loss/r1_penalty_semantic', r1_penalty_semantic)
                    training_stats.report(
                        'Loss/D/reg_semantic', loss_Dr1_semantic)

            with torch.autograd.profiler.record_function(name + '_backward'):
                (loss_Dreal_semantic + loss_Dr1_semantic).mean().mul(gain).backward()
#----------------------------------------------------------------------------
