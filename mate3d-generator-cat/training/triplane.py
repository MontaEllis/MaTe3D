# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import torch
from torch_utils import persistence
from training.networks_stylegan2 import Generator as StyleGAN2Backbone
from training.volumetric_rendering.renderer import ImportanceRenderer
from training.volumetric_rendering.ray_sampler import RaySampler
import dnnlib
from hparams import hparams as hp
import torch.autograd as autograd
import torch.nn.functional as F

@persistence.persistent_class
class TriPlaneGenerator(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality.
        c_dim,                      # Conditioning label (C) dimensionality.
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output resolution.
        img_channels,               # Number of output color channels.
        sr_num_fp16_res     = 0,
        mapping_kwargs      = {},   # Arguments for MappingNetwork.
        rendering_kwargs    = {},
        sr_kwargs = {},
        **synthesis_kwargs,         # Arguments for SynthesisNetwork.
    ):
        super().__init__()
        self.z_dim=z_dim
        self.c_dim=c_dim
        self.w_dim=w_dim
        self.img_resolution=img_resolution
        self.img_channels=img_channels
        self.renderer = ImportanceRenderer()
        self.ray_sampler = RaySampler()
        self.backbone = StyleGAN2Backbone(z_dim, c_dim, w_dim, img_resolution=256, img_channels=32*3, mapping_kwargs=mapping_kwargs, **synthesis_kwargs)
        self.superresolution = dnnlib.util.construct_class_by_name(class_name=rendering_kwargs['superresolution_module'], channels=32, img_resolution=img_resolution, sr_num_fp16_res=sr_num_fp16_res, sr_antialias=rendering_kwargs['sr_antialias'], **sr_kwargs)
        self.superresolution_semantic = dnnlib.util.construct_class_by_name(class_name=rendering_kwargs['superresolution_module_semantic'], channels=32, img_resolution=img_resolution, sr_num_fp16_res=sr_num_fp16_res, sr_antialias=rendering_kwargs['sr_antialias'],semantic_channels=hp.number, **sr_kwargs)
        self.decoder = OSGDecoder(32, {'decoder_lr_mul': rendering_kwargs.get('decoder_lr_mul', 1), 'decoder_output_dim': 32})
        self.neural_rendering_resolution = 64
        self.rendering_kwargs = rendering_kwargs
    
        self._last_planes = None
    
    def mapping(self, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False):
        if self.rendering_kwargs['c_gen_conditioning_zero']:
                c = torch.zeros_like(c)
        return  self.backbone.mapping(z, c * self.rendering_kwargs.get('c_scale', 0), truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)


    def synthesis_ablation(self, ws, c, eik=False, neural_rendering_resolution=None, update_emas=False, cache_backbone=False, use_cached_backbone=False, **synthesis_kwargs):
        cam2world_matrix = c[:, :16].view(-1, 4, 4)
        intrinsics = c[:, 16:25].view(-1, 3, 3)

        if neural_rendering_resolution is None:
            neural_rendering_resolution = self.neural_rendering_resolution
        else:
            self.neural_rendering_resolution = neural_rendering_resolution

        # Create a batch of rays for volume rendering
        ray_origins, ray_directions = self.ray_sampler(cam2world_matrix, intrinsics, neural_rendering_resolution)



        # Create triplanes by running StyleGAN backbone
        N, M, _ = ray_origins.shape
        if use_cached_backbone and self._last_planes is not None:
            planes = self._last_planes
        else:
            planes_tex, planes_sdf = self.backbone.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
        if cache_backbone:
            self._last_planes = planes

        # Reshape output into three 32-channel planes
        planes_tex = planes_tex.view(len(planes_tex), 3, 32, planes_tex.shape[-2], planes_tex.shape[-1])
        planes_sdf = planes_sdf.view(len(planes_sdf), 3, 32, planes_sdf.shape[-2], planes_sdf.shape[-1])


        planes_sdf += torch.rand_like(planes_sdf)*50
        # Perform volume rendering
        # feature_samples, depth_samples, weights_samples,all_densities = self.renderer(planes, self.decoder, ray_origins, ray_directions, self.rendering_kwargs) # channels last
        feature_samples, depth_samples, weights_samples, feature_samples_aux, depth_samples_aux, semantic_sample, eikonal, eikonal_aux, all_sdf, all_densities, all_pts, all_sdf_aux, all_densities_aux = self.renderer(planes_tex, planes_sdf, self.decoder, ray_origins, ray_directions, eik, self.rendering_kwargs) # channels last

        # Reshape into 'raw' neural-rendered image
        H = W = self.neural_rendering_resolution
        feature_image = feature_samples.permute(0, 2, 1).reshape(N, feature_samples.shape[-1], H, W).contiguous()
        semantic_sample = semantic_sample.permute(0, 2, 1).reshape(N, semantic_sample.shape[-1], H, W).contiguous()
        depth_image = depth_samples.permute(0, 2, 1).reshape(N, 1, H, W)
        # Run superresolution to get final image
        rgb_image = feature_image[:, :3]
        
        rgb_image_seg_list = []
        depth_image_seg_list = []
        feature_image_seg_list = []
        
        for i in range(hp.number):
            feature_samples_temp = feature_samples_aux[str(i)]
            depth_samples_temp = depth_samples_aux[str(i)]
            feature_image_temp = feature_samples_temp.permute(0, 2, 1).reshape(N, feature_samples_temp.shape[-1], H, W).contiguous()
            feature_image_seg_list.append(feature_image_temp)
            depth_image_temp = depth_samples_temp.permute(0, 2, 1).reshape(N, 1, H, W)
            # Run superresolution to get final image
            rgb_image_temp = feature_image_temp[:, :3]

            depth_image_seg_list.append(depth_image_temp)
            rgb_image_seg_list.append(rgb_image_temp)            
        

        # semantic_sample += torch.rand_like(semantic_sample)*1.5

        sr_image = self.superresolution(rgb_image, feature_image, ws, noise_mode=self.rendering_kwargs['superresolution_noise_mode'], **{k:synthesis_kwargs[k] for k in synthesis_kwargs.keys() if k != 'noise_mode'})
        if hp.semantic_feat_detach:
            sr_semantic = self.superresolution_semantic(semantic_sample, feature_image.detach(), ws, noise_mode=self.rendering_kwargs['superresolution_noise_mode'], **{k:synthesis_kwargs[k] for k in synthesis_kwargs.keys() if k != 'noise_mode'})
        else:
            sr_semantic = self.superresolution_semantic(semantic_sample, feature_image, ws, noise_mode=self.rendering_kwargs['superresolution_noise_mode'], **{k:synthesis_kwargs[k] for k in synthesis_kwargs.keys() if k != 'noise_mode'})            
        return {'image': sr_image, 'image_raw': rgb_image, 'image_depth': depth_image, 'image_raw_seg_list': rgb_image_seg_list, 'image_depth_seg_list': depth_image_seg_list, 'semantic_raw': semantic_sample, 'semantic':sr_semantic, 'eikonal': eikonal,'eikonal_aux': eikonal_aux, 'all_sdf': all_sdf, 'all_sdf_aux':all_sdf_aux, 'all_densities':all_densities, 'all_densities_aux':all_densities_aux, 'pts':all_pts, 'ray_end':self.rendering_kwargs['ray_end'], 'ray_start':self.rendering_kwargs['ray_start']}
        # return {'image': sr_image, 'image_raw': rgb_image, 'image_depth': depth_image,'density_origin':all_densities}
    
    
    def synthesis(self, ws, c, eik=False, neural_rendering_resolution=None, update_emas=False, cache_backbone=False, use_cached_backbone=False, **synthesis_kwargs):
        cam2world_matrix = c[:, :16].view(-1, 4, 4)
        intrinsics = c[:, 16:25].view(-1, 3, 3)

        if neural_rendering_resolution is None:
            neural_rendering_resolution = self.neural_rendering_resolution
        else:
            self.neural_rendering_resolution = neural_rendering_resolution

        # Create a batch of rays for volume rendering
        ray_origins, ray_directions = self.ray_sampler(cam2world_matrix, intrinsics, neural_rendering_resolution)



        # Create triplanes by running StyleGAN backbone
        N, M, _ = ray_origins.shape
        if use_cached_backbone and self._last_planes is not None:
            planes = self._last_planes
        else:
            planes_tex, planes_sdf = self.backbone.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
        if cache_backbone:
            self._last_planes = planes

        # Reshape output into three 32-channel planes
        planes_tex = planes_tex.view(len(planes_tex), 3, 32, planes_tex.shape[-2], planes_tex.shape[-1])
        planes_sdf = planes_sdf.view(len(planes_sdf), 3, 32, planes_sdf.shape[-2], planes_sdf.shape[-1])

        # Perform volume rendering
        # feature_samples, depth_samples, weights_samples,all_densities = self.renderer(planes, self.decoder, ray_origins, ray_directions, self.rendering_kwargs) # channels last
        feature_samples, depth_samples, weights_samples, feature_samples_aux, depth_samples_aux, semantic_sample, eikonal, eikonal_aux, all_sdf, all_densities, all_pts, all_sdf_aux, all_densities_aux = self.renderer(planes_tex, planes_sdf, self.decoder, ray_origins, ray_directions, eik, self.rendering_kwargs) # channels last

        # Reshape into 'raw' neural-rendered image
        H = W = self.neural_rendering_resolution
        feature_image = feature_samples.permute(0, 2, 1).reshape(N, feature_samples.shape[-1], H, W).contiguous()
        semantic_sample = semantic_sample.permute(0, 2, 1).reshape(N, semantic_sample.shape[-1], H, W).contiguous()
        depth_image = depth_samples.permute(0, 2, 1).reshape(N, 1, H, W)
        # Run superresolution to get final image
        rgb_image = feature_image[:, :3]
        
        rgb_image_seg_list = []
        depth_image_seg_list = []
        feature_image_seg_list = []
        
        for i in range(hp.number):
            feature_samples_temp = feature_samples_aux[str(i)]
            depth_samples_temp = depth_samples_aux[str(i)]
            feature_image_temp = feature_samples_temp.permute(0, 2, 1).reshape(N, feature_samples_temp.shape[-1], H, W).contiguous()
            feature_image_seg_list.append(feature_image_temp)
            depth_image_temp = depth_samples_temp.permute(0, 2, 1).reshape(N, 1, H, W)
            # Run superresolution to get final image
            rgb_image_temp = feature_image_temp[:, :3]

            depth_image_seg_list.append(depth_image_temp)
            rgb_image_seg_list.append(rgb_image_temp)            
        

        sr_image = self.superresolution(rgb_image, feature_image, ws, noise_mode=self.rendering_kwargs['superresolution_noise_mode'], **{k:synthesis_kwargs[k] for k in synthesis_kwargs.keys() if k != 'noise_mode'})
        if hp.semantic_feat_detach:
            sr_semantic = self.superresolution_semantic(semantic_sample, feature_image.detach(), ws, noise_mode=self.rendering_kwargs['superresolution_noise_mode'], **{k:synthesis_kwargs[k] for k in synthesis_kwargs.keys() if k != 'noise_mode'})
        else:
            sr_semantic = self.superresolution_semantic(semantic_sample, feature_image, ws, noise_mode=self.rendering_kwargs['superresolution_noise_mode'], **{k:synthesis_kwargs[k] for k in synthesis_kwargs.keys() if k != 'noise_mode'})            
        return {'image': sr_image, 'image_raw': rgb_image, 'image_depth': depth_image, 'image_raw_seg_list': rgb_image_seg_list, 'image_depth_seg_list': depth_image_seg_list, 'semantic_raw': semantic_sample, 'semantic':sr_semantic, 'eikonal': eikonal,'eikonal_aux': eikonal_aux, 'all_sdf': all_sdf, 'all_sdf_aux':all_sdf_aux, 'all_densities':all_densities, 'all_densities_aux':all_densities_aux, 'pts':all_pts, 'ray_end':self.rendering_kwargs['ray_end'], 'ray_start':self.rendering_kwargs['ray_start']}
        # return {'image': sr_image, 'image_raw': rgb_image, 'image_depth': depth_image,'density_origin':all_densities}

    def synthesis_normal(self, ws, c, eik=True, neural_rendering_resolution=None, update_emas=False, cache_backbone=False, use_cached_backbone=False, **synthesis_kwargs):
        cam2world_matrix = c[:, :16].view(-1, 4, 4)
        intrinsics = c[:, 16:25].view(-1, 3, 3)

        if neural_rendering_resolution is None:
            neural_rendering_resolution = self.neural_rendering_resolution
        else:
            self.neural_rendering_resolution = neural_rendering_resolution

        # Create a batch of rays for volume rendering
        ray_origins, ray_directions = self.ray_sampler(cam2world_matrix, intrinsics, neural_rendering_resolution)



        # Create triplanes by running StyleGAN backbone
        N, M, _ = ray_origins.shape
        if use_cached_backbone and self._last_planes is not None:
            planes = self._last_planes
        else:
            planes_tex, planes_sdf = self.backbone.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
        if cache_backbone:
            self._last_planes = planes

        # Reshape output into three 32-channel planes
        planes_tex = planes_tex.view(len(planes_tex), 3, 32, planes_tex.shape[-2], planes_tex.shape[-1])
        planes_sdf = planes_sdf.view(len(planes_sdf), 3, 32, planes_sdf.shape[-2], planes_sdf.shape[-1])

        # Perform volume rendering
        # feature_samples, depth_samples, weights_samples,all_densities = self.renderer(planes, self.decoder, ray_origins, ray_directions, self.rendering_kwargs) # channels last
        feature_samples, depth_samples, weights_samples, feature_samples_aux, depth_samples_aux, semantic_sample, eikonal, eikonal_aux, all_sdf, all_densities, all_pts, all_sdf_aux, all_densities_aux = self.renderer(planes_tex, planes_sdf, self.decoder, ray_origins, ray_directions, eik, self.rendering_kwargs) # channels last

        # Reshape into 'raw' neural-rendered image
        H = W = self.neural_rendering_resolution
        feature_image = feature_samples.permute(0, 2, 1).reshape(N, feature_samples.shape[-1], H, W).contiguous()
        semantic_sample = semantic_sample.permute(0, 2, 1).reshape(N, semantic_sample.shape[-1], H, W).contiguous()
        depth_image = depth_samples.permute(0, 2, 1).reshape(N, 1, H, W)
        # Run superresolution to get final image
        rgb_image = feature_image[:, :3]
        
        rgb_image_seg_list = []
        depth_image_seg_list = []
        feature_image_seg_list = []
        
        for i in range(hp.number):
            feature_samples_temp = feature_samples_aux[str(i)]
            depth_samples_temp = depth_samples_aux[str(i)]
            feature_image_temp = feature_samples_temp.permute(0, 2, 1).reshape(N, feature_samples_temp.shape[-1], H, W).contiguous()
            feature_image_seg_list.append(feature_image_temp)
            depth_image_temp = depth_samples_temp.permute(0, 2, 1).reshape(N, 1, H, W)
            # Run superresolution to get final image
            rgb_image_temp = feature_image_temp[:, :3]

            depth_image_seg_list.append(depth_image_temp)
            rgb_image_seg_list.append(rgb_image_temp)            
        

        sr_image = self.superresolution(rgb_image, feature_image, ws, noise_mode=self.rendering_kwargs['superresolution_noise_mode'], **{k:synthesis_kwargs[k] for k in synthesis_kwargs.keys() if k != 'noise_mode'})
        if hp.semantic_feat_detach:
            sr_semantic = self.superresolution_semantic(semantic_sample, feature_image.detach(), ws, noise_mode=self.rendering_kwargs['superresolution_noise_mode'], **{k:synthesis_kwargs[k] for k in synthesis_kwargs.keys() if k != 'noise_mode'})
        else:
            sr_semantic = self.superresolution_semantic(semantic_sample, feature_image, ws, noise_mode=self.rendering_kwargs['superresolution_noise_mode'], **{k:synthesis_kwargs[k] for k in synthesis_kwargs.keys() if k != 'noise_mode'})            
        return {'image': sr_image, 'image_raw': rgb_image, 'image_depth': depth_image, 'image_raw_seg_list': rgb_image_seg_list, 'image_depth_seg_list': depth_image_seg_list, 'semantic_raw': semantic_sample, 'semantic':sr_semantic, 'eikonal': eikonal,'eikonal_aux': eikonal_aux, 'all_sdf': all_sdf, 'all_sdf_aux':all_sdf_aux, 'all_densities':all_densities, 'all_densities_aux':all_densities_aux, 'pts':all_pts, 'ray_end':self.rendering_kwargs['ray_end'], 'ray_start':self.rendering_kwargs['ray_start']}
        # return {'image': sr_image, 'image_raw': rgb_image, 'image_depth': depth_image,'density_origin':all_densities}
        
    def synthesis_with_coord_first(self, ws, c, coord, eik=False, neural_rendering_resolution=None, update_emas=False, cache_backbone=False, use_cached_backbone=False, **synthesis_kwargs):
        cam2world_matrix = c[:, :16].view(-1, 4, 4)
        intrinsics = c[:, 16:25].view(-1, 3, 3)
        batch_size = 1
        num_rays = 16384
        samples_per_ray = 48
        N_importance = 48

        coordinates_all, direction, depths_all = coord

        if neural_rendering_resolution is None:
            neural_rendering_resolution = self.neural_rendering_resolution
        else:
            self.neural_rendering_resolution = neural_rendering_resolution

        # Create a batch of rays for volume rendering
        ray_origins, ray_directions = self.ray_sampler(cam2world_matrix, intrinsics, neural_rendering_resolution)



        # Create triplanes by running StyleGAN backbone
        N, M, _ = ray_origins.shape
        if use_cached_backbone and self._last_planes is not None:
            planes = self._last_planes
        else:
            planes_tex, planes_sdf = self.backbone.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
        if cache_backbone:
            self._last_planes = planes

        # Reshape output into three 32-channel planes
        planes_tex = planes_tex.view(len(planes_tex), 3, 32, planes_tex.shape[-2], planes_tex.shape[-1])
        planes_sdf = planes_sdf.view(len(planes_sdf), 3, 32, planes_sdf.shape[-2], planes_sdf.shape[-1])

        # Perform volume rendering
        # feature_samples, depth_samples, weights_samples,all_densities = self.renderer(planes, self.decoder, ray_origins, ray_directions, self.rendering_kwargs) # channels last
        sampled_features_tex,sampled_features_shp = self.renderer.run_model_get_planes(planes_tex, planes_sdf, self.decoder, coordinates_all.reshape(1,-1,3), ray_directions.unsqueeze(-2).expand(-1, -1, samples_per_ray*2, -1).reshape(batch_size, -1, 3), self.rendering_kwargs) # channels last
        




        return sampled_features_tex,sampled_features_shp

    def synthesis_with_coord(self, ws, c, coord, eik=False, neural_rendering_resolution=None, update_emas=False, cache_backbone=False, use_cached_backbone=False, **synthesis_kwargs):
        cam2world_matrix = c[:, :16].view(-1, 4, 4)
        intrinsics = c[:, 16:25].view(-1, 3, 3)
        batch_size = 1
        num_rays = 16384
        samples_per_ray = 48
        N_importance = 48

        coordinates_coarse, coordinates_fine, directions, depths_coarse, depths_fine = coord

        if neural_rendering_resolution is None:
            neural_rendering_resolution = self.neural_rendering_resolution
        else:
            self.neural_rendering_resolution = neural_rendering_resolution

        # Create a batch of rays for volume rendering
        ray_origins, ray_directions = self.ray_sampler(cam2world_matrix, intrinsics, neural_rendering_resolution)



        # Create triplanes by running StyleGAN backbone
        N, M, _ = ray_origins.shape
        if use_cached_backbone and self._last_planes is not None:
            planes = self._last_planes
        else:
            planes_tex, planes_sdf = self.backbone.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
        if cache_backbone:
            self._last_planes = planes

        # Reshape output into three 32-channel planes
        planes_tex = planes_tex.view(len(planes_tex), 3, 32, planes_tex.shape[-2], planes_tex.shape[-1])
        planes_sdf = planes_sdf.view(len(planes_sdf), 3, 32, planes_sdf.shape[-2], planes_sdf.shape[-1])

        # Perform volume rendering
        # feature_samples, depth_samples, weights_samples,all_densities = self.renderer(planes, self.decoder, ray_origins, ray_directions, self.rendering_kwargs) # channels last
        out_coarse = self.renderer.run_model(planes_tex, planes_sdf, self.decoder, coordinates_coarse, directions, self.rendering_kwargs) # channels last
        
        
        colors_coarse = out_coarse['rgb']
        densities_coarse = out_coarse['sigma']
        sdfs_coarse = out_coarse['sdf'] 
        densities_aux_coarse = out_coarse['aux_sigma']
        sdfs_aux_coarse = out_coarse['aux_sdf']
        colors_coarse = colors_coarse.reshape(batch_size, num_rays, samples_per_ray, colors_coarse.shape[-1])
        densities_coarse = densities_coarse.reshape(batch_size, num_rays, samples_per_ray, 1)
        sdfs_coarse = sdfs_coarse.reshape(batch_size, num_rays, samples_per_ray, 1)
        densities_aux_coarse = densities_aux_coarse.reshape(batch_size, num_rays, samples_per_ray, hp.number)
        sdfs_aux_coarse = sdfs_aux_coarse.reshape(batch_size, num_rays, samples_per_ray, hp.number)




        out_fine = self.renderer.run_model(planes_tex, planes_sdf, self.decoder, coordinates_fine, directions, self.rendering_kwargs) # channels last

        colors_fine = out_fine['rgb']
        densities_fine = out_fine['sigma']
        sdfs_fine = out_fine['sdf']
        densities_aux_fine = out_fine['aux_sigma']
        sdfs_aux_fine = out_fine['aux_sdf']
        colors_fine = colors_fine.reshape(batch_size, num_rays, N_importance, colors_fine.shape[-1])
        densities_fine = densities_fine.reshape(batch_size, num_rays, N_importance, 1)
        sdfs_fine = sdfs_fine.reshape(batch_size, num_rays, N_importance, 1)
        densities_aux_fine = densities_aux_fine.reshape(batch_size, num_rays, N_importance, hp.number)
        sdfs_aux_fine = sdfs_aux_fine.reshape(batch_size, num_rays, N_importance, hp.number)


        all_depths, all_colors, all_densities, all_sdfs, all_densities_aux, all_sdfs_aux, all_pts = self.renderer.unify_samples_add_sdf_add_pts(depths_coarse, coordinates_coarse.reshape(batch_size, num_rays, samples_per_ray, coordinates_coarse.shape[-1]), colors_coarse, densities_coarse, sdfs_coarse,densities_aux_coarse,sdfs_aux_coarse,
                                                                  depths_fine, coordinates_fine.reshape(batch_size, num_rays, samples_per_ray, coordinates_fine.shape[-1]), colors_fine, densities_fine, sdfs_fine,densities_aux_fine,sdfs_aux_fine)


        return all_depths, all_colors, all_densities

    def synthesis_with_coord_second(self, ws, c, coord, eik=False, neural_rendering_resolution=None, update_emas=False, cache_backbone=False, use_cached_backbone=False, **synthesis_kwargs):
        cam2world_matrix = c[:, :16].view(-1, 4, 4)
        intrinsics = c[:, 16:25].view(-1, 3, 3)
        batch_size = 1
        num_rays = 16384
        samples_per_ray = 48
        N_importance = 48

        coordinates_all, directions, depths_all = coord

        if neural_rendering_resolution is None:
            neural_rendering_resolution = self.neural_rendering_resolution
        else:
            self.neural_rendering_resolution = neural_rendering_resolution

        # Create a batch of rays for volume rendering
        ray_origins, ray_directions = self.ray_sampler(cam2world_matrix, intrinsics, neural_rendering_resolution)



        # Create triplanes by running StyleGAN backbone
        N, M, _ = ray_origins.shape
        if use_cached_backbone and self._last_planes is not None:
            planes = self._last_planes
        else:
            planes_tex, planes_sdf = self.backbone.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
        if cache_backbone:
            self._last_planes = planes

        # Reshape output into three 32-channel planes
        planes_tex = planes_tex.view(len(planes_tex), 3, 32, planes_tex.shape[-2], planes_tex.shape[-1])
        planes_sdf = planes_sdf.view(len(planes_sdf), 3, 32, planes_sdf.shape[-2], planes_sdf.shape[-1])

        # Perform volume rendering
        # feature_samples, depth_samples, weights_samples,all_densities = self.renderer(planes, self.decoder, ray_origins, ray_directions, self.rendering_kwargs) # channels last
        out_coarse = self.renderer.run_model(planes_tex, planes_sdf, self.decoder, coordinates_all.reshape(1,-1,3), directions.unsqueeze(-2).expand(-1, -1, samples_per_ray*2, -1).reshape(batch_size, -1, 3), self.rendering_kwargs) # channels last
        

        colors_coarse = out_coarse['rgb']
        densities_coarse = out_coarse['sigma']
        sdfs_coarse = out_coarse['sdf'] 
        densities_aux_coarse = out_coarse['aux_sigma']
        sdfs_aux_coarse = out_coarse['aux_sdf']

        return colors_coarse,densities_coarse,sdfs_aux_coarse,densities_aux_coarse,sdfs_coarse

    def synthesis_coord(self, ws, c, eik=False, neural_rendering_resolution=None, update_emas=False, cache_backbone=False, use_cached_backbone=False, **synthesis_kwargs):
        cam2world_matrix = c[:, :16].view(-1, 4, 4)
        intrinsics = c[:, 16:25].view(-1, 3, 3)

        # coordinates_coarse, coordinates_fine, directions = coord

        if neural_rendering_resolution is None:
            neural_rendering_resolution = self.neural_rendering_resolution
        else:
            self.neural_rendering_resolution = neural_rendering_resolution

        # Create a batch of rays for volume rendering
        ray_origins, ray_directions = self.ray_sampler(cam2world_matrix, intrinsics, neural_rendering_resolution)



        # Create triplanes by running StyleGAN backbone
        N, M, _ = ray_origins.shape
        if use_cached_backbone and self._last_planes is not None:
            planes = self._last_planes
        else:
            planes_tex, planes_sdf = self.backbone.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
        if cache_backbone:
            self._last_planes = planes

        # Reshape output into three 32-channel planes
        planes_tex = planes_tex.view(len(planes_tex), 3, 32, planes_tex.shape[-2], planes_tex.shape[-1])
        planes_sdf = planes_sdf.view(len(planes_sdf), 3, 32, planes_sdf.shape[-2], planes_sdf.shape[-1])

        sample_coordinates_coarse, sample_coordinates_fine, sample_directions, depths_coarse, depths_fine  = self.renderer.forward_coord(planes_tex, planes_sdf, self.decoder, ray_origins, ray_directions, eik, self.rendering_kwargs) # channels last


        
        return sample_coordinates_coarse, sample_coordinates_fine, sample_directions, depths_coarse, depths_fine


    def synthesis_coord_first(self, ws, c, eik=False, neural_rendering_resolution=None, update_emas=False, cache_backbone=False, use_cached_backbone=False, **synthesis_kwargs):
        cam2world_matrix = c[:, :16].view(-1, 4, 4)
        intrinsics = c[:, 16:25].view(-1, 3, 3)

        # coordinates_coarse, coordinates_fine, directions = coord

        if neural_rendering_resolution is None:
            neural_rendering_resolution = self.neural_rendering_resolution
        else:
            self.neural_rendering_resolution = neural_rendering_resolution

        # Create a batch of rays for volume rendering
        ray_origins, ray_directions = self.ray_sampler(cam2world_matrix, intrinsics, neural_rendering_resolution)



        # Create triplanes by running StyleGAN backbone
        N, M, _ = ray_origins.shape
        if use_cached_backbone and self._last_planes is not None:
            planes = self._last_planes
        else:
            planes_tex, planes_sdf = self.backbone.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
        if cache_backbone:
            self._last_planes = planes

        # Reshape output into three 32-channel planes
        planes_tex = planes_tex.view(len(planes_tex), 3, 32, planes_tex.shape[-2], planes_tex.shape[-1])
        planes_sdf = planes_sdf.view(len(planes_sdf), 3, 32, planes_sdf.shape[-2], planes_sdf.shape[-1])

        all_depths, all_colors, all_densities, all_pts, sample_directions  = self.renderer.forward_coord_first(planes_tex, planes_sdf, self.decoder, ray_origins, ray_directions, eik, self.rendering_kwargs) # channels last


        
        return all_depths, sample_directions, all_pts

    def synthesis_result(self, ws, c, all_colors_volume_blend, all_densities_volume_blend, all_depths_volume, neural_rendering_resolution=None, update_emas=False, cache_backbone=False, use_cached_backbone=False, **synthesis_kwargs):

        feature_samples, depth_samples, weights_samples,_ = self.renderer.ray_marcher(all_colors_volume_blend, all_densities_volume_blend, all_depths_volume, self.rendering_kwargs)

        # Reshape into 'raw' neural-rendered image
        H = W = self.neural_rendering_resolution
        feature_image = feature_samples.permute(0, 2, 1).reshape(1, feature_samples.shape[-1], H, W).contiguous()
        depth_image = depth_samples.permute(0, 2, 1).reshape(1, 1, H, W)
        # Run superresolution to get final image
        rgb_image = feature_image[:, :3]

        sr_image = self.superresolution(rgb_image, feature_image, ws, noise_mode=self.rendering_kwargs['superresolution_noise_mode'], **{k:synthesis_kwargs[k] for k in synthesis_kwargs.keys() if k != 'noise_mode'})
        # if hp.semantic_feat_detach:
        #     sr_semantic = self.superresolution_semantic(semantic_sample, feature_image.detach(), ws, noise_mode=self.rendering_kwargs['superresolution_noise_mode'], **{k:synthesis_kwargs[k] for k in synthesis_kwargs.keys() if k != 'noise_mode'})
        # else:
        #     sr_semantic = self.superresolution_semantic(semantic_sample, feature_image, ws, noise_mode=self.rendering_kwargs['superresolution_noise_mode'], **{k:synthesis_kwargs[k] for k in synthesis_kwargs.keys() if k != 'noise_mode'})            
        return {'image': sr_image, 'image_raw': rgb_image, 'image_depth': depth_image}

    def synthesis_result_second(self, ws, c, all_colors_volume_blend,all_densities_volume_blend, all_semantic_volume_blend, all_density_aux_volume_blend, all_sdf_volume_blend, all_depths_volume,neural_rendering_resolution=None, update_emas=False, cache_backbone=False, use_cached_backbone=False, **synthesis_kwargs):

        batch_size = 1
        num_rays = 16384
        samples_per_ray = 48+48


        all_colors_volume_blend = all_colors_volume_blend.reshape(batch_size, num_rays, samples_per_ray, all_colors_volume_blend.shape[-1])
        all_densities_volume_blend = all_densities_volume_blend.reshape(batch_size, num_rays, samples_per_ray, all_densities_volume_blend.shape[-1])
        all_semantic_volume_blend = all_semantic_volume_blend.reshape(batch_size, num_rays, samples_per_ray, all_semantic_volume_blend.shape[-1])
        all_density_aux_volume_blend = all_density_aux_volume_blend.reshape(batch_size, num_rays, samples_per_ray, all_density_aux_volume_blend.shape[-1])
        all_sdf_volume_blend = all_sdf_volume_blend.reshape(batch_size, num_rays, samples_per_ray, all_sdf_volume_blend.shape[-1])

        sdf_aux_list = torch.split(all_semantic_volume_blend,1,dim=-1)
        sigma_aux_list = torch.split(all_density_aux_volume_blend,1,dim=-1)
        semantic = torch.sigmoid(-20 * torch.cat(sdf_aux_list,3))

        all_depths_volume = all_depths_volume

        feature_samples, depth_samples, weights_samples,_ = self.renderer.ray_marcher(all_colors_volume_blend, all_densities_volume_blend, all_depths_volume, self.rendering_kwargs)
        semantic_sample, _, _,_ = self.renderer.ray_marcher(semantic, all_densities_volume_blend, all_depths_volume, self.rendering_kwargs)

        # Reshape into 'raw' neural-rendered image
        H = W = self.neural_rendering_resolution
        feature_image = feature_samples.permute(0, 2, 1).reshape(1, feature_samples.shape[-1], H, W).contiguous()
        semantic_sample = semantic_sample.permute(0, 2, 1).reshape(1, semantic_sample.shape[-1], H, W).contiguous()
        depth_image = depth_samples.permute(0, 2, 1).reshape(1, 1, H, W)
        # Run superresolution to get final image
        rgb_image = feature_image[:, :3]

        sr_image = self.superresolution(rgb_image, feature_image, ws, noise_mode=self.rendering_kwargs['superresolution_noise_mode'], **{k:synthesis_kwargs[k] for k in synthesis_kwargs.keys() if k != 'noise_mode'})

        sr_semantic = self.superresolution_semantic(semantic_sample, feature_image, ws, noise_mode=self.rendering_kwargs['superresolution_noise_mode'], **{k:synthesis_kwargs[k] for k in synthesis_kwargs.keys() if k != 'noise_mode'})            
        return {'image': sr_image, 'image_raw': rgb_image, 'image_depth': depth_image,'semantic_raw': semantic_sample, 'semantic':sr_semantic,'all_sdf_aux':sdf_aux_list,'all_densities_aux':sigma_aux_list,'all_densities':all_densities_volume_blend,'all_sdf':all_sdf_volume_blend}



    def synthesis_result_first(self, ws, c, tex_plane, shp_plane, all_depths_volume,neural_rendering_resolution=None, update_emas=False, cache_backbone=False, use_cached_backbone=False, **synthesis_kwargs):

        batch_size = 1
        num_rays = 16384
        samples_per_ray = 48+48

        out = self.renderer.run_model_from_plane_to_feat(tex_plane, shp_plane, self.decoder)
        all_colors_volume_blend,all_densities_volume_blend, all_semantic_volume_blend, all_density_aux_volume_blend, all_sdf_volume_blend = out['rgb'], out['sigma'], out['aux_sdf'], out['aux_sigma'], out['sdf']

        all_colors_volume_blend = all_colors_volume_blend.reshape(batch_size, num_rays, samples_per_ray, all_colors_volume_blend.shape[-1])
        all_densities_volume_blend = all_densities_volume_blend.reshape(batch_size, num_rays, samples_per_ray, all_densities_volume_blend.shape[-1])
        all_semantic_volume_blend = all_semantic_volume_blend.reshape(batch_size, num_rays, samples_per_ray, all_semantic_volume_blend.shape[-1])
        all_density_aux_volume_blend = all_density_aux_volume_blend.reshape(batch_size, num_rays, samples_per_ray, all_density_aux_volume_blend.shape[-1])
        all_sdf_volume_blend = all_sdf_volume_blend.reshape(batch_size, num_rays, samples_per_ray, all_sdf_volume_blend.shape[-1])

        sdf_aux_list = torch.split(all_semantic_volume_blend,1,dim=-1)
        sigma_aux_list = torch.split(all_density_aux_volume_blend,1,dim=-1)
        semantic = torch.sigmoid(-20 * torch.cat(sdf_aux_list,3))

        all_depths_volume = all_depths_volume

        feature_samples, depth_samples, weights_samples,_ = self.renderer.ray_marcher(all_colors_volume_blend, all_densities_volume_blend, all_depths_volume, self.rendering_kwargs)
        semantic_sample, _, _,_ = self.renderer.ray_marcher(semantic, all_densities_volume_blend, all_depths_volume, self.rendering_kwargs)

        # Reshape into 'raw' neural-rendered image
        H = W = self.neural_rendering_resolution
        feature_image = feature_samples.permute(0, 2, 1).reshape(1, feature_samples.shape[-1], H, W).contiguous()
        semantic_sample = semantic_sample.permute(0, 2, 1).reshape(1, semantic_sample.shape[-1], H, W).contiguous()
        depth_image = depth_samples.permute(0, 2, 1).reshape(1, 1, H, W)
        # Run superresolution to get final image
        rgb_image = feature_image[:, :3]

        sr_image = self.superresolution(rgb_image, feature_image, ws, noise_mode=self.rendering_kwargs['superresolution_noise_mode'], **{k:synthesis_kwargs[k] for k in synthesis_kwargs.keys() if k != 'noise_mode'})

        sr_semantic = self.superresolution_semantic(semantic_sample, feature_image, ws, noise_mode=self.rendering_kwargs['superresolution_noise_mode'], **{k:synthesis_kwargs[k] for k in synthesis_kwargs.keys() if k != 'noise_mode'})            
        return {'image': sr_image, 'image_raw': rgb_image, 'image_depth': depth_image,'semantic_raw': semantic_sample, 'semantic':sr_semantic,'all_sdf_aux':sdf_aux_list,'all_densities_aux':sigma_aux_list,'all_densities':all_densities_volume_blend,'all_sdf':all_sdf_volume_blend}

    def synthesis_result_first_inf(self, ws, c, tex_plane, shp_plane, all_depths_volume,neural_rendering_resolution=None, update_emas=False, cache_backbone=False, use_cached_backbone=False, **synthesis_kwargs):

        batch_size = 1
        num_rays = 16384
        samples_per_ray = (48+48)*2

        out = self.renderer.run_model_from_plane_to_feat(tex_plane, shp_plane, self.decoder)
        all_colors_volume_blend,all_densities_volume_blend, all_semantic_volume_blend, all_density_aux_volume_blend, all_sdf_volume_blend = out['rgb'], out['sigma'], out['aux_sdf'], out['aux_sigma'], out['sdf']

        all_colors_volume_blend = all_colors_volume_blend.reshape(batch_size, num_rays, samples_per_ray, all_colors_volume_blend.shape[-1])
        all_densities_volume_blend = all_densities_volume_blend.reshape(batch_size, num_rays, samples_per_ray, all_densities_volume_blend.shape[-1])
        all_semantic_volume_blend = all_semantic_volume_blend.reshape(batch_size, num_rays, samples_per_ray, all_semantic_volume_blend.shape[-1])
        all_density_aux_volume_blend = all_density_aux_volume_blend.reshape(batch_size, num_rays, samples_per_ray, all_density_aux_volume_blend.shape[-1])
        all_sdf_volume_blend = all_sdf_volume_blend.reshape(batch_size, num_rays, samples_per_ray, all_sdf_volume_blend.shape[-1])

        sdf_aux_list = torch.split(all_semantic_volume_blend,1,dim=-1)
        sigma_aux_list = torch.split(all_density_aux_volume_blend,1,dim=-1)
        semantic = torch.sigmoid(-20 * torch.cat(sdf_aux_list,3))

        all_depths_volume = all_depths_volume

        feature_samples, depth_samples, weights_samples,_ = self.renderer.ray_marcher(all_colors_volume_blend, all_densities_volume_blend, all_depths_volume, self.rendering_kwargs)
        semantic_sample, _, _,_ = self.renderer.ray_marcher(semantic, all_densities_volume_blend, all_depths_volume, self.rendering_kwargs)

        # Reshape into 'raw' neural-rendered image
        H = W = self.neural_rendering_resolution
        feature_image = feature_samples.permute(0, 2, 1).reshape(1, feature_samples.shape[-1], H, W).contiguous()
        semantic_sample = semantic_sample.permute(0, 2, 1).reshape(1, semantic_sample.shape[-1], H, W).contiguous()
        depth_image = depth_samples.permute(0, 2, 1).reshape(1, 1, H, W)
        # Run superresolution to get final image
        rgb_image = feature_image[:, :3]

        sr_image = self.superresolution(rgb_image, feature_image, ws, noise_mode=self.rendering_kwargs['superresolution_noise_mode'], **{k:synthesis_kwargs[k] for k in synthesis_kwargs.keys() if k != 'noise_mode'})

        sr_semantic = self.superresolution_semantic(semantic_sample, feature_image, ws, noise_mode=self.rendering_kwargs['superresolution_noise_mode'], **{k:synthesis_kwargs[k] for k in synthesis_kwargs.keys() if k != 'noise_mode'})            
        return {'image': sr_image, 'image_raw': rgb_image, 'image_depth': depth_image,'semantic_raw': semantic_sample, 'semantic':sr_semantic,'all_sdf_aux':sdf_aux_list,'all_densities_aux':sigma_aux_list,'all_densities':all_densities_volume_blend,'all_sdf':all_sdf_volume_blend}

    def synthesis_result_first_normal(self, ws, c, tex_plane, shp_plane, all_coord, all_depths_volume,neural_rendering_resolution=None, update_emas=False, cache_backbone=False, use_cached_backbone=False, **synthesis_kwargs):

        batch_size = 1
        num_rays = 16384
        samples_per_ray = 48+48

        out = self.renderer.run_model_from_plane_to_feat(tex_plane, shp_plane, self.decoder)
        all_colors_volume_blend,all_densities_volume_blend, all_semantic_volume_blend, all_density_aux_volume_blend, all_sdf_volume_blend = out['rgb'], out['sigma'], out['aux_sdf'], out['aux_sigma'], out['sdf']

        all_colors_volume_blend = all_colors_volume_blend.reshape(batch_size, num_rays, samples_per_ray, all_colors_volume_blend.shape[-1])
        all_densities_volume_blend = all_densities_volume_blend.reshape(batch_size, num_rays, samples_per_ray, all_densities_volume_blend.shape[-1])
        all_semantic_volume_blend = all_semantic_volume_blend.reshape(batch_size, num_rays, samples_per_ray, all_semantic_volume_blend.shape[-1])
        all_density_aux_volume_blend = all_density_aux_volume_blend.reshape(batch_size, num_rays, samples_per_ray, all_density_aux_volume_blend.shape[-1])
        all_sdf_volume_blend = all_sdf_volume_blend.reshape(batch_size, num_rays, samples_per_ray, all_sdf_volume_blend.shape[-1])

        sdf_aux_list = torch.split(all_semantic_volume_blend,1,dim=-1)
        sigma_aux_list = torch.split(all_density_aux_volume_blend,1,dim=-1)
        semantic = torch.sigmoid(-20 * torch.cat(sdf_aux_list,3))
        
        ############## normal ##############
        eikonal_term = autograd.grad(outputs=all_sdf_volume_blend, inputs=all_coord,
                             grad_outputs=torch.ones_like(all_sdf_volume_blend,requires_grad=False),
                             create_graph=True,
                             retain_graph=True,
                             only_inputs=True)[0]
        ##########################################
            
            
        all_depths_volume = all_depths_volume

        feature_samples, depth_samples, weights_samples,_ = self.renderer.ray_marcher(all_colors_volume_blend, all_densities_volume_blend, all_depths_volume, self.rendering_kwargs)
        semantic_sample, _, _,_ = self.renderer.ray_marcher(semantic, all_densities_volume_blend, all_depths_volume, self.rendering_kwargs)

        H = W = self.neural_rendering_resolution

        ############## normal ##############
        eikonal_term = (eikonal_term[:, :, :-1] + eikonal_term[:, :, 1:]) / 2
        # eikonal_term = eikonal_term.norm(-1)
        eikonal_term_output = eikonal_term.clone()
        eikonal_term = F.normalize(eikonal_term,p=2,dim=-1)
        normal = torch.sum(weights_samples * eikonal_term, -2)
        normal = normal.permute(0, 2, 1).reshape(1, normal.shape[-1], H, W).contiguous()
        ##########################################



        # Reshape into 'raw' neural-rendered image
        feature_image = feature_samples.permute(0, 2, 1).reshape(1, feature_samples.shape[-1], H, W).contiguous()
        semantic_sample = semantic_sample.permute(0, 2, 1).reshape(1, semantic_sample.shape[-1], H, W).contiguous()
        depth_image = depth_samples.permute(0, 2, 1).reshape(1, 1, H, W)
        # Run superresolution to get final image
        rgb_image = feature_image[:, :3]

        sr_image = self.superresolution(rgb_image, feature_image, ws, noise_mode=self.rendering_kwargs['superresolution_noise_mode'], **{k:synthesis_kwargs[k] for k in synthesis_kwargs.keys() if k != 'noise_mode'})

        sr_semantic = self.superresolution_semantic(semantic_sample, feature_image, ws, noise_mode=self.rendering_kwargs['superresolution_noise_mode'], **{k:synthesis_kwargs[k] for k in synthesis_kwargs.keys() if k != 'noise_mode'})            
        return {'image': sr_image, 'image_raw': rgb_image, 'image_depth': depth_image,'semantic_raw': semantic_sample, 'semantic':sr_semantic,'all_sdf_aux':sdf_aux_list,'all_densities_aux':sigma_aux_list,'all_densities':all_densities_volume_blend,'all_sdf':all_sdf_volume_blend,'normal':normal,'eikonal': eikonal_term_output}

    # def synthesis_result_first_normal(self, ws, c, tex_plane, shp_plane, all_coord, all_depths_volume,neural_rendering_resolution=None, update_emas=False, cache_backbone=False, use_cached_backbone=False, **synthesis_kwargs):

    #     batch_size = 1
    #     num_rays = 16384
    #     samples_per_ray = 48+48

    #     out = self.renderer.run_model_from_plane_to_feat(tex_plane, shp_plane, self.decoder)
    #     all_colors_volume_blend,all_densities_volume_blend, all_semantic_volume_blend, all_density_aux_volume_blend, all_sdf_volume_blend = out['rgb'], out['sigma'], out['aux_sdf'], out['aux_sigma'], out['sdf']

    #     all_colors_volume_blend = all_colors_volume_blend.reshape(batch_size, num_rays, samples_per_ray, all_colors_volume_blend.shape[-1])
    #     all_densities_volume_blend = all_densities_volume_blend.reshape(batch_size, num_rays, samples_per_ray, all_densities_volume_blend.shape[-1])
    #     all_semantic_volume_blend = all_semantic_volume_blend.reshape(batch_size, num_rays, samples_per_ray, all_semantic_volume_blend.shape[-1])
    #     all_density_aux_volume_blend = all_density_aux_volume_blend.reshape(batch_size, num_rays, samples_per_ray, all_density_aux_volume_blend.shape[-1])
    #     all_sdf_volume_blend = all_sdf_volume_blend.reshape(batch_size, num_rays, samples_per_ray, all_sdf_volume_blend.shape[-1])

    #     sdf_aux_list = torch.split(all_semantic_volume_blend,1,dim=-1)
    #     sigma_aux_list = torch.split(all_density_aux_volume_blend,1,dim=-1)
    #     semantic = torch.sigmoid(-20 * torch.cat(sdf_aux_list,3))
        
    #     ############## normal ##############
    #     eikonal_term = autograd.grad(outputs=all_sdf_volume_blend, inputs=all_coord,
    #                          grad_outputs=torch.ones_like(all_sdf_volume_blend,requires_grad=False),
    #                          create_graph=True,
    #                          retain_graph=True,
    #                          only_inputs=True)[0]
    #     ##########################################
            
            
    #     all_depths_volume = all_depths_volume

    #     feature_samples, depth_samples, weights_samples,_ = self.renderer.ray_marcher(all_colors_volume_blend, all_densities_volume_blend, all_depths_volume, self.rendering_kwargs)
    #     semantic_sample, _, _,_ = self.renderer.ray_marcher(semantic, all_densities_volume_blend, all_depths_volume, self.rendering_kwargs)

    #     H = W = self.neural_rendering_resolution

    #     ############## normal ##############
    #     eikonal_term = (eikonal_term[:, :, :-1] + eikonal_term[:, :, 1:]) / 2
    #     # eikonal_term = eikonal_term.norm(-1)
    #     eikonal_term = F.normalize(eikonal_term,p=2,dim=-1)
    #     normal = torch.sum(weights_samples * eikonal_term, -2)
    #     normal = normal.permute(0, 2, 1).reshape(1, normal.shape[-1], H, W).contiguous()
    #     ##########################################



    #     # Reshape into 'raw' neural-rendered image
    #     feature_image = feature_samples.permute(0, 2, 1).reshape(1, feature_samples.shape[-1], H, W).contiguous()
    #     semantic_sample = semantic_sample.permute(0, 2, 1).reshape(1, semantic_sample.shape[-1], H, W).contiguous()
    #     depth_image = depth_samples.permute(0, 2, 1).reshape(1, 1, H, W)
    #     # Run superresolution to get final image
    #     rgb_image = feature_image[:, :3]

    #     sr_image = self.superresolution(rgb_image, feature_image, ws, noise_mode=self.rendering_kwargs['superresolution_noise_mode'], **{k:synthesis_kwargs[k] for k in synthesis_kwargs.keys() if k != 'noise_mode'})

    #     sr_semantic = self.superresolution_semantic(semantic_sample, feature_image, ws, noise_mode=self.rendering_kwargs['superresolution_noise_mode'], **{k:synthesis_kwargs[k] for k in synthesis_kwargs.keys() if k != 'noise_mode'})            
    #     return {'image': sr_image, 'image_raw': rgb_image, 'image_depth': depth_image,'semantic_raw': semantic_sample, 'semantic':sr_semantic,'all_sdf_aux':sdf_aux_list,'all_densities_aux':sigma_aux_list,'all_densities':all_densities_volume_blend,'all_sdf':all_sdf_volume_blend,'normal':normal}

    def synthesis_result_first_normal_inf(self, ws, c, tex_plane, shp_plane, all_coord, all_depths_volume,neural_rendering_resolution=None, update_emas=False, cache_backbone=False, use_cached_backbone=False, **synthesis_kwargs):

        batch_size = 1
        num_rays = 16384
        samples_per_ray = (48+48)*2

        out = self.renderer.run_model_from_plane_to_feat(tex_plane, shp_plane, self.decoder)
        all_colors_volume_blend,all_densities_volume_blend, all_semantic_volume_blend, all_density_aux_volume_blend, all_sdf_volume_blend = out['rgb'], out['sigma'], out['aux_sdf'], out['aux_sigma'], out['sdf']

        all_colors_volume_blend = all_colors_volume_blend.reshape(batch_size, num_rays, samples_per_ray, all_colors_volume_blend.shape[-1])
        all_densities_volume_blend = all_densities_volume_blend.reshape(batch_size, num_rays, samples_per_ray, all_densities_volume_blend.shape[-1])
        all_semantic_volume_blend = all_semantic_volume_blend.reshape(batch_size, num_rays, samples_per_ray, all_semantic_volume_blend.shape[-1])
        all_density_aux_volume_blend = all_density_aux_volume_blend.reshape(batch_size, num_rays, samples_per_ray, all_density_aux_volume_blend.shape[-1])
        all_sdf_volume_blend = all_sdf_volume_blend.reshape(batch_size, num_rays, samples_per_ray, all_sdf_volume_blend.shape[-1])

        sdf_aux_list = torch.split(all_semantic_volume_blend,1,dim=-1)
        sigma_aux_list = torch.split(all_density_aux_volume_blend,1,dim=-1)
        semantic = torch.sigmoid(-20 * torch.cat(sdf_aux_list,3))
        
        ############## normal ##############
        eikonal_term = autograd.grad(outputs=all_sdf_volume_blend, inputs=all_coord,
                             grad_outputs=torch.ones_like(all_sdf_volume_blend,requires_grad=False),
                             create_graph=True,
                             retain_graph=True,
                             only_inputs=True)[0]
        ##########################################
            
            
        all_depths_volume = all_depths_volume

        feature_samples, depth_samples, weights_samples,_ = self.renderer.ray_marcher(all_colors_volume_blend, all_densities_volume_blend, all_depths_volume, self.rendering_kwargs)
        semantic_sample, _, _,_ = self.renderer.ray_marcher(semantic, all_densities_volume_blend, all_depths_volume, self.rendering_kwargs)

        H = W = self.neural_rendering_resolution

        ############## normal ##############
        eikonal_term = (eikonal_term[:, :, :-1] + eikonal_term[:, :, 1:]) / 2
        # eikonal_term = eikonal_term.norm(-1)
        eikonal_term = F.normalize(eikonal_term,p=2,dim=-1)
        normal = torch.sum(weights_samples * eikonal_term, -2)
        normal = normal.permute(0, 2, 1).reshape(1, normal.shape[-1], H, W).contiguous()
        ##########################################



        # Reshape into 'raw' neural-rendered image
        feature_image = feature_samples.permute(0, 2, 1).reshape(1, feature_samples.shape[-1], H, W).contiguous()
        semantic_sample = semantic_sample.permute(0, 2, 1).reshape(1, semantic_sample.shape[-1], H, W).contiguous()
        depth_image = depth_samples.permute(0, 2, 1).reshape(1, 1, H, W)
        # Run superresolution to get final image
        rgb_image = feature_image[:, :3]

        sr_image = self.superresolution(rgb_image, feature_image, ws, noise_mode=self.rendering_kwargs['superresolution_noise_mode'], **{k:synthesis_kwargs[k] for k in synthesis_kwargs.keys() if k != 'noise_mode'})

        sr_semantic = self.superresolution_semantic(semantic_sample, feature_image, ws, noise_mode=self.rendering_kwargs['superresolution_noise_mode'], **{k:synthesis_kwargs[k] for k in synthesis_kwargs.keys() if k != 'noise_mode'})            
        return {'image': sr_image, 'image_raw': rgb_image, 'image_depth': depth_image,'semantic_raw': semantic_sample, 'semantic':sr_semantic,'all_sdf_aux':sdf_aux_list,'all_densities_aux':sigma_aux_list,'all_densities':all_densities_volume_blend,'all_sdf':all_sdf_volume_blend,'normal':normal}


    def sample(self, coordinates, directions, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        # Compute RGB features, density for arbitrary 3D coordinates. Mostly used for extracting shapes. 
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        planes_tex, planes_sdf = self.backbone.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
            
        planes_tex = planes_tex.view(len(planes_tex), 3, 32, planes_tex.shape[-2], planes_tex.shape[-1])
        planes_sdf = planes_sdf.view(len(planes_sdf), 3, 32, planes_sdf.shape[-2], planes_sdf.shape[-1])
        
        return self.renderer.run_model(planes_tex, planes_sdf, self.decoder, coordinates, directions, self.rendering_kwargs)

    def sample_mixed(self, coordinates, directions, ws, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        # Same as sample, but expects latent vectors 'ws' instead of Gaussian noise 'z'
        planes_tex, planes_sdf = self.backbone.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
        
        planes_tex = planes_tex.view(len(planes_tex), 3, 32, planes_tex.shape[-2], planes_tex.shape[-1])
        planes_sdf = planes_sdf.view(len(planes_sdf), 3, 32, planes_sdf.shape[-2], planes_sdf.shape[-1])
        return self.renderer.run_model(planes_tex, planes_sdf, self.decoder, coordinates, directions, self.rendering_kwargs)

    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, neural_rendering_resolution=None, update_emas=False, cache_backbone=False, use_cached_backbone=False, **synthesis_kwargs):
        # Render a batch of generated images.
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        return self.synthesis(ws, c, update_emas=update_emas, neural_rendering_resolution=neural_rendering_resolution, cache_backbone=cache_backbone, use_cached_backbone=use_cached_backbone, **synthesis_kwargs)


from training.networks_stylegan2 import FullyConnectedLayer

class OSGDecoder(torch.nn.Module):
    def __init__(self, n_features, options):
        super().__init__()
        self.hidden_dim = 64

        self.net_tex = torch.nn.Sequential(
            FullyConnectedLayer(n_features, self.hidden_dim, lr_multiplier=options['decoder_lr_mul']),
            torch.nn.Softplus(),
            FullyConnectedLayer(self.hidden_dim, options['decoder_output_dim'], lr_multiplier=options['decoder_lr_mul'])
        )

        self.net_main_shp = torch.nn.Sequential(
            FullyConnectedLayer(n_features, self.hidden_dim, lr_multiplier=options['decoder_lr_mul']),
            torch.nn.Softplus(),
            FullyConnectedLayer(self.hidden_dim, 1, lr_multiplier=options['decoder_lr_mul'])
        )

        # self.net_aux_shp1 = torch.nn.Sequential(
        #     FullyConnectedLayer(n_features, self.hidden_dim, lr_multiplier=options['decoder_lr_mul']),
        #     torch.nn.Softplus(),
        #     FullyConnectedLayer(self.hidden_dim, hp.number+1, lr_multiplier=options['decoder_lr_mul'])
        # )
        for i in range(hp.number):
            block = torch.nn.Sequential(
                FullyConnectedLayer(n_features, self.hidden_dim, lr_multiplier=options['decoder_lr_mul']),
                torch.nn.Softplus(),
                FullyConnectedLayer(self.hidden_dim, 1, lr_multiplier=options['decoder_lr_mul'])
            )
            setattr(self, f'net_main_shp_bbbb_{i}', block)
            
        # self.sigmoid_beta = torch.nn.Parameter(0.1 * torch.ones(1))

        
        sigmoid_beta = 0.1 * torch.ones(1)
        self.register_buffer('sigmoid_beta', sigmoid_beta)
        
        
    def sdf_activation(self, input):
        sigma = torch.sigmoid(input / (torch.abs(self.sigmoid_beta)+0.0001)) / (torch.abs(self.sigmoid_beta)+0.0001)

        return sigma

    def forward(self, sampled_features_tex, sampled_features_shp, ray_directions):
        # Aggregate features
        sampled_features_tex = sampled_features_tex.half()
        sampled_features_shp = sampled_features_shp.half()
        sampled_features_tex = sampled_features_tex.mean(1)
        sampled_features_shp = sampled_features_shp.mean(1)
        x_tex = sampled_features_tex
        x_shp = sampled_features_shp

        N_tex, M_tex, C_tex = x_tex.shape
        x_tex = x_tex.view(N_tex*M_tex, C_tex)

        N_shp, M_shp, C_shp = x_shp.shape
        x_shp = x_shp.view(N_shp*M_shp, C_shp)


        x_tex = self.net_tex(x_tex)
        x_tex = x_tex.view(N_tex, M_tex, -1)
        rgb = torch.sigmoid(x_tex)*(1 + 2*0.001) - 0.001 # Uses sigmoid clamping from MipNeRF
        
        x_main_shp = self.net_main_shp(x_shp)
        x_sdf = x_main_shp.view(N_shp, M_shp, -1)
        
        x_aux_sdf_list = []
        for i in range(hp.number):
            block = getattr(self, f'net_main_shp_bbbb_{i}')
            x_aux_shp = block(x_shp)
            x_aux_sdf = x_aux_shp.view(N_shp, M_shp, -1)
            x_aux_sdf_list.append(x_aux_sdf)
        
        x_aux_sdf = torch.cat(x_aux_sdf_list,-1)
        
        
        
        x_sigma = self.sdf_activation(-x_sdf)
        
        # x_aux_sdf = x_aux_sdf + x_sdf
        x_aux_sigma = self.sdf_activation(-x_aux_sdf)
        return {'rgb': rgb, 'sigma': x_sigma, 'sdf':x_sdf,'aux_sigma': x_aux_sigma, 'aux_sdf':x_aux_sdf}




class TexDecoder(torch.nn.Module):
    def __init__(self, n_features):
        super().__init__()

        self.net_tex = torch.nn.Sequential(
            FullyConnectedLayer(n_features, n_features, lr_multiplier=1),
            torch.nn.Softplus(),
            FullyConnectedLayer(n_features, n_features, lr_multiplier=1),
            torch.nn.Softplus(),
            FullyConnectedLayer(n_features, n_features, lr_multiplier=1),
            torch.nn.Softplus()
        )

 


    def forward(self, sampled_features_tex):
        # Aggregate features
        x_tex = sampled_features_tex

        N_tex, CC_tex ,M_tex, C_tex = x_tex.shape
        x_tex = x_tex.reshape(N_tex*M_tex*CC_tex, C_tex)


        x_tex = self.net_tex(x_tex)
        x_tex = x_tex.view(N_tex, CC_tex, M_tex, -1)
        


        return x_tex