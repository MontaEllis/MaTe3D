# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""
The renderer is a module that takes in rays, decides where to sample along each
ray, and computes pixel colors using the volume rendering equation.
"""

import math
import torch
import torch.nn as nn
from collections import OrderedDict
from training.volumetric_rendering.ray_marcher import MipRayMarcher2
from training.volumetric_rendering import math_utils
import torch.autograd as autograd
from cuda_gridsample_grad2 import cuda_gridsample as cu
from hparams import hparams as hp

# def grid_sample(image, optical):
#     N, C, IH, IW = image.shape
#     _, H, W, _ = optical.shape

#     ix = optical[..., 0]
#     iy = optical[..., 1]

#     ix = ((ix + 1) / 2) * (IW-1);
#     iy = ((iy + 1) / 2) * (IH-1);
#     with torch.no_grad():
#         ix_nw = torch.floor(ix);
#         iy_nw = torch.floor(iy);
#         ix_ne = ix_nw + 1;
#         iy_ne = iy_nw;
#         ix_sw = ix_nw;
#         iy_sw = iy_nw + 1;
#         ix_se = ix_nw + 1;
#         iy_se = iy_nw + 1;

#     nw = (ix_se - ix)    * (iy_se - iy)
#     ne = (ix    - ix_sw) * (iy_sw - iy)
#     sw = (ix_ne - ix)    * (iy    - iy_ne)
#     se = (ix    - ix_nw) * (iy    - iy_nw)
    
#     with torch.no_grad():
#         torch.clamp(ix_nw, 0, IW-1, out=ix_nw)
#         torch.clamp(iy_nw, 0, IH-1, out=iy_nw)

#         torch.clamp(ix_ne, 0, IW-1, out=ix_ne)
#         torch.clamp(iy_ne, 0, IH-1, out=iy_ne)
 
#         torch.clamp(ix_sw, 0, IW-1, out=ix_sw)
#         torch.clamp(iy_sw, 0, IH-1, out=iy_sw)
 
#         torch.clamp(ix_se, 0, IW-1, out=ix_se)
#         torch.clamp(iy_se, 0, IH-1, out=iy_se)

#     image = image.view(N, C, IH * IW)


#     nw_val = torch.gather(image, 2, (iy_nw * IW + ix_nw).long().view(N, 1, H * W).repeat(1, C, 1))
#     ne_val = torch.gather(image, 2, (iy_ne * IW + ix_ne).long().view(N, 1, H * W).repeat(1, C, 1))
#     sw_val = torch.gather(image, 2, (iy_sw * IW + ix_sw).long().view(N, 1, H * W).repeat(1, C, 1))
#     se_val = torch.gather(image, 2, (iy_se * IW + ix_se).long().view(N, 1, H * W).repeat(1, C, 1))

#     out_val = (nw_val.view(N, C, H, W) * nw.view(N, 1, H, W) + 
#                ne_val.view(N, C, H, W) * ne.view(N, 1, H, W) +
#                sw_val.view(N, C, H, W) * sw.view(N, 1, H, W) +
#                se_val.view(N, C, H, W) * se.view(N, 1, H, W))

#     return out_val


def generate_planes():
    """
    Defines planes by the three vectors that form the "axes" of the
    plane. Should work with arbitrary number of planes and planes of
    arbitrary orientation.
    """
    return torch.tensor([[[1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1]],
                            [[1, 0, 0],
                            [0, 0, 1],
                            [0, 1, 0]],
                            [[0, 0, 1],
                            [1, 0, 0],
                            [0, 1, 0]]], dtype=torch.float32)

def project_onto_planes(planes, coordinates):
    """
    Does a projection of a 3D point onto a batch of 2D planes,
    returning 2D plane coordinates.

    Takes plane axes of shape n_planes, 3, 3
    # Takes coordinates of shape N, M, 3
    # returns projections of shape N*n_planes, M, 2
    """
    N, M, C = coordinates.shape
    n_planes, _, _ = planes.shape
    coordinates = coordinates.unsqueeze(1).expand(-1, n_planes, -1, -1).reshape(N*n_planes, M, 3)
    inv_planes = torch.linalg.inv(planes).unsqueeze(0).expand(N, -1, -1, -1).reshape(N*n_planes, 3, 3)
    projections = torch.bmm(coordinates, inv_planes)
    return projections[..., :2]

def sample_from_planes(plane_axes, plane_features, coordinates, mode='bilinear', padding_mode='zeros', box_warp=None):
    assert padding_mode == 'zeros'
    N, n_planes, C, H, W = plane_features.shape
    _, M, _ = coordinates.shape
    plane_features = plane_features.view(N*n_planes, C, H, W)

    coordinates = (2/box_warp) * coordinates # TODO: add specific box bounds

    projected_coordinates = project_onto_planes(plane_axes, coordinates).unsqueeze(1)
    # output_features = torch.nn.functional.grid_sample(plane_features, projected_coordinates.float(), mode=mode, padding_mode=padding_mode, align_corners=False).permute(0, 3, 2, 1).reshape(N, n_planes, M, C)
    # output_features = grid_sample(plane_features, projected_coordinates.float()).permute(0, 3, 2, 1).reshape(N, n_planes, M, C)
    output_features = cu.grid_sample_2d(plane_features, projected_coordinates.float(), padding_mode=padding_mode, align_corners=False).permute(0, 3, 2, 1).reshape(N, n_planes, M, C)
    return output_features

def sample_from_3dgrid(grid, coordinates):
    """
    Expects coordinates in shape (batch_size, num_points_per_batch, 3)
    Expects grid in shape (1, channels, H, W, D)
    (Also works if grid has batch size)
    Returns sampled features of shape (batch_size, num_points_per_batch, feature_channels)
    """
    batch_size, n_coords, n_dims = coordinates.shape
    sampled_features = torch.nn.functional.grid_sample(grid.expand(batch_size, -1, -1, -1, -1),
                                                       coordinates.reshape(batch_size, 1, 1, -1, n_dims),
                                                       mode='bilinear', padding_mode='zeros', align_corners=False)
    N, C, H, W, D = sampled_features.shape
    sampled_features = sampled_features.permute(0, 4, 3, 2, 1).reshape(N, H*W*D, C)
    return sampled_features

class ImportanceRenderer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ray_marcher = MipRayMarcher2()
        self.plane_axes = generate_planes().cuda()
        self.sigmoid = 20

    def get_eikonal_term(self, pts, sdf):
        eikonal_term = autograd.grad(outputs=sdf, inputs=pts,
                                     grad_outputs=torch.ones_like(sdf,requires_grad=False),
                                     create_graph=True,
                                     retain_graph=True,
                                     only_inputs=True)[0]
        eikonal_term = torch.nan_to_num(eikonal_term)
        return eikonal_term
        


    def forward(self, planes_tex, planes_shp, decoder, ray_origins, ray_directions, eik, rendering_options):
        self.plane_axes = self.plane_axes.to(ray_origins.device)

        if rendering_options['ray_start'] == rendering_options['ray_end'] == 'auto':
            ray_start, ray_end = math_utils.get_ray_limits_box(ray_origins, ray_directions, box_side_length=rendering_options['box_warp'])
            is_ray_valid = ray_end > ray_start
            if torch.any(is_ray_valid).item():
                ray_start[~is_ray_valid] = ray_start[is_ray_valid].min()
                ray_end[~is_ray_valid] = ray_start[is_ray_valid].max()
            depths_coarse = self.sample_stratified(ray_origins, ray_start, ray_end, rendering_options['depth_resolution'], rendering_options['disparity_space_sampling'])
        else:
            # Create stratified depth samples
            depths_coarse = self.sample_stratified(ray_origins, rendering_options['ray_start'], rendering_options['ray_end'], rendering_options['depth_resolution'], rendering_options['disparity_space_sampling'])

        batch_size, num_rays, samples_per_ray, _ = depths_coarse.shape

        # Coarse Pass
        sample_coordinates_coarse = (ray_origins.unsqueeze(-2) + depths_coarse * ray_directions.unsqueeze(-2)).reshape(batch_size, -1, 3)
        sample_directions = ray_directions.unsqueeze(-2).expand(-1, -1, samples_per_ray, -1).reshape(batch_size, -1, 3)

        ## SDF!!!
        if eik:
            sample_coordinates_coarse.requires_grad = True


        out = self.run_model(planes_tex, planes_shp, decoder, sample_coordinates_coarse, sample_directions, rendering_options)
        colors_coarse = out['rgb']
        densities_coarse = out['sigma']
        sdfs_coarse = out['sdf'] 
        densities_aux_coarse = out['aux_sigma']
        sdfs_aux_coarse = out['aux_sdf']

        colors_coarse = colors_coarse.reshape(batch_size, num_rays, samples_per_ray, colors_coarse.shape[-1])
        densities_coarse = densities_coarse.reshape(batch_size, num_rays, samples_per_ray, 1)
        sdfs_coarse = sdfs_coarse.reshape(batch_size, num_rays, samples_per_ray, 1)
        densities_aux_coarse = densities_aux_coarse.reshape(batch_size, num_rays, samples_per_ray, hp.number)
        sdfs_aux_coarse = sdfs_aux_coarse.reshape(batch_size, num_rays, samples_per_ray, hp.number)

        # Fine Pass
        N_importance = rendering_options['depth_resolution_importance']
        rgb_final_rgbsegs = OrderedDict()
        depth_final_rgbsegs = OrderedDict()
        if N_importance > 0:
            _, _, weights,_ = self.ray_marcher(colors_coarse, densities_coarse, depths_coarse, rendering_options)

            depths_fine = self.sample_importance(depths_coarse, weights, N_importance)

            sample_directions = ray_directions.unsqueeze(-2).expand(-1, -1, N_importance, -1).reshape(batch_size, -1, 3)
            sample_coordinates_fine = (ray_origins.unsqueeze(-2) + depths_fine * ray_directions.unsqueeze(-2)).reshape(batch_size, -1, 3)

            ### SDF!!!
            if eik:
                sample_coordinates_fine.requires_grad = True

            out = self.run_model(planes_tex, planes_shp, decoder, sample_coordinates_fine, sample_directions, rendering_options)
            colors_fine = out['rgb']
            densities_fine = out['sigma']
            sdfs_fine = out['sdf']
            densities_aux_fine = out['aux_sigma']
            sdfs_aux_fine = out['aux_sdf']
        
            colors_fine = colors_fine.reshape(batch_size, num_rays, N_importance, colors_fine.shape[-1])
            densities_fine = densities_fine.reshape(batch_size, num_rays, N_importance, 1)
            sdfs_fine = sdfs_fine.reshape(batch_size, num_rays, N_importance, 1)
            densities_aux_fine = densities_aux_fine.reshape(batch_size, num_rays, N_importance, hp.number)
            sdfs_aux_fine = sdfs_aux_fine.reshape(batch_size, num_rays, N_importance, hp.number)


            # all_depths, all_colors, all_densities = self.unify_samples(depths_coarse, colors_coarse, densities_coarse,
            #                                                       depths_fine, colors_fine, densities_fine)


            if eik:
                eikonal1 = self.get_eikonal_term(sample_coordinates_coarse, sdfs_coarse)
                eikonal2 = self.get_eikonal_term(sample_coordinates_fine, sdfs_fine)
                
                if hp.eik_seg:
                    eikonal1_aux = self.get_eikonal_term(sample_coordinates_coarse, sdfs_aux_coarse[...,1:])
                    eikonal2_aux = self.get_eikonal_term(sample_coordinates_fine, sdfs_aux_fine[...,1:])
                else:
                    eikonal1_aux = torch.ones_like(sdfs_coarse)
                    eikonal2_aux = torch.ones_like(sdfs_fine)
            else:
                eikonal1 = torch.ones_like(sdfs_coarse)
                eikonal2 = torch.ones_like(sdfs_fine)
                eikonal1_aux = torch.ones_like(sdfs_coarse)
                eikonal2_aux = torch.ones_like(sdfs_fine)
                
            # all_depths, all_colors, all_densities, all_sdfs, all_pts = self.unify_samples_add_sdf_add_pts(depths_coarse, sample_coordinates_coarse.reshape(batch_size, num_rays, samples_per_ray, sample_coordinates_coarse.shape[-1]), colors_coarse, densities_coarse, sdfs_coarse,
                                                                  # depths_fine, sample_coordinates_fine.reshape(batch_size, num_rays, samples_per_ray, sample_coordinates_coarse.shape[-1]), colors_fine, densities_fine, sdfs_fine)
            all_depths, all_colors, all_densities, all_sdfs, all_densities_aux, all_sdfs_aux, all_pts = self.unify_samples_add_sdf_add_pts(depths_coarse, sample_coordinates_coarse.reshape(batch_size, num_rays, samples_per_ray, sample_coordinates_coarse.shape[-1]), colors_coarse, densities_coarse, sdfs_coarse,densities_aux_coarse,sdfs_aux_coarse,
                                                                  depths_fine, sample_coordinates_fine.reshape(batch_size, num_rays, samples_per_ray, sample_coordinates_coarse.shape[-1]), colors_fine, densities_fine, sdfs_fine,densities_aux_fine,sdfs_aux_fine)

            # Aggregate
            rgb_final, depth_final, weights,_ = self.ray_marcher(all_colors, all_densities, all_depths, rendering_options)
            
            densities_aux_list = []
            sdf_aux_list = torch.split(all_sdfs_aux,1,dim=-1)
            for i in range(hp.number):
                rgb_final_rgbsegs[str(i)], depth_final_rgbsegs[str(i)], _ , densities_local= self.ray_marcher(all_colors, all_densities_aux[...,i:i+1], all_depths, rendering_options)
                densities_aux_list.append(densities_local)
            semantic = torch.sigmoid(-self.sigmoid * torch.cat(sdf_aux_list,3))
            if hp.semantic_density_detach:
                semantic_final, _, _,_ = self.ray_marcher(semantic, all_densities.detach(), all_depths, rendering_options)
            else:
                semantic_final, _, _,_ = self.ray_marcher(semantic, all_densities, all_depths, rendering_options)
            # semantic_final = torch.softmax(semantic_final,dim=-1)
            # semantic_final = semantic_final*2 -1   
            
        else:
            rgb_final, depth_final, weights,_ = self.ray_marcher(colors_coarse, densities_coarse, depths_coarse, rendering_options)


        # return rgb_final, depth_final, weights.sum(2), [eikonal1,eikonal2], all_sdfs, all_densities, all_pts
        return rgb_final, depth_final, weights.sum(2),rgb_final_rgbsegs, depth_final_rgbsegs, semantic_final, [eikonal1,eikonal2], [eikonal1_aux,eikonal2_aux], all_sdfs, all_densities, all_pts, sdf_aux_list, densities_aux_list

    def forward_coord_first(self, planes_tex, planes_shp, decoder, ray_origins, ray_directions, eik, rendering_options):
        self.plane_axes = self.plane_axes.to(ray_origins.device)

        if rendering_options['ray_start'] == rendering_options['ray_end'] == 'auto':
            ray_start, ray_end = math_utils.get_ray_limits_box(ray_origins, ray_directions, box_side_length=rendering_options['box_warp'])
            is_ray_valid = ray_end > ray_start
            if torch.any(is_ray_valid).item():
                ray_start[~is_ray_valid] = ray_start[is_ray_valid].min()
                ray_end[~is_ray_valid] = ray_start[is_ray_valid].max()
            depths_coarse = self.sample_stratified(ray_origins, ray_start, ray_end, rendering_options['depth_resolution'], rendering_options['disparity_space_sampling'])
        else:
            # Create stratified depth samples
            depths_coarse = self.sample_stratified(ray_origins, rendering_options['ray_start'], rendering_options['ray_end'], rendering_options['depth_resolution'], rendering_options['disparity_space_sampling'])

        batch_size, num_rays, samples_per_ray, _ = depths_coarse.shape

        # Coarse Pass
        sample_coordinates_coarse = (ray_origins.unsqueeze(-2) + depths_coarse * ray_directions.unsqueeze(-2)).reshape(batch_size, -1, 3)
        sample_directions = ray_directions.unsqueeze(-2).expand(-1, -1, samples_per_ray, -1).reshape(batch_size, -1, 3)

        ## SDF!!!
        if eik:
            sample_coordinates_coarse.requires_grad = True


        out = self.run_model(planes_tex, planes_shp, decoder, sample_coordinates_coarse, sample_directions, rendering_options)
        colors_coarse = out['rgb']
        densities_coarse = out['sigma']
        sdfs_coarse = out['sdf'] 
        densities_aux_coarse = out['aux_sigma']
        sdfs_aux_coarse = out['aux_sdf']

        colors_coarse = colors_coarse.reshape(batch_size, num_rays, samples_per_ray, colors_coarse.shape[-1])
        densities_coarse = densities_coarse.reshape(batch_size, num_rays, samples_per_ray, 1)
        sdfs_coarse = sdfs_coarse.reshape(batch_size, num_rays, samples_per_ray, 1)
        densities_aux_coarse = densities_aux_coarse.reshape(batch_size, num_rays, samples_per_ray, hp.number)
        sdfs_aux_coarse = sdfs_aux_coarse.reshape(batch_size, num_rays, samples_per_ray, hp.number)

        # Fine Pass
        N_importance = rendering_options['depth_resolution_importance']
        rgb_final_rgbsegs = OrderedDict()
        depth_final_rgbsegs = OrderedDict()
        if N_importance > 0:
            _, _, weights,_ = self.ray_marcher(colors_coarse, densities_coarse, depths_coarse, rendering_options)

            depths_fine = self.sample_importance(depths_coarse, weights, N_importance)

            sample_directions = ray_directions.unsqueeze(-2).expand(-1, -1, N_importance, -1).reshape(batch_size, -1, 3)
            sample_coordinates_fine = (ray_origins.unsqueeze(-2) + depths_fine * ray_directions.unsqueeze(-2)).reshape(batch_size, -1, 3)

            ### SDF!!!
            if eik:
                sample_coordinates_fine.requires_grad = True

            out = self.run_model(planes_tex, planes_shp, decoder, sample_coordinates_fine, sample_directions, rendering_options)
            colors_fine = out['rgb']
            densities_fine = out['sigma']
            sdfs_fine = out['sdf']
            densities_aux_fine = out['aux_sigma']
            sdfs_aux_fine = out['aux_sdf']
        
            colors_fine = colors_fine.reshape(batch_size, num_rays, N_importance, colors_fine.shape[-1])
            densities_fine = densities_fine.reshape(batch_size, num_rays, N_importance, 1)
            sdfs_fine = sdfs_fine.reshape(batch_size, num_rays, N_importance, 1)
            densities_aux_fine = densities_aux_fine.reshape(batch_size, num_rays, N_importance, hp.number)
            sdfs_aux_fine = sdfs_aux_fine.reshape(batch_size, num_rays, N_importance, hp.number)


            # all_depths, all_colors, all_densities = self.unify_samples(depths_coarse, colors_coarse, densities_coarse,
            #                                                       depths_fine, colors_fine, densities_fine)


            if eik:
                eikonal1 = self.get_eikonal_term(sample_coordinates_coarse, sdfs_coarse)
                eikonal2 = self.get_eikonal_term(sample_coordinates_fine, sdfs_fine)
                
                if hp.eik_seg:
                    eikonal1_aux = self.get_eikonal_term(sample_coordinates_coarse, sdfs_aux_coarse[...,1:])
                    eikonal2_aux = self.get_eikonal_term(sample_coordinates_fine, sdfs_aux_fine[...,1:])
                else:
                    eikonal1_aux = torch.ones_like(sdfs_coarse)
                    eikonal2_aux = torch.ones_like(sdfs_fine)
            else:
                eikonal1 = torch.ones_like(sdfs_coarse)
                eikonal2 = torch.ones_like(sdfs_fine)
                eikonal1_aux = torch.ones_like(sdfs_coarse)
                eikonal2_aux = torch.ones_like(sdfs_fine)
                
            # all_depths, all_colors, all_densities, all_sdfs, all_pts = self.unify_samples_add_sdf_add_pts(depths_coarse, sample_coordinates_coarse.reshape(batch_size, num_rays, samples_per_ray, sample_coordinates_coarse.shape[-1]), colors_coarse, densities_coarse, sdfs_coarse,
                                                                  # depths_fine, sample_coordinates_fine.reshape(batch_size, num_rays, samples_per_ray, sample_coordinates_coarse.shape[-1]), colors_fine, densities_fine, sdfs_fine)
            all_depths, all_colors, all_densities, all_sdfs, all_densities_aux, all_sdfs_aux, all_pts = self.unify_samples_add_sdf_add_pts(depths_coarse, sample_coordinates_coarse.reshape(batch_size, num_rays, samples_per_ray, sample_coordinates_coarse.shape[-1]), colors_coarse, densities_coarse, sdfs_coarse,densities_aux_coarse,sdfs_aux_coarse,
                                                                  depths_fine, sample_coordinates_fine.reshape(batch_size, num_rays, samples_per_ray, sample_coordinates_coarse.shape[-1]), colors_fine, densities_fine, sdfs_fine,densities_aux_fine,sdfs_aux_fine)

            # Aggregate
            rgb_final, depth_final, weights,_ = self.ray_marcher(all_colors, all_densities, all_depths, rendering_options)
            
            densities_aux_list = []
            sdf_aux_list = torch.split(all_sdfs_aux,1,dim=-1)
            for i in range(hp.number):
                rgb_final_rgbsegs[str(i)], depth_final_rgbsegs[str(i)], _ , densities_local= self.ray_marcher(all_colors, all_densities_aux[...,i:i+1], all_depths, rendering_options)
                densities_aux_list.append(densities_local)
            semantic = torch.sigmoid(-self.sigmoid * torch.cat(sdf_aux_list,3))
            if hp.semantic_density_detach:
                semantic_final, _, _,_ = self.ray_marcher(semantic, all_densities.detach(), all_depths, rendering_options)
            else:
                semantic_final, _, _,_ = self.ray_marcher(semantic, all_densities, all_depths, rendering_options)
            # semantic_final = torch.softmax(semantic_final,dim=-1)
            # semantic_final = semantic_final*2 -1   
            
        else:
            rgb_final, depth_final, weights,_ = self.ray_marcher(colors_coarse, densities_coarse, depths_coarse, rendering_options)



        # return rgb_final, depth_final, weights.sum(2), [eikonal1,eikonal2], all_sdfs, all_densities, all_pts
        # return sample_coordinates_coarse, sample_coordinates_fine, sample_directions, depths_coarse, depths_fine
        return all_depths, all_colors, all_densities, all_pts, sample_directions

    def forward_coord(self, planes_tex, planes_shp, decoder, ray_origins, ray_directions, eik, rendering_options):
        self.plane_axes = self.plane_axes.to(ray_origins.device)

        if rendering_options['ray_start'] == rendering_options['ray_end'] == 'auto':
            ray_start, ray_end = math_utils.get_ray_limits_box(ray_origins, ray_directions, box_side_length=rendering_options['box_warp'])
            is_ray_valid = ray_end > ray_start
            if torch.any(is_ray_valid).item():
                ray_start[~is_ray_valid] = ray_start[is_ray_valid].min()
                ray_end[~is_ray_valid] = ray_start[is_ray_valid].max()
            depths_coarse = self.sample_stratified(ray_origins, ray_start, ray_end, rendering_options['depth_resolution'], rendering_options['disparity_space_sampling'])
        else:
            # Create stratified depth samples
            depths_coarse = self.sample_stratified(ray_origins, rendering_options['ray_start'], rendering_options['ray_end'], rendering_options['depth_resolution'], rendering_options['disparity_space_sampling'])

        batch_size, num_rays, samples_per_ray, _ = depths_coarse.shape

        # Coarse Pass
        sample_coordinates_coarse = (ray_origins.unsqueeze(-2) + depths_coarse * ray_directions.unsqueeze(-2)).reshape(batch_size, -1, 3)
        sample_directions = ray_directions.unsqueeze(-2).expand(-1, -1, samples_per_ray, -1).reshape(batch_size, -1, 3)

        ## SDF!!!
        if eik:
            sample_coordinates_coarse.requires_grad = True


        out = self.run_model(planes_tex, planes_shp, decoder, sample_coordinates_coarse, sample_directions, rendering_options)
        colors_coarse = out['rgb']
        densities_coarse = out['sigma']
        sdfs_coarse = out['sdf'] 
        densities_aux_coarse = out['aux_sigma']
        sdfs_aux_coarse = out['aux_sdf']

        colors_coarse = colors_coarse.reshape(batch_size, num_rays, samples_per_ray, colors_coarse.shape[-1])
        densities_coarse = densities_coarse.reshape(batch_size, num_rays, samples_per_ray, 1)
        sdfs_coarse = sdfs_coarse.reshape(batch_size, num_rays, samples_per_ray, 1)
        densities_aux_coarse = densities_aux_coarse.reshape(batch_size, num_rays, samples_per_ray, hp.number)
        sdfs_aux_coarse = sdfs_aux_coarse.reshape(batch_size, num_rays, samples_per_ray, hp.number)

        # Fine Pass
        N_importance = rendering_options['depth_resolution_importance']
        rgb_final_rgbsegs = OrderedDict()
        depth_final_rgbsegs = OrderedDict()
        if N_importance > 0:
            _, _, weights,_ = self.ray_marcher(colors_coarse, densities_coarse, depths_coarse, rendering_options)

            depths_fine = self.sample_importance(depths_coarse, weights, N_importance)

            sample_directions = ray_directions.unsqueeze(-2).expand(-1, -1, N_importance, -1).reshape(batch_size, -1, 3)
            sample_coordinates_fine = (ray_origins.unsqueeze(-2) + depths_fine * ray_directions.unsqueeze(-2)).reshape(batch_size, -1, 3)

            ### SDF!!!
            if eik:
                sample_coordinates_fine.requires_grad = True

            out = self.run_model(planes_tex, planes_shp, decoder, sample_coordinates_fine, sample_directions, rendering_options)
            colors_fine = out['rgb']
            densities_fine = out['sigma']
            sdfs_fine = out['sdf']
            densities_aux_fine = out['aux_sigma']
            sdfs_aux_fine = out['aux_sdf']
        
            colors_fine = colors_fine.reshape(batch_size, num_rays, N_importance, colors_fine.shape[-1])
            densities_fine = densities_fine.reshape(batch_size, num_rays, N_importance, 1)
            sdfs_fine = sdfs_fine.reshape(batch_size, num_rays, N_importance, 1)
            densities_aux_fine = densities_aux_fine.reshape(batch_size, num_rays, N_importance, hp.number)
            sdfs_aux_fine = sdfs_aux_fine.reshape(batch_size, num_rays, N_importance, hp.number)


            # all_depths, all_colors, all_densities = self.unify_samples(depths_coarse, colors_coarse, densities_coarse,
            #                                                       depths_fine, colors_fine, densities_fine)


            if eik:
                eikonal1 = self.get_eikonal_term(sample_coordinates_coarse, sdfs_coarse)
                eikonal2 = self.get_eikonal_term(sample_coordinates_fine, sdfs_fine)
                
                if hp.eik_seg:
                    eikonal1_aux = self.get_eikonal_term(sample_coordinates_coarse, sdfs_aux_coarse[...,1:])
                    eikonal2_aux = self.get_eikonal_term(sample_coordinates_fine, sdfs_aux_fine[...,1:])
                else:
                    eikonal1_aux = torch.ones_like(sdfs_coarse)
                    eikonal2_aux = torch.ones_like(sdfs_fine)
            else:
                eikonal1 = torch.ones_like(sdfs_coarse)
                eikonal2 = torch.ones_like(sdfs_fine)
                eikonal1_aux = torch.ones_like(sdfs_coarse)
                eikonal2_aux = torch.ones_like(sdfs_fine)
                
            # all_depths, all_colors, all_densities, all_sdfs, all_pts = self.unify_samples_add_sdf_add_pts(depths_coarse, sample_coordinates_coarse.reshape(batch_size, num_rays, samples_per_ray, sample_coordinates_coarse.shape[-1]), colors_coarse, densities_coarse, sdfs_coarse,
                                                                  # depths_fine, sample_coordinates_fine.reshape(batch_size, num_rays, samples_per_ray, sample_coordinates_coarse.shape[-1]), colors_fine, densities_fine, sdfs_fine)
            all_depths, all_colors, all_densities, all_sdfs, all_densities_aux, all_sdfs_aux, all_pts = self.unify_samples_add_sdf_add_pts(depths_coarse, sample_coordinates_coarse.reshape(batch_size, num_rays, samples_per_ray, sample_coordinates_coarse.shape[-1]), colors_coarse, densities_coarse, sdfs_coarse,densities_aux_coarse,sdfs_aux_coarse,
                                                                  depths_fine, sample_coordinates_fine.reshape(batch_size, num_rays, samples_per_ray, sample_coordinates_coarse.shape[-1]), colors_fine, densities_fine, sdfs_fine,densities_aux_fine,sdfs_aux_fine)

            # Aggregate
            rgb_final, depth_final, weights,_ = self.ray_marcher(all_colors, all_densities, all_depths, rendering_options)
            
            densities_aux_list = []
            sdf_aux_list = torch.split(all_sdfs_aux,1,dim=-1)
            for i in range(hp.number):
                rgb_final_rgbsegs[str(i)], depth_final_rgbsegs[str(i)], _ , densities_local= self.ray_marcher(all_colors, all_densities_aux[...,i:i+1], all_depths, rendering_options)
                densities_aux_list.append(densities_local)
            semantic = torch.sigmoid(-self.sigmoid * torch.cat(sdf_aux_list,3))
            if hp.semantic_density_detach:
                semantic_final, _, _,_ = self.ray_marcher(semantic, all_densities.detach(), all_depths, rendering_options)
            else:
                semantic_final, _, _,_ = self.ray_marcher(semantic, all_densities, all_depths, rendering_options)
            # semantic_final = torch.softmax(semantic_final,dim=-1)
            # semantic_final = semantic_final*2 -1   
            
        else:
            rgb_final, depth_final, weights,_ = self.ray_marcher(colors_coarse, densities_coarse, depths_coarse, rendering_options)



        # return rgb_final, depth_final, weights.sum(2), [eikonal1,eikonal2], all_sdfs, all_densities, all_pts
        return sample_coordinates_coarse, sample_coordinates_fine, sample_directions, depths_coarse, depths_fine


    def run_model(self, planes_tex, planes_shp, decoder, sample_coordinates, sample_directions, options):
        sampled_features_tex = sample_from_planes(self.plane_axes, planes_tex, sample_coordinates, padding_mode='zeros', box_warp=options['box_warp'])
        sampled_features_shp = sample_from_planes(self.plane_axes, planes_shp, sample_coordinates, padding_mode='zeros', box_warp=options['box_warp'])

        out = decoder(sampled_features_tex, sampled_features_shp, sample_directions)
        if options.get('density_noise', 0) > 0:
            out['sigma'] += torch.randn_like(out['sigma']) * options['density_noise']
        return out

    def run_model_get_planes(self, planes_tex, planes_shp, decoder, sample_coordinates, sample_directions, options):
        sampled_features_tex = sample_from_planes(self.plane_axes, planes_tex, sample_coordinates, padding_mode='zeros', box_warp=options['box_warp'])
        sampled_features_shp = sample_from_planes(self.plane_axes, planes_shp, sample_coordinates, padding_mode='zeros', box_warp=options['box_warp'])

        return sampled_features_tex, sampled_features_shp

    def run_model_from_plane_to_feat(self, planes_tex, planes_shp, decoder):
        out = decoder(planes_tex, planes_shp, 0)

        return out

    def forward_blend(self, planes_tex, planes_shp, decoder, ray_origins, ray_directions, depths_coarse, rendering_options):
        self.plane_axes = self.plane_axes.to(ray_origins.device)

        # if rendering_options['ray_start'] == rendering_options['ray_end'] == 'auto':
        #     ray_start, ray_end = math_utils.get_ray_limits_box(ray_origins, ray_directions, box_side_length=rendering_options['box_warp'])
        #     is_ray_valid = ray_end > ray_start
        #     if torch.any(is_ray_valid).item():
        #         ray_start[~is_ray_valid] = ray_start[is_ray_valid].min()
        #         ray_end[~is_ray_valid] = ray_start[is_ray_valid].max()
        #     depths_coarse = self.sample_offset(ray_origins, ray_start, ray_end, rendering_options['depth_resolution'], rendering_options['disparity_space_sampling'])
        # else:
        #     # Create stratified depth samples
        #     depths_coarse = self.sample_offset(ray_origins, rendering_options['ray_start'], rendering_options['ray_end'], rendering_options['depth_resolution'], rendering_options['disparity_space_sampling'])

        batch_size, num_rays, samples_per_ray, _ = depths_coarse.shape

        # Coarse Pass
        sample_coordinates_coarse = (ray_origins.unsqueeze(-2) + depths_coarse * ray_directions.unsqueeze(-2)).reshape(batch_size, -1, 3)
        sample_directions = ray_directions.unsqueeze(-2).expand(-1, -1, samples_per_ray, -1).reshape(batch_size, -1, 3)



        out = self.run_model(planes_tex, planes_shp, decoder, sample_coordinates_coarse, sample_directions, rendering_options)
        colors_coarse = out['rgb']
        densities_coarse = out['sigma']
        sdfs_coarse = out['sdf'] 
        densities_aux_coarse = out['aux_sigma']
        sdfs_aux_coarse = out['aux_sdf']

        colors_coarse = colors_coarse.reshape(batch_size, num_rays, samples_per_ray, colors_coarse.shape[-1])
        densities_coarse = densities_coarse.reshape(batch_size, num_rays, samples_per_ray, 1)
        sdfs_coarse = sdfs_coarse.reshape(batch_size, num_rays, samples_per_ray, 1)
        densities_aux_coarse = densities_aux_coarse.reshape(batch_size, num_rays, samples_per_ray, hp.number)
        sdfs_aux_coarse = sdfs_aux_coarse.reshape(batch_size, num_rays, samples_per_ray, hp.number)

        # Fine Pass
        N_importance = rendering_options['depth_resolution_importance']
        rgb_final_rgbsegs = OrderedDict()
        depth_final_rgbsegs = OrderedDict()
        if N_importance > 0:
            _, _, weights,_ = self.ray_marcher(colors_coarse, densities_coarse, depths_coarse, rendering_options)

            depths_fine = self.sample_importance(depths_coarse, weights, N_importance)

            sample_directions = ray_directions.unsqueeze(-2).expand(-1, -1, N_importance, -1).reshape(batch_size, -1, 3)
            sample_coordinates_fine = (ray_origins.unsqueeze(-2) + depths_fine * ray_directions.unsqueeze(-2)).reshape(batch_size, -1, 3)


            out = self.run_model(planes_tex, planes_shp, decoder, sample_coordinates_fine, sample_directions, rendering_options)
            colors_fine = out['rgb']
            densities_fine = out['sigma']
            sdfs_fine = out['sdf']
            densities_aux_fine = out['aux_sigma']
            sdfs_aux_fine = out['aux_sdf']
        
            colors_fine = colors_fine.reshape(batch_size, num_rays, N_importance, colors_fine.shape[-1])
            densities_fine = densities_fine.reshape(batch_size, num_rays, N_importance, 1)
            sdfs_fine = sdfs_fine.reshape(batch_size, num_rays, N_importance, 1)
            densities_aux_fine = densities_aux_fine.reshape(batch_size, num_rays, N_importance, hp.number)
            sdfs_aux_fine = sdfs_aux_fine.reshape(batch_size, num_rays, N_importance, hp.number)


            # all_depths, all_colors, all_densities = self.unify_samples(depths_coarse, colors_coarse, densities_coarse,
            #                                                       depths_fine, colors_fine, densities_fine)
                
            # all_depths, all_colors, all_densities, all_sdfs, all_pts = self.unify_samples_add_sdf_add_pts(depths_coarse, sample_coordinates_coarse.reshape(batch_size, num_rays, samples_per_ray, sample_coordinates_coarse.shape[-1]), colors_coarse, densities_coarse, sdfs_coarse,
                                                                  # depths_fine, sample_coordinates_fine.reshape(batch_size, num_rays, samples_per_ray, sample_coordinates_coarse.shape[-1]), colors_fine, densities_fine, sdfs_fine)
            all_depths, all_colors, all_densities, all_sdfs, all_densities_aux, all_sdfs_aux, all_pts = self.unify_samples_add_sdf_add_pts(depths_coarse, sample_coordinates_coarse.reshape(batch_size, num_rays, samples_per_ray, sample_coordinates_coarse.shape[-1]), colors_coarse, densities_coarse, sdfs_coarse,densities_aux_coarse,sdfs_aux_coarse,
                                                                  depths_fine, sample_coordinates_fine.reshape(batch_size, num_rays, samples_per_ray, sample_coordinates_coarse.shape[-1]), colors_fine, densities_fine, sdfs_fine,densities_aux_fine,sdfs_aux_fine)


            # semantic_final = torch.softmax(semantic_final,dim=-1)
            # semantic_final = semantic_final*2 -1   
            
        else:
            rgb_final, depth_final, weights,_ = self.ray_marcher(colors_coarse, densities_coarse, depths_coarse, rendering_options)


        return all_colors, all_densities, all_depths, rendering_options

 
    def sort_samples(self, all_depths, all_colors, all_densities):
        _, indices = torch.sort(all_depths, dim=-2)
        all_depths = torch.gather(all_depths, -2, indices)
        all_colors = torch.gather(all_colors, -2, indices.expand(-1, -1, -1, all_colors.shape[-1]))
        all_densities = torch.gather(all_densities, -2, indices.expand(-1, -1, -1, 1))
        return all_depths, all_colors, all_densities

    def unify_samples(self, depths1, colors1, densities1, depths2, colors2, densities2):
        all_depths = torch.cat([depths1, depths2], dim = -2)
        all_colors = torch.cat([colors1, colors2], dim = -2)
        all_densities = torch.cat([densities1, densities2], dim = -2)

        _, indices = torch.sort(all_depths, dim=-2)
        all_depths = torch.gather(all_depths, -2, indices)
        all_colors = torch.gather(all_colors, -2, indices.expand(-1, -1, -1, all_colors.shape[-1]))
        all_densities = torch.gather(all_densities, -2, indices.expand(-1, -1, -1, 1))

        return all_depths, all_colors, all_densities

    def unify_samples_add_sdf_add_pts(self, depths1, pts1, colors1, densities1, sdf1,densities_aux_1,sdf_aux_1,  depths2, pts2, colors2, densities2, sdf2, densities_aux_2, sdf_aux_2):
        all_depths = torch.cat([depths1, depths2], dim = -2)
        all_colors = torch.cat([colors1, colors2], dim = -2)
        all_densities = torch.cat([densities1, densities2], dim = -2)
        all_sdfs = torch.cat([sdf1, sdf2], dim = -2)
        all_densities_aux = torch.cat([densities_aux_1, densities_aux_2], dim = -2)
        all_sdfs_aux = torch.cat([sdf_aux_1, sdf_aux_2], dim = -2)
        all_pts = torch.cat([pts1, pts2], dim = -2)

        _, indices = torch.sort(all_depths, dim=-2)
        all_depths = torch.gather(all_depths, -2, indices)
        all_colors = torch.gather(all_colors, -2, indices.expand(-1, -1, -1, all_colors.shape[-1]))
        all_densities = torch.gather(all_densities, -2, indices.expand(-1, -1, -1, all_densities.shape[-1]))
        all_sdfs = torch.gather(all_sdfs, -2, indices.expand(-1, -1, -1, all_sdfs.shape[-1]))
        all_densities_aux = torch.gather(all_densities_aux, -2, indices.expand(-1, -1, -1, all_densities_aux.shape[-1]))
        all_sdfs_aux = torch.gather(all_sdfs_aux, -2, indices.expand(-1, -1, -1, all_sdfs_aux.shape[-1]))
        all_pts = torch.gather(all_pts, -2, indices.expand(-1, -1, -1, all_pts.shape[-1]))

        return all_depths, all_colors, all_densities, all_sdfs, all_densities_aux, all_sdfs_aux, all_pts
    
    
    def unify_samples_add_sdfs(self, depths1, colors1, densities1, sdf1, depths2, colors2, densities2, sdf2):
        all_depths = torch.cat([depths1, depths2], dim = -2)
        all_colors = torch.cat([colors1, colors2], dim = -2)
        all_densities = torch.cat([densities1, densities2], dim = -2)
        all_sdfs = torch.cat([sdf1, sdf2], dim = -2)

        _, indices = torch.sort(all_depths, dim=-2)
        all_depths = torch.gather(all_depths, -2, indices)
        all_colors = torch.gather(all_colors, -2, indices.expand(-1, -1, -1, all_colors.shape[-1]))
        all_densities = torch.gather(all_densities, -2, indices.expand(-1, -1, -1, all_densities.shape[-1]))
        all_sdfs = torch.gather(all_sdfs, -2, indices.expand(-1, -1, -1, all_sdfs.shape[-1]))

        return all_depths, all_colors, all_densities, all_sdfs


    def sample_stratified(self, ray_origins, ray_start, ray_end, depth_resolution, disparity_space_sampling=False):
        """
        Return depths of approximately uniformly spaced samples along rays.
        """
        N, M, _ = ray_origins.shape
        if disparity_space_sampling:
            depths_coarse = torch.linspace(0,
                                    1,
                                    depth_resolution,
                                    device=ray_origins.device).reshape(1, 1, depth_resolution, 1).repeat(N, M, 1, 1)
            depth_delta = 1/(depth_resolution - 1)
            depths_coarse += torch.rand_like(depths_coarse) * depth_delta
            depths_coarse = 1./(1./ray_start * (1. - depths_coarse) + 1./ray_end * depths_coarse)
        else:
            if type(ray_start) == torch.Tensor:
                depths_coarse = math_utils.linspace(ray_start, ray_end, depth_resolution).permute(1,2,0,3)
                depth_delta = (ray_end - ray_start) / (depth_resolution - 1)
                depths_coarse += torch.rand_like(depths_coarse) * depth_delta[..., None]
            else:
                depths_coarse = torch.linspace(ray_start, ray_end, depth_resolution, device=ray_origins.device).reshape(1, 1, depth_resolution, 1).repeat(N, M, 1, 1)
                depth_delta = (ray_end - ray_start)/(depth_resolution - 1)
                depths_coarse += torch.rand_like(depths_coarse) * depth_delta

        return depths_coarse

    
    def sample_offset(self, ray_origins, ray_start, ray_end, depth_resolution, disparity_space_sampling=False):
        """
        Return depths of approximately uniformly spaced samples along rays.
        """
        N, M, _ = ray_origins.shape
        if disparity_space_sampling:
            depths_coarse = torch.linspace(0,
                                    1,
                                    depth_resolution,
                                    device=ray_origins.device).reshape(1, 1, depth_resolution, 1).repeat(N, M, 1, 1)
            depth_delta = 1/(depth_resolution - 1)
            depths_coarse += torch.rand_like(depths_coarse) * depth_delta
            depths_coarse = 1./(1./ray_start * (1. - depths_coarse) + 1./ray_end * depths_coarse)
        else:
            if type(ray_start) == torch.Tensor:
                depths_coarse = math_utils.linspace(ray_start, ray_end, depth_resolution).permute(1,2,0,3)
                depth_delta = (ray_end - ray_start) / (depth_resolution)
                depths_coarse += torch.rand_like(depths_coarse) * depth_delta[..., None]
            else:
                depths_coarse = torch.linspace(ray_start, ray_end, depth_resolution, device=ray_origins.device).reshape(1, 1, depth_resolution, 1).repeat(N, M, 1, 1)
                depth_delta = (ray_end - ray_start)/(depth_resolution)
                depths_coarse += torch.rand(depths_coarse.shape[0],depths_coarse.shape[1],1,depths_coarse.shape[-1], device=ray_origins.device) * depth_delta
        return depths_coarse

    def sample_importance(self, z_vals, weights, N_importance):
        """
        Return depths of importance sampled points along rays. See NeRF importance sampling for more.
        """
        with torch.no_grad():
            batch_size, num_rays, samples_per_ray, _ = z_vals.shape

            z_vals = z_vals.reshape(batch_size * num_rays, samples_per_ray)
            weights = weights.reshape(batch_size * num_rays, -1) # -1 to account for loss of 1 sample in MipRayMarcher

            # smooth weights
            weights = torch.nn.functional.max_pool1d(weights.unsqueeze(1).float(), 2, 1, padding=1)
            weights = torch.nn.functional.avg_pool1d(weights, 2, 1).squeeze()
            weights = weights + 0.01

            z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:])
            importance_z_vals = self.sample_pdf(z_vals_mid, weights[:, 1:-1],
                                             N_importance).detach().reshape(batch_size, num_rays, N_importance, 1)
        return importance_z_vals

    def sample_pdf(self, bins, weights, N_importance, det=False, eps=1e-5):
        """
        Sample @N_importance samples from @bins with distribution defined by @weights.
        Inputs:
            bins: (N_rays, N_samples_+1) where N_samples_ is "the number of coarse samples per ray - 2"
            weights: (N_rays, N_samples_)
            N_importance: the number of samples to draw from the distribution
            det: deterministic or not
            eps: a small number to prevent division by zero
        Outputs:
            samples: the sampled samples
        """
        N_rays, N_samples_ = weights.shape
        weights = weights + eps # prevent division by zero (don't do inplace op!)
        pdf = weights / torch.sum(weights, -1, keepdim=True) # (N_rays, N_samples_)
        cdf = torch.cumsum(pdf, -1) # (N_rays, N_samples), cumulative distribution function
        cdf = torch.cat([torch.zeros_like(cdf[: ,:1]), cdf], -1)  # (N_rays, N_samples_+1)
                                                                   # padded to 0~1 inclusive

        if det:
            u = torch.linspace(0, 1, N_importance, device=bins.device)
            u = u.expand(N_rays, N_importance)
        else:
            u = torch.rand(N_rays, N_importance, device=bins.device)
        u = u.contiguous()

        inds = torch.searchsorted(cdf, u, right=True)
        below = torch.clamp_min(inds-1, 0)
        above = torch.clamp_max(inds, N_samples_)

        inds_sampled = torch.stack([below, above], -1).view(N_rays, 2*N_importance)
        cdf_g = torch.gather(cdf, 1, inds_sampled).view(N_rays, N_importance, 2)
        bins_g = torch.gather(bins, 1, inds_sampled).view(N_rays, N_importance, 2)

        denom = cdf_g[...,1]-cdf_g[...,0]
        denom[denom<eps] = 1 # denom equals 0 means a bin has weight 0, in which case it will not be sampled
                             # anyway, therefore any value for it is fine (set to 1 here)

        samples = bins_g[...,0] + (u-cdf_g[...,0])/denom * (bins_g[...,1]-bins_g[...,0])
        return samples