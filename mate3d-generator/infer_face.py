# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Generate images and shapes using pretrained network pickle."""

import os
import re
from typing import List, Optional, Tuple, Union

import click
import dnnlib
import numpy as np
import PIL.Image
import torch
from tqdm import tqdm
import mrcfile

import legacy
from camera_utils import LookAtPoseSampler, FOV_to_intrinsics


#----------------------------------------------------------------------------
COLOR_MAP_NEW = {
    0: [0, 0, 0],
    1: [204, 0, 0],
    2: [76, 153, 0], 
    3: [204, 204, 0], 
    4: [51, 51, 255], 
    5: [204, 0, 204], 
    6: [0, 255, 255], 
    7: [255, 204, 204], 
    8: [102, 51, 0], 
    9: [255, 0, 0], 
    10: [102, 204, 0], 
    11: [255, 255, 0], 
    12: [0, 0, 153], 
    13: [0, 0, 204], 
    14: [255, 51, 153], 
    15: [0, 204, 204], 
    16: [0, 51, 0], 
    17: [255, 153, 51], 
    18: [0, 204, 0]}

def mask2color(masks):
    # masks = torch.argmax(masks, dim=1).float()
    # print(masks.shape)
    # masks = remap_list[masks.long()]
    sample_mask = torch.zeros((masks.shape[0], masks.shape[1], masks.shape[2], 3), dtype=torch.float, device=masks.device)
    for key in COLOR_MAP_NEW:
        sample_mask[masks==key] = torch.tensor(COLOR_MAP_NEW[key], dtype=torch.float, device=masks.device)
    # sample_mask = sample_mask.permute(0,3,1,2)
    return sample_mask


def parse_range(s: Union[str, List]) -> List[int]:
    '''Parse a comma separated list of numbers or ranges and return a list of ints.

    Example: '1,2,5-10' returns [1, 2, 5, 6, 7]
    '''
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        if m := range_re.match(p):
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------

def parse_vec2(s: Union[str, Tuple[float, float]]) -> Tuple[float, float]:
    '''Parse a floating point 2-vector of syntax 'a,b'.

    Example:
        '0,1' returns (0,1)
    '''
    if isinstance(s, tuple): return s
    parts = s.split(',')
    if len(parts) == 2:
        return (float(parts[0]), float(parts[1]))
    raise ValueError(f'cannot parse 2-vector {s}')

#----------------------------------------------------------------------------

def make_transform(translate: Tuple[float,float], angle: float):
    m = np.eye(3)
    s = np.sin(angle/360.0*np.pi*2)
    c = np.cos(angle/360.0*np.pi*2)
    m[0][0] = c
    m[0][1] = s
    m[0][2] = translate[0]
    m[1][0] = -s
    m[1][1] = c
    m[1][2] = translate[1]
    return m

#----------------------------------------------------------------------------

def create_samples(N=256, voxel_origin=[0, 0, 0], cube_length=2.0):
    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = np.array(voxel_origin) - cube_length/2
    voxel_size = cube_length / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 3)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.float() / N) % N
    samples[:, 0] = ((overall_index.float() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N ** 3

    return samples.unsqueeze(0), voxel_origin, voxel_size

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=False, default='/openbayes/input/input0/mate3d-mask-face.pkl')
@click.option('--seeds', type=parse_range, help='List of random seeds (e.g., \'0,1,4-6\')', required=False, default = '666')
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=0.5, show_default=True)
@click.option('--trunc-cutoff', 'truncation_cutoff', type=int, help='Truncation cutoff', default=14, show_default=True)
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)')
@click.option('--outdir', help='Where to save the output images', type=str, required=False,default='out-sample-quality', metavar='DIR')
@click.option('--shapes', help='Export shapes as .mrc files viewable in ChimeraX', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
@click.option('--shape-res', help='', type=int, required=False, metavar='int', default=512, show_default=True)
@click.option('--fov-deg', help='Field of View of camera in degrees', type=int, required=False, metavar='float', default=18.837, show_default=True)
@click.option('--shape-format', help='Shape Format', type=click.Choice(['.mrc', '.ply']), default='.ply')
@click.option('--reload_modules', help='Overload persistent modules?', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
def generate_images(
    network_pkl: str,
    seeds: List[int],
    truncation_psi: float,
    truncation_cutoff: int,
    outdir: str,
    shapes: bool,
    shape_res: int,
    fov_deg: float,
    shape_format: str,
    class_idx: Optional[int],
    reload_modules: bool,
):
    """Generate images using pretrained network pickle.

    Examples:

    \b
    # Generate an image using pre-trained FFHQ model.
    python gen_samples.py --outdir=output --trunc=0.7 --seeds=0-5 --shapes=True\\
        --network=ffhq-rebalanced-128.pkl
    """

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

 

    os.makedirs(outdir, exist_ok=True)

    cam2world_pose = LookAtPoseSampler.sample(3.14/2, 3.14/2, torch.tensor([0, 0, 0.2], device=device), radius=2.7, device=device)
    intrinsics = FOV_to_intrinsics(fov_deg, device=device)

    # Generate images.
    for seed_idx, seed in enumerate(seeds):
        print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)

        imgs = []
        imgs_raw = []
        segs_raw = []
        segs = []
        mask_color_rgbs = []
        angle_p = -0.2
        
            
        # for angle_y, angle_p in [(30/180*np.pi, angle_p),(20/180*np.pi, angle_p),(10/180*np.pi, angle_p), (0, angle_p), (-10/180*np.pi, angle_p),(-20/180*np.pi, angle_p), (-30/180*np.pi, angle_p)]:
        for angle_y, angle_p in [(20/180*np.pi, angle_p),(10/180*np.pi, angle_p), (0, angle_p), (-10/180*np.pi, angle_p),(-20/180*np.pi, angle_p)]:
            cam_pivot = torch.tensor(G.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0]), device=device)
            cam_radius = G.rendering_kwargs.get('avg_camera_radius', 2.7)
            cam2world_pose = LookAtPoseSampler.sample(np.pi/2 + angle_y, np.pi/2 + angle_p, cam_pivot, radius=cam_radius, device=device)
            conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi/2, np.pi/2, cam_pivot, radius=cam_radius, device=device)
            camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
            conditioning_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

            ws = G.mapping(z, conditioning_params,truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
            temp = G.synthesis(ws, camera_params)
            img = temp['image']
            img_raw = temp['image_raw']
            seg_raw = temp['semantic_raw']
            seg = temp['semantic']
            seg_ori = temp['semantic']

            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            img_raw = (img_raw.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            seg_raw = (torch.argmax(seg_raw,1).unsqueeze(1).permute(0, 2, 3, 1) /19*255).clamp(0, 255).to(torch.uint8)
            seg = (torch.argmax(seg,1).unsqueeze(1).permute(0, 2, 3, 1) /19*255).clamp(0, 255).to(torch.uint8)
            mask_color_rgb = mask2color(torch.argmax(seg_ori,1)).to(torch.uint8)

            imgs.append(img)
            imgs_raw.append(img_raw)
            segs.append(seg)
            segs_raw.append(seg_raw)
            mask_color_rgbs.append(mask_color_rgb)
            
        imgs = torch.cat(imgs, dim=2)
        imgs_raw = torch.cat(imgs_raw, dim=2)
        segs = torch.cat(segs, dim=2).repeat(1,1,1,3)
        segs_raw = torch.cat(segs_raw, dim=2).repeat(1,1,1,3)
        mask_color_rgbs = torch.cat(mask_color_rgbs, dim=2)
        # print(img.shape)
        # print(imgs_raw.shape)
        # print(segs.shape)
        # print(segs_raw.shape)
        PIL.Image.fromarray(imgs[0].cpu().numpy(), 'RGB').save(f'{outdir}/seed{seed:04d}.png')
        PIL.Image.fromarray(imgs_raw[0].cpu().numpy(), 'RGB').save(f'{outdir}/seed{seed:04d}_raw.png')
        PIL.Image.fromarray(segs[0].cpu().numpy(), 'RGB').save(f'{outdir}/seed{seed:04d}_segs.png')
        PIL.Image.fromarray(segs_raw[0].cpu().numpy(), 'RGB').save(f'{outdir}/seed{seed:04d}_seg_raw.png')
        PIL.Image.fromarray(mask_color_rgbs[0].cpu().numpy(), 'RGB').save(f'{outdir}/mask_color_rgb_seed{seed:04d}.png')

        if shapes:
            # extract a shape.mrc with marching cubes. You can view the .mrc file using ChimeraX from UCSF.
            max_batch=1000000

            samples, voxel_origin, voxel_size = create_samples(N=shape_res, voxel_origin=[0, 0, 0], cube_length=G.rendering_kwargs['box_warp'] * 1)#.reshape(1, -1, 3)
            samples = samples.to(z.device)
            sigmas = torch.zeros((samples.shape[0], samples.shape[1], 1), device=z.device).int()

                
            transformed_ray_directions_expanded = torch.zeros((samples.shape[0], max_batch, 3), device=z.device)
            transformed_ray_directions_expanded[..., -1] = -1

            head = 0
            with tqdm(total = samples.shape[1]) as pbar:
                with torch.no_grad():
                    while head < samples.shape[1]:
                        torch.manual_seed(0)
                        temp = G.sample(samples[:, head:head+max_batch], transformed_ray_directions_expanded[:, :samples.shape[1]-head], z, conditioning_params, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, noise_mode='const')
                        sigmas[:, head:head+max_batch] = temp['sigma'].int()
                        
                        head += max_batch
                        pbar.update(max_batch)

            sigmas = sigmas.reshape((shape_res, shape_res, shape_res)).cpu().numpy()
            sigmas = np.flip(sigmas, 0)

            # Trim the border of the extracted cube
            pad = int(30 * shape_res / 256)
            pad_value = -1000
            sigmas[:pad] = pad_value
            sigmas[-pad:] = pad_value
            sigmas[:, :pad] = pad_value
            sigmas[:, -pad:] = pad_value
            sigmas[:, :, :pad] = pad_value
            sigmas[:, :, -pad:] = pad_value

            if shape_format == '.ply':
                from shape_utils import convert_sdf_samples_to_ply
                convert_sdf_samples_to_ply(np.transpose(sigmas, (2, 1, 0)), [0, 0, 0], 1, os.path.join(outdir, f'seed{seed:04d}.ply'), level=10)
            elif shape_format == '.mrc': # output mrc
                with mrcfile.new_mmap(os.path.join(outdir, f'seed{seed:04d}.mrc'), overwrite=True, shape=sigmas.shape, mrc_mode=2) as mrc:
                    mrc.data[:] = sigmas
            


#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
