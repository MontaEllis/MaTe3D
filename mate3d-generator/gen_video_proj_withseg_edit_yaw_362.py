''' Generate videos using pretrained network pickle.
Code adapted from following paper
"Efficient Geometry-aware 3D Generative Adversarial Networks."
See LICENSES/LICENSE_EG3D for original license.
'''

import os
import re
from typing import List, Optional, Tuple, Union
import torch.nn.functional as F
import click
import dnnlib
import imageio
import numpy as np
import scipy.interpolate
import torch
from tqdm import tqdm
import mrcfile
from training.triplane import TriPlaneGenerator
import legacy
from hparams import hparams as hp
from camera_utils import LookAtPoseSampler
from torch_utils import misc
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

def dilate_3d(bin_img, ksize=5):
    src_size = bin_img.numpy().shape
    pad = (ksize - 1) // 2
    # bin_img = F.pad(bin_img, pad=[pad, pad, pad, pad], mode='reflect')
    out = F.max_pool3d(bin_img, kernel_size=ksize, stride=1, padding=0)
    out = F.interpolate(out,
                        size=src_size[2:],
                        mode="trilinear")
    return out

def erode_3d(bin_img, ksize=5):
    out = 1 - dilate_3d(1 - bin_img, ksize)
    return out

def mask3d_labels(mask_np):
    label_size = 19
    # label_size = 8
    # label_size = 6
    labels = np.zeros((label_size, mask_np.shape[0], mask_np.shape[1], mask_np.shape[2]))
    for i in range(label_size):
        labels[i][mask_np==i] = 1.0
    return labels

def layout_grid(img, grid_w=None, grid_h=1, float_to_uint8=True, chw_to_hwc=True, to_numpy=True):
    batch_size, channels, img_h, img_w = img.shape
    if grid_w is None:
        grid_w = batch_size // grid_h
    assert batch_size == grid_w * grid_h
    if float_to_uint8:
        img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    img = img.reshape(grid_h, grid_w, channels, img_h, img_w)
    img = img.permute(2, 0, 3, 1, 4)
    img = img.reshape(channels, grid_h * img_h, grid_w * img_w)
    if chw_to_hwc:
        img = img.permute(1, 2, 0)
    if to_numpy:
        img = img.cpu().numpy()
    return img

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

def gen_interp_video(G_origin, G, mp4: str, ws,ws_aux, w_frames=60*4, kind='cubic', grid_dims=(1,1), num_keyframes=None, wraps=2, psi=1, truncation_cutoff=14, cfg='FFHQ', image_mode='image', gen_shapes=False, device=torch.device('cuda'), **video_kwargs):
    grid_w = grid_dims[0]
    grid_h = grid_dims[1]

    if num_keyframes is None:
        if len(ws) % (grid_w*grid_h) != 0:
            raise ValueError('Number of input seeds must be divisible by grid W*H')
        num_keyframes = len(ws) // (grid_w*grid_h)

    camera_lookat_point = torch.tensor([0, 0, 0], device=device) if cfg == 'FFHQ' else torch.tensor([0, 0, 0], device=device)

    cam2world_pose = LookAtPoseSampler.sample(3.14/2, 3.14/2, camera_lookat_point, radius=2.7, device=device)
    intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=device)
    c = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
    c = c.repeat(len(ws), 1)
    # ws = G.mapping(z=zs, c=c, truncation_psi=psi, truncation_cutoff=truncation_cutoff)
    # _ = G.synthesis(ws[:1], c[:1]) # warm up
    ws = ws.reshape(grid_h, grid_w, num_keyframes, *ws.shape[1:])
    ws_aux = ws_aux.reshape(grid_h, grid_w, num_keyframes, *ws_aux.shape[1:])


    # create new folder
    outdirs = os.path.dirname(mp4)
    os.makedirs(outdirs, exist_ok=True)
    # add delta_c
    # z_samples = np.random.RandomState(123).randn(10000, G.z_dim)
    # delta_c = G.t_mapping(torch.from_numpy(np.mean(z_samples, axis=0, keepdims=True)).to(device), c[:1], truncation_psi=1.0, truncation_cutoff=None, update_emas=False)
    # delta_c = torch.squeeze(delta_c, 1)
    # c[:,3] += delta_c[:,0]
    # c[:,7] += delta_c[:,1]
    # c[:,11] += delta_c[:,2]

    # Interpolation.
    grid = []
    for yi in range(grid_h):
        row = []
        for xi in range(grid_w):
            x = np.arange(-num_keyframes * wraps, num_keyframes * (wraps + 1))
            y = np.tile(ws[yi][xi].cpu().numpy(), [wraps * 2 + 1, 1, 1])
            interp = scipy.interpolate.interp1d(x, y, kind=kind, axis=0)
            row.append(interp)
        grid.append(row)

    grid_aux = []
    for yi in range(grid_h):
        row_aux = []
        for xi in range(grid_w):
            x = np.arange(-num_keyframes * wraps, num_keyframes * (wraps + 1))
            y = np.tile(ws_aux[yi][xi].cpu().numpy(), [wraps * 2 + 1, 1, 1])
            interp = scipy.interpolate.interp1d(x, y, kind=kind, axis=0)
            row_aux.append(interp)
        grid_aux.append(row_aux)

    # Render video.
    max_batch = 10000000
    voxel_resolution = 512
    video_out = imageio.get_writer(mp4, mode='I', fps=60, codec='libx264', **video_kwargs)
    video_out_normal = imageio.get_writer(mp4.replace('.mp4','_normal.mp4'), mode='I', fps=60, codec='libx264', **video_kwargs)
    video_out_mask = imageio.get_writer(mp4.replace('.mp4','_mask.mp4'), mode='I', fps=60, codec='libx264', **video_kwargs)


    all_poses = []
    for frame_idx in tqdm(range(num_keyframes * w_frames)):
        imgs = []
        normals = []
        masks = []
        for yi in range(grid_h):
            for xi in range(grid_w):
                if cfg == "Head":
                    cam2world_pose = LookAtPoseSampler.sample(3.14/2 + 2 * 3.14 * frame_idx / (num_keyframes * w_frames), 3.14/2,
                                                            camera_lookat_point, radius=2.75, device=device)
                else:
                    pitch_range = 0.25
                    # yaw_range = 1.5 # 0.35
                    yaw_range = 0.35
                    cam2world_pose = LookAtPoseSampler.sample(3.14/2 + yaw_range * np.sin(2 * 3.14 * frame_idx / (num_keyframes * w_frames)),
                                                            3.14/2 -0.05 + pitch_range * np.cos(2 * 3.14 * frame_idx / (num_keyframes * w_frames)),
                                                            camera_lookat_point, radius=2.7, device=device)
                    # cam2world_pose = LookAtPoseSampler.sample(3.14/2 + 3.14/2/3 - 3.14/3 * frame_idx / (num_keyframes * w_frames),
                    #                                         3.14/2 -0.05,
                    #                                         camera_lookat_point, radius=2.7, device=device)

                all_poses.append(cam2world_pose.squeeze().cpu().numpy())
                intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=device)
                c = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

                interp = grid[yi][xi]
                w = torch.from_numpy(interp(frame_idx / w_frames)).to(device)
                interp_aux = grid_aux[yi][xi]
                w_aux = torch.from_numpy(interp_aux(frame_idx / w_frames)).to(device)

                print(w.shape)
                print(w_aux.shape)
                # img = G.synthesis(ws=w.unsqueeze(0), c=c[0:1], noise_mode='const')[image_mode][0]
                
                # fix delta_c
                # c[:,3] += delta_c[:,0]
                # c[:,7] += delta_c[:,1]
                # c[:,11] += delta_c[:,2]
                


                all_depths, coord_direction, all_coord = G_origin.synthesis_coord_first(w.unsqueeze(0), c=c, noise_mode='const')
                all_coord.requires_grad = True

                tex_plane, shp_plane = G.synthesis_with_coord_first(w_aux.unsqueeze(0), c, [all_coord, coord_direction, all_depths],noise_mode='const')
                temp = G.synthesis_result_first_inf(w_aux.unsqueeze(0), c, tex_plane,shp_plane,all_depths)
                tex_plane_origin, shp_plane_origin = G_origin.synthesis_with_coord_first(w.unsqueeze(0), c, [all_coord, coord_direction, all_depths],noise_mode='const')
                temp_origin = G_origin.synthesis_result_first_inf(w.unsqueeze(0), c, tex_plane_origin,shp_plane_origin,all_depths)



                mask_3d = torch.sigmoid(-20 * torch.cat(temp['all_sdf_aux'],3))
                control3d = mask_3d.reshape(1,128,128,96*2,hp.number)
                mask_3d_origin = torch.sigmoid(-20 * torch.cat(temp_origin['all_sdf_aux'],3))
                control3d_origin = mask_3d_origin.reshape(1,128,128,96*2,hp.number)



                control3d = torch.argmax(control3d,-1).to(torch.uint8)[0]
                control3d = control3d[:, :, :, np.newaxis]
                control3d = control3d.permute(3, 0, 1, 2)
                control3d = mask3d_labels(control3d[0].cpu())
                control3d = torch.from_numpy(control3d).permute(1, 2, 3, 0).numpy()
                control3d = torch.from_numpy(control3d).unsqueeze(0).permute(0,4,1,2,3)

                control3d_origin = torch.argmax(control3d_origin,-1).to(torch.uint8)[0]
                control3d_origin = control3d_origin[:, :, :, np.newaxis]
                control3d_origin = control3d_origin.permute(3, 0, 1, 2)
                control3d_origin = mask3d_labels(control3d_origin[0].cpu())
                control3d_origin = torch.from_numpy(control3d_origin).permute(1, 2, 3, 0).numpy()
                control3d_origin = torch.from_numpy(control3d_origin).unsqueeze(0).permute(0,4,1,2,3)


                
                control3d_max = torch.argmax(control3d,1).int()
                control3d_origin_max = torch.argmax(control3d_origin,1).int()

                ############################### TODO ###########################################
                # mask_3d_modify = (1-(control3d_origin_max==1).cuda().int())
                # mask_3d_modify_1 = (1-(control3d_max==1).cuda().int())
                mask_3d_modify_2 = (1-(control3d_origin_max==13).cuda().int())
                mask_3d_modify_3 = (1-(control3d_max==13).cuda().int())
                # mask_3d_modify_4 = (1-(control3d_origin_max==3).cuda().int())
                # mask_3d_modify_5 = (1-(control3d_max==3).cuda().int())
                # mask_3d_modify_6 = (1-(control3d_origin_max==4).cuda().int())
                # mask_3d_modify_7 = (1-(control3d_max==4).cuda().int())
                # mask_3d_modify_8 = (1-(control3d_origin_max==5).cuda().int())
                # mask_3d_modify_9 = (1-(control3d_max==5).cuda().int())
                mask_3d_modify = mask_3d_modify_2*mask_3d_modify_3#*mask_3d_modify_4*mask_3d_modify_5*mask_3d_modify_6*mask_3d_modify_7*mask_3d_modify_8*mask_3d_modify_9
                #############################################################################################
                
                mask_3d_modify = erode_3d(dilate_3d(mask_3d_modify.cpu().float().unsqueeze(1),2),10).cuda()[:,0,:,:,:]


                tex_plane_blend = mask_3d_modify.reshape(1,1,128*128*96*2,1).repeat(1,3,1,32) * tex_plane_origin + (1- mask_3d_modify.reshape(1,1,128*128*96*2,1).repeat(1,3,1,32))*tex_plane
                shp_plane_blend = mask_3d_modify.reshape(1,1,128*128*96*2,1).repeat(1,3,1,32) * shp_plane_origin + (1- mask_3d_modify.reshape(1,1,128*128*96*2,1).repeat(1,3,1,32))*shp_plane


                temp_blend = G_origin.synthesis_result_first_normal_inf(w.unsqueeze(0), c, tex_plane_blend,shp_plane_blend,all_coord,all_depths)                        


                img = temp_blend[image_mode][0]
                normal = temp_blend['normal'][0]

                mask = temp_blend['semantic']
                mask_color_rgb = mask2color(torch.argmax(mask,1)).permute(0,3,1,2)[0]/255.*2-1


                if image_mode == 'image_depth':
                    img = -img
                    img = (img - img.min()) / (img.max() - img.min()) * 2 - 1

                imgs.append(img)
                normals.append(normal)
                masks.append(mask_color_rgb)

                torch.cuda.empty_cache()
                del all_coord


        video_out.append_data(layout_grid(torch.stack(imgs), grid_w=grid_w, grid_h=grid_h))
        video_out_normal.append_data(layout_grid(torch.stack(normals), grid_w=grid_w, grid_h=grid_h))
        video_out_mask.append_data(layout_grid(torch.stack(masks), grid_w=grid_w, grid_h=grid_h))
    video_out.close()
    video_out_normal.close()
    all_poses = np.stack(all_poses)

    if gen_shapes:
        print(all_poses.shape)
        with open(mp4.replace('.mp4', '_trajectory.npy'), 'wb') as f:
            np.save(f, all_poses)

#----------------------------------------------------------------------------

def parse_range(s: Union[str, List[int]]) -> List[int]:
    '''Parse a comma separated list of numbers or ranges and return a list of ints.

    Example: '1,2,5-10' returns [1, 2, 5, 6, 7]
    '''
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------

def parse_tuple(s: Union[str, Tuple[int,int]]) -> Tuple[int, int]:
    '''Parse a 'M,N' or 'MxN' integer tuple.

    Example:
        '4x2' returns (4,2)
        '0,1' returns (0,1)
    '''
    if isinstance(s, tuple): return s
    m = re.match(r'^(\d+)[x,](\d+)$', s)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    raise ValueError(f'cannot parse tuple {s}')

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=False, default='/openbayes/input/input0/mate3d-mask-face.pkl')
@click.option('--network_aux', 'network_pkl_aux', help='Network pickle filename', required=False, default='/openbayes/input/input1/mate3d-mask-face.pkl')
@click.option('--latent', type=str, help='latent code', required=False,default='/mnt/workspace/xuanchuang/temp_workspace/DRGeRFface-sdf-simple-stratified-sdfreg-add-mask/out/0826_front/latent/seed0010.npz')
@click.option('--latent_aux', type=str, help='latent code', required=False,default='/mnt/workspace/xuanchuang/temp_workspace_face/Mate3D-mask-19/out/1112_front/latent/seed0047.npz')
@click.option('--output', help='Output path', type=str, required=False,default='videos/5_inpaint-flower-hair-stage2.mp4')
@click.option('--grid', type=parse_tuple, help='Grid width/height, e.g. \'4x3\' (default: 1x1)', default=(1,1))
@click.option('--num-keyframes', type=int, help='Number of seeds to interpolate through.  If not specified, determine based on the length of the seeds array given by --seeds.', default=None)
@click.option('--w-frames', type=int, help='Number of frames to interpolate between latents', default=240)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--trunc-cutoff', 'truncation_cutoff', type=int, help='Truncation cutoff', default=14, show_default=True)
@click.option('--reload_modules', help='Overload persistent modules?', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
@click.option('--cfg', help='Config', type=click.Choice(['FFHQ', 'Cats', 'Head']), required=False, metavar='STR', default='FFHQ', show_default=True)
@click.option('--image_mode', help='Image mode', type=click.Choice(['image', 'image_depth', 'image_raw']), required=False, metavar='STR', default='image', show_default=True)
@click.option('--sample_mult', 'sampling_multiplier', type=float, help='Multiplier for depth sampling in volume rendering', default=2, show_default=True)
@click.option('--nrr', type=int, help='Neural rendering resolution override', default=None, show_default=True)
@click.option('--shapes', type=bool, help='Gen shapes for shape interpolation', default=False, show_default=True)
@click.option('--interpolate', type=bool, help='Interpolate between seeds', default=True, show_default=True)

def generate_images(
    network_pkl: str,
    network_pkl_aux: str,
    latent: str,
    latent_aux: str,
    output: str,
    truncation_psi: float,
    truncation_cutoff: int,
    grid: Tuple[int,int],
    num_keyframes: Optional[int],
    w_frames: int,
    reload_modules: bool,
    cfg: str,
    image_mode: str,
    sampling_multiplier: float,
    nrr: Optional[int],
    shapes: bool,
    interpolate: bool,
):
    """Render a latent vector interpolation video.

    """

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda:0')
    with dnnlib.util.open_url(network_pkl) as f:
        G_origin = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore
    with dnnlib.util.open_url(network_pkl_aux) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore


    print("Reloading Modules!")
    G_new = TriPlaneGenerator(*G.init_args, **G.init_kwargs).eval().requires_grad_(False).to(device)
    misc.copy_params_and_buffers(G_origin, G_new, require_all=True)
    G_new.neural_rendering_resolution = G_origin.neural_rendering_resolution
    G_new.rendering_kwargs = G_origin.rendering_kwargs
    G_origin = G_new


    G_new = TriPlaneGenerator(*G.init_args, **G.init_kwargs).eval().requires_grad_(False).to(device)
    misc.copy_params_and_buffers(G, G_new, require_all=True)
    G_new.neural_rendering_resolution = G.neural_rendering_resolution
    G_new.rendering_kwargs = G.rendering_kwargs
    G = G_new


    G_origin.rendering_kwargs['depth_resolution'] = int(G_origin.rendering_kwargs['depth_resolution'] * sampling_multiplier)
    G_origin.rendering_kwargs['depth_resolution_importance'] = int(G_origin.rendering_kwargs['depth_resolution_importance'] * sampling_multiplier)
    G.rendering_kwargs['depth_resolution'] = int(G.rendering_kwargs['depth_resolution'] * sampling_multiplier)
    G.rendering_kwargs['depth_resolution_importance'] = int(G.rendering_kwargs['depth_resolution_importance'] * sampling_multiplier)

    if nrr is not None: G.neural_rendering_resolution = nrr

    if truncation_cutoff == 0:
        truncation_psi = 1.0 # truncation cutoff of 0 means no truncation anyways
    if truncation_psi == 1.0:
        truncation_cutoff = 14 # no truncation so doesn't matter where we cutoff

    ws = torch.tensor(np.load(latent)['w']).to(device)[0]
    if network_pkl_aux == network_pkl:
        ws_aux = torch.tensor(np.load(latent_aux)['w']).to(device)[0]
    else:
        ws_aux = torch.tensor(np.load(latent_aux)['w']).to(device)
    gen_interp_video(G_origin=G_origin, G=G, mp4=output, ws=ws,ws_aux=ws_aux, bitrate='100M', grid_dims=grid, num_keyframes=num_keyframes, w_frames=w_frames, psi=truncation_psi, truncation_cutoff=truncation_cutoff, cfg=cfg, image_mode=image_mode, gen_shapes=shapes, device=device)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------