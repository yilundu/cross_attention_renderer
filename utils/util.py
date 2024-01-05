import matplotlib.colors as colors
import torch.nn.functional as F
import geometry
import os, struct, math
import numpy as np
import torch
from glob import glob
import collections
import collections.abc
from copy import deepcopy


def parse_comma_separated_integers(string):
    return list(map(int, string.split(',')))

def normalize_for_grid_sample(pixel_coords, H, W):
    pixel_coords[..., 0] = (pixel_coords[..., 0] / (W - 1)) * 2 - 1
    pixel_coords[..., 1] = (pixel_coords[..., 1] / (H - 1)) * 2 - 1
    return pixel_coords

def normalize_imagenet(x):
    ''' Normalize input images according to ImageNet standards.

    Args:
        x (tensor): input images
    '''
    x = x.clone()
    x[:, 0] = (x[:, 0] - 0.485) / 0.229
    x[:, 1] = (x[:, 1] - 0.456) / 0.224
    x[:, 2] = (x[:, 2] - 0.406) / 0.225
    return x


def convert_image(img, type):
    '''Expects single batch dimesion'''
    img = img.squeeze(0)

    if not 'normal' in type:
        img = detach_all(lin2img(img, mode='np'))

    if 'rgb' in type or 'normal' in type:
        img += 1.
        img /= 2.
    elif type == 'depth':
        img = (img - np.amin(img)) / (np.amax(img) - np.amin(img))
    img *= 255.
    img = np.clip(img, 0., 255.).astype(np.uint8)
    return img


def flatten_first_two(tensor):
    b, s, *rest = tensor.shape
    return tensor.view(b * s, *rest)


def parse_intrinsics(filepath, trgt_sidelength=None, invert_y=False):
    # Get camera intrinsics
    with open(filepath, 'r') as file:
        f, cx, cy, _ = map(float, file.readline().split())
        grid_barycenter = torch.Tensor(list(map(float, file.readline().split())))
        scale = float(file.readline())
        height, width = map(float, file.readline().split())

        try:
            world2cam_poses = int(file.readline())
        except ValueError:
            world2cam_poses = None

    if world2cam_poses is None:
        world2cam_poses = False

    world2cam_poses = bool(world2cam_poses)

    if trgt_sidelength is not None:
        cx = cx / width * trgt_sidelength
        cy = cy / height * trgt_sidelength
        f = trgt_sidelength / height * f

    fx = f
    if invert_y:
        fy = -f
    else:
        fy = f

    # Build the intrinsic matrices
    full_intrinsic = np.array([[fx, 0., cx, 0.],
                               [0., fy, cy, 0],
                               [0., 0, 1, 0],
                               [0, 0, 0, 1]])

    return full_intrinsic, grid_barycenter, scale, world2cam_poses


def num_divisible_by_2(number):
    i = 0
    while not number % 2:
        number = number // 2
        i += 1

    return i


def cond_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def normalize(img):
    return (img - img.min()) / (img.max() - img.min())


def print_network(net):
    model_parameters = filter(lambda p: p.requires_grad, net.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("%d" % params)


def add_batch_dim_to_dict(ob):
    if isinstance(ob, collections.abc.Mapping):
        return {k: add_batch_dim_to_dict(v) for k, v in ob.items()}
    elif isinstance(ob, tuple):
        return tuple(add_batch_dim_to_dict(k) for k in ob)
    elif isinstance(ob, list):
        return [add_batch_dim_to_dict(k) for k in ob]
    else:
        try:
            return ob[None, ...]
        except:
            return ob


def detach_all(tensor):
    return tensor.detach().cpu().numpy()


def lin2img(tensor, image_resolution=None, mode='torch'):
    if len(tensor.shape) == 3:
        batch_size, num_samples, channels = tensor.shape
    elif len(tensor.shape) == 2:
        num_samples, channels = tensor.shape

    if image_resolution is None:
        width = np.sqrt(num_samples).astype(int)
        height = width
    else:
        height = image_resolution[0]
        width = image_resolution[1]

    if len(tensor.shape) == 3:
        if mode == 'torch':
            tensor = tensor.permute(0, 2, 1).view(batch_size, channels, height, width)
        elif mode == 'np':
            tensor = tensor.view(batch_size, height, width, channels)
    elif len(tensor.shape) == 2:
        if mode == 'torch':
            tensor = tensor.permute(1, 0).view(channels, height, width)
        elif mode == 'np':
            tensor = tensor.view(height, width, channels)

    return tensor


def pick(list, item_idcs):
    if not list:
        return list
    return [list[i] for i in item_idcs]


def get_mgrid(sidelen, dim=2, flatten=False):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.'''
    if isinstance(sidelen, int):
        sidelen = dim * (sidelen,)

    if dim == 2:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[0, :, :, 0] = pixel_coords[0, :, :, 0] / (sidelen[0] - 1)
        pixel_coords[0, :, :, 1] = pixel_coords[0, :, :, 1] / (sidelen[1] - 1)
    elif dim == 3:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1], :sidelen[2]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[..., 0] = pixel_coords[..., 0] / max(sidelen[0] - 1, 1)
        pixel_coords[..., 1] = pixel_coords[..., 1] / (sidelen[1] - 1)
        pixel_coords[..., 2] = pixel_coords[..., 2] / (sidelen[2] - 1)
    else:
        raise NotImplementedError('Not implemented for dim=%d' % dim)

    pixel_coords -= 0.5
    pixel_coords *= 2.
    pixel_coords = torch.from_numpy(pixel_coords)

    if flatten:
        pixel_coords = pixel_coords.view(-1, dim)
    return pixel_coords


def dict_to_gpu(ob):
    if isinstance(ob, collections.abc.Mapping):
        return {k: dict_to_gpu(v) for k, v in ob.items()}
    elif isinstance(ob, tuple):
        return tuple(dict_to_gpu(k) for k in ob)
    elif isinstance(ob, list):
        return [dict_to_gpu(k) for k in ob]
    else:
        try:
            return ob.cuda()
        except:
            return ob

def dict_to_gpu_expand(ob):
    if isinstance(ob, collections.abc.Mapping):
        return {k: dict_to_gpu_expand(v) for k, v in ob.items()}
    elif isinstance(ob, tuple):
        return tuple(dict_to_gpu_expand(k) for k in ob)
    elif isinstance(ob, list):
        return [dict_to_gpu_expand(k) for k in ob]
    else:
        try:
            return ob[None].cuda()
        except:
            return ob


def assemble_model_input(context, query, gpu=True):
    context = deepcopy(context)
    query = deepcopy(query)
    context['mask'] = torch.Tensor([1.])
    query['mask'] = torch.Tensor([1.])

    context = add_batch_dim_to_dict(context)
    context = add_batch_dim_to_dict(context)

    query = add_batch_dim_to_dict(query)
    query = add_batch_dim_to_dict(query)

    context['rgb'] = context['rgb_other']
    context['cam2world'] = context['cam2world_other']

    model_input = {'context': context, 'query': query, 'post_input': query}

    if gpu:
        model_input = dict_to_gpu(model_input)
    return model_input
