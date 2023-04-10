import torch.nn.functional as F
import time
import torch
import torch.nn as nn
import numpy as np
from midas import dpt_depth, midas_net, midas_net_custom

from utils import util

import geometry
from epipolar import project_rays
from encoder import SpatialEncoder, ImageEncoder, UNetEncoder
from resnet_block_fc import ResnetFC
import timm

from copy import deepcopy


def encode_relative_ray(ray, transform):
    s = ray.size()
    b, ncontext = transform.size()[:2]

    ray = ray.view(b, ncontext, *s[1:])
    ray = (ray[:, :, :, :, None, :] * transform[:, :, None, None, :3, :3]).sum(dim=-1)

    ray = ray.view(*s)
    return ray


def encode_relative_point(ray, transform):
    s = ray.size()
    b, ncontext = transform.size()[:2]

    ray = ray.view(b, ncontext, *s[1:])
    ray = torch.cat([ray, torch.ones_like(ray[..., :1])], dim=-1)
    ray = (ray[:, :, :, :, None, :] * transform[:, :, None, None, :4, :4]).sum(dim=-1)[..., :3]

    ray = ray.view(*s)
    return ray


class CrossAttentionRenderer(nn.Module):
    def __init__(self, no_sample=False, no_latent_concat=False, no_multiview=False, no_high_freq=False, model="midas_vit", uv=None, repeat_attention=True, n_view=1, npoints=64, num_hidden_units_phi=128):
        super().__init__()

        self.n_view = n_view

        if self.n_view == 2 or self.n_view == 1:
            self.npoints = 64
        else:
            self.npoints = 48

        if npoints:
            self.npoints = npoints

        self.repeat_attention = repeat_attention

        self.no_sample = no_sample
        self.no_latent_concat = no_latent_concat
        self.no_multiview = no_multiview
        self.no_high_freq = no_high_freq

        if model == "resnet":
            self.encoder = SpatialEncoder(use_first_pool=False, num_layers=4)
            self.latent_dim = 512
        elif model == 'midas':
            self.encoder = midas_net_custom.MidasNet_small(
                path=None,
                features=64,
                backbone="efficientnet_lite3",
                exportable=True,
                non_negative=True,
                blocks={'expand': True}
            )
            checkpoint = (
                    "https://github.com/AlexeyAB/MiDaS/releases/download/midas_dpt/midas_v21_small-70d6b9c8.pt"
            )
            state_dict = torch.hub.load_state_dict_from_url(
                checkpoint, map_location=torch.device('cpu'), progress=True, check_hash=True
            )
            self.encoder.load_state_dict(state_dict)
            self.latent_dim = 512
        elif model == 'midas_vit':
            self.encoder = dpt_depth.DPTDepthModel(
                path=None,
                backbone="vitb_rn50_384",
                non_negative=True,
            )
            checkpoint = (
                "https://github.com/intel-isl/DPT/releases/download/1_0/dpt_hybrid-midas-501f0c75.pt"
            )

            self.encoder.pretrained.model.patch_embed.backbone.stem.conv = timm.models.layers.std_conv.StdConv2dSame(3, 64, kernel_size=(7, 7), stride=(2, 2), bias=False)
            self.latent_dim = 512 + 64

            self.conv_map = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3)
        else:
            self.encoder = UNetEncoder()
            self.latent_dim = 32

        if self.n_view > 1 and (not self.no_latent_concat):
            self.query_encode_latent = nn.Conv2d(self.latent_dim + 3, self.latent_dim, 1)
            self.query_encode_latent_2 = nn.Conv2d(self.latent_dim, self.latent_dim // 2 , 1)
            self.latent_dim = self.latent_dim // 2
            self.update_val_merge = nn.Conv2d(self.latent_dim * 2 + 6, self.latent_dim, 1)
        elif self.no_latent_concat:
            self.feature_map = nn.Conv2d(self.latent_dim, self.latent_dim // 2 , 1)
        else:
            self.update_val_merge = nn.Conv2d(self.latent_dim + 6, self.latent_dim, 1)

        self.model = model
        self.num_hidden_units_phi = num_hidden_units_phi

        hidden_dim = 128

        if not self.no_latent_concat:
            self.latent_value = nn.Conv2d(self.latent_dim * self.n_view, self.latent_dim, 1)
            self.key_map = nn.Conv2d(self.latent_dim * self.n_view, hidden_dim, 1)
            self.key_map_2 = nn.Conv2d(hidden_dim, hidden_dim, 1)
        else:
            self.latent_value = nn.Conv2d(self.latent_dim, self.latent_dim, 1)
            self.key_map = nn.Conv2d(self.latent_dim, hidden_dim, 1)
            self.key_map_2 = nn.Conv2d(hidden_dim, hidden_dim, 1)


        self.query_embed = nn.Conv2d(16, hidden_dim, 1)
        self.query_embed_2 = nn.Conv2d(hidden_dim, hidden_dim, 1)
        self.hidden_dim = hidden_dim

        self.latent_avg_query = nn.Conv2d(9+16, hidden_dim, 1)
        self.latent_avg_query_2 = nn.Conv2d(hidden_dim, hidden_dim, 1)

        self.latent_avg_key = nn.Conv2d(self.latent_dim, hidden_dim, 1)
        self.latent_avg_key_2 = nn.Conv2d(hidden_dim, hidden_dim, 1)

        self.query_repeat_embed = nn.Conv2d(16+128, hidden_dim, 1)
        self.query_repeat_embed_2 = nn.Conv2d(hidden_dim, hidden_dim, 1)

        self.latent_avg_repeat_query = nn.Conv2d(9+16+128, hidden_dim, 1)
        self.latent_avg_repeat_query_2 = nn.Conv2d(hidden_dim, hidden_dim, 1)

        self.encode_latent = nn.Conv1d(self.latent_dim, 128, 1)

        self.phi = ResnetFC(self.n_view * 9, n_blocks=3, d_out=3,
                            d_latent=self.latent_dim * self.n_view, d_hidden=self.num_hidden_units_phi)


    def get_z(self, input, val=False):
        # self.normalize_input(input)
        rgb = input['context']['rgb']
        intrinsics = input['context']['intrinsics']
        context = input['context']

        cam2world = context['cam2world']
        rel_cam2world = torch.matmul(torch.inverse(cam2world[:, :1]), cam2world)

        # Flatten first two dims (batch and number of context)
        rgb = torch.flatten(rgb, 0, 1)
        intrinsics = torch.flatten(intrinsics, 0, 1)
        intrinsics = intrinsics[:, None, :, :]
        rgb = rgb.permute(0, -1, 1, 2) # (b*n_ctxt, ch, H, W)
        self.H, self.W = rgb.shape[-2], rgb.shape[-1]

        if self.model == "resnet":
            rgb = (rgb + 1) / 2.
            rgb = util.normalize_imagenet(rgb)
            rgb = torch.cat([rgb], dim=1)
        elif self.model == "midas" or self.model == "midas_vit":
            rgb = (rgb + 1) / 2
            rgb = util.normalize_imagenet(rgb)

        if self.no_multiview:
            cam2world_encode = rel_cam2world.view(-1, 16)
            cam2world_encode = torch.zeros_like(cam2world_encode)
        else:
            cam2world_encode = rel_cam2world.view(-1, 16)

        z = self.encoder.forward(rgb, cam2world_encode, self.n_view) # (b*n_ctxt, self.latent_dim, H, W)

        if self.model == "midas" or self.model == "midas_vit":
            z_conv = self.conv_map(rgb)

            if self.no_high_freq:
                z_conv = torch.zeros_like(z_conv)

            z = z + [z_conv]

        return z

    def forward(self, input, z=None, val=False, debug=False):

        out_dict = {}
        input = deepcopy(input)

        query = input['query']
        context = input['context']
        b, n_context = input['context']["rgb"].shape[:2]
        n_qry, n_qry_rays = query["uv"].shape[1:3]

        # Get img features
        if z is None:
            z = z_orig = self.get_z(input)
        else:
            z_orig = z

        # Get relative coordinates of the query and context ray in each context camera coordinate system
        context_cam2world = torch.matmul(torch.inverse(context['cam2world']), context['cam2world'])
        query_cam2world = torch.matmul(torch.inverse(context['cam2world']), query['cam2world'])

        # Compute each context relative to the first view
        context_rel_cam2world = torch.matmul(torch.inverse(context['cam2world'][:, :1]), context['cam2world'])

        lf_coords = geometry.plucker_embedding(torch.flatten(query_cam2world, 0, 1), torch.flatten(query['uv'].expand(-1, query_cam2world.size(1), -1, -1).contiguous(), 0, 1), torch.flatten(query['intrinsics'].expand(-1, query_cam2world.size(1), -1, -1).contiguous(), 0, 1))
        lf_coords = lf_coords.reshape(b, n_context, n_qry_rays, 6)

        lf_coords.requires_grad_(True)
        out_dict['coords'] = lf_coords.reshape(b*n_context, n_qry_rays, 6)
        out_dict['uv'] = query['uv']

        # Compute epi line
        if self.no_sample:
            start, end, diff, valid_mask, pixel_val = geometry.get_epipolar_lines_volumetric(lf_coords, query_cam2world, context['intrinsics'], self.H, self.W, self.npoints, debug=debug)
        else:

            # Prepare arguments for epipolar line computation
            intrinsics_norm = context['intrinsics'].clone()
            # Normalize intrinsics for a 0-1 image
            intrinsics_norm[:, :, :2, :] = intrinsics_norm[:, :, :2, :] / self.H

            camera_origin = geometry.get_ray_origin(query_cam2world)
            ray_dir = lf_coords[..., :3]
            extrinsics = torch.eye(4).to(ray_dir.device)[None, None, :, :].expand(ray_dir.size(0), ray_dir.size(1), -1, -1)
            camera_origin = camera_origin[:, :, None, :].expand(-1, -1, ray_dir.size(2), -1)

            s = camera_origin.size()

            # Compute 2D epipolar line samples for the image
            output = project_rays(torch.flatten(camera_origin, 0, 1), torch.flatten(ray_dir, 0, 1), torch.flatten(extrinsics, 0, 1), torch.flatten(intrinsics_norm, 0, 1))

            valid_mask = output['overlaps_image']
            start, end = output['xy_min'], output['xy_max']

            start = start.view(*s[:2], *start.size()[1:])
            end = end.view(*s[:2], *end.size()[1:])
            valid_mask = valid_mask.view(*s[:2], valid_mask.size(1))
            start = (start - 0.5) * 2
            end = (end - 0.5) * 2

            start[torch.isnan(start)] = 0
            start[torch.isinf(start)] = 0
            end[torch.isnan(end)] = 0
            end[torch.isinf(end)] = 0

            diff = end - start

            valid_mask = valid_mask.float()
            start = start[..., :2]
            end = end[..., :2]

        diff = end - start
        interval = torch.linspace(0, 1, self.npoints, device=lf_coords.device)

        if (not self.no_sample):
            pixel_val = None
        else:
            pixel_val = torch.flatten(pixel_val, 0, 1)

        latents_out = []
        at_wts = []

        diff = end[:, :, :, None, :] - start[:, :, :, None, :]

        if pixel_val is None and (not self.no_sample):
            pixel_val = start[:, :, :, None, :] + diff * interval[None, None, None, :, None]
            pixel_val = torch.flatten(pixel_val, 0, 1)

        # Gather corresponding features on line
        interp_val_orig = interp_val = torch.cat([F.grid_sample(latent, pixel_val, mode='bilinear', padding_mode='border', align_corners=False) for latent in z], dim=1)

        # Find the 3D point correspondence in every other camera view
        if self.n_view == 2 and (not self.no_latent_concat):
            # Find the nearest neighbor latent in the other frame when given 2 views
            pt, _, _, _ = geometry.get_3d_point_epipolar(lf_coords.flatten(0, 1), pixel_val, context_cam2world.flatten(0, 1), self.H, self.W, context['intrinsics'].flatten(0, 1))

            context_rel_cam2world_view1 = torch.matmul(torch.inverse(context['cam2world'][:, 0:1]), context['cam2world'])
            context_rel_cam2world_view2 = torch.matmul(torch.inverse(context['cam2world'][:, 1:2]), context['cam2world'])

            pt_view1 = encode_relative_point(pt, context_rel_cam2world_view1)
            pt_view2 = encode_relative_point(pt, context_rel_cam2world_view2)

            intrinsics_view1 = context['intrinsics'][:, 0]
            intrinsics_view2 = context['intrinsics'][:, 1]

            s = pt_view1.size()
            pt_view1 = pt_view1.view(b, n_context, *s[1:])
            pt_view2 = pt_view2.view(b, n_context, *s[1:])

            s = interp_val.size()
            interp_val = interp_val.view(b, n_context, *s[1:])

            interp_val_1 = interp_val[:, 0]
            interp_val_2 = interp_val[:, 1]

            pt_view1_context1 = pt_view1[:, 0]
            pt_view1_context2 = pt_view1[:, 1]

            pt_view2_context1 = pt_view2[:, 0]
            pt_view2_context2 = pt_view2[:, 1]

            pixel_val_view2_context1 = geometry.project(pt_view2_context1[..., 0], pt_view2_context1[..., 1], pt_view2_context1[..., 2], intrinsics_view2)
            pixel_val_view2_context1 = util.normalize_for_grid_sample(pixel_val_view2_context1[..., :2], self.H, self.W)

            pixel_val_view1_context2 = geometry.project(pt_view1_context2[..., 0], pt_view1_context2[..., 1], pt_view1_context2[..., 2], intrinsics_view1)
            pixel_val_view1_context2 = util.normalize_for_grid_sample(pixel_val_view1_context2[..., :2], self.H, self.W)

            pixel_val_stack = torch.stack([pixel_val_view1_context2, pixel_val_view2_context1], dim=1).flatten(0, 1)
            interp_val_nearest = torch.cat([F.grid_sample(latent, pixel_val_stack, mode='bilinear', padding_mode='zeros', align_corners=False) for latent in z], dim=1)
            interp_val_nearest = interp_val_nearest.view(b, n_context, *s[1:])
            interp_val_nearest_1 = interp_val_nearest[:, 0]
            interp_val_nearest_2 = interp_val_nearest[:, 1]

            pt_view1_context1 = torch.nan_to_num(pt_view1_context1, 0)
            pt_view2_context2 = torch.nan_to_num(pt_view2_context2, 0)
            pt_view1_context2 = torch.nan_to_num(pt_view1_context2, 0)
            pt_view2_context1 = torch.nan_to_num(pt_view2_context1, 0)

            pt_view1_context1 = pt_view1_context1.detach()
            pt_view2_context2 = pt_view2_context2.detach()

            interp_val_1_view_1 = torch.cat([interp_val_1, torch.tanh(pt_view1_context1 / 5.).permute(0, 3, 1, 2)], dim=1)
            interp_val_1_view_2 = torch.cat([interp_val_nearest_2, torch.tanh(pt_view2_context1 / 5.).permute(0, 3, 1, 2)], dim=1)

            interp_val_1_encode_1 = self.query_encode_latent_2(F.relu(self.query_encode_latent(interp_val_1_view_1)))
            interp_val_1_encode_2 = self.query_encode_latent_2(F.relu(self.query_encode_latent(interp_val_1_view_2)))
            interp_val_1_avg = torch.stack([interp_val_1_encode_1, interp_val_1_encode_2], dim=1).flatten(1, 2)

            interp_val_2_view_2 = torch.cat([interp_val_2, torch.tanh(pt_view2_context2 / 5.).permute(0, 3, 1, 2)], dim=1)
            interp_val_2_view_1 = torch.cat([interp_val_nearest_1, torch.tanh(pt_view1_context2 / 5.).permute(0, 3, 1, 2)], dim=1)

            interp_val_2_encode_1 = self.query_encode_latent_2(F.relu(self.query_encode_latent(interp_val_2_view_1)))
            interp_val_2_encode_2 = self.query_encode_latent_2(F.relu(self.query_encode_latent(interp_val_2_view_2)))
            interp_val_2_avg = torch.stack([interp_val_2_encode_1, interp_val_2_encode_2], dim=1).flatten(1, 2)

            interp_val = torch.stack([interp_val_1_avg, interp_val_2_avg], dim=1).flatten(0, 1)
        elif (self.n_view == 3) and not self.no_latent_concat:
            # Find the nearest neighbor latent in the other 2 frames when given 3 views
            pt, _, _, _ = geometry.get_3d_point_epipolar(lf_coords.flatten(0, 1), pixel_val, context_cam2world.flatten(0, 1), self.H, self.W, context['intrinsics'].flatten(0, 1))

            context_rel_cam2world_view1 = torch.matmul(torch.inverse(context['cam2world'][:, 0:1]), context['cam2world'])
            context_rel_cam2world_view2 = torch.matmul(torch.inverse(context['cam2world'][:, 1:2]), context['cam2world'])
            context_rel_cam2world_view3 = torch.matmul(torch.inverse(context['cam2world'][:, 2:3]), context['cam2world'])

            pt_view1 = encode_relative_point(pt, context_rel_cam2world_view1)
            pt_view2 = encode_relative_point(pt, context_rel_cam2world_view2)
            pt_view3 = encode_relative_point(pt, context_rel_cam2world_view3)

            intrinsics_view1 = context['intrinsics'][:, 0]
            intrinsics_view2 = context['intrinsics'][:, 1]
            intrinsics_view3 = context['intrinsics'][:, 2]

            s = pt_view1.size()
            pt_view1 = pt_view1.view(b, n_context, *s[1:])
            pt_view2 = pt_view2.view(b, n_context, *s[1:])
            pt_view3 = pt_view3.view(b, n_context, *s[1:])

            s = interp_val.size()
            interp_val = interp_val.view(b, n_context, *s[1:])
            interp_val_1 = interp_val[:, 0]
            interp_val_2 = interp_val[:, 1]
            interp_val_3 = interp_val[:, 2]

            pt_view1_context1 = pt_view1[:, 0]
            pt_view1_context2 = pt_view1[:, 1]
            pt_view1_context3 = pt_view1[:, 2]

            pt_view2_context1 = pt_view2[:, 0]
            pt_view2_context2 = pt_view2[:, 1]
            pt_view2_context3 = pt_view2[:, 2]

            pt_view3_context1 = pt_view3[:, 0]
            pt_view3_context2 = pt_view3[:, 1]
            pt_view3_context3 = pt_view3[:, 2]

            # Compute the coordinates to gather for view 2 and 3 on view 1
            pt_view1_context = torch.flatten(torch.stack([pt_view2_context1, pt_view3_context1], dim=1), 1, 2)
            pt_view2_context = torch.flatten(torch.stack([pt_view1_context2, pt_view3_context2], dim=1), 1, 2)
            pt_view3_context = torch.flatten(torch.stack([pt_view1_context3, pt_view2_context3], dim=1), 1, 2)


            pixel_val_view2_context = geometry.project(pt_view2_context[..., 0], pt_view2_context[..., 1], pt_view2_context[..., 2], intrinsics_view2)
            pixel_val_view2_context = util.normalize_for_grid_sample(pixel_val_view2_context[..., :2], self.H, self.W)

            pixel_val_view1_context = geometry.project(pt_view1_context[..., 0], pt_view1_context[..., 1], pt_view1_context[..., 2], intrinsics_view1)
            pixel_val_view1_context = util.normalize_for_grid_sample(pixel_val_view1_context[..., :2], self.H, self.W)

            pixel_val_view3_context = geometry.project(pt_view3_context[..., 0], pt_view3_context[..., 1], pt_view3_context[..., 2], intrinsics_view3)
            pixel_val_view3_context = util.normalize_for_grid_sample(pixel_val_view3_context[..., :2], self.H, self.W)

            pixel_val_stack = torch.stack([pixel_val_view1_context, pixel_val_view2_context, pixel_val_view3_context], dim=1).flatten(0, 1)
            interp_val_nearest = torch.cat([F.grid_sample(latent, pixel_val_stack, mode='bilinear', padding_mode='zeros', align_corners=False) for latent in z], dim=1)

            s = interp_val_nearest.size()
            interp_val_nearest = interp_val_nearest.view(s[0] // 3, 3, *s[1:])

            interp_val_nearest_1 = interp_val_nearest[:, 0]
            interp_val_nearest_2 = interp_val_nearest[:, 1]
            interp_val_nearest_3 = interp_val_nearest[:, 2]

            # Features on each point
            interp_val_view_2_context_1, interp_val_view_3_context_1 = torch.chunk(interp_val_nearest_1, 2, dim=2)
            interp_val_view_1_context_2, interp_val_view_3_context_2 = torch.chunk(interp_val_nearest_2, 2, dim=2)
            interp_val_view_1_context_3, interp_val_view_2_context_3 = torch.chunk(interp_val_nearest_3, 2, dim=2)

            # Gather the right 3D pts along each image
            pt_view1_context = torch.flatten(torch.stack([pt_view1_context2, pt_view1_context3], dim=1), 1, 2)
            pt_view2_context = torch.flatten(torch.stack([pt_view2_context1, pt_view2_context3], dim=1), 1, 2)
            pt_view3_context = torch.flatten(torch.stack([pt_view3_context1, pt_view3_context2], dim=1), 1, 2)

            interp_val_nearest_1 = torch.cat([interp_val_view_1_context_2, interp_val_view_1_context_3], dim=2)
            interp_val_nearest_2 = torch.cat([interp_val_view_2_context_1, interp_val_view_2_context_3], dim=2)
            interp_val_nearest_3 = torch.cat([interp_val_view_3_context_1, interp_val_view_3_context_2], dim=2)

            pt_view1_context1 = torch.nan_to_num(pt_view1_context1, 0)
            pt_view2_context2 = torch.nan_to_num(pt_view2_context2, 0)
            pt_view3_context3 = torch.nan_to_num(pt_view3_context3, 0)

            pt_view1_context = torch.nan_to_num(pt_view1_context, 0)
            pt_view2_context = torch.nan_to_num(pt_view2_context, 0)
            pt_view3_context = torch.nan_to_num(pt_view3_context, 0)

            pt_view1_context = pt_view1_context.detach()
            pt_view2_context = pt_view2_context.detach()
            pt_view3_context = pt_view3_context.detach()

            # Compute average latent for first view
            interp_val_1_view_1 = torch.cat([interp_val_1, torch.tanh(pt_view1_context1 / 5.).permute(0, 3, 1, 2)], dim=1)
            interp_val_1_view_context = torch.cat([interp_val_nearest_1, torch.tanh(pt_view1_context / 5.).permute(0, 3, 1, 2)], dim=1)

            interp_val_1_encode_1 = self.query_encode_latent_2(F.relu(self.query_encode_latent(interp_val_1_view_1)))
            interp_val_1_encode_context = self.query_encode_latent_2(F.relu(self.query_encode_latent(interp_val_1_view_context)))

            interp_val_1_encode_1 = interp_val_1_encode_1[:, :, None, :, :]
            s = interp_val_1_encode_context.size()
            interp_val_1_encode_context = interp_val_1_encode_context.view(s[0], s[1], 2, s[2] // 2, s[3])

            interp_val_1_avg = torch.cat([interp_val_1_encode_1, interp_val_1_encode_context], dim=2).flatten(1, 2)

            # Compute average latent for second view
            interp_val_2_view_2 = torch.cat([interp_val_2, torch.tanh(pt_view2_context2 / 5.).permute(0, 3, 1, 2)], dim=1)
            interp_val_2_view_context = torch.cat([interp_val_nearest_2, torch.tanh(pt_view2_context / 5.).permute(0, 3, 1, 2)], dim=1)

            interp_val_2_encode_2 = self.query_encode_latent_2(F.relu(self.query_encode_latent(interp_val_2_view_2)))
            interp_val_2_encode_context = self.query_encode_latent_2(F.relu(self.query_encode_latent(interp_val_2_view_context)))

            interp_val_2_encode_2 = interp_val_2_encode_2[:, :, None, :, :]
            s = interp_val_2_encode_context.size()
            interp_val_2_encode_context = interp_val_2_encode_context.view(s[0], s[1], 2, s[2] // 2, s[3])

            interp_val_2_avg = torch.cat([interp_val_2_encode_2, interp_val_2_encode_context], dim=2).flatten(1, 2)

            # Compute average latent for third view

            interp_val_3_view_3 = torch.cat([interp_val_3, torch.tanh(pt_view3_context3 / 5.).permute(0, 3, 1, 2)], dim=1)
            interp_val_3_view_context = torch.cat([interp_val_nearest_3, torch.tanh(pt_view3_context / 5.).permute(0, 3, 1, 2)], dim=1)

            interp_val_3_encode_3 = self.query_encode_latent_2(F.relu(self.query_encode_latent(interp_val_3_view_3)))
            interp_val_3_encode_context = self.query_encode_latent_2(F.relu(self.query_encode_latent(interp_val_3_view_context)))

            interp_val_3_encode_3 = interp_val_3_encode_3[:, :, None, :, :]
            s = interp_val_3_encode_context.size()
            interp_val_3_encode_context = interp_val_3_encode_context.view(s[0], s[1], 2, s[2] // 2, s[3])

            interp_val_3_avg = torch.cat([interp_val_3_encode_3, interp_val_3_encode_context], dim=2).flatten(1, 2)

            interp_val = torch.stack([interp_val_1_avg, interp_val_2_avg, interp_val_3_avg], dim=1).flatten(0, 1)
        elif self.no_latent_concat:
            pass
        else:
            # Find the nearest neighbor latent for a single view (null operation)
            pt, _, _, _ = geometry.get_3d_point_epipolar(lf_coords.flatten(0, 1), pixel_val, context_cam2world.flatten(0, 1), self.H, self.W, context['intrinsics'].flatten(0, 1))

            pt[torch.isnan(pt)] = 0
            pt_context = torch.cat([torch.tanh(pt / 5.), torch.tanh(pt / 100.)], dim=-1)
            interp_val = torch.cat([interp_val, pt_context.permute(0, 3, 1, 2)], dim=1)
            interp_val = self.update_val_merge(interp_val)

        joint_latent = self.latent_value(interp_val)
        s = interp_val.size()

        # Compute key value
        key_val = self.key_map_2(F.relu(self.key_map(interp_val))) # (b*n_ctxt, n_pix, interval_steps, latent)

        # Get camera ray direction of each epipolar pixel coordinate 
        cam_rays = geometry.get_ray_directions_cam(pixel_val, context['intrinsics'].flatten(0, 1), self.H, self.W)

        # Ray direction of the query ray to be rendered 
        ray_dir = lf_coords[..., :3].flatten(0, 1)
        ray_dir = ray_dir[:, :, None]
        ray_dir = ray_dir.expand(-1, -1, cam_rays.size(2), -1)

        # 3D coordinate of each epipolar point in 3D
        # depth, _, _ = geometry.get_depth_epipolar(lf_coords.flatten(0, 1), pixel_val, query_cam2world, self.H, self.W, context['intrinsics'].flatten(0, 1))
        pt, dist, parallel, equivalent = geometry.get_3d_point_epipolar(lf_coords.flatten(0, 1), pixel_val, context_cam2world.flatten(0, 1), self.H, self.W, context['intrinsics'].flatten(0, 1))

        # Compute the origin of the query ray
        query_ray_orig = geometry.get_ray_origin(query_cam2world).flatten(0, 1)
        query_ray_orig = query_ray_orig[:, None, None]
        query_ray_orig_ex = torch.broadcast_to(query_ray_orig, cam_rays.size())

        # Compute depth of the computed 3D coordinate (with respect to query camera)
        depth = torch.norm(pt - query_ray_orig, p=2, dim=-1)[..., None]

        # Set NaN and large depth values to a finite value
        depth[torch.isnan(depth)] = 1000000
        depth[torch.isinf(depth)] = 1000000
        depth = depth.detach()

        pixel_dist = pixel_val[:, :, :1, :] - pixel_val[:, :, -1:, :]
        pixel_dist = torch.norm(pixel_dist, p=2, dim=-1)

        # Origin of the context camera ray (always zeros)
        cam_origin = torch.zeros_like(query_ray_orig_ex)

        # Encode depth with tanh to encode different scales of depth values depth values
        depth_encode = torch.cat([torch.tanh(depth), torch.tanh(depth / 10.), torch.tanh(depth / 100.), torch.tanh(depth / 1000.)], dim=-1)

        # Compute query coordinates by combining context ray info, query ray info, and 3D depth of epipolar line
        local_coords = torch.cat([cam_rays, cam_origin, ray_dir, depth_encode, query_ray_orig_ex], dim=-1).permute(0, 3, 1, 2)
        coords_embed = self.query_embed_2(F.relu(self.query_embed(local_coords)))

        # Multiply key and value pairs
        dot_at_joint = torch.einsum('bijk,bijk->bjk', key_val, coords_embed) / 16.
        dot_at_joint = dot_at_joint.view(b, n_context, n_qry_rays, self.npoints).permute(0, 2, 1, 3).reshape(b, n_qry_rays, n_context * (self.npoints))
        at_wt_joint = F.softmax(dot_at_joint, dim=-1)
        at_wt_joint = torch.flatten(at_wt_joint.view(b, n_qry_rays, self.n_view, self.npoints).permute(0, 2, 1, 3), 0, 1)

        z_local = (joint_latent * at_wt_joint[:, None, :, :]).sum(dim=-1)
        s = z_local.size()
        z_local = z_local.view(b, n_context, s[1], n_qry_rays)
        z_sum = z_local.sum(dim=1)
        z_local = torch.cat([z_sum for i in range(n_context)], dim=1).view(*s)

        at_wt = at_wt_joint
        at_wts.append(at_wt)

        # A second round of attention to gather additional information
        if self.repeat_attention:
            z_embed = self.encode_latent(z_local)
            z_embed_local = z_embed[:, :, :, None].expand(-1, -1, -1, local_coords.size(-1))

            # Concatenate the previous cross-attention vector as context for second round of attention
            query_embed_local = torch.cat([z_embed_local, local_coords], dim=1)
            query_embed_local = self.query_repeat_embed_2(F.relu(self.query_repeat_embed(query_embed_local)))

            dot_at = torch.einsum('bijk,bijk->bjk', query_embed_local, coords_embed) / 16
            dot_at = dot_at.view(b, n_context, n_qry_rays, self.npoints).permute(0, 2, 1, 3).reshape(b, n_qry_rays, n_context * (self.npoints))
            at_wt_joint = F.softmax(dot_at, dim=-1)

            # Compute second averaged feature after cross-attention 
            at_wt_joint = torch.flatten(at_wt_joint.view(b, n_qry_rays, self.n_view, self.npoints).permute(0, 2, 1, 3), 0, 1)
            z_local = (joint_latent * at_wt_joint[:, None, :, :]).sum(dim=-1) + z_local
            z_local = z_local.view(b, n_context, s[1], n_qry_rays)

            z_sum = z_local.sum(dim=1)
            z_local = torch.cat([z_sum for i in range(n_context)], dim=1).view(*s)

        latents_out.append(z_local)

        z = torch.cat(latents_out, dim=1).permute(0, 2, 1).contiguous()
        out_dict['pixel_val'] = pixel_val.cpu()
        out_dict['at_wts'] = at_wts

        depth_squeeze = depth[..., 0]
        at_max_idx = at_wt[..., :].argmax(dim=-1)[..., None, None].expand(-1, -1, -1, 3)

        # Ignore points that are super far away
        pt_clamp = torch.clamp(pt, -100, 100)
        # Get the 3D point that is the average (along attention weight) across epipolar points
        world_point_3d_max = (at_wt[..., None] * pt_clamp).sum(dim=-2)

        s = world_point_3d_max.size()
        world_point_3d_max = world_point_3d_max.view(b, n_context, *s[1:]).sum(dim=1)
        world_point_3d_max = world_point_3d_max[:, :, None, :]

        # Compute the depth for epipolar line visualization
        world_point_3d_max = geometry.project_cam2world(world_point_3d_max[:, :, 0, :], query['cam2world'][:, 0])
        depth_ray = world_point_3d_max[:, :, 2]

        # Clamp depth to make sure things don't get too large due to numerical instability
        depth_ray = torch.clamp(depth_ray, 0, 10)

        out_dict['at_wt'] = at_wt
        out_dict['at_wt_max'] = at_max_idx[:, :, :, 0]
        out_dict['depth_ray'] = depth_ray[..., None]

        # Append to the origin of ray into coords that we query the MLP so that it can reason by disocclusion
        out_dict['coords'] = torch.cat([out_dict['coords'], query_ray_orig_ex[:, :, 0, :]], dim=-1)

        # Plucker embedding for query ray 
        coords = out_dict['coords']
        s = coords.size()
        coords = torch.flatten(coords.view(b, n_context, n_qry_rays, s[-1]).permute(0, 2, 1, 3), -2, -1)

        zsize = z.size()
        z_flat = z.view(b, n_context, *zsize[1:]).permute(0, 2, 1, 3)
        z_flat = torch.flatten(z_flat, -2, -1)

        coords = torch.cat((z_flat, coords), dim=-1)

        # Light field decoder using the gather geometric context
        lf_out = self.phi(coords)
        rgb = lf_out[..., :3]

        # Mask invalid regions (no epipolar line correspondence) to be white
        valid_mask = valid_mask.bool().any(dim=1).float()
        rgb = rgb * valid_mask[:, :, None] + 1 * (1 - valid_mask[:, :, None])
        out_dict['valid_mask'] = valid_mask[..., None]

        rgb = rgb.view(b, n_qry, n_qry_rays, 3)

        out_dict['rgb'] = rgb

        # Return the multiview latent for each image (so we can cache computation of multiview encoder)
        out_dict['z'] = z_orig

        return out_dict


