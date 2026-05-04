# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np
from gs import GaussianRenderer
from graphics_utils import getWorld2View2, getProjectionMatrix, focal2fov

# numpy printing options
np.set_printoptions(threshold=np.inf)


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2, downsample=False, residual=False):
        super().__init__()
        self.residual = residual
        self.downsample = downsample
        self.gs = GaussianRenderer()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        if self.downsample:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)

        if self.residual:
            if self.downsample:
                identity = self.down(x)
            out += identity
        return F.relu(out)


class Decoder(nn.Module):
    """4x transposed-conv stack that lifts ``em`` (B, 512, 4, 16) back to (B, C, 64, 256)."""

    def __init__(self, in_channels, output_channels):
        super().__init__()
        # Decoder upsampling stack
        self.decoder = nn.Sequential(
            # upsample step 1
            nn.ConvTranspose2d(in_channels, 256, kernel_size=2, stride=2),
            BasicBlock(256, 256, stride=1, downsample=False, residual=True),

            # upsample step 2
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            BasicBlock(128, 128, stride=1, downsample=False, residual=True),

            # upsample step 3
            nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2),
            BasicBlock(128, 128, stride=1, downsample=False, residual=True),

            # upsample step 4
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            BasicBlock(64, 64, stride=1, downsample=False, residual=True),

            # final 1x1 to project to ``output_channels``
            nn.Conv2d(64, output_channels, kernel_size=1),
        )

    def forward(self, x):
        return self.decoder(x)


class BaseNet(nn.Module):
    def __init__(self, output_channels=22):
        super().__init__()
        # Load pretrained ResNet18
        resnet = models.resnet18(pretrained=True)
        self.initial_layers = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
        )
        self.encoder_layers = nn.ModuleList([
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        ])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 1)
        self.gs = GaussianRenderer()


        # render parameters
        self.znear = 0.01
        self.zfar = 1000
        self.time_length = 256
        self.spatial_size = 64
        fx, fy = self.time_length/2, self.spatial_size/2
        # get fov
        fovx = focal2fov(fx, self.time_length)
        fovy = focal2fov(fy, self.spatial_size)
        self.intrinsic = torch.tensor([[fx, 0, fx],
                                        [0, fy, fy],
                                        [0, 0, 1]], dtype=torch.float32)
        self.c2ws = torch.eye(4)
        self.FovY = fovy
        self.FovX = fovx
        # identity rotation R (3x3)
        self.R = np.eye(3)
        # zero translation T (3,)
        self.T = np.zeros(3)
        # w2c: world -> opencv camera
        self.trans = np.array([0.0, 0.0, 0.0])  # no translation
        self.world_view_transform = torch.tensor(getWorld2View2(self.R, self.T, self.trans)).transpose(0, 1)
        # proj : opencv_cam to 0-1-NDC
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar,
                                                     fovX=self.FovX, fovY=self.FovY).transpose(0, 1)
        # w2c + c2pixel : X_world * full_proj_transform = pixel
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        # camera centre in world coordinates
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        self.render_paras = {
            "world_view_transform": self.world_view_transform,
            "projection_matrix": self.projection_matrix,
            "full_proj_transform": self.full_proj_transform,
            "camera_center": self.camera_center,
            "FovX": self.FovX,
            "FovY": self.FovY,
            "time_length": self.time_length,
            "spatial_size": self.spatial_size,
        }
        # decoder
        self.decoder = Decoder(in_channels=512, output_channels=output_channels)

        self.bvp_head = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=[1, 2], stride=[1, 2]),
            BasicBlock(512, 256, [2, 1], downsample=1),
            nn.ConvTranspose2d(256, 256, kernel_size=[1, 2], stride=[1, 2]),
            BasicBlock(256, 64, [1, 1], downsample=1),
            nn.ConvTranspose2d(64, 64, kernel_size=[1, 2], stride=[1, 2]),
            BasicBlock(64, 32, [2, 1], downsample=1),
            nn.ConvTranspose2d(32, 32, kernel_size=[1, 2], stride=[1, 2]),
            BasicBlock(32, 1, [1, 1], downsample=1),
        )

        # Gaussian Renderer
        self.gs = GaussianRenderer()

        # activations...
        self.pos_act = lambda x: x.clamp(-1, 1)
        self.scale_act = lambda x: 0.1 * F.softplus(x)
        self.opacity_act = lambda x: torch.sigmoid(x)
        self.rot_act = lambda x: F.normalize(x, dim=-1)
        self.rgb_act = lambda x: 0.5 * torch.tanh(x) + 0.5  # NOTE: may use sigmoid if train again

    def depth2xyz(self, Depth, intrinsics, c2ws, N=1, H=64, W=256, scale=1):
        """Lift a per-pixel depth map to world-space xyz.

        Args:
            Depth: ``(N, H, W)`` per-pixel depth values.
            intrinsics: ``(3, 3)`` camera intrinsic matrix.
            c2ws: ``(4, 4)`` camera-to-world transform.
        Returns:
            ``(N, H, W, 3)`` world-space xyz coordinates.
        """
        channel_1 = torch.arange(W).unsqueeze(0).expand(H, -1).repeat(N, 1, 1).to(Depth.device)
        channel_2 = torch.arange(H).unsqueeze(0).expand(W, -1).permute(1, 0).repeat(N, 1, 1).to(Depth.device)
        # stack pixel grid (u, v, depth) → (N, H, W, 3)
        uv_map = torch.stack((channel_1, channel_2, Depth), dim=-1).to(torch.bfloat16)

        cam_map = torch.zeros_like(uv_map).to(torch.bfloat16).to(Depth.device)
        cam_map[..., 2] += uv_map[..., 2]
        cam_map[..., 0:2] += torch.mul(uv_map[..., 0:2], uv_map[..., 2].unsqueeze(-1))
        cam_map = cam_map.reshape(N, -1, 3)
        xyzs = torch.zeros_like(cam_map).to(torch.bfloat16)
        xyzs_world = torch.zeros_like(cam_map).to(torch.bfloat16)
        for index in range(N):
            intrinsic, c2w = intrinsics, c2ws
            xyzs[index,] += torch.mm(torch.inverse(intrinsic.to(cam_map.device).float()).to(torch.bfloat16), cam_map[index,].T).T
            temp_one = torch.ones_like(xyzs[index, :, 0]).unsqueeze(-1)
            temp = torch.cat((xyzs[index], temp_one), dim=-1).float()
            temp = torch.mm(c2w.to(uv_map.device), temp.T)
            xyzs_world[index] += temp[0:3, :].T
        xyzs = xyzs_world.reshape(N, H, W, 3)

        return xyzs

    def forward_gaussians(self, x):
        """Split decoder output into Gaussian primitives and motion flow.

        Input ``x``: ``(B, 22, H=spatial_size, W=time_length)``.
        22 channels (RhythmGaussian paper):
        ``depth(1) + v_s(3) + v_d(3) + v_n(3) + alpha(1) + scale(3) + rotation(3) + motion_flow(5)``.
        Returns the original Gaussian tensor, the motion-corrected Gaussian tensor
        (after EMM), and per-pixel grids of ``v_d``, ``v_n``, ``motion_flow`` for the
        chroma decomposition losses.
        """
        B, C, H, W = x.shape

        # rearrange to (B, N, 22) where N = H*W
        x = x.permute(0, 2, 3, 1).reshape(B, -1, C)

        depth_raw = x[..., 0]                         # (B, N)
        v_s = self.rgb_act(x[..., 1:4])               # (B, N, 3)
        v_d = self.rgb_act(x[..., 4:7])               # (B, N, 3)
        v_n = self.rgb_act(x[..., 7:10])              # (B, N, 3)
        opacity = self.opacity_act(x[..., 10:11])     # (B, N, 1)
        scale = self.scale_act(x[..., 11:14])         # (B, N, 3)
        rot3 = x[..., 14:17]                          # (B, N, 3) — paper r∈R^3
        dH = torch.tanh(x[..., 17:18])                # (B, N, 1)
        dW = torch.tanh(x[..., 18:19])                # (B, N, 1)
        dS = x[..., 19:22]                            # (B, N, 3)

        # rasterizer needs 4-quat: pad w=1 in front then normalize
        rotation = self.rot_act(F.pad(rot3, (1, 0), value=1.0))  # (B, N, 4)

        # color = v_s + v_d + v_n
        rgbs = (v_s + v_d + v_n).clamp(0.0, 1.0)

        # 1-channel depth → world xyz
        depth = 10.5 * torch.tanh(depth_raw) + 10.5
        depth = depth.reshape(B, self.spatial_size, self.time_length)
        pos = self.depth2xyz(depth, self.intrinsic, self.c2ws,
                             N=B, H=self.spatial_size, W=self.time_length)
        pos = pos.reshape(B, -1, 3).float()           # (B, N, 3)

        # modified position p̂: world-space additive motion flow
        dW_flat = dW.squeeze(-1)
        dH_flat = dH.squeeze(-1)
        pos_mod = torch.stack([
            pos[..., 0] + dW_flat,
            pos[..., 1] + dH_flat,
            pos[..., 2],
        ], dim=-1)
        scale_mod = scale + dS

        gaussians_orig = torch.cat([pos,     opacity, scale,     rotation, rgbs], dim=-1)
        gaussians_mod  = torch.cat([pos_mod, opacity, scale_mod, rotation, rgbs], dim=-1)

        v_d_grid = v_d.reshape(B, H, W, 3)
        v_n_grid = v_n.reshape(B, H, W, 3)
        motion_flow_grid = torch.cat([dH, dW, dS], dim=-1).reshape(B, H, W, 5)

        return gaussians_orig, gaussians_mod, v_d_grid, v_n_grid, motion_flow_grid


    def forward(self, x):
        # ResNet18 encoder
        x = self.initial_layers(x)
        for layer in self.encoder_layers:
            x = layer(x)

        em = x
        bvp = self.bvp_head(em).squeeze(1)
        hr = self.fc(self.avgpool(em).view(em.size(0), -1))

        # decode back to the input spatial size and split into Gaussian channels
        decoded = self.decoder(em)
        g_orig, g_mod, v_d, v_n, motion_flow = self.forward_gaussians(decoded)

        # render twice: original Gaussian points and motion-corrected (EMM) points
        r_orig = self.gs.render(g_orig, render_paras=self.render_paras)
        r_mod = self.gs.render(g_mod, render_paras=self.render_paras)

        rendered = {
            "image_orig":   r_orig["image"],
            "indices_orig": r_orig["indices"],
            "image_mod":    r_mod["image"],
            "indices_mod":  r_mod["indices"],
            "v_d":          v_d,
            "v_n":          v_n,
            "motion_flow":  motion_flow,
        }
        return bvp, hr, rendered
