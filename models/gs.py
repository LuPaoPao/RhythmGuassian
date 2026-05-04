import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)



class GaussianRenderer:
    """Thin wrapper around the Inria differentiable 3D Gaussian rasterizer.

    Renders a batch of Gaussians into the 4D virtual camera defined by
    ``render_paras`` (built once in ``model.BaseNet.__init__``).
    """

    def __init__(self):
        self.zfar = 100
        self.fovy = 45
        self.bg_color = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
        # Number of Gaussians sampled per render call (sub-sample for memory/speed).
        self.gs_num = 128

    def render(self, gaussians, render_paras, bg_color=None, scale_modifier=1):
        """Rasterize Gaussians into images.

        Args:
            gaussians: ``(B, N, 14)`` packed primitives —
                ``[xyz(3), opacity(1), scale(3), rotation_quat(4), rgb(3)]``.
            render_paras: dict produced by ``BaseNet`` containing the
                world-view / projection matrices and FOVs of the 4D virtual camera.
        Returns:
            dict with ``image`` ``(B', 3, H, W)``, ``indices`` of the sampled
            Gaussians and their ``means3D`` numpy array (for debugging).
        """
        device = gaussians.device
        B = gaussians.shape[0]
        indices = torch.randperm(B)[:self.gs_num]
        gaussians = gaussians[indices]
        B = gaussians.shape[0]
        # loop of loop...
        images = []
        alphas = []
        for b in range(B):

            # pos, opacity, scale, rotation, shs
            means3D = gaussians[b, :, 0:3].contiguous().float()
            opacity = gaussians[b, :, 3:4].contiguous().float()
            scales = gaussians[b, :, 4:7].contiguous().float()
            rotations = gaussians[b, :, 7:11].contiguous().float()
            rgbs = gaussians[b, :, 11:].contiguous().float() # [N, 3]
            self.bg_color = self.bg_color.to(device)
            tanfovx = math.tan(render_paras["FovX"] * 0.5)
            tanfovy = math.tan(render_paras["FovY"] * 0.5)
            raster_settings = GaussianRasterizationSettings(
                image_height=torch.tensor(render_paras["spatial_size"], dtype=torch.float32, device=device),
                image_width=torch.tensor(render_paras["time_length"], dtype=torch.float32, device=device),
                tanfovx=torch.tensor(tanfovx, dtype=torch.float32, device=device),
                tanfovy=torch.tensor(tanfovy, dtype=torch.float32, device=device),
                bg=self.bg_color if bg_color is None else bg_color,
                scale_modifier=scale_modifier,
                viewmatrix=torch.tensor(render_paras["world_view_transform"], dtype=torch.float32, device=device),
                projmatrix=torch.tensor(render_paras["full_proj_transform"], dtype=torch.float32, device=device),
                sh_degree=0,
                campos=torch.tensor(render_paras["camera_center"], dtype=torch.float32, device=device),
                prefiltered=False,
                debug=False,
            )

            rasterizer = GaussianRasterizer(raster_settings=raster_settings)

            # Rasterize visible Gaussians to image, obtain their radii (on screen).
            rendered_image, radii, rendered_depth = rasterizer(
                means3D=means3D, 
                means2D=torch.zeros_like(means3D, dtype=torch.float32, device=device),
                shs=None,
                colors_precomp=rgbs,
                opacities=opacity,
                scales=scales,
                rotations=rotations,
                cov3D_precomp=None,
            )
            # rendered_alpha
            rendered_image = rendered_image.clamp(0, 1)

            images.append(rendered_image)

        images = torch.stack(images, dim=0).view(B, 3, render_paras["spatial_size"], render_paras["time_length"])

        return {
            "image": images,                                       # (B, 3, H, W)
            "indices": indices,                                    # (B,) sampled batch indices
            "means3D": gaussians[:, :, 0:3].detach().cpu().numpy(),
        }

