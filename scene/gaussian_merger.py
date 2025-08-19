import torch
import torch.nn as nn

from simple_knn._C import distCUDA2
from utils.general import inverse_sigmoid
from utils.sh import RGB2SH
from arguments import GSParams
from .gaussian_model import GaussianModel


class GaussianMerger:
    def __init__(self, gaussians: GaussianModel):
        self.gaussians = gaussians

    def init_from_pcd(self, points, colors):
        """
        Params:
            points: [N, 3] Tensor of 3D positions
            colors: [N, 3] Tensor of RGB colors in [0,1] range
        """
        colors = RGB2SH(colors)
        features = (
            torch.zeros((colors.shape[0], 3, (self.gaussians.max_sh_degree + 1) ** 2))
            .float()
            .cuda()
        )
        features[:, :3, 0] = colors
        features[:, 3:, 1:] = 0.0
        features_dc = features[:, :, 0:1].transpose(1, 2).contiguous()
        features_rest = features[:, :, 1:].transpose(1, 2).contiguous()

        print(
            f"Adding {points.shape[0]} points into existing {self.gaussians._xyz.shape[0]} gaussians"
        )

        dist2 = torch.clamp_min(distCUDA2(points), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((points.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(
            0.1 * torch.ones((points.shape[0], 1), dtype=torch.float, device="cuda")
        )
        return {
            "xyz": points,
            "features_dc": features_dc,
            "features_rest": features_rest,
            "scaling": scales,
            "rotation": rots,
            "opacity": opacities,
        }

    def merge_gaussians(self, new_gaussian_attrs: dict, opt: GSParams):
        gs = self.gaussians
        xyz = torch.cat([gs._xyz, new_gaussian_attrs["xyz"]], dim=0)
        features_dc = torch.cat(
            [gs._features_dc, new_gaussian_attrs["features_dc"]], dim=0
        )
        features_rest = torch.cat(
            [gs._features_rest, new_gaussian_attrs["features_rest"]], dim=0
        )
        scaling = torch.cat([gs._scaling, new_gaussian_attrs["scaling"]], dim=0)
        rotation = torch.cat([gs._rotation, new_gaussian_attrs["rotation"]], dim=0)
        opacity = torch.cat([gs._opacity, new_gaussian_attrs["opacity"]], dim=0)

        gs._xyz = nn.Parameter(xyz.requires_grad_(True))
        gs._features_dc = nn.Parameter(features_dc.requires_grad_(True))
        gs._features_rest = nn.Parameter(features_rest.requires_grad_(True))
        gs._scaling = nn.Parameter(scaling.requires_grad_(True))
        gs._rotation = nn.Parameter(rotation.requires_grad_(True))
        gs._opacity = nn.Parameter(opacity.requires_grad_(True))
        gs.max_radii2D = torch.zeros((gs.get_xyz.shape[0]), device="cuda")

        # high-order SH is not really necessary for generative tasks, so simply reset it
        # spatial_lr_scale is a constant and has been set when initializing Scene
        gs.active_sh_degree = 0

        # take care of percent_dense, xyz_gradient_accum, denom, and initialize optimizer
        gs.training_setup(opt)
