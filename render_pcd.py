import imageio
import torch
import numpy as np
from arguments import GSParams
from scene.gaussian_model import GaussianModel
from utils.trajectory import generate_llff_scaling
from utils.camera import get_render_cam
from gaussian_renderer import render
from utils.depth import colorize


pcd_path = "outputs/hawaii_lookdown_83920174658/v5_no_grad/gsplat.ply"

trajectory = "diving_llff"  # level_wise, diving_llff

save_path_rgb = f"tmp/v5_no_grad_rgb_{trajectory}.mp4"
save_path_depth = f"tmp/v5_no_grad_depth_{trajectory}.mp4"


if __name__ == "__main__":
    opt = GSParams()
    gaussians = GaussianModel(opt.sh_degree)
    gaussians.load_ply(pcd_path)

    # generate zoom-in trajectory with interpolated cameras
    n_levels = 7
    p_min = 2
    p_max = p_min ** (n_levels - 1)
    focal = 5.8269e02
    H, W = 512, 512

    n_views = 601
    if trajectory == "diving_llff":
        focals = np.linspace(focal, focal * p_max, n_views)
        z_scales = [f / focal for f in focals]

        w2c = generate_llff_scaling(1, 0.001 / p_max, n_views, round=4, d=0)
        c2w = [
            np.linalg.inv(np.concatenate((w2c[i], np.array([[0, 0, 0, 1]])), axis=0))
            for i in range(n_views)
        ]
        cams = [
            get_render_cam(focal=focals[i], c2w=c2w[i], H=H, W=W, z_scale=z_scales[i])
            for i in range(n_views)
        ]
    else:
        # TODO this is very coarse, find a better scheme for angle range
        n_steps = 5
        focals = np.linspace(focal, focal * p_max, n_steps)
        z_scales = [f / focal for f in focals]

        w2c = []
        f = []
        z_s = []
        for i in range(n_steps):
            w2c.append(
                generate_llff_scaling(
                    4 / z_scales[i], 4 / z_scales[i], n_views // n_steps, round=4, d=0
                )
            )
            f.extend([focals[i]] * (n_views // n_steps))
            z_s.extend([z_scales[i]] * (n_views // n_steps))
        w2c = np.concatenate(w2c, axis=0)
        c2w = [
            np.linalg.inv(np.concatenate((w2c[i], np.array([[0, 0, 0, 1]])), axis=0))
            for i in range(n_steps * (n_views // n_steps))
        ]
        cams = [
            get_render_cam(focal=f[i], c2w=c2w[i], H=H, W=W, z_scale=z_s[i])
            for i in range(n_steps * (n_views // n_steps))
        ]

    framelist = []
    depthlist = []
    dmin, dmax = 1e8, -1e8

    bg_color = [1, 1, 1] if opt.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    for view in cams:
        results = render(view, gaussians, opt, background)
        frame, depth = results["render"], results["depth"]
        framelist.append(
            np.round(
                frame.permute(1, 2, 0).detach().cpu().numpy().clip(0, 1) * 255.0
            ).astype(np.uint8)
        )
        depth = -(depth * (depth > 0)).detach().cpu().numpy()
        dmin_local = depth.min().item()
        dmax_local = depth.max().item()
        if dmin_local < dmin:
            dmin = dmin_local
        if dmax_local > dmax:
            dmax = dmax_local
        depthlist.append(depth)

    depthlist = [colorize(depth) for depth in depthlist]
    imageio.mimwrite(save_path_rgb, framelist, fps=60, quality=8)
    imageio.mimwrite(save_path_depth, depthlist, fps=60, quality=8)
