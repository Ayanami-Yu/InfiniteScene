import os
import gc
import datetime
import torch
import numpy as np
import gradio as gr
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from luciddreamer import LucidDreamer
from arguments import CameraParams, GSParams
from scene import Scene
from scene.gaussian_merger import GaussianMerger
from gaussian_renderer import render
from utils.trajectory import (
    get_pcdGenPoses,
    generate_seed_llff,
    generate_lookdown_specified,
)
from utils.camera import get_render_cam, get_train_cam, prepare_cameras_zoom_in
from gen_powers_10.model import GenPowers10Pipeline
from gen_powers_10.utils import save_images
from murre.pipeline import MurrePipeline
from utils.murre import get_sparse_depth
from utils.scipy import interp_grid, maximum_filter, minimum_filter
from utils.image import crop_images
from crafter import Crafter


class Dreamer(LucidDreamer):
    def __init__(
        self,
        for_gradio=True,
        save_dir=None,
        torch_hub_local=True,
        version="DFIF_XL_L_X4",
        murre_ckpt_path=None,
        dtype=torch.float32,
        device="cuda",
    ):
        super().__init__(
            for_gradio=for_gradio,
            save_dir=save_dir,
            torch_hub_local=torch_hub_local,
            dtype=dtype,
            device=device,
        )
        self.version = version
        self.zoom_model = GenPowers10Pipeline(version)
        self.murre_model = MurrePipeline.from_pretrained(
            murre_ckpt_path, variant=None, torch_dtype=dtype
        ).to(device)

    def generate_pcd_torch(
        self,
        rgb_cond,
        prompt,
        negative_prompt,
        pcdgenpath,
        seed,
        diff_steps,
        progress=gr.Progress(),
        deg_scale=1,
        fovx=None,
        z_scale=1.0,
        depth_model="zoedepth",
    ):
        """
        Params:
            pcdgenpath: Union[str, np.ndarray]
                Either the name of the camera trajectory or a numpy array of shape [N, 3, 4] that corresponds to a series of camera poses.
        """
        # processing inputs
        dtype, device = self.dtype, self.device
        generator = torch.Generator(device=device).manual_seed(seed)
        fovx = self.cam.fov[0] if not fovx else fovx

        image_curr_pil = self.resize_image(rgb_cond, prompt, negative_prompt, generator)
        render_poses = torch.tensor(
            (
                get_pcdGenPoses(pcdgenpath, deg_denom=deg_scale)
                if type(pcdgenpath) is str
                else pcdgenpath
            ),
            dtype=dtype,
            device=device,
        )
        # the initial depth can't be estimated by Murre since there is no existing point cloud
        depth_curr = self.d(image_curr_pil)
        center_depth_np = np.mean(
            depth_curr[
                self.cam.H // 2 - 10 : self.cam.H // 2 + 10,
                self.cam.W // 2 - 10 : self.cam.W // 2 + 10,
            ]
        )
        depth_curr = torch.tensor(depth_curr, device=device)
        image_curr = torch.tensor(np.array(image_curr_pil), dtype=dtype, device=device)

        H, W, K = self.cam.H, self.cam.W, torch.tensor(self.cam.K, device=device)
        K_inv = torch.linalg.inv(K)
        x, y = torch.meshgrid(
            torch.arange(W, dtype=dtype, device=device),
            torch.arange(H, dtype=dtype, device=device),
            indexing="xy",
        )  # pixels
        edgeN = 2
        edgemask = torch.ones((H - 2 * edgeN, W - 2 * edgeN), device=device)
        edgemask = F.pad(
            edgemask, (edgeN, edgeN, edgeN, edgeN), mode="constant", value=0
        )

        # initialize
        R0, T0 = render_poses[0, :3, :3], render_poses[0, :3, 3:4]
        R0_inv = torch.linalg.inv(R0)
        pts_coord_cam = K_inv @ torch.stack(
            (x * depth_curr, y * depth_curr, 1 * depth_curr), dim=0
        ).reshape(3, -1)

        new_pts_coord_world2 = R0_inv @ pts_coord_cam - R0_inv @ T0
        new_pts_colors2 = image_curr.reshape(-1, 3) / 255.0

        # pts_colors will be used to initialize the colors of corresponding 3D points. It remains the same during projecting points to/from 3D, because the correspondence between a color and a point remains binded.
        pts_coord_world, pts_colors = (
            new_pts_coord_world2.clone(),
            new_pts_colors2.clone(),
        )  # initial point cloud P_0

        if self.for_gradio:
            progress(0, desc="[1/4] Dreaming...")
            iterable_dream = progress.tqdm(
                range(1, len(render_poses)), desc="[1/4] Dreaming"
            )
        else:
            iterable_dream = range(1, len(render_poses))

        for i in iterable_dream:
            R, T = render_poses[i, :3, :3], render_poses[i, :3, 3:4]  # tensors
            R_inv = torch.linalg.inv(R)

            # transform world to pixel
            # same as c2w x world_coord (in homogeneous space)
            pts_coord_cam2 = R @ pts_coord_world + T
            pixel_coord_cam2 = (
                K @ pts_coord_cam2
            )  # [3, N] the previous 3D points in image coord

            z = pixel_coord_cam2[2]
            x_homo = pixel_coord_cam2[0] / z
            y_homo = pixel_coord_cam2[1] / z
            valid_idx = torch.nonzero(
                (z > 0)
                & (x_homo >= 0)
                & (x_homo <= W - 1)
                & (y_homo >= 0)
                & (y_homo <= H - 1),
                as_tuple=False,
            ).squeeze(1)

            # divide by z coord to get homogeneous coord
            pixel_coord_cam2 = (
                pixel_coord_cam2[:2, valid_idx] / pixel_coord_cam2[-1:, valid_idx]
            )
            round_coord_cam2 = torch.round(pixel_coord_cam2).to(torch.int32)

            x, y = torch.meshgrid(
                torch.arange(W, dtype=torch.float32, device=device),
                torch.arange(H, dtype=torch.float32, device=device),
                indexing="xy",
            )
            grid = torch.stack((x, y), dim=-1).reshape(-1, 2)  # [N, 2] row-major
            image2 = interp_grid(
                pixel_coord_cam2.transpose(1, 0),
                pts_colors[valid_idx],
                grid,
                method="linear",
                fill_value=0,
            ).reshape(H, W, 3)
            # note that F.pad starts from the last dimension whereas np.pad starts from the first
            image2_CHW = image2.permute(2, 0, 1)
            image2 = edgemask[..., None] * image2 + (1 - edgemask[..., None]) * F.pad(
                image2_CHW[:, 1:-1, 1:-1], (1, 1, 1, 1), mode="replicate"
            ).permute(1, 2, 0)

            round_mask2 = torch.zeros((H, W), dtype=torch.float32, device=device)
            round_mask2[round_coord_cam2[1], round_coord_cam2[0]] = 1

            round_mask2 = maximum_filter(round_mask2, size=(9, 9), axes=(0, 1))
            image2 = round_mask2[..., None] * image2 + (1 - round_mask2[..., None]) * (
                -1
            )

            mask2 = minimum_filter(  # M_i
                (image2.sum(-1) != -3).to(dtype), size=(11, 11), axes=(0, 1)
            )
            image2 = mask2[..., None] * image2 + (1 - mask2[..., None]) * 0  # \hat{I}_i

            mask_hf = torch.abs(
                mask2[: H - 1, : W - 1] - mask2[1:, : W - 1]
            ) + torch.abs(mask2[: H - 1, : W - 1] - mask2[: H - 1, 1:])
            mask_hf = F.pad(
                mask_hf.unsqueeze(0), (0, 1, 0, 1), mode="replicate"
            ).squeeze(0)
            mask_hf = torch.where(mask_hf < 0.3, 0, 1)
            # use valid_idx[border_valid_idx] for world1
            border_valid_idx = torch.where(
                mask_hf[round_coord_cam2[1], round_coord_cam2[0]] == 1
            )[0]

            image_curr_pil = self.rgb(  # inpainting  # I_i
                prompt=prompt,
                image=image2.detach().cpu().numpy(),
                negative_prompt=negative_prompt,
                generator=generator,
                num_inference_steps=diff_steps,
                mask_image=mask2.detach().cpu().numpy(),
            )
            # TODO test Murre for generating pcd
            depth_curr = torch.tensor(
                (
                    self.pred_depth_with_pcd(
                        image_curr_pil,
                        points=pts_coord_world.T,
                        ixt=K,
                        ext=render_poses[i],
                    )
                    if depth_model == "murre"
                    else self.d(image_curr_pil)
                ),
                device=device,
            )  # \hat{D}_i
            image_curr = torch.tensor(
                np.array(image_curr_pil), dtype=dtype, device=device
            )

            # depth optimize  # TODO remove it if Murre works
            t_z2 = torch.tensor(depth_curr, device=device)
            sc = torch.ones(
                1, dtype=torch.float32, device=device, requires_grad=True
            )  # d_i
            optimizer = torch.optim.Adam(params=[sc], lr=0.001)

            for idx in range(100):  # d_i optimization loop
                trans3d = torch.tensor(
                    [[sc, 0, 0, 0], [0, sc, 0, 0], [0, 0, sc, 0], [0, 0, 0, 1]],
                    device=device,
                ).requires_grad_(True)
                coord_cam2 = K_inv @ torch.stack(
                    (x * t_z2, y * t_z2, 1 * t_z2),
                    dim=0,
                )[:, round_coord_cam2[1], round_coord_cam2[0]].reshape(3, -1)

                coord_world2 = R_inv @ coord_cam2 - R_inv @ T
                coord_world2_warp = torch.cat(
                    (coord_world2, torch.ones((1, valid_idx.shape[0]), device=device)),
                    dim=0,
                )
                coord_world2_trans = trans3d @ coord_world2_warp
                coord_world2_trans = (
                    coord_world2_trans[:3] / coord_world2_trans[-1]
                )  # \tilde{P}_i
                loss = torch.mean(
                    (
                        torch.tensor(pts_coord_world[:, valid_idx]).to(dtype)
                        - coord_world2_trans  # P_{i - 1} - \tilde{P}_i
                    )
                    ** 2
                )  # align with the old 3d points backprojected from the previous camera

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                coord_cam2 = K_inv @ torch.stack(
                    (x * t_z2, y * t_z2, 1 * t_z2),
                    dim=0,
                )[
                    :,
                    round_coord_cam2[1, border_valid_idx],
                    round_coord_cam2[0, border_valid_idx],
                ].reshape(
                    3, -1
                )
                coord_world2 = R_inv @ coord_cam2 - R_inv @ T
                coord_world2_warp = torch.cat(
                    (
                        coord_world2,
                        torch.ones((1, border_valid_idx.shape[0]), device=device),
                    ),
                    dim=0,
                )
                coord_world2_trans = trans3d @ coord_world2_warp
                coord_world2_trans = (
                    coord_world2_trans[:3] / coord_world2_trans[-1]
                )  # rectified \tilde{P}_i (no new points added)

            # trans3d = trans3d.detach().numpy()

            # backproject new 3D points from inpainted region
            pts_coord_cam2 = K_inv @ torch.stack(
                (x * depth_curr, y * depth_curr, 1 * depth_curr), dim=0
            ).reshape(3, -1)
            pts_coord_cam2 = pts_coord_cam2[
                :, torch.where(1 - mask2.reshape(-1))[0]
            ]  # select M_i == 0

            camera_origin_coord_world2 = (
                -R_inv @ T
            )  # [3, 1] camera origin in world coord
            new_pts_coord_world2 = R_inv @ pts_coord_cam2 - R_inv @ T
            new_pts_coord_world2_warp = torch.cat(
                (
                    new_pts_coord_world2,
                    torch.ones((1, new_pts_coord_world2.shape[1]), device=device),
                ),
                dim=0,
            )
            new_pts_coord_world2 = trans3d @ new_pts_coord_world2_warp
            new_pts_coord_world2 = (
                new_pts_coord_world2[:3] / new_pts_coord_world2[-1]
            )  # \hat{P}_i
            new_pts_colors2 = (image_curr.reshape(-1, 3) / 255.0)[
                torch.where(1 - mask2.reshape(-1))[0]
            ]

            vector_camorigin_to_campixels = (
                coord_world2_trans - camera_origin_coord_world2
            )  # the ray lines from cam center to corresponding points
            vector_camorigin_to_pcdpixels = (
                pts_coord_world[:, valid_idx[border_valid_idx]]  # P_{i - 1}
                - camera_origin_coord_world2
            )

            compensate_depth_coeff = torch.sum(
                vector_camorigin_to_pcdpixels * vector_camorigin_to_campixels, dim=0
            ) / torch.sum(
                vector_camorigin_to_campixels * vector_camorigin_to_campixels, dim=0
            )  # N_correspond
            compensate_pts_coord_world2_correspond = (
                camera_origin_coord_world2
                + vector_camorigin_to_campixels * compensate_depth_coeff.reshape(1, -1)
            )

            compensate_coord_cam2_correspond = (
                R @ compensate_pts_coord_world2_correspond + T
            )
            homography_coord_cam2_correspond = R @ coord_world2_trans + T

            compensate_depth_correspond = (
                compensate_coord_cam2_correspond[-1]
                - homography_coord_cam2_correspond[-1]
            )  # N_correspond
            compensate_depth_zero = torch.zeros(4, device=device)
            compensate_depth = torch.cat(
                (compensate_depth_correspond, compensate_depth_zero), dim=0
            )  # N_correspond + 4

            pixel_cam2_correspond = pixel_coord_cam2[
                :, border_valid_idx
            ]  # [2, N_correspond] (xy)  # points corresponding to the previous image (in cam coord)
            pixel_cam2_zero = torch.tensor(
                [[0, 0, W - 1, W - 1], [0, H - 1, 0, H - 1]], device=device
            )
            pixel_cam2 = torch.cat(
                (pixel_cam2_correspond, pixel_cam2_zero), dim=1
            ).transpose(
                1, 0
            )  # N + H, 2

            # calculate for each pixel how much the depth value should change using linear interpolation
            masked_pixels_xy = torch.stack(torch.where(1 - mask2), dim=1)[:, [1, 0]]
            new_depth_linear, new_depth_nearest = interp_grid(
                pixel_cam2, compensate_depth, masked_pixels_xy, method="linear"
            ), interp_grid(
                pixel_cam2, compensate_depth, masked_pixels_xy, method="nearest"
            )
            new_depth = torch.where(
                torch.isnan(new_depth_linear), new_depth_nearest, new_depth_linear
            )

            pts_coord_cam2 = K_inv @ torch.stack(
                (x * depth_curr, y * depth_curr, 1 * depth_curr), dim=0
            ).reshape(3, -1)
            pts_coord_cam2 = pts_coord_cam2[
                :, torch.where(1 - mask2.reshape(-1))[0]
            ]  # corresponds to the newly added points
            x_nonmask, y_nonmask = (
                x.reshape(-1)[torch.where(1 - mask2.reshape(-1))[0]],
                y.reshape(-1)[torch.where(1 - mask2.reshape(-1))[0]],
            )  # the points without GT counterparts (M_i == 0)
            compensate_pts_coord_cam2 = K_inv @ torch.stack(
                (x_nonmask * new_depth, y_nonmask * new_depth, 1 * new_depth),
                dim=0,
            )
            new_warp_pts_coord_cam2 = pts_coord_cam2 + compensate_pts_coord_cam2

            new_pts_coord_world2 = R_inv @ new_warp_pts_coord_cam2 - R_inv @ T
            new_pts_coord_world2_warp = torch.cat(
                (
                    new_pts_coord_world2,
                    torch.ones((1, new_pts_coord_world2.shape[1]), device=device),
                ),
                dim=0,
            )
            new_pts_coord_world2 = trans3d @ new_pts_coord_world2_warp
            new_pts_coord_world2 = (
                new_pts_coord_world2[:3] / new_pts_coord_world2[-1]
            )  # W(\hat{P}_i)
            new_pts_colors2 = (image_curr.reshape(-1, 3) / 255.0)[
                torch.where(1 - mask2.reshape(-1))[0]
            ]

            pts_coord_world = torch.cat(
                (pts_coord_world, new_pts_coord_world2), dim=-1
            )  # P_i = P_{i - 1} \cup W(\hat{P}_i)
            pts_colors = torch.cat((pts_colors, new_pts_colors2), dim=0)

        yz_reverse = torch.tensor(
            [[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=torch.float32, device=device
        )
        traindata = {
            "camera_angle_x": [],  # for intrinsics
            "W": W,
            "H": H,
            "pcd_points": pts_coord_world.detach().cpu().numpy(),  # for 3DGS init
            "pcd_colors": pts_colors.detach().cpu().numpy(),
            "frames": [],  # contains extrinsics
            "z_scale": [],  # for scaling znear and zfar
        }

        internal_render_poses = torch.tensor(
            get_pcdGenPoses(
                "hemisphere", {"center_depth": center_depth_np}, deg_denom=deg_scale
            ),
            dtype=dtype,
            device=device,
        )

        if self.for_gradio:
            progress(0, desc="[2/4] Aligning...")
            iterable_align = progress.tqdm(
                range(len(render_poses)), desc="[2/4] Aligning"
            )
        else:
            iterable_align = range(len(render_poses))

        # generate additional image-mask pairs by reprojecting from P_N by a new camera sequence
        for i in iterable_align:
            for j in range(len(internal_render_poses)):
                idx = i * len(internal_render_poses) + j
                print(f"{idx + 1} / {len(render_poses) * len(internal_render_poses)}")

                # transform world to pixel
                Rw2i = render_poses[i, :3, :3]
                Tw2i = render_poses[i, :3, 3:4]
                Ri2j = internal_render_poses[j, :3, :3]
                Ti2j = internal_render_poses[j, :3, 3:4]

                Rw2j = Ri2j @ Rw2i
                Tw2j = Ri2j @ Tw2i + Ti2j

                # Transfrom cam2 to world + change sign of yz axis
                Rj2w = (yz_reverse @ Rw2j).T
                Tj2w = -Rj2w @ (yz_reverse @ Tw2j)
                Pc2w = torch.cat((Rj2w, Tj2w), dim=1)
                Pc2w = torch.cat(
                    (
                        Pc2w,
                        torch.tensor([[0, 0, 0, 1]], dtype=Pc2w.dtype, device=device),
                    ),
                    dim=0,
                )

                # project P_N onto camj
                pts_coord_camj = Rw2j @ pts_coord_world + Tw2j
                pixel_coord_camj = K @ pts_coord_camj

                z_j = pixel_coord_camj[2]
                x_homo_j = pixel_coord_camj[0] / z_j
                y_homo_j = pixel_coord_camj[1] / z_j
                valid_idxj = torch.where(
                    (z_j > 0)
                    & (x_homo_j >= 0)
                    & (x_homo_j <= W - 1)
                    & (y_homo_j >= 0)
                    & (y_homo_j <= H - 1)
                )[0]
                if len(valid_idxj) == 0:
                    continue
                pts_depthsj = pixel_coord_camj[-1:, valid_idxj]
                pixel_coord_camj = (
                    pixel_coord_camj[:2, valid_idxj] / pixel_coord_camj[-1:, valid_idxj]
                )
                round_coord_camj = torch.round(pixel_coord_camj).to(torch.int32)

                x, y = torch.meshgrid(
                    torch.arange(W, dtype=torch.float32, device=device),
                    torch.arange(H, dtype=torch.float32, device=device),
                    indexing="xy",
                )  # pixels
                grid = torch.stack((x, y), dim=-1).reshape(-1, 2)
                imagej = interp_grid(
                    pixel_coord_camj.transpose(1, 0),
                    pts_colors[valid_idxj],
                    grid,
                    method="linear",
                    fill_value=0,
                ).reshape(H, W, 3)
                imagej_CHW = imagej.permute(2, 0, 1)
                imagej = edgemask[..., None] * imagej + (
                    1 - edgemask[..., None]
                ) * F.pad(
                    imagej_CHW[:, 1:-1, 1:-1], (1, 1, 1, 1), mode="replicate"
                ).permute(
                    1, 2, 0
                )

                depthj = interp_grid(
                    pixel_coord_camj.transpose(1, 0),
                    pts_depthsj.T,
                    grid,
                    method="linear",
                    fill_value=0,
                ).reshape(H, W)
                depthj = edgemask * depthj + (1 - edgemask) * F.pad(
                    depthj[None, 1:-1, 1:-1], (1, 1, 1, 1), mode="replicate"
                ).squeeze(0)

                maskj = torch.zeros((H, W), dtype=torch.float32, device=device)
                maskj[round_coord_camj[1], round_coord_camj[0]] = 1
                maskj = maximum_filter(maskj, size=(9, 9), axes=(0, 1))
                imagej = maskj[..., None] * imagej + (1 - maskj[..., None]) * (-1)

                maskj = minimum_filter(
                    (imagej.sum(-1) != -3).to(dtype), size=(11, 11), axes=(0, 1)
                )
                imagej = maskj[..., None] * imagej + (1 - maskj[..., None]) * 0

                traindata["camera_angle_x"].append(fovx)
                traindata["z_scale"].append(z_scale)
                traindata["frames"].append(
                    {
                        "image": Image.fromarray(
                            np.round(imagej.detach().cpu().numpy() * 255.0).astype(
                                np.uint8
                            )
                        ),
                        "transform_matrix": Pc2w.tolist(),
                    }
                )

        progress(1, desc="[3/4] Baking Gaussians...")
        return traindata

    def create(
        self,
        rgb_cond,
        txt_cond,
        neg_txt_cond,
        pcdgenpath,
        seed,
        diff_steps,
        p=None,
    ):
        if rgb_cond and type(txt_cond) is str:
            self.traindata = self.generate_pcd_torch(
                rgb_cond, txt_cond, neg_txt_cond, pcdgenpath, seed, diff_steps
            )
            self.scene = Scene(self.traindata, self.gaussians, self.opt)
            self.training()
            self.timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
            outfile = self.save_ply(os.path.join(self.save_dir, "gsplat.ply"))
        else:  # use Powers of Ten
            self.create_scene_v6(  # TODO
                rgb_cond, txt_cond, neg_txt_cond, pcdgenpath, seed, diff_steps, p
            )
            os.makedirs(self.save_dir, exist_ok=True)
            outfile = self.save_ply(os.path.join(self.save_dir, "gsplat.ply"))
        return outfile

    # TODO add arguments to argparse
    def create_scene_v6(
        self,
        rgb_cond,
        txt_cond,
        neg_txt_cond,
        pcdgenpath,
        seed,
        diff_steps,
        p,
        nvs_cfg="configs/inference_pvd_512.yaml",
    ):
        """
        Process:
            1) Incorporate ViewCrafter to expand training dataset at each zoom-in level.
            2) Train Gaussians jointly at all zoom-in levels with expanded dataset.
        """
        # generate camera trajectory
        n_levels = len(txt_cond)
        H, W, focal = self.cam.H, self.cam.W, self.cam.focal[0]
        focals, c2w_init, cams_ext_init, cams_ixt, cams_diving, cams_diving_llff = (
            prepare_cameras_zoom_in(
                pcdgenpath,
                n_levels,
                n_views=301,
                focal=focal,
                H=self.cam.H,
                W=self.cam.W,
                p=p,
            )
        )

        # generate dataset for training
        imgs_zoomed = self.generate_zoom_stack(
            txt_cond, neg_txt_cond, diff_steps, p, rgb_cond=rgb_cond
        )
        imgs_zoomed = [self.resize_image(img) for img in imgs_zoomed]
        cams_train = [
            get_train_cam(
                img=imgs_zoomed[i],
                focal=focals[i],
                c2w=c2w_init,
                H=H,
                W=W,
                white_background=self.opt.white_background,
                z_scale=p**i,
            )
            for i in range(n_levels)
        ]

        # generate the initial scene
        traindata = self.generate_pcd_torch(
            imgs_zoomed[0],
            txt_cond[0],
            neg_txt_cond,
            cams_ext_init,
            seed,
            diff_steps,
            depth_model="murre",
        )
        self.scene = Scene(traindata, self.gaussians, self.opt)
        self.training()

        self.render_video(cams_diving_llff, "llff_zoom_before")
        self.render_video(cams_diving, "diving_zoom_before")

        # zoom in (forward in a straight line)
        merger = GaussianMerger(self.scene.gaussians)
        for i in range(1, n_levels):
            with torch.no_grad():  # TODO test again
                points_new, colors_new = self.backproject(
                    image=imgs_zoomed[i],
                    ixt=cams_ixt[i],
                    ext=torch.tensor(
                        cams_ext_init[0], dtype=self.dtype, device=self.device
                    ),
                    points=self.gaussians.get_xyz,
                )
            merger.merge_gaussians(
                new_gaussian_attrs=merger.init_from_pcd(
                    points_new.reshape(-1, 3), colors_new
                ),
                opt=self.opt,
            )

        # delete models that will no longer be used
        del self.d_model
        del self.zoom_model
        del self.rgb_model
        gc.collect()
        torch.cuda.empty_cache()

        # TODO check disabling densification
        self.opt = GSParams(iterations=2990, densify_until_iter=0)
        # self.opt = GSParams(iterations=2990)
        self.training(cameras=cams_train)

        # generate all camera poses and render Gaussians
        # 2D lists: outer dimensions corresponds to each zoom-in level
        n_frames = 25
        imgs_render = []
        c2ws_zoom = []
        with torch.no_grad():
            for i in range(n_levels):
                ext = generate_lookdown_specified(deg_denom=p**i)
                arr_ones = np.repeat(np.array([[[0, 0, 0, 1]]]), ext.shape[0], axis=0)
                c2ws = np.linalg.inv(np.concatenate((ext, arr_ones), axis=-2))
                c2ws_zoom.append(c2ws)
                cams_render = [
                    get_render_cam(
                        focal=focals[i],
                        c2w=c2ws[j],
                        H=H,
                        W=W,
                        z_scale=focals[i] / focal,
                    )
                    for j in range(n_frames)
                ]
                imgs_render.append([
                    render(cams_render[j], self.gaussians, self.opt, self.background)[
                        "render"
                    ]
                    for j in range(n_frames)
                ])

        # load VDM after rendering Gaussians since it will occupy large memory
        self.nvs_model = Crafter(cfg_path=nvs_cfg)  # TODO test
        to_pil = transforms.ToPILImage()
        for i in range(n_levels):
            # TODO resize
            imgs_cond = torch.stack(imgs_render[i], dim=0)
            imgs_cond = crop_images(imgs_cond).permute(0, 2, 3, 1)
            imgs_nvs = (self.nvs_model(imgs_cond) + 1.0) / 2.0
            imgs_nvs = [to_pil(img.permute(2, 0, 1)) for img in imgs_nvs]
            cams_train.extend(
                [
                    get_train_cam(
                        img=imgs_nvs[j],
                        focal=focals[i],
                        c2w=c2ws_zoom[i][j],
                        H=H,
                        W=W,
                        white_background=self.opt.white_background,
                        z_scale=focals[i] / focal,
                    )
                    for j in range(n_frames)
                ]
            )
        del self.nvs_model
        gc.collect()
        torch.cuda.empty_cache()

        self.opt = GSParams(iterations=8990, densify_until_iter=0)
        # self.opt = GSParams(iterations=8990)
        self.training(cameras=cams_train)

        # save zoom-in videos
        self.render_video(cams_diving, "diving_zoom_after")
        self.render_video(cams_diving_llff, "llff_zoom_after")

    def create_scene_v5(
        self,
        rgb_cond,
        txt_cond,
        neg_txt_cond,
        pcdgenpath,
        seed,
        diff_steps,
        p,
    ):
        """
        Process:
            Train Gaussians jointly at all zoom-in levels.
        """
        # generate camera trajectory
        num_levels = len(txt_cond)
        H, W = self.cam.H, self.cam.W
        cam_ext = get_pcdGenPoses(pcdgenpath=pcdgenpath)
        c2w_init = np.linalg.inv(
            np.concatenate((cam_ext[0], np.array([[0, 0, 0, 1]])), axis=0)
        )
        focalx = [self.cam.focal[0] * (p**i) for i in range(num_levels)]
        cam_ixt = [
            torch.tensor(
                [
                    [focalx[i], 0.0, self.cam.W / 2],
                    [0.0, focalx[i], self.cam.H / 2],
                    [0.0, 0.0, 1.0],
                ],
                dtype=self.dtype,
                device=self.device,
            )
            for i in range(num_levels)
        ]
        # generate zoom-in trajectory with interpolated cameras
        n_views = 201
        focals = np.linspace(focalx[0], focalx[num_levels - 1], n_views)
        z_scales = [f / focalx[0] for f in focals]
        cams_diving = [
            get_render_cam(focal=f, c2w=c2w_init, H=H, W=W, z_scale=z_s)
            for f, z_s in zip(focals, z_scales)
        ]
        w2c_llff = generate_seed_llff(5, n_views, round=4, d=0)
        c2w_llff = [
            np.linalg.inv(
                np.concatenate((w2c_llff[i], np.array([[0, 0, 0, 1]])), axis=0)
            )
            for i in range(n_views)
        ]
        cams_diving_llff = [
            get_render_cam(
                focal=focals[i], c2w=c2w_llff[i], H=H, W=W, z_scale=z_scales[i]
            )
            for i in range(n_views)
        ]

        # generate dataset for training
        imgs_zoomed = self.generate_zoom_stack(
            txt_cond, neg_txt_cond, diff_steps, p, rgb_cond=rgb_cond
        )
        imgs_zoomed = [self.resize_image(img) for img in imgs_zoomed]
        cam_train = [
            get_train_cam(
                img=imgs_zoomed[i],
                focal=focalx[i],
                c2w=c2w_init,
                H=H,
                W=W,
                white_background=self.opt.white_background,
                z_scale=p**i,
            )
            for i in range(num_levels)
        ]

        # generate the initial scene
        traindata = self.generate_pcd_torch(
            imgs_zoomed[0],
            txt_cond[0],
            neg_txt_cond,
            cam_ext,
            seed,
            diff_steps,
            depth_model="murre",
        )
        self.scene = Scene(traindata, self.gaussians, self.opt)
        self.training()

        self.render_video(cams_diving_llff, "llff_zoom_before")
        self.render_video(cams_diving, "diving_zoom_before")

        # zoom in (forward in a straight line)
        merger = GaussianMerger(self.scene.gaussians)
        for i in range(1, num_levels):
            with torch.no_grad():  # TODO test again
                points_new, colors_new = self.backproject(
                    image=imgs_zoomed[i],
                    ixt=cam_ixt[i],
                    ext=torch.tensor(cam_ext[0], dtype=self.dtype, device=self.device),
                    points=self.gaussians.get_xyz,
                )
            merger.merge_gaussians(
                new_gaussian_attrs=merger.init_from_pcd(
                    points_new.reshape(-1, 3), colors_new
                ),
                opt=self.opt,
            )
        # self.opt = GSParams(iterations=5990)
        self.opt = GSParams(iterations=8990)
        self.training(cameras=cam_train)

        # save zoom-in videos
        self.render_video(cams_diving, "diving_zoom_after")
        self.render_video(cams_diving_llff, "llff_zoom_after")

    def create_scene_v4(
        self,
        rgb_cond,
        txt_cond,
        neg_txt_cond,
        pcdgenpath,
        seed,
        diff_steps,
        p,
    ):
        """
        Process:
            Progressively and individually train Gaussians one zoom-in level at a time.
        Caveat:
            - Although V4 may produce clearer object shapes at higher zoom-in levels (the Gaussians can be better optimized), it also introduces evident inconsistency near the boundaries of zoomed-in images, and it suffers from long-shaped floating black artifacts because the Gaussians are not jointly trained.
            - Generally, training jointly across all zoom-in levels will be better.
        """
        # generate camera trajectory
        n_levels = len(txt_cond)
        H, W = self.cam.H, self.cam.W
        focals, c2w_init, cams_ext_init, cams_ixt, cams_diving, cams_diving_llff = (
            prepare_cameras_zoom_in(
                pcdgenpath,
                n_levels,
                n_views=301,
                focal=self.cam.focal[0],
                H=self.cam.H,
                W=self.cam.W,
                p=p,
            )
        )
        imgs_zoomed = self.generate_zoom_stack(
            txt_cond, neg_txt_cond, diff_steps, p, rgb_cond=rgb_cond
        )

        # generate the initial scene
        traindata = self.generate_pcd_torch(
            imgs_zoomed[0],
            txt_cond[0],
            neg_txt_cond,
            cams_ext_init,
            seed,
            diff_steps,
            depth_model="murre",
        )
        self.scene = Scene(traindata, self.gaussians, self.opt)
        self.training()

        self.render_video(cams_diving_llff, "llff_zoom_before")
        self.render_video(cams_diving, "diving_zoom_before")

        # zoom in (forward in a straight line)
        merger = GaussianMerger(self.scene.gaussians)
        for i in range(1, n_levels):
            with torch.no_grad():
                img_zoomed = self.resize_image(imgs_zoomed[i])
                points_new, colors_new = self.backproject(
                    image=img_zoomed,
                    ixt=cams_ixt[i],
                    ext=torch.tensor(
                        cams_ext_init[0], dtype=self.dtype, device=self.device
                    ),
                    points=self.gaussians.get_xyz,
                )
            # merge Gaussians and jointly train
            cam_train = get_train_cam(
                img=img_zoomed,
                focal=focals[i],
                c2w=c2w_init,
                H=H,
                W=W,
                white_background=self.opt.white_background,
                z_scale=p**i,
            )
            merger.merge_gaussians(
                new_gaussian_attrs=merger.init_from_pcd(
                    points_new.reshape(-1, 3), colors_new
                ),
                opt=self.opt,
            )
            self.training(cameras=[cam_train])

        # save zoom-in video
        self.render_video(cams_diving, "diving_zoom_after")
        self.render_video(cams_diving_llff, "llff_zoom_after")

    def create_scene_v3(
        self,
        rgb_cond,
        txt_cond,
        neg_txt_cond,
        pcdgenpath,
        seed,
        diff_steps,
        p,
        save_imgs=True,
    ):
        """
        Process:
            Generate zoomed-in image one at a time (the number of zoom-in levels of each zoom stack is 2).
        Caveat:
            Because zoomed-in images don't belong to the same zoom stack, inconsistency on the borders of zoomed-in images can be observed.
        """
        # generate camera trajectory
        n_levels = len(txt_cond)
        H, W = self.cam.H, self.cam.W
        focals, c2w_init, cams_ext_init, cams_ixt, cams_diving, cams_diving_llff = (
            prepare_cameras_zoom_in(
                pcdgenpath,
                n_levels,
                n_views=301,
                focal=self.cam.focal[0],
                H=self.cam.H,
                W=self.cam.W,
                p=p,
            )
        )
        cams_render = [
            get_render_cam(focal=focals[i], c2w=c2w_init, H=H, W=W, z_scale=p**i)
            for i in range(n_levels)
        ]

        # generate the initial scene
        traindata = self.generate_pcd_torch(
            rgb_cond, txt_cond[0], neg_txt_cond, cams_ext_init, seed, diff_steps, depth_model="murre"
        )
        self.scene = Scene(traindata, self.gaussians, self.opt)
        self.training()

        # delete models that will no longer be used
        del self.d_model
        del self.rgb_model
        gc.collect()
        torch.cuda.empty_cache()

        self.render_video(cams_diving_llff, "llff_zoom_before")
        self.render_video(cams_diving, "diving_zoom_before")

        # zoom in (forward in a straight line)
        merger = GaussianMerger(self.scene.gaussians)
        to_pil = transforms.ToPILImage()
        for i in range(n_levels - 1):
            with torch.no_grad():
                img_cur = render(cams_render[i], self.gaussians, self.opt, self.background)[
                    "render"
                ]
                img_zoomed = self.zoom_model(
                    txt_cond[i : i + 2],
                    neg_txt_cond,
                    p,
                    num_inference_steps=diff_steps,
                    guidance_scale=7,
                    photograph=to_pil(img_cur),
                    viz_step=0,
                )[1]
                if save_imgs:
                    save_images(
                        to_pil(img_cur), self.save_dir, name=f"{i}_render_{txt_cond[i]}.png"
                    )
                    save_images(
                        img_zoomed, self.save_dir, name=f"{i + 1}_zoom_{txt_cond[i + 1]}.png"
                    )

                img_zoomed = self.resize_image(img_zoomed)
                ext_zoomed = torch.tensor(cams_ext_init[0], dtype=self.dtype, device=self.device)
                points_new, colors_new = self.backproject(
                    image=img_zoomed,
                    ixt=cams_ixt[i + 1],
                    ext=ext_zoomed,
                    points=self.gaussians.get_xyz,
                )
            # merge Gaussians and jointly train
            cam_train = get_train_cam(
                img=img_zoomed,
                focal=focals[i + 1],
                c2w=c2w_init,
                H=H,
                W=W,
                white_background=self.opt.white_background,
                z_scale=p ** (i + 1),
            )
            merger.merge_gaussians(
                new_gaussian_attrs=merger.init_from_pcd(
                    points_new.reshape(-1, 3), colors_new
                ),
                opt=self.opt,
            )
            self.training(cameras=[cam_train])

        # save zoom-in video
        self.render_video(cams_diving, "diving_zoom_after")
        self.render_video(cams_diving_llff, "llff_zoom_after")

    def create_scene_v2(
        self, rgb_cond, txt_cond, neg_txt_cond, pcdgenpath, seed, diff_steps, p
    ):
        # generate camera trajectories for all zoom-in levels
        dtype, device = self.dtype, self.device
        num_levels = len(txt_cond)
        cam_extri = [
            torch.tensor(
                get_pcdGenPoses(pcdgenpath=pcdgenpath, deg_denom=p**i),
                dtype=dtype,
                device=device,
            )
            for i in range(num_levels)
        ]
        # note that since self.cam only affects H, W and generate_pcd, so no need to modify it for zooming process
        # by default, focals of x and y are the same
        cam_focal = [self.cam.focal[0] / (p**i) for i in range(num_levels)]
        cam_intri = [
            torch.tensor(
                [
                    [cam_focal[i], 0.0, self.cam.W / 2],
                    [0.0, cam_focal[i], self.cam.H / 2],
                    [0.0, 0.0, 1.0],
                ],
                dtype=dtype,
                device=device,
            )
            for i in range(num_levels)
        ]
        minicams = [
            [
                get_render_cam(
                    focal=cam_focal[i], c2w=cam_extri[i][j], H=self.cam.H, W=self.cam.W
                )
                for j in range(len(cam_extri[i]))
            ]
            for i in range(num_levels)
        ]

        # generate the initial scene
        traindata = self.generate_pcd_torch(
            rgb_cond, txt_cond[0], neg_txt_cond, cam_extri[0], seed, diff_steps
        )
        self.scene = Scene(traindata, self.gaussians, self.opt)
        self.training()

        # zoom in
        to_pil = transforms.ToPILImage()
        for i in range(num_levels - 1):
            for j in range(len(cam_extri[i])):
                img = render(minicams[i][j], self.gaussians, self.opt, self.background)[
                    "render"
                ]
                img_gt = self.zoom_model(
                    txt_cond[i : i + 2],
                    neg_txt_cond,
                    p,
                    num_inference_steps=diff_steps,
                    guidance_scale=7,
                    photograph=to_pil(img),
                    viz_step=0,
                )[1]
                # TODO enforce multi-view consistency

                points_new, colors_new = self.backproject(
                    image=img_gt,
                    H=self.cam.H,
                    W=self.cam.W,
                    ext=cam_extri[i + 1][j],
                )
                # TODO implement version 2

    def create_scene_v1(self, txt_cond, neg_txt_cond, pcdgenpath, seed, diff_steps, p):
        """
        Params:
            pcdgenpath: The name of the camera trajectory.
        Process:
            1) Generate the complete zoom stack for the first camera.
            2) For each zoom-in level, treat the zoom stack image at this level as image condition for generating point cloud by LucidDreamer.
            3) The rotation degrees of cameras are scaled down accordingly.
            4) Stack all point clouds and camera trajectories at once, and use them to initialize and train 3DGS.
        Caveat:
            - Without point cloud pruning, four zoom-in levels will requires around 23GB VRAM.
            - The inpainted images across different zoom-in levels are inconsistent.
        """
        # generate the initial zoom stack
        images = self.generate_zoom_stack(txt_cond, neg_txt_cond, diff_steps, p)

        # generate complete point clouds at all zoom-level
        traindata = None
        fx, fy = self.cam.focal
        for i in range(len(images)):
            # self.cam will be used inside generate_pcd
            self.cam = CameraParams(focal=(fx * (p**i), fy * (p**i)))
            data = self.generate_pcd_torch(
                images[i],
                txt_cond[i],
                neg_txt_cond,
                pcdgenpath,
                seed,
                diff_steps,
                deg_scale=p**i,
            )
            if traindata is None:
                traindata = data
            else:
                # W and H remain the same
                traindata["camera_angle_x"].extend(data["camera_angle_x"])
                traindata["pcd_points"] = np.concatenate(
                    (traindata["pcd_points"], data["pcd_points"]), axis=-1
                )  # [3, N1 + N2]
                traindata["pcd_colors"] = np.concatenate(
                    (traindata["pcd_colors"], data["pcd_colors"]), axis=0
                )  # [N1 + N2, 3]
                traindata["frames"].extend(data["frames"])

        self.scene = Scene(traindata, self.gaussians, self.opt)
        self.training()

    def generate_zoom_stack(
        self, txt_cond, neg_txt_cond, diff_steps, p, rgb_cond=None, save_dir=None
    ):
        """
        Params:
            save_dir: The folder to save the zoom stack images. If the folder already exists, open and return the images inside that folder.
            rgb_cond: The most zoomed-out image. If not provided, generate the zoom stack solely based on txt_cond.
        Returns: A list of PIL images corresponding to each zoom-in levels.
        """
        if save_dir is None:
            save_dir = f"{self.save_dir}/{self.version}"
        if not os.path.exists(save_dir):
            images = self.zoom_model(
                txt_cond,
                neg_txt_cond,
                p,
                save_dir,
                num_inference_steps=diff_steps,
                guidance_scale=7,
                photograph=rgb_cond,
                viz_step=0,
            )
            save_images(images, save_dir, txt_cond)
        else:
            images = [
                Image.open(os.path.join(save_dir, img))
                for img in sorted(os.listdir(save_dir))
                if img.split("_")[0].isdigit()
            ]
        return images

    def backproject(self, image, ixt, ext, points=None):
        """
        Params:
            image: PIL image for depth prediction.
            ixt: Tensor[3, 3] Camera to image transformation.
            ext: Tensor[3, 4] World to camera transformation.
            points: Tensor[N, 3] Existing point cloud coords which serves as the condition for Murre. If not provided, use ZoeDepth instead.
        Returns:
            coord_world: Tensor[3, N] The backprojected 3D points, where N is the number of points.
            colors: Tensor[N, 3] The colors corresponding to the 3D points.
        """
        dtype, device = self.dtype, self.device
        depth = (
            self.pred_depth_with_pcd(image, points, ixt, ext)
            if points is not None
            else self.d(image)
        )
        depth = torch.tensor(depth, device=device)
        image = torch.tensor(np.array(image), dtype=dtype, device=device)
        H, W = image.shape[0], image.shape[1]  # TODO (H, W) or (W, H)

        K_inv = torch.linalg.inv(ixt)
        x, y = torch.meshgrid(
            torch.arange(W, dtype=dtype, device=device),
            torch.arange(H, dtype=dtype, device=device),
            indexing="xy",
        )  # pixels
        R, T = ext[:3, :3], ext[:3, 3:4]
        R_inv = torch.linalg.inv(R)
        coord_cam = K_inv @ torch.stack(
            (x * depth, y * depth, 1 * depth), dim=0
        ).reshape(3, -1)

        coord_world = R_inv @ coord_cam - R_inv @ T
        colors = image.reshape(-1, 3) / 255.0
        return coord_world, colors

    def pred_depth_with_pcd(
        self,
        im: Image,
        points=None,
        ixt=None,
        ext=None,
        denoise_steps=10,
        ensemble_size=1,
    ):
        sdpt = get_sparse_depth(
            points=points, ixt=ixt, ext=ext, H=im.size[1], W=im.size[0]
        )
        depth_np = self.murre_model(
            input_image=im,
            input_sparse_depth=sdpt,
            max_depth=None,
            denoising_steps=denoise_steps,
            ensemble_size=ensemble_size,
            processing_res=max(im.size[0], im.size[1]),
            model_dtype=self.dtype,
            show_progress_bar=True,
        ).depth_np
        return depth_np
