import os
import datetime
import torch
import numpy as np
from PIL import Image

from luciddreamer import LucidDreamer
from arguments import CameraParams
from scene import Scene
from scene.gaussian_merger import GaussianMerger
from gaussian_renderer import render
from utils.trajectory import get_pcdGenPoses
from utils.camera import get_render_cam, get_train_cam
from gen_powers_10.model import GenPowers10Pipeline
from gen_powers_10.utils import save_images


class Dreamer(LucidDreamer):
    def __init__(
        self,
        for_gradio=True,
        save_dir=None,
        torch_hub_local=True,
        version="DFIF_XL_L_X4",
        dtype=torch.float32,
        device="cuda",
    ):
        super().__init__(
            for_gradio=for_gradio, save_dir=save_dir, torch_hub_local=torch_hub_local
        )
        self.version = version
        self.zoom_model = GenPowers10Pipeline(version)
        self.dtype = dtype
        self.device = device

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
            self.create_scene(txt_cond, neg_txt_cond, pcdgenpath, seed, diff_steps, p)
            os.makedirs(self.save_dir, exist_ok=True)
            outfile = self.save_ply(os.path.join(self.save_dir, "gsplat.ply"))
        return outfile
    
    def create_scene(self, rgb_cond, txt_cond, neg_txt_cond, pcdgenpath, seed, diff_steps, p):
        # generate the initial scene
        dtype, device = self.dtype, self.device
        H, W = self.cam.H, self.cam.W
        cam_extri = torch.tensor(get_pcdGenPoses(pcdgenpath=pcdgenpath), dtype=dtype, device=device)
        traindata = self.generate_pcd_torch(
            rgb_cond, txt_cond[0], neg_txt_cond, cam_extri, seed, diff_steps
        )
        self.scene = Scene(traindata, self.gaussians, self.opt)
        self.training()
        merger = GaussianMerger(self.scene.gaussians)

        num_levels = len(txt_cond)
        focalx = [self.cam.focal[0] * (p ** (i + 1)) for i in range(num_levels)]
        cam_render = [get_render_cam(focalx=focalx[i], c2w=cam_extri[0], H=H, W=W) for i in range(num_levels)]
        cam_train = [get_train_cam(img=img_zoomed, focalx=focalx[i], c2w=cam_extri[0], H=H, W=W, white_background=self.opt.white_background) for i in range(num_levels)]

        # zoom in (forward in a straight line)
        for i in range(num_levels - 1):
            img_cur = render(cam_render[i], self.gaussians, self.opt, self.background)
            img_zoomed = self.zoom_model(
                    txt_cond[i : i + 2],
                    neg_txt_cond,
                    p,
                    num_inference_steps=diff_steps,
                    guidance_scale=7,
                    photograph=img_cur,
                    viz_step=0,
                )[1]
            # TODO test results of ZoeDepth
            points_new, colors_new = self.backproject(image=img_zoomed, H=H, W=W, cam_pose=cam_extri[0])

            # merge Gaussians and jointly train
            merger.merge_gaussians(new_gaussian_attrs=merger.init_from_pcd(points_new, colors_new), opt=self.opt)
            self.training(cameras=[cam_train[i]])





    def create_scene_v2(
        self, rgb_cond, txt_cond, neg_txt_cond, pcdgenpath, seed, diff_steps, p
    ):
        # generate camera trajectories for all zoom-in levels
        dtype, device = self.dtype, self.device
        num_levels = len(txt_cond)
        cam_extri = [
            torch.tensor(get_pcdGenPoses(pcdgenpath=pcdgenpath, deg_denom=p**i), dtype=dtype, device=device)
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
            [get_render_cam(
                focalx=cam_focal[i], c2w=cam_extri[i][j], H=self.cam.H, W=self.cam.W
            )
            for j in range(len(cam_extri[i]))] for i in range(num_levels)
        ]

        # generate the initial scene
        traindata = self.generate_pcd_torch(
            rgb_cond, txt_cond[0], neg_txt_cond, cam_extri[0], seed, diff_steps
        )
        self.scene = Scene(traindata, self.gaussians, self.opt)
        self.training()

        # zoom in
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
                    photograph=img,
                    viz_step=0,
                )[1]
                # TODO enforce multi-view consistency

                points_new, colors_new = self.backproject(
                    image=img_gt,
                    H=self.cam.H,
                    W=self.cam.W,
                    cam_pose=cam_extri[i][j],
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

    def generate_zoom_stack(self, txt_cond, neg_txt_cond, diff_steps, p, save_dir=None):
        """
        Params:
            save_dir: The folder to save the zoom stack images. If the folder already exists, open and return the images inside that folder.
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
                photograph=None,
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

    def backproject(self, image, H, W, K, cam_pose):
        """
        Params:
            image: PIL image for depth prediction.
            K: Tensor[3, 3] Camera to image transformation.
            cam_pose: Tensor[3, 4] World to camera transformation.
        Returns:
            coord_world: Tensor[3, N] The backprojected 3D points, where N is the number of points.
            colors: Tensor[N, 3] The colors corresponding to the 3D points.
        """
        dtype, device = self.dtype, self.device
        depth = torch.tensor(self.d(image), device=device)
        image = torch.tensor(np.array(image), dtype=dtype, device=device)

        K_inv = torch.linalg.inv(K)
        x, y = torch.meshgrid(
            torch.arange(W, dtype=dtype, device=device),
            torch.arange(H, dtype=dtype, device=device),
            indexing="xy",
        )  # pixels
        R, T = cam_pose[:3, :3], cam_pose[:3, 3:4]
        R_inv = torch.linalg.inv(R)
        coord_cam = K_inv @ torch.stack(
            (x * depth, y * depth, 1 * depth), dim=0
        ).reshape(3, -1)

        coord_world = R_inv @ coord_cam - R_inv @ T
        colors = image.reshape(-1, 3) / 255.0
        return coord_world, colors
