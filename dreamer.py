import os
import datetime
import numpy as np
from PIL import Image

from luciddreamer import LucidDreamer
from arguments import CameraParams
from scene import Scene
from gaussian_renderer import render
from utils.trajectory import get_pcdGenPoses
from gen_powers_10.model import GenPowers10Pipeline
from gen_powers_10.utils import save_images


class Dreamer(LucidDreamer):
    def __init__(self, for_gradio=True, save_dir=None, version="DFIF_XL_L_X4"):
        super().__init__(for_gradio=for_gradio, save_dir=save_dir)
        self.version = version
        self.zoom_model = GenPowers10Pipeline(version)

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

    def create_scene(
        self, rgb_cond, txt_cond, neg_txt_cond, pcdgenpath, seed, diff_steps, p
    ):
        # generate camera trajectories for all zoom-in levels
        num_levels = len(txt_cond)
        cameras = [
            get_pcdGenPoses(pcdgenpath=pcdgenpath, deg_denom=p**i)
            for i in range(num_levels)
        ]

        # generate the initial scene
        traindata = self.generate_pcd_torch(
            rgb_cond, txt_cond, neg_txt_cond, cameras[0], seed, diff_steps
        )
        self.scene = Scene(traindata, self.gaussians, self.opt)
        self.training()

        # zoom in
        for i in range(num_levels - 1):
            for j in range(len(cameras[i])):
                img_gt = render()

                images = self.zoom_model(
                    txt_cond[i : i + 2],
                    neg_txt_cond,
                    p,
                    num_inference_steps=diff_steps,
                    guidance_scale=7,
                    photograph=img_gt,
                    viz_step=0,
                )

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
