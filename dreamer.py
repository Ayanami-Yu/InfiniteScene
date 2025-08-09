import os
import datetime
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from luciddreamer import LucidDreamer
from arguments import CameraParams
from scene import Scene
from scene.gaussian_merger import GaussianMerger
from gaussian_renderer import render
from utils.trajectory import get_pcdGenPoses
from utils.camera import get_render_cam, get_train_cam
from gen_powers_10.model import GenPowers10Pipeline
from gen_powers_10.utils import save_images
from murre.pipeline import MurrePipeline
from utils.murre import get_sparse_depth


class Dreamer(LucidDreamer):
    def __init__(
        self,
        for_gradio=True,
        save_dir=None,
        torch_hub_local=True,
        version="DFIF_XL_L_X4",
        murre_ckpt_path=None, dtype=torch.float32, device="cuda"
    ):
        super().__init__(
            for_gradio=for_gradio, save_dir=save_dir, torch_hub_local=torch_hub_local, dtype=dtype, device=device
        )
        self.version = version
        self.zoom_model = GenPowers10Pipeline(version)
        self.murre_model = MurrePipeline.from_pretrained(murre_ckpt_path, variant=None, torch_dtype=dtype).to(device)

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
            self.create_scene(
                rgb_cond, txt_cond, neg_txt_cond, pcdgenpath, seed, diff_steps, p
            )
            os.makedirs(self.save_dir, exist_ok=True)
            outfile = self.save_ply(os.path.join(self.save_dir, "gsplat.ply"))
        return outfile
    
    def create_scene(
        self, rgb_cond, txt_cond, neg_txt_cond, pcdgenpath, seed, diff_steps, p, save_imgs=True
    ):
        # generate camera trajectory
        num_levels = len(txt_cond)
        H, W = self.cam.H, self.cam.W
        cam_extri = get_pcdGenPoses(pcdgenpath=pcdgenpath)
        c2w = np.linalg.inv(np.concatenate((cam_extri[0], np.array([[0, 0, 0, 1]])), axis=0))
        focalx = [self.cam.focal[0] * (p ** i) for i in range(num_levels)]
        cam_intri = [
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
        focals = np.linspace(focalx[0], focalx[num_levels - 1], 201)
        z_scales = [f / focalx[0] for f in focals]
        minicams = [get_render_cam(focalx=f, c2w=c2w, H=H, W=W, z_scale=z_s) for f, z_s in zip(focals, z_scales)]
        
        imgs_zoomed = self.zoom_model(
                txt_cond,
                neg_txt_cond,
                p,
                num_inference_steps=diff_steps,
                guidance_scale=7,
                photograph=rgb_cond,
                viz_step=0,
            )
        if save_imgs:
            save_dir = f"{self.save_dir}/{self.version}"
            save_images(imgs_zoomed, save_dir, txt_cond)

        # generate the initial scene
        traindata = self.generate_pcd_torch(
            imgs_zoomed[0], txt_cond[0], neg_txt_cond, cam_extri, seed, diff_steps
        )
        self.scene = Scene(traindata, self.gaussians, self.opt)
        self.training()
        self.render_video(minicams, "zoom_before")
        merger = GaussianMerger(self.scene.gaussians)

        # zoom in (forward in a straight line)
        for i in range(num_levels - 1):
            # TODO test results of ZoeDepth
            img_zoomed = self.resize_image(imgs_zoomed[i])
            points_new, colors_new = self.backproject(
                image=img_zoomed, ixt=cam_intri[i], ext=torch.tensor(cam_extri[0], dtype=self.dtype, device=self.device), points=self.gaussians.get_xyz
            )
            # merge Gaussians and jointly train
            cam_train = get_train_cam(
                img=img_zoomed,
                focalx=focalx[i + 1],
                c2w=c2w,
                H=H,
                W=W,
                white_background=self.opt.white_background,
                z_scale=p ** (i + 1),
            )
            merger.merge_gaussians(
                new_gaussian_attrs=merger.init_from_pcd(points_new.reshape(-1, 3), colors_new),
                opt=self.opt,
            )
            self.training(cameras=[cam_train])

        # save zoom-in video
        self.render_video(minicams, "zoom_after")

    def create_scene_v3(
        self, rgb_cond, txt_cond, neg_txt_cond, pcdgenpath, seed, diff_steps, p, save_imgs=True
    ):
        """
        Process:
            Generate zoomed-in image one at a time (the number of zoom-in levels of each zoom stack is 2).
        Caveat:
            Because zoomed-in images don't belong to the same zoom stack, inconsistency on the borders of zoomed-in images can be observed.
        """
        # generate camera trajectory
        num_levels = len(txt_cond)
        H, W = self.cam.H, self.cam.W
        cam_extri = get_pcdGenPoses(pcdgenpath=pcdgenpath)
        c2w = np.linalg.inv(np.concatenate((cam_extri[0], np.array([[0, 0, 0, 1]])), axis=0))
        focalx = [self.cam.focal[0] * (p ** i) for i in range(num_levels)]
        cam_intri = [
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
        cam_render = [
            get_render_cam(
                focalx=focalx[i], c2w=c2w, H=H, W=W, z_scale=p ** i
            )
            for i in range(num_levels)
        ]
        # generate zoom-in trajectory with interpolated cameras
        focals = np.linspace(focalx[0], focalx[num_levels - 1], 201)
        z_scales = [f / focalx[0] for f in focals]
        minicams = [get_render_cam(focalx=f, c2w=c2w, H=H, W=W, z_scale=z_s) for f, z_s in zip(focals, z_scales)]

        # generate the initial scene
        traindata = self.generate_pcd_torch(
            rgb_cond, txt_cond[0], neg_txt_cond, cam_extri, seed, diff_steps
        )
        self.scene = Scene(traindata, self.gaussians, self.opt)
        self.training()
        self.render_video(minicams, "zoom_before")
        merger = GaussianMerger(self.scene.gaussians)

        # zoom in (forward in a straight line)
        to_pil = transforms.ToPILImage()
        for i in range(num_levels - 1):
            img_cur = render(cam_render[i], self.gaussians, self.opt, self.background)["render"]
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
                save_dir = f"{self.save_dir}/{self.version}"
                save_images(to_pil(img_cur), save_dir, name=f"{i}_render_{txt_cond[i]}.png")
                save_images(img_zoomed, save_dir, name=f"{i + 1}_zoom_{txt_cond[i + 1]}.png")

            # TODO test results of ZoeDepth
            img_zoomed = self.resize_image(img_zoomed)
            points_new, colors_new = self.backproject(
                image=img_zoomed, ixt=cam_intri[i], ext=torch.tensor(cam_extri[0], dtype=self.dtype, device=self.device)
            )
            # merge Gaussians and jointly train
            cam_train = get_train_cam(
                img=img_zoomed,
                focalx=focalx[i + 1],
                c2w=c2w,
                H=H,
                W=W,
                white_background=self.opt.white_background,
                z_scale=p ** (i + 1),
            )
            merger.merge_gaussians(
                new_gaussian_attrs=merger.init_from_pcd(points_new.reshape(-1, 3), colors_new),
                opt=self.opt,
            )
            self.training(cameras=[cam_train])

        # save zoom-in video
        self.render_video(minicams, "zoom_after")

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
                    focalx=cam_focal[i], c2w=cam_extri[i][j], H=self.cam.H, W=self.cam.W
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
                    ext=cam_extri[i][j],
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
        depth = self.pred_depth_with_pcd(image, points, ixt, ext) if points is not None else self.d(image)
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
    
    def pred_depth_with_pcd(self, im: Image, points=None, ixt=None, ext=None, denoise_steps=10, ensemble_size=1):
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

