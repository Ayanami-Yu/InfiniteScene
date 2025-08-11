import os
from omegaconf import OmegaConf
from torchvision import transforms
from PIL import Image
from viewcrafter.diffusion_utils import (
    instantiate_from_config,
    load_model_checkpoint,
    image_guided_synthesis,
)
from utils.image import crop_image
from utils.pvd_utils import world_point_to_obj, generate_traj_circular


class Crafter:
    def __init__(self, cfg_path, device="cuda"):
        self.device = device
        self.setup_diffusion(cfg_path)

    def setup_diffusion(self, cfg_path):
        config = OmegaConf.load(cfg_path)
        model_config = config.pop("model", OmegaConf.create())

        model = instantiate_from_config(model_config)
        model = model.to(self.device)
        model.cond_stage_model.device = self.device
        model.perframe_ae = self.opts.perframe_ae
        assert os.path.exists(self.opts.ckpt_path), "Error: checkpoint Not Found!"
        model = load_model_checkpoint(model, self.opts.ckpt_path)
        model.eval()
        self.diffusion = model

        h, w = self.opts.height // 8, self.opts.width // 8
        channels = model.model.diffusion_model.out_channels
        n_frames = self.opts.video_length
        self.noise_shape = [self.opts.bs, channels, n_frames, h, w]

    def __call__(
        self,
        pts2d,
        depth2d,
        colors2d,
        img: Image,
        c2w,
        focal,
        H=320,
        W=512,
        n_frame=5,
        d_theta=30,
        d_phi=30,
        d_r=0,
    ):
        """
        Args:
            pts2d (Tensor[H, W, C]): Corresponding 3D coords of each pixel
            depth2d (Tensor[H, W]): Corresponding depth of each pixel
            c2w (Tensor[4, 4]): Camera to world transformation
        """
        to_tensor = transforms.ToTensor()
        img = to_tensor(crop_image(img, H=H, W=W))  # [C, H, W]
        depth_center = depth2d[H // 2, W // 2]

        # TODO k should be 0 if it's refering to the first frame
        c2w, pcd = world_point_to_obj(
            poses=c2w.unsqueeze(0),
            points=pts2d.unsqueeze(0),
            k=-1,  # TODO remove arg
            r=depth_center,
            elevation=5.0,  # TODO tune it
            device=self.device,
        )
        # TODO ideally cameras should be provided from the outside
        cameras = generate_traj_circular(
            c2w,
            H,
            W,
            focal,
            # principal_points,
            d_theta=d_theta,
            d_phi=d_phi,
            d_r=d_r,
            n_frame=9,
            device=self.device,
        )
        # TODO finish implementation
