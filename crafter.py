import os
import torch
from warnings import warn
from omegaconf import OmegaConf
from viewcrafter.diffusion_utils import (
    instantiate_from_config,
    load_model_checkpoint,
    image_guided_synthesis,
)


class Crafter:
    def __init__(self, cfg_path, device="cuda"):
        self.device = device
        self.prompt = "Rotating view of a scene"  # fixed
        self.setup_diffusion(cfg_path)

    def setup_diffusion(self, cfg_path):
        config = OmegaConf.load(cfg_path)
        model_config = config.pop("model", OmegaConf.create())
        self.opts = config.pop("opts", OmegaConf.create())

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

    def run_diffusion(self, renderings):
        """
        Args:
            renderings: Tensor[n_frames, H, W, C] in range [0, 1]
        """
        prompts = [self.prompt]
        videos = (
            (renderings * 2.0 - 1.0).permute(3, 0, 1, 2).unsqueeze(0).to(self.device)
        )
        condition_index = [0]
        with torch.no_grad(), torch.cuda.amp.autocast():
            batch_samples = image_guided_synthesis(
                self.diffusion,
                prompts,
                videos,
                self.noise_shape,
                self.opts.n_samples,
                self.opts.ddim_steps,
                self.opts.ddim_eta,
                self.opts.unconditional_guidance_scale,
                self.opts.cfg_img,
                self.opts.frame_stride,
                self.opts.text_input,
                self.opts.multiple_cond_cfg,
                self.opts.timestep_spacing,
                self.opts.guidance_rescale,
                condition_index,
            )

        return torch.clamp(batch_samples[0][0].permute(1, 2, 3, 0), -1.0, 1.0)

    def __call__(self, imgs):
        """
        Args:
            imgs: Tensor[n_frames, H, W, C] in range [0, 1]
        Returns:
            Generated video frames (Tensor[n_frames, H, W, C]) in range [-1, 1]
        """
        if len(imgs) != 25:
            warn(
                "If the number of frames is not 25, the generation quality of ViewCrafter will be significantly worse"
            )

        return self.run_diffusion(imgs)
