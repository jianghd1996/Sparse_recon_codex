"""Full-attention Control pipeline with the native Wan2.2 VAE."""

import torch
from einops import rearrange

from .pipeline_wan2_2_fun_control_infer_wjx_stream_light import (
    Wan2_2FunControlPipeline_infer as _ControlPipeline,
    resize_mask,
)
from .standard_wan_vae import StandardWanVaeCodec


class Wan2_2FunControlPipeline_infer(StandardWanVaeCodec, _ControlPipeline):
    def _ensure_mask_cache(self, *args, **kwargs):
        # The native VAE encodes the complete condition on every invocation.
        # Keep the old call signature for the worker warmup and diffusion loop.
        return None

    def _prepare_mask_and_latents(
        self, init_video, mask_video, video_length, height, width, batch_size,
        weight_dtype, device, generator, do_classifier_free_guidance,
    ):
        mask_condition = self.mask_processor.preprocess(
            rearrange(mask_video, "b c f h w -> (b f) c h w"),
            height=height, width=width,
        )
        mask_condition = rearrange(
            mask_condition, "(b f) c h w -> b c f h w", f=video_length,
        ).to(device=device, dtype=weight_dtype)
        masked_video = init_video * (mask_condition < 0.5)
        masked_video_latents = self.encode_videos_streaming(masked_video)
        # Match the existing Wan condition layout: four mask channels for the
        # temporal compression groups, with the first frame repeated.
        mask_for_latent = torch.cat([
            mask_condition[:, :, :1].repeat_interleave(4, dim=2),
            mask_condition[:, :, 1:],
        ], dim=2)
        groups = mask_for_latent.shape[2] // 4
        mask_for_latent = mask_for_latent.reshape(
            mask_condition.shape[0], groups, 4, height, width,
        ).transpose(1, 2)
        mask_latents = resize_mask(
            1 - mask_for_latent, masked_video_latents, True,
        ).to(device=device, dtype=weight_dtype)
        return mask_condition, masked_video_latents, mask_latents
