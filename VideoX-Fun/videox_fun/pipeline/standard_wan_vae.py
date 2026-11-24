"""Native Wan VAE codec shared by the dense Control and LivePhoto pipelines."""

import torch


class StandardWanVaeCodec:
    # Method names match the existing diffusion loops; only native encode/decode
    # are called. No LightVAE decoder or encode_first/next_step API is required.
    def encode_videos_streaming(
        self, videos, K=-1, converged_mu=None, converged_caches=None,
    ):
        if K >= 0 or converged_mu is not None or converged_caches is not None:
            raise ValueError("Native Wan VAE does not use skipped zero-frame caches")
        latents = []
        for sample in videos.split(1):
            encoded = self.vae.encode(sample.to(dtype=self.vae.dtype), return_dict=False)[0]
            latents.append(encoded.mode())
        return torch.cat(latents, dim=0)

    def decode_latents_streaming(self, latents, decode_to_cpu=True):
        decoded = self.vae.decode(latents.to(dtype=self.vae.dtype), return_dict=False)[0]
        # The reused video processor expects [B,C,T,H,W] in [0,1].
        return (decoded / 2 + 0.5).clamp(0, 1).cpu().float().numpy()
