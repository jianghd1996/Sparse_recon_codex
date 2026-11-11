import inspect
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch, os
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.embeddings import get_1d_rotary_pos_embed
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import BaseOutput, logging, replace_example_docstring
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from einops import rearrange
from PIL import Image
from transformers import T5Tokenizer
import time

from ..models import (AutoencoderKLWan, AutoTokenizer,
                      Wan2_2Transformer3DModel, WanT5EncoderModel)
from ..utils.fm_solvers import (FlowDPMSolverMultistepScheduler,
                                get_sampling_sigmas)
from ..utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def _move_tensor_tree(value, device):
    """Move persistent VAE caches without converting their dtypes."""
    if isinstance(value, torch.Tensor):
        return value.to(device=device)
    if isinstance(value, dict):
        return {key: _move_tensor_tree(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_move_tensor_tree(item, device) for item in value)
    if isinstance(value, list):
        return [_move_tensor_tree(item, device) for item in value]
    return value


def _cuda_tensor_bytes(value):
    """Estimate logical CUDA tensor bytes for the cache diagnostic."""
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size() if value.device.type == "cuda" else 0
    if isinstance(value, dict):
        return sum(_cuda_tensor_bytes(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_cuda_tensor_bytes(item) for item in value)
    return 0


def _activation_dtype(dtype):
    """FP8 weight storage must not select the dtype of pipeline activations."""
    fp8_dtypes = tuple(
        getattr(torch, name)
        for name in (
            "float8_e4m3fn", "float8_e5m2",
            "float8_e4m3fnuz", "float8_e5m2fnuz",
        )
        if hasattr(torch, name)
    )
    return torch.bfloat16 if dtype in fp8_dtypes else dtype


EXAMPLE_DOC_STRING = """
    Examples:
        ```python
        pass
        ```
"""


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


def resize_mask(mask, latent, process_first_frame_only=True):
    latent_size = latent.size()
    batch_size, channels, num_frames, height, width = mask.shape

    if process_first_frame_only:
        target_size = list(latent_size[2:])
        target_size[0] = 1
        first_frame_resized = F.interpolate(
            mask[:, :, 0:1, :, :],
            size=target_size,
            mode='trilinear',
            align_corners=False
        )
        
        target_size = list(latent_size[2:])
        target_size[0] = target_size[0] - 1
        if target_size[0] != 0:
            remaining_frames_resized = F.interpolate(
                mask[:, :, 1:, :, :],
                size=target_size,
                mode='trilinear',
                align_corners=False
            )
            resized_mask = torch.cat([first_frame_resized, remaining_frames_resized], dim=2)
        else:
            resized_mask = first_frame_resized
    else:
        target_size = list(latent_size[2:])
        resized_mask = F.interpolate(
            mask,
            size=target_size,
            mode='trilinear',
            align_corners=False
        )
    return resized_mask


@dataclass
class WanPipelineOutput(BaseOutput):
    r"""
    Output class for CogVideo pipelines.

    Args:
        video (`torch.Tensor`, `np.ndarray`, or List[List[PIL.Image.Image]]):
            List of video outputs - It can be a nested list of length `batch_size,` with each sub-list containing
            denoised PIL image sequences of length `num_frames.` It can also be a NumPy array or Torch tensor of shape
            `(batch_size, num_frames, channels, height, width)`.
    """

    videos: torch.Tensor


class Wan2_2FunControlPipeline_infer(DiffusionPipeline):
    r"""
    Pipeline for text-to-video generation using Wan.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)
    """

    _optional_components = ["transformer_2"]
    model_cpu_offload_seq = "text_encoder->transformer_2->transformer"
    _exclude_from_cpu_offload = ["vae"]

    _callback_tensor_inputs = [
        "latents",
        "prompt_embeds",
        "negative_prompt_embeds",
    ]

    def __init__(
        self,
        # tokenizer: AutoTokenizer,
        # text_encoder: WanT5EncoderModel,
        vae: AutoencoderKLWan,
        transformer: Wan2_2Transformer3DModel,
        transformer_2: Wan2_2Transformer3DModel = None,
        scheduler: FlowMatchEulerDiscreteScheduler = None,
    ):
        super().__init__()

        self.register_modules(
            vae=vae, transformer=transformer, 
            transformer_2=transformer_2, scheduler=scheduler
        )
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae.config.spatial_compression_ratio)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae.config.spatial_compression_ratio)
        self.mask_processor = VaeImageProcessor(
            vae_scale_factor=self.vae.config.spatial_compression_ratio, do_normalize=False, do_binarize=True, do_convert_grayscale=True
        )

    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        prompt_attention_mask = text_inputs.attention_mask
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, max_sequence_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {max_sequence_length} tokens: {removed_text}"
            )

        seq_lens = prompt_attention_mask.gt(0).sum(dim=1).long()
        prompt_embeds = self.text_encoder(text_input_ids.to(device), attention_mask=prompt_attention_mask.to(device))[0]
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        return [u[:v] for u, v in zip(prompt_embeds, seq_lens)]

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            do_classifier_free_guidance (`bool`, *optional*, defaults to `True`):
                Whether to use classifier free guidance or not.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                Number of videos that should be generated per prompt. torch device to place the resulting embeddings on
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            device: (`torch.device`, *optional*):
                torch device
            dtype: (`torch.dtype`, *optional*):
                torch dtype
        """
        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            negative_prompt_embeds = self._get_t5_prompt_embeds(
                prompt=negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        return prompt_embeds, negative_prompt_embeds

    # def encode_videos_streaming(self, videos: torch.Tensor, K: int = -1) -> torch.Tensor:
    #     """
    #     Args:
    #         K: -1 = 不跳过
    #         >=0 = iter 0 之后再正常跑 K 步，然后跳到最后一步
    #                 K=0: 2步（iter 0 + 最后）
    #                 K=1: 3步（iter 0 + 1步 + 最后）
    #     """
    #     assert videos.dim() == 5, f"Expected [B,C,T,H,W], got {videos.shape}"

    #     videos = videos.to(self.vae.dtype)
    #     B, C, T, H, W = videos.shape
    #     step = 4
    #     total_iters = 1 + (T - 1 + step - 1) // step

    #     # iter 0: first frame
    #     out = self.vae.encode_first_step(videos[:, :, :1, :, :])
    #     z_stats = out[0]
    #     enc_caches = out[1:]
    #     mu, _ = z_stats.chunk(2, dim=1)
    #     latents_list = [mu]

    #     if K < 0:
    #         # 不跳过
    #         start = 1
    #         for it in range(1, total_iters):
    #             end = min(start + step, T)
    #             out = self.vae.encode_next_step(videos[:, :, start:end, :, :], *enc_caches)
    #             z_stats = out[0]
    #             enc_caches = out[1:]
    #             mu, _ = z_stats.chunk(2, dim=1)
    #             latents_list.append(mu)
    #             start = end
    #     else:
    #         start = 1

    #         # iter 1 ~ K: 正常跑
    #         for it in range(1, K + 1):
    #             end = min(start + step, T)
    #             out = self.vae.encode_next_step(videos[:, :, start:end, :, :], *enc_caches)
    #             z_stats = out[0]
    #             enc_caches = out[1:]
    #             mu, _ = z_stats.chunk(2, dim=1)
    #             latents_list.append(mu)
    #             start = end

    #         # 此时 mu 和 enc_caches 就是收敛值
    #         converged_mu = mu
    #         converged_caches = enc_caches

    #         # iter K+1 ~ total-2: 跳过
    #         skip_count = total_iters - 1 - (K + 1)
    #         for _ in range(skip_count):
    #             latents_list.append(converged_mu)

    #         # 最后一个 iter: 用收敛 cache 跑
    #         last_start = 1 + step * (total_iters - 2)
    #         out = self.vae.encode_next_step(videos[:, :, last_start:, :, :], *converged_caches)
    #         z_stats = out[0]
    #         mu, _ = z_stats.chunk(2, dim=1)
    #         latents_list.append(mu)

    #     return torch.cat(latents_list, dim=2)

    def encode_videos_streaming(self, videos: torch.Tensor, K: int = -1,
                             converged_mu=None, converged_caches=None) -> torch.Tensor:
        """
        Args:
            K: -1 = 不跳过；>=0 = iter 0 之后跑 K 步再跳到最后
            converged_mu, converged_caches: 预热好的零帧收敛状态。
                如果都不为 None，且 K>=0，则跳过 iter 1~K 的实际计算，
                直接用预热结果填充。
        """
        assert videos.dim() == 5, f"Expected [B,C,T,H,W], got {videos.shape}"
        # print(f"[encode_stream] T={videos.shape[2]} iters={1 + (videos.shape[2] - 1 + 3) // 4}")
        videos = videos.to(self.vae.dtype)
        B, C, T, H, W = videos.shape
        step = 4
        total_iters = 1 + (T - 1 + step - 1) // step

        # iter 0: first frame（必算，依赖 frame 0 内容）
        out = self.vae.encode_first_step(videos[:, :, :1, :, :])
        z_stats = out[0]
        enc_caches = out[1:]
        mu, _ = z_stats.chunk(2, dim=1)
        latents_list = [mu]

        if K < 0:
            # 不跳过
            start = 1
            for it in range(1, total_iters):
                end = min(start + step, T)
                out = self.vae.encode_next_step(videos[:, :, start:end, :, :], *enc_caches)
                z_stats = out[0]
                enc_caches = out[1:]
                mu, _ = z_stats.chunk(2, dim=1)
                latents_list.append(mu)
                start = end
        else:
            use_warm_cache = (converged_mu is not None and converged_caches is not None)

            if use_warm_cache:
                # 直接用预热的 cache，跳过 iter 1~K 的计算
                for _ in range(K):
                    latents_list.append(converged_mu)
                enc_caches = converged_caches
                mu = converged_mu
            else:
                # 没有预热缓存，正常跑 iter 1~K
                start = 1
                for it in range(1, K + 1):
                    end = min(start + step, T)
                    out = self.vae.encode_next_step(videos[:, :, start:end, :, :], *enc_caches)
                    z_stats = out[0]
                    enc_caches = out[1:]
                    mu, _ = z_stats.chunk(2, dim=1)
                    latents_list.append(mu)
                    start = end

            converged_mu_local = mu
            converged_caches_local = enc_caches

            # iter K+1 ~ total-2: 跳过
            skip_count = total_iters - 1 - (K + 1)
            for _ in range(skip_count):
                latents_list.append(converged_mu_local)

            # 最后一个 iter: 用收敛 cache 跑 frame 80
            last_start = 1 + step * (total_iters - 2)
            out = self.vae.encode_next_step(videos[:, :, last_start:, :, :], *converged_caches_local)
            z_stats = out[0]
            mu, _ = z_stats.chunk(2, dim=1)
            latents_list.append(mu)

        return torch.cat(latents_list, dim=2)

    def decode_latents_streaming(self, latents: torch.Tensor, decode_to_cpu: bool = True):
        # Same first/warmup/steady calls, cache handoff, and padding as before.
        # Normalize each output in its original dtype/device, then write it to
        # one FP32 destination. Avoid full-video cat and arithmetic copies.
        latents = latents.to(self.vae.dtype)
        B, C, T, H, W = latents.shape
        output = None
        written = 0
        ratio = int(self.vae.config.temporal_compression_ratio)

        def append_frames(frames):
            nonlocal output, written
            if decode_to_cpu:
                frames = frames.cpu()
            # Do not cast to float32 before this expression: BF16 rounding must
            # match the original whole-video expression exactly.
            normalized = (frames / 2 + 0.5).clamp(0, 1)
            chunk = normalized.cpu().float().numpy()
            if output is None:
                capacity = chunk.shape[2] + (T - 1) * ratio
                output = np.empty((chunk.shape[0], chunk.shape[1], capacity,
                                   chunk.shape[3], chunk.shape[4]), dtype=np.float32)
            end = written + chunk.shape[2]
            if end > output.shape[2]:
                raise RuntimeError("Decoder output exceeds configured temporal compression ratio")
            output[:, :, written:end] = chunk
            written = end

        out = self.vae.decode_first_step(latents[:, :, :1, :, :])
        dec_caches = out[1:]
        append_frames(out[0])
        out = None
        padded = False
        if T >= 2:
            out = self.vae.decode_warmup_step(latents[:, :, 1:2], *dec_caches)
            dec_caches = out[1:]
            append_frames(out[0])
            out = None
            decode_step = 1
            if (T - 2)//decode_step * 2 + 3 == T:
                pad = torch.zeros(B, C, 1, H, W,
                                  device=latents.device, dtype=latents.dtype)
                latents = torch.cat([latents, pad], dim=2)
                padded = True
            for t in range(2, T, decode_step):
                chunk = latents[:, :, t:t+decode_step, :, :]
                out = self.vae.decode_steady_step(chunk, *dec_caches)
                dec_caches = out[1:]
                append_frames(out[0])
                out = None
        dec_caches = None
        if padded:
            written = max(0, written - 4)
        return output[:, :, :written]

    def prepare_latents(
        self, batch_size, num_channels_latents, num_frames, height, width, dtype, device, generator, latents=None
    ):
        st_T = time.time()
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        shape = (
            batch_size,
            num_channels_latents,
            (num_frames - 1) // self.vae.config.temporal_compression_ratio + 1,
            height // self.vae.config.spatial_compression_ratio,
            width // self.vae.config.spatial_compression_ratio,
        )

        dtype = _activation_dtype(dtype)
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device=device, dtype=dtype)

        # scale the initial noise by the standard deviation required by the scheduler
        if hasattr(self.scheduler, "init_noise_sigma"):
            latents = latents * self.scheduler.init_noise_sigma
        
        # print("calling prepare latents", time.time()-st_T)
        return latents

    def _build_mask_cache(self, mask_video, video_length, height, width,
                        batch_size, weight_dtype, device, bs):
        weight_dtype = _activation_dtype(weight_dtype)
        mask_condition = self.mask_processor.preprocess(
            rearrange(mask_video, "b c f h w -> (b f) c h w"), height=height, width=width
        ).to(dtype=torch.float32)
        mask_condition = rearrange(mask_condition, "(b f) c h w -> b c f h w", f=video_length).to(device, weight_dtype)

        mask_cond_for_latent = torch.concat([
            torch.repeat_interleave(mask_condition[:, :, 0:1], repeats=4, dim=2),
            mask_condition[:, :, 1:]
        ], dim=2)
        mask_cond_for_latent = mask_cond_for_latent.view(
            bs, mask_cond_for_latent.shape[2] // 4, 4, height, width
        ).transpose(1, 2).to(device, weight_dtype)

        # Broadcast one boolean channel over RGB in masked_video multiplication.
        mask_tile = mask_condition < 0.5

        # ---- 预热零帧 cache ----
        converged_mu, converged_caches = self._warmup_zero_cache(
            video_length, height, width, weight_dtype, device, K=video_length//4-2
        )

        self._mask_cache = {
            'video_length': video_length,
            'sample_size': (height, width),   # 记录构建时的尺寸
            'mask_condition': mask_condition,
            'mask_cond_for_latent': mask_cond_for_latent,
            'mask_tile': mask_tile,
            'mask_latents': None,
            # 零帧 encode 缓存
            'K_zero': video_length//4-2,
            'converged_mu': converged_mu,
            'converged_caches': converged_caches,
        }

    @torch.no_grad()
    def _warmup_zero_cache(self, video_length, height, width, weight_dtype, device, K):
        """
        用零的 frame 0 跑 iter 0 + K 步零帧，得到收敛 cache 和 mu。
        这些缓存只取决于 (video_length, H, W)，与真实 frame 0 内容无关。
        """
        weight_dtype = _activation_dtype(weight_dtype)
        step = 4
        # 构造全零输入
        zero_first = torch.zeros(1, 3, 1, height, width, device=device, dtype=weight_dtype)
        zero_chunk = torch.zeros(1, 3, step, height, width, device=device, dtype=weight_dtype)

        # iter 0
        out = self.vae.encode_first_step(zero_first)
        enc_caches = out[1:]

        # iter 1 ~ K
        for _ in range(K):
            out = self.vae.encode_next_step(zero_chunk, *enc_caches)
            enc_caches = out[1:]

        # 收敛后再多跑一次拿 mu（也可以直接用最后一次的 mu）
        z_stats = out[0]
        converged_mu, _ = z_stats.chunk(2, dim=1)

        return converged_mu.detach().clone(), tuple(c.detach().clone() for c in enc_caches)


    def _ensure_mask_cache(self, mask_video, video_length, height, width,
                        batch_size, weight_dtype, device, bs):
        need_rebuild = (
            not hasattr(self, '_mask_cache')
            or self._mask_cache is None
            or self._mask_cache['sample_size'] != (height, width)
            or self._mask_cache.get('video_length') != video_length
        )
        if need_rebuild:
            self._mask_cache = None
            self._build_mask_cache(mask_video, video_length, height, width,
                                batch_size, weight_dtype, device, bs)
        
    def _prepare_mask_and_latents(self, init_video, mask_video, video_length,
                               height, width, batch_size, weight_dtype, device,
                               generator, do_classifier_free_guidance):
        weight_dtype = _activation_dtype(weight_dtype)
        # A previous inference may have parked these caches on the CPU.
        self._mask_cache = _move_tensor_tree(self._mask_cache, device)
        cache = self._mask_cache
        # Normalize mask tensors too if this persistent worker has an older cache.
        for key in ('mask_condition', 'mask_cond_for_latent', 'mask_latents'):
            if cache.get(key) is not None:
                cache[key] = cache[key].to(device=device, dtype=weight_dtype)
        mask_condition = cache['mask_condition']
        if init_video.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            init_video = init_video.to(torch.bfloat16)

        # Rebuild from mask_condition to support caches made by the older version.
        cache['mask_tile'] = mask_condition < 0.5
        masked_video = init_video * cache['mask_tile']

        # 把预热的 cache 传给 encode
        _, masked_video_latents = self.prepare_mask_latents(
            None, masked_video, batch_size, height, width,
            weight_dtype, device, generator, do_classifier_free_guidance,
            noise_aug_strength=None,
            K_skip=cache['K_zero'],
            converged_mu=cache['converged_mu'],
            converged_caches=cache['converged_caches'],
        )

        if cache['mask_latents'] is None:
            cache['mask_latents'] = resize_mask(
                1 - cache['mask_cond_for_latent'], masked_video_latents, True
            ).to(device, weight_dtype)
        mask_latents = cache['mask_latents']

        return mask_condition, masked_video_latents, mask_latents
    
    def prepare_mask_latents(
        self, mask, masked_image, batch_size, height, width, dtype, device, generator, do_classifier_free_guidance, noise_aug_strength,
        K_skip=-1, converged_mu=None, converged_caches=None
    ):
        # resize the mask to latents shape as we concatenate the mask to the latents
        # we do that before converting to dtype to avoid breaking in case we're using cpu_offload
        # and half precision
        st_T = time.time()

        if mask is not None:
            mask = mask.to(device=device, dtype=self.vae.dtype)
            bs = 1
            new_mask = []
            for i in range(0, mask.shape[0], bs):
                mask_bs = mask[i : i + bs]
                # mask_bs = self.vae.encode(mask_bs)[0]
                # mask_bs = mask_bs.mode()
                mask_bs = self.encode_videos_streaming(mask_bs)
                new_mask.append(mask_bs)
            mask = torch.cat(new_mask, dim = 0)
            # mask = mask * self.vae.config.scaling_factor

        if masked_image is not None:
            masked_image = masked_image.to(device=device, dtype=self.vae.dtype)
            bs = 1
            new_mask_pixel_values = []
            for i in range(0, masked_image.shape[0], bs):
                mask_pixel_values_bs = masked_image[i : i + bs]
                # mask_pixel_values_bs = self.vae.encode(mask_pixel_values_bs)[0]
                # mask_pixel_values_bs = mask_pixel_values_bs.mode()
                mask_pixel_values_bs = self.encode_videos_streaming(
                    mask_pixel_values_bs, K_skip,
                    converged_mu=converged_mu, converged_caches=converged_caches,
                )
                new_mask_pixel_values.append(mask_pixel_values_bs)
            masked_image_latents = torch.cat(new_mask_pixel_values, dim = 0)
            # masked_image_latents = masked_image_latents * self.vae.config.scaling_factor
        else:
            masked_image_latents = None

        # print("calling prepare mask latents", time.time()-st_T)

        return mask, masked_image_latents

    def prepare_control_latents(
        self, control, control_image, batch_size, height, width, dtype, device, generator, do_classifier_free_guidance
    ):
        # resize the control to latents shape as we concatenate the control to the latents
        # we do that before converting to dtype to avoid breaking in case we're using cpu_offload
        # and half precision
        st_T = time.time()

        if control is not None:
            control = control.to(device=device, dtype=dtype)
            bs = 1
            new_control = []
            for i in range(0, control.shape[0], bs):
                control_bs = control[i : i + bs]
                # control_bs = self.vae.encode(control_bs)[0]
                # control_bs = control_bs.mode()
                control_bs = self.encode_videos_streaming(control_bs)
                new_control.append(control_bs)
            control = torch.cat(new_control, dim = 0)
            # print(f"control: shape={control.shape} dtype={control.dtype} "
            #     f"device={control.device} contiguous={control.is_contiguous()} "
            #     f"min={control.min().item():.3f} max={control.max().item():.3f}"
            # )

        if control_image is not None:
            torch.cuda.synchronize(); t0 = time.time()
            control_image = control_image.to(device=device, dtype=dtype)
            torch.cuda.synchronize(); t1 = time.time()
            bs = 1
            new_control_pixel_values = []
            for i in range(0, control_image.shape[0], bs):
                control_pixel_values_bs = control_image[i : i + bs]
                # control_pixel_values_bs = self.vae.encode(control_pixel_values_bs)[0]
                # control_pixel_values_bs = control_pixel_values_bs.mode()
                control_pixel_values_bs = self.encode_videos_streaming(control_pixel_values_bs)
                new_control_pixel_values.append(control_pixel_values_bs)
            torch.cuda.synchronize(); t2 = time.time()
            control_image_latents = torch.cat(new_control_pixel_values, dim = 0)
            torch.cuda.synchronize(); t3 = time.time()

            # print(f"control_image: shape={control_image.shape} dtype={control_image.dtype} "
            #     f"device={control_image.device} contiguous={control_image.is_contiguous()} "
            #     f"min={control_image.min().item():.3f} max={control_image.max().item():.3f}"
            # )
            # print(f"  to_device={t1-t0:.3f}  encode={t2-t1:.3f}   cat={t3-t2:.3f}")
        else:
            control_image_latents = None

        # print("calling prepare control latents", time.time()-st_T)

        return control, control_image_latents

    # def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
    #     frames = self.vae.decode(latents.to(self.vae.dtype)).sample
    #     frames = (frames / 2 + 0.5).clamp(0, 1)
    #     # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
    #     frames = frames.cpu().float().numpy()
    #     return frames

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    # Copied from diffusers.pipelines.latte.pipeline_latte.LattePipeline.check_inputs
    def check_inputs(
        self,
        prompt,
        height,
        width,
        negative_prompt,
        callback_on_step_end_tensor_inputs,
        prompt_embeds=None,
        negative_prompt_embeds=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )
        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @property
    def interrupt(self):
        return self._interrupt

    # @torch.no_grad()
    @torch.inference_mode()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 480,
        width: int = 720,
        video: Union[torch.FloatTensor] = None,
        mask_video: Union[torch.FloatTensor] = None,
        control_video: Union[torch.FloatTensor] = None,
        control_camera_video: Union[torch.FloatTensor] = None,
        start_image: Union[torch.FloatTensor] = None,
        ref_image: Union[torch.FloatTensor] = None,
        num_frames: int = 49,
        num_inference_steps: int = 50,
        timesteps: Optional[List[int]] = None,
        guidance_scale: float = 6,
        num_videos_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: str = "numpy",
        return_dict: bool = False,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        boundary: float = 0.875,
        comfyui_progressbar: bool = False,
        shift: int = 5,
        low_vram: bool = True,
        decode_to_cpu: bool = True,
        mask_cache_video: Optional[torch.FloatTensor] = None,
        offload_vae: bool = True,
    ) -> Union[WanPipelineOutput, Tuple]:
        """
        Function invoked when calling the pipeline for generation.
        Args:

        Examples:

        Returns:

        """
        begin_T = time.time()
        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs
        num_videos_per_prompt = 1

        # 1. Check inputs. Raise error if not correct
        '''
        self.check_inputs(
            prompt,
            height,
            width,
            negative_prompt,
            callback_on_step_end_tensor_inputs,
            prompt_embeds,
            negative_prompt_embeds,
        )
        '''
        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._interrupt = False

        # 2. Default call parameters
        '''
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        '''
        batch_size = 1

        device = self._execution_device
        # Model weights may be FP8; video/latent/mask arithmetic and autocast
        # still use a regular floating-point compute dtype.
        weight_dtype = _activation_dtype(self.transformer.dtype)

        # ============================================================
        # Manual VAE offload:
        #   preprocessing / encode -> encoder + conv1 on GPU in low-VRAM mode
        #   transformer denoising  -> whole VAE on CPU
        #   streaming decode       -> decoder + conv2 on GPU in low-VRAM mode
        #   finish                 -> whole VAE on CPU
        #
        # VAE is detached from Accelerate sequential-offload hooks so
        # its internal streaming state is not disturbed layer by layer.
        # ============================================================
        if not getattr(self, "_manual_vae_offload_initialized", False):
            try:
                from accelerate.hooks import remove_hook_from_module
                remove_hook_from_module(self.vae, recurse=True)
                # print("[VAE offload] removed Accelerate hooks from VAE")
            except Exception as e:
                print(f"[VAE offload] warning: failed to remove VAE hooks: {e}")
            self._manual_vae_offload_initialized = True

        offload_vae = offload_vae or low_vram
        if low_vram and hasattr(self.vae, 'enable_low_vram'):
            self.vae.enable_low_vram(cache_device=None, cpu_shortcut=False)
        elif hasattr(self.vae, 'disable_low_vram'):
            self.vae.disable_low_vram()

        def _vae_device():
            try:
                return next(self.vae.parameters()).device
            except StopIteration:
                return torch.device("cpu")

        def _vae_to_gpu(phase='encode'):
            target = torch.device(device)
            if not offload_vae and _vae_device() == target:
                return
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize(target)
                torch.cuda.empty_cache()
            # _print_cuda_mem(f'before VAE {phase} -> GPU')
            prepare = getattr(self.vae, f'prepare_stream_{phase}', None)
            if low_vram and callable(prepare):
                prepare(target)
            else:
                self.vae.to(target)
            if torch.cuda.is_available():
                torch.cuda.synchronize(target)
            # _print_cuda_mem(f'after VAE {phase} -> GPU')

        def _vae_to_cpu():
            if not offload_vae:
                return
            import gc
            # Encoder and decoder may reside on different devices. Checking only
            # the first parameter would miss a decoder still on CUDA.
            self.vae.to('cpu')
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
                torch.cuda.empty_cache()
            # _print_cuda_mem('after whole VAE -> CPU')

        def _print_cuda_mem(tag):
            if torch.cuda.is_available():
                free_b, total_b = torch.cuda.mem_get_info(device)
                alloc_b = torch.cuda.memory_allocated(device)
                reserv_b = torch.cuda.memory_reserved(device)
                print(
                    f"[CUDA mem] {tag}: "
                    f"free={free_b / 1024**3:.2f}G, "
                    f"used_global={(total_b-free_b) / 1024**3:.2f}G, "
                    f"allocated={alloc_b / 1024**3:.2f}G, "
                    f"reserved={reserv_b / 1024**3:.2f}G"
                )

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        if do_classifier_free_guidance:
            # print('### negative_prompt_embeds:', negative_prompt_embeds[0].shape, negative_prompt_embeds[0].dtype)
            # print('### prompt_embeds:', prompt_embeds[0].shape, prompt_embeds[0].dtype)
            in_prompt_embeds = negative_prompt_embeds + prompt_embeds
        else:
            in_prompt_embeds = prompt_embeds
        # print("prompt:", time.time()-st_T)
        # 4. Prepare timesteps
        if isinstance(self.scheduler, FlowMatchEulerDiscreteScheduler):
            timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps, mu=1)
        elif isinstance(self.scheduler, FlowUniPCMultistepScheduler):
            self.scheduler.set_timesteps(num_inference_steps, device=device, shift=shift)
            timesteps = self.scheduler.timesteps
        elif isinstance(self.scheduler, FlowDPMSolverMultistepScheduler):
            sampling_sigmas = get_sampling_sigmas(num_inference_steps, shift)
            timesteps, _ = retrieve_timesteps(
                self.scheduler,
                device=device,
                sigmas=sampling_sigmas)
        else:
            timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        self._num_timesteps = len(timesteps)
        if comfyui_progressbar:
            from comfy.utils import ProgressBar
            pbar = ProgressBar(num_inference_steps + 2)

        st_T = time.time()
        # 5. Prepare latents.
        if video is not None:
            video_length = video.shape[2]
            # init_video = self.image_processor.preprocess(rearrange(video, "b c f h w -> (b f) c h w"), height=height, width=width) 
            # init_video = init_video.to(device=device, dtype=torch.float32)
            # init_video = rearrange(init_video, "(b f) c h w -> b c f h w", f=video_length)

            # torch.cuda.synchronize()
            # t0 = time.time()
            # x = rearrange(video, "b c f h w -> (b f) c h w")
            # print("rearrange 1:", time.time() - t0)

            # torch.cuda.synchronize()
            # t1 = time.time()
            # x = self.image_processor.preprocess(x, height=height, width=width)
            # print("preprocess:", time.time() - t1, x.device, x.dtype)

            # torch.cuda.synchronize()
            # t2 = time.time()
            # x = x.to(device=device, dtype=torch.float32)
            # print("to device:", time.time() - t2, x.device, x.dtype)

            # torch.cuda.synchronize()
            # t3 = time.time()
            # x = rearrange(x, "(b f) c h w -> b c f h w", f=video_length)
            # print("rearrange 2:", time.time() - t3)

            init_video = video * 2
            init_video.sub_(1)  # same rounded multiply/subtract, one fewer CPU temporary
            init_video = init_video.to(device=device, dtype=weight_dtype)
        else:
            init_video = None
        print("input video:", time.time()-st_T)

        st_T = time.time()
        latent_channels = self.vae.config.latent_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            latent_channels,
            num_frames,
            height,
            width,
            weight_dtype,
            device,
            generator,
            latents,
        )
        # print("prepare noise:", time.time()-st_T)

        if comfyui_progressbar:
            pbar.update(1)

        st_T = time.time()

        # Encoding uses the encoder and encode projection on GPU; the decoder
        # remains on CPU when stage-specific low-VRAM placement is available.
        _vae_to_gpu('encode')
        if mask_cache_video is not None:
            self._ensure_mask_cache(
                mask_cache_video, mask_cache_video.shape[2], height, width,
                batch_size, weight_dtype, device, batch_size)
            mask_cache_video = None
        # _print_cuda_mem("before VAE preprocessing")

        # Prepare mask latent variables
        if init_video is not None:
            if (mask_video == 255).all():
                mask_latents = torch.tile(
                    torch.zeros_like(latents)[:, :1].to(device, weight_dtype), [1, 4, 1, 1, 1]
                )
                masked_video_latents = torch.zeros_like(latents).to(device, weight_dtype)
                if self.vae.config.spatial_compression_ratio >= 16:
                    mask = torch.ones_like(latents).to(device, weight_dtype)[:, :1].to(device, weight_dtype)
            else:
                bs, _, video_length, height, width = video.size()
                # mask_condition = self.mask_processor.preprocess(rearrange(mask_video, "b c f h w -> (b f) c h w"), height=height, width=width) 
                # mask_condition = mask_condition.to(dtype=torch.float32)
                # mask_condition = rearrange(mask_condition, "(b f) c h w -> b c f h w", f=video_length)
                # # print("prepare mask_condition:", time.time()-st_T)

                # masked_video = init_video * (torch.tile(mask_condition, [1, 3, 1, 1, 1]) < 0.5)
                # _, masked_video_latents = self.prepare_mask_latents(
                #     None,
                #     masked_video,
                #     batch_size,
                #     height,
                #     width,
                #     weight_dtype,
                #     device,
                #     generator,
                #     do_classifier_free_guidance,
                #     noise_aug_strength=None,
                # )
                # # print("prepare masked_video_latents:", time.time()-st_T)
                
                # mask_condition = torch.concat(
                #     [
                #         torch.repeat_interleave(mask_condition[:, :, 0:1], repeats=4, dim=2), 
                #         mask_condition[:, :, 1:]
                #     ], dim=2
                # )
                # mask_condition = mask_condition.view(bs, mask_condition.shape[2] // 4, 4, height, width)
                # mask_condition = mask_condition.transpose(1, 2)
                # mask_latents = resize_mask(1 - mask_condition, masked_video_latents, True).to(device, weight_dtype) 
                # print("prepare mask_latents:", time.time()-st_T)
                mask_condition, masked_video_latents, mask_latents = self._prepare_mask_and_latents(
                    init_video, mask_video, video_length,
                    height, width, batch_size, weight_dtype, device,
                    generator, do_classifier_free_guidance
                )

                if self.vae.config.spatial_compression_ratio >= 16:
                    mask = F.interpolate(mask_condition[:, :1], size=latents.size()[-3:], mode='trilinear', align_corners=True).to(device, weight_dtype)  

                    if mask.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                        mask = mask.to(torch.bfloat16)
                    if masked_video_latents.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                        masked_video_latents = masked_video_latents.to(torch.bfloat16)
                    if latents.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                        latents = latents.to(torch.bfloat16)
                    if not mask[:, :, 0, :, :].any():
                        mask[:, :, 1:, :, :] = 1
                        latents = (1 - mask) * masked_video_latents + mask * latents
                        # print("reconstruct latents:", time.time()-st_T)
                    latents = latents.to(torch.bfloat16)

        st_T = time.time()
        # Prepare mask latent variables
        if control_camera_video is not None:
            control_latents = None
            # Rearrange dimensions
            # Concatenate and transpose dimensions
            control_camera_latents = torch.concat(
                [
                    torch.repeat_interleave(control_camera_video[:, :, 0:1], repeats=4, dim=2),
                    control_camera_video[:, :, 1:]
                ], dim=2
            ).transpose(1, 2)

            # Reshape, transpose, and view into desired shape
            b, f, c, h, w = control_camera_latents.shape
            control_camera_latents = control_camera_latents.contiguous().view(b, f // 4, 4, c, h, w).transpose(2, 3)
            control_camera_latents = control_camera_latents.contiguous().view(b, f // 4, c * 4, h, w).transpose(1, 2)
        elif control_video is not None:
            video_length = control_video.shape[2]
            control_video = self.image_processor.preprocess(rearrange(control_video, "b c f h w -> (b f) c h w"), height=height, width=width) 
            control_video = control_video.to(dtype=weight_dtype)
            control_video = rearrange(control_video, "(b f) c h w -> b c f h w", f=video_length)
            control_video_latents = self.prepare_control_latents(
                None,
                control_video,
                batch_size,
                height,
                width,
                weight_dtype,
                device,
                generator,
                do_classifier_free_guidance
            )[1]
            control_camera_latents = None
        else:
            control_video_latents = torch.zeros_like(latents).to(device, weight_dtype)
            control_camera_latents = None
        print("control video:", time.time()-st_T)

        st_T = time.time()
        if start_image is not None:
            video_length = start_image.shape[2]
            start_image = self.image_processor.preprocess(rearrange(start_image, "b c f h w -> (b f) c h w"), height=height, width=width) 
            start_image = start_image.to(dtype=torch.float32)
            start_image = rearrange(start_image, "(b f) c h w -> b c f h w", f=video_length)
            
            start_image_latentes = self.prepare_control_latents(
                None,
                start_image,
                batch_size,
                height,
                width,
                weight_dtype,
                device,
                generator,
                do_classifier_free_guidance
            )[1]

            start_image_latentes_conv_in = torch.zeros_like(latents)
            if latents.size()[2] != 1:
                start_image_latentes_conv_in[:, :, :1] = start_image_latentes
        else:
            start_image_latentes_conv_in = torch.zeros_like(latents)

        if self.transformer.config.get("add_ref_conv", False):
            if ref_image is not None:
                video_length = ref_image.shape[2]
                ref_image = self.image_processor.preprocess(rearrange(ref_image, "b c f h w -> (b f) c h w"), height=height, width=width) 
                ref_image = ref_image.to(dtype=torch.float32)
                ref_image = rearrange(ref_image, "(b f) c h w -> b c f h w", f=video_length)
                
                ref_image_latentes = self.prepare_control_latents(
                    None,
                    ref_image,
                    batch_size,
                    height,
                    width,
                    weight_dtype,
                    device,
                    generator,
                    do_classifier_free_guidance
                )[1]
                ref_image_latentes = ref_image_latentes[:, :, 0]
            else:
                ref_image_latentes = torch.zeros_like(latents)[:, :, 0]
        else:
            if ref_image is not None:
                raise ValueError("The add_ref_conv is False, but ref_image is not None")
            else:
                ref_image_latentes = None
        print("images:", time.time()-st_T)

        # Only latent-sized tensors are needed for denoising from this point.
        has_init_video = init_video is not None
        has_camera_control = control_camera_video is not None
        if low_vram:
            cache = getattr(self, "_mask_cache", None)
            cache_mib = _cuda_tensor_bytes(cache) / 1024**2
            if cache is not None:
                self._mask_cache = _move_tensor_tree(cache, "cpu")
            cache = None
            # These assignments release only this call's references. Inputs still
            # held by the caller cannot be freed by the pipeline.
            video = init_video = mask_video = control_video = None
            control_camera_video = start_image = ref_image = None
            mask_condition = start_image_latentes = None
            print(f"[low VRAM] VAE/mask cache parked on CPU: ~{cache_mib:.1f} MiB")

        _vae_to_cpu()
        # _print_cuda_mem("after VAE preprocessing offload")

        if comfyui_progressbar:
            pbar.update(1)

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        target_shape = (self.vae.config.latent_channels, (num_frames - 1) // self.vae.config.temporal_compression_ratio + 1, width // self.vae.config.spatial_compression_ratio, height // self.vae.config.spatial_compression_ratio)
        seq_len = math.ceil((target_shape[2] * target_shape[3]) / (self.transformer.config.patch_size[1] * self.transformer.config.patch_size[2]) * target_shape[1]) 
        # 7. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self.transformer.num_inference_steps = num_inference_steps
        print("total preparation:", time.time()-begin_T)

        condition_parts = []
        if not has_camera_control:
            condition_parts.append(control_video_latents.to(device=device, dtype=weight_dtype))
        if has_init_video:
            condition_parts.extend([
                mask_latents.to(device=device, dtype=weight_dtype),
                masked_video_latents.to(device=device, dtype=weight_dtype),
            ])
        else:
            condition_parts.append(start_image_latentes_conv_in.to(device=device, dtype=weight_dtype))
        condition_latents = torch.cat(condition_parts, dim=1) if condition_parts else None
        condition_parts.clear()
        condition_camera = control_camera_latents.to(device=device, dtype=weight_dtype) if has_camera_control else None
        condition_ref = ref_image_latentes.to(device=device, dtype=weight_dtype) if ref_image_latentes is not None else None
        if do_classifier_free_guidance:
            if condition_latents is not None:
                condition_latents = torch.cat([condition_latents, condition_latents], dim=0)
            if condition_camera is not None:
                condition_camera = torch.cat([condition_camera, condition_camera], dim=0)
            if condition_ref is not None:
                condition_ref = torch.cat([condition_ref, condition_ref], dim=0)
        # The packed conditioning now owns the tensors required by Transformer.
        control_video_latents = control_camera_latents = ref_image_latentes = None
        start_image_latentes_conv_in = mask_latents = None
        needs_mask_blend = has_init_video and self.vae.config.spatial_compression_ratio >= 16
        if not needs_mask_blend:
            masked_video_latents = None
        if low_vram and torch.cuda.is_available():
            torch.cuda.empty_cache()
        # _print_cuda_mem("before Transformer / conditioning packed")

        st_T = time.time()
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                self.transformer.current_steps = i

                if self.interrupt:
                    continue

                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                if hasattr(self.scheduler, "scale_model_input"):
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                latent_model_input = latent_model_input.to(device=device, dtype=weight_dtype)

                control_latents_input = condition_latents
                control_camera_latents_input = condition_camera
                full_ref = condition_ref

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                if self.vae.config.spatial_compression_ratio >= 16 and has_init_video:
                    temp_ts = ((mask[0][0][:, ::2, ::2]) * t).flatten()
                    temp_ts = torch.cat([
                        temp_ts,
                        temp_ts.new_ones(seq_len - temp_ts.size(0)) * t
                    ])
                    temp_ts = temp_ts.unsqueeze(0)
                    timestep = temp_ts.expand(latent_model_input.shape[0], temp_ts.size(1))
                else:
                    timestep = t.expand(latent_model_input.shape[0])
                
                if self.transformer_2 is not None:
                    if t >= boundary * self.scheduler.config.num_train_timesteps:
                        local_transformer = self.transformer_2
                    else:
                        local_transformer = self.transformer
                else:
                    local_transformer = self.transformer
                
                # predict noise model_output
                with torch.cuda.device(device), torch.amp.autocast(device_type='cuda', dtype=weight_dtype):
                    noise_pred = local_transformer(
                        x=latent_model_input,
                        context=in_prompt_embeds,
                        t=timestep,
                        seq_len=seq_len,
                        y=control_latents_input,
                        y_camera=control_camera_latents_input, 
                        full_ref=full_ref,
                    )

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                if has_init_video and self.vae.config.spatial_compression_ratio >= 16 and not mask[:, :, 0, :, :].any():
                    latents = (1 - mask) * masked_video_latents + mask * latents

                # chunk() returns views; clear both views and their parent output
                # once scheduler/guidance have consumed them.
                latent_model_input = timestep = noise_pred = None
                noise_pred_uncond = noise_pred_text = None

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                if comfyui_progressbar:
                    pbar.update(1)
        print("denoising:", time.time()-st_T)

        condition_latents = condition_camera = condition_ref = None
        control_latents_input = control_camera_latents_input = full_ref = None
        latent_model_input = timestep = noise_pred = None
        noise_pred_uncond = noise_pred_text = None
        masked_video_latents = mask = None
        if low_vram:
            from accelerate.hooks import CpuOffload
            for user_hook in getattr(self, '_all_hooks', ()):
                if getattr(user_hook, 'model', None) is self.vae:
                    continue
                offload = getattr(user_hook, 'offload', None)
                if callable(offload):
                    offload()
            final_hook = getattr(self, 'final_offload_hook', None)
            final_offload = getattr(final_hook, 'offload', None)
            if callable(final_offload):
                final_offload()
            for name in ('transformer', 'transformer_2'):
                transformer = getattr(self, name, None)
                if transformer is None:
                    continue
                hook = getattr(transformer, '_hf_hook', None)
                if isinstance(hook, CpuOffload):
                    hook.init_hook(transformer)
                gpu_bytes = sum(
                    tensor.numel() * tensor.element_size()
                    for tensor in (*transformer.parameters(), *transformer.buffers())
                    if tensor.device.type == 'cuda'
                )
                print(f'[decode preparation] {name}: CUDA parameters+buffers={gpu_bytes / 1024**3:.3f} GiB')

        # Transformer sequential-offload hooks manage Transformer layers.
        # Clear cached blocks before reloading the full VAE for decode.
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
            if offload_vae:
                torch.cuda.empty_cache()
        # _print_cuda_mem("after denoising / before VAE decode")

        st_T = time.time()
        if output_type == "latent":
            video = latents
        else:
            _vae_to_gpu('decode')
            # _print_cuda_mem("VAE decode start")
            try:
                video = self.decode_latents_streaming(latents, decode_to_cpu=decode_to_cpu)
                if output_type != "numpy":
                    video = self.video_processor.postprocess_video(
                        video=video,
                        output_type=output_type,
                    )
            finally:
                # Offload all VAE modules, including a decoder on a different
                # device from the first (encoder) parameter.
                _vae_to_cpu()
                # _print_cuda_mem("after VAE decode offload")

        print(output_type, "decode:", time.time()-st_T)

        if not return_dict and isinstance(video, np.ndarray):
            video = torch.from_numpy(video)

        return WanPipelineOutput(videos=video)
