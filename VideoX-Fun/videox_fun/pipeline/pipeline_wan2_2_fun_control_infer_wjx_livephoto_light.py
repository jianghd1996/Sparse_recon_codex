import inspect
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
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


def get_continuous_selected_indices(num_total_frames, num_selected_frames):
    if num_total_frames >= num_selected_frames:
            # 情况 1：原视频足够长，计算居中的起始点，切出连续的帧
        start_idx = (num_total_frames - num_selected_frames) // 2
        selected_indices = np.arange(start_idx, start_idx + num_selected_frames)
    else:
        # 情况 2：边界情况（原视频总帧数不足）
        # 保持原视频居中，两边对称地用首尾帧进行边缘填充（Edge Padding）
        core_indices = np.arange(num_total_frames)
        # 2. 计算左边和右边各需要填充多少帧
        total_padding = num_selected_frames - num_total_frames
        left_padding_len = total_padding // 2
        right_padding_len = total_padding - left_padding_len
        # 3. 左边全部用第 0 帧填充，右边全部用最后一帧（num_total_frames - 1）填充
        left_padding = np.zeros(left_padding_len, dtype=int)
        right_padding = np.ones(right_padding_len, dtype=int) * (num_total_frames - 1)
        # 4. 拼接成最终长度为 num_selected_frames 的连续索引
        selected_indices = np.concatenate([left_padding, core_indices, right_padding])
    return selected_indices


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


class Wan2_2LivePhotoPipeline_infer(DiffusionPipeline):
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
        tokenizer: AutoTokenizer,
        text_encoder: WanT5EncoderModel,
        vae: AutoencoderKLWan,
        transformer: Wan2_2Transformer3DModel,
        transformer_2: Wan2_2Transformer3DModel = None,
        scheduler: FlowMatchEulerDiscreteScheduler = None,
    ):
        super().__init__()

        self.register_modules(
            tokenizer=tokenizer, text_encoder=text_encoder, vae=vae, transformer=transformer, 
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

    def prepare_latents(
        self, batch_size, num_channels_latents, num_frames, num_selected_frames, height, width, dtype, device, generator, latents=None, is_frame_cat_compress_mode=False
    ):
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
        if is_frame_cat_compress_mode:
            shape = (
                batch_size,
                num_channels_latents,
                ((num_frames - 1) // self.vae.config.temporal_compression_ratio + 1) + ((num_selected_frames - 1) // self.vae.config.temporal_compression_ratio + 1),
                height // self.vae.config.spatial_compression_ratio,
                width // self.vae.config.spatial_compression_ratio,
            )
        else:
            shape = (
                batch_size,
                num_channels_latents,
                ((num_frames - 1) // self.vae.config.temporal_compression_ratio + 1) + num_selected_frames,
                height // self.vae.config.spatial_compression_ratio,
                width // self.vae.config.spatial_compression_ratio,
            )
        
        if dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            dtype = torch.bfloat16
        
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        if hasattr(self.scheduler, "init_noise_sigma"):
            latents = latents * self.scheduler.init_noise_sigma
        return latents


    def prepare_video_latents(
        self, control_image, batch_size, height, width, dtype, device,
    ):
        # resize the control to latents shape as we concatenate the control to the latents
        # we do that before converting to dtype to avoid breaking in case we're using cpu_offload
        # and half precision
        if control_image is not None:
            control_image = control_image.to(device=device, dtype=dtype)
            bs = 1
            new_control_pixel_values = []
            for i in range(0, control_image.shape[0], bs):
                control_pixel_values_bs = self.encode_videos_streaming(control_image[i : i + bs])
                new_control_pixel_values.append(control_pixel_values_bs)
            control_image_latents = torch.cat(new_control_pixel_values, dim = 0)
        else:
            control_image_latents = None

        return control_image_latents

    def prepare_control_latents(
        self, control, control_image, batch_size, height, width, dtype, device, generator, do_classifier_free_guidance
    ):
        # resize the control to latents shape as we concatenate the control to the latents
        # we do that before converting to dtype to avoid breaking in case we're using cpu_offload
        # and half precision

        if control is not None:
            control = control.to(device=device, dtype=dtype)
            bs = 1
            new_control = []
            for i in range(0, control.shape[0], bs):
                control_bs = control[i : i + bs]
                control_bs = self.vae.encode(control_bs)[0]
                control_bs = control_bs.mode()
                new_control.append(control_bs)
            control = torch.cat(new_control, dim = 0)

        if control_image is not None:
            control_image = control_image.to(device=device, dtype=dtype)
            bs = 1
            new_control_pixel_values = []
            for i in range(0, control_image.shape[0], bs):
                control_pixel_values_bs = control_image[i : i + bs]
                control_pixel_values_bs = self.vae.encode(control_pixel_values_bs)[0]
                control_pixel_values_bs = control_pixel_values_bs.mode()
                new_control_pixel_values.append(control_pixel_values_bs)
            control_image_latents = torch.cat(new_control_pixel_values, dim = 0)
        else:
            control_image_latents = None

        return control, control_image_latents

    # def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
    #     frames = self.vae.decode(latents.to(self.vae.dtype)).sample
    #     frames = (frames / 2 + 0.5).clamp(0, 1)
    #     # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
    #     frames = frames.cpu().float().numpy()
    #     return frames
    def decode_latents_streaming(self, latents: torch.Tensor):
        latents = latents.to(self.vae.dtype)
        B, C, T, H, W = latents.shape

        first_chunk = latents[:, :, :1, :, :]
        # t0 = time.time()
        out = self.vae.decode_first_step(first_chunk)
        # print("decode_first_step:", time.time() - t0)

        frames = out[0]
        dec_caches = out[1:]
        # print("first output shape:", tuple(frames.shape))
        # print_cache_stats("after first", dec_caches)

        frames_list = [frames]

        padded = False
        if T >= 2:
            out = self.vae.decode_warmup_step(latents[:, :, 1:2], *dec_caches)
            frames = out[0]
            dec_caches = out[1:]
            frames_list.append(frames)
            decode_step = 1
            if (T - 2)//decode_step * 2 + 3 == T:
                pad = torch.zeros(B, C, 1, H, W,
                                    device=latents.device, dtype=latents.dtype)
                latents = torch.cat([latents, pad], dim=2)
                padded = True

            for t in range(2, T, decode_step):
                chunk = latents[:, :, t:t+decode_step, :, :]
                # t1 = time.time()
                out = self.vae.decode_steady_step(chunk, *dec_caches)
                # dt = time.time() - t1

                frames = out[0]
                dec_caches = out[1:]

                # print(f"decode_next_step {t}: {dt:.4f}s, output={tuple(frames.shape)}")
                # print_cache_stats(f"after step {t}", dec_caches)

                frames_list.append(frames)
        

        frames = torch.cat(frames_list, dim=2)
        if padded:
            frames = frames[:,:,:-4]
        frames = (frames / 2 + 0.5).clamp(0, 1)
        return frames.cpu().float().numpy()

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
        cond_video: Union[torch.FloatTensor] = None,
        input_video: Union[torch.FloatTensor] = None,
        input_image: Union[torch.FloatTensor] = None,
        cond_index: int = 0,
        num_frames: int = 49,
        num_inference_steps: int = 50,
        timesteps: Optional[List[int]] = None,
        guidance_scale: float = 5.0,
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
        camera_movement=None,
        num_selected_frames=5,
        use_fixed_key_frames=False,
        reverse_width=False,
        denoising_strength=1.0,
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
        self.check_inputs(
            prompt,
            height,
            width,
            negative_prompt,
            callback_on_step_end_tensor_inputs,
            prompt_embeds,
            negative_prompt_embeds,
        )
        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._interrupt = False

        # 2. Default call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        weight_dtype = self.text_encoder.dtype

        # ============================================================
        # Manual VAE offload
        #
        # Transformer/text encoder keep their existing sequential CPU
        # offload behavior. The custom streaming VAE is detached from
        # Accelerate hooks and moved only as a complete module:
        #
        #   input latent encode  -> VAE on GPU
        #   transformer denoise  -> VAE on CPU
        #   streaming decode     -> VAE on GPU
        #   call finished        -> VAE on CPU
        # With offload_vae=False, the VAE stays on GPU until the worker releases it.
        #
        # This avoids layer-wise offload inside the stateful LightVAE.
        # ============================================================
        if not getattr(self, "_manual_vae_offload_initialized", False):
            try:
                from accelerate.hooks import remove_hook_from_module
                remove_hook_from_module(self.vae, recurse=True)
                print("[VAE offload] removed Accelerate hooks from VAE")
            except Exception as e:
                print(f"[VAE offload] warning: failed to remove VAE hooks: {e}")
            self._manual_vae_offload_initialized = True

        def _vae_device():
            try:
                return next(self.vae.parameters()).device
            except StopIteration:
                return torch.device("cpu")

        def _vae_to_gpu():
            target = torch.device(device)
            if _vae_device() != target:
                if torch.cuda.is_available():
                    torch.cuda.synchronize(target)
                self.vae.to(target)
                if torch.cuda.is_available():
                    torch.cuda.synchronize(target)
            print(f"[VAE offload] VAE -> {_vae_device()}")

        def _vae_to_cpu():
            if not offload_vae:
                return
            current = _vae_device()
            if current.type != "cpu":
                if torch.cuda.is_available():
                    torch.cuda.synchronize(current)
                self.vae.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"[VAE offload] VAE -> {_vae_device()}")

        def _print_cuda_mem(tag):
            if torch.cuda.is_available():
                free_b, total_b = torch.cuda.mem_get_info(device)
                alloc_b = torch.cuda.memory_allocated(device)
                reserved_b = torch.cuda.memory_reserved(device)
                print(
                    f"[CUDA mem] {tag}: "
                    f"free={free_b / 1024**3:.2f}G, "
                    f"used_global={(total_b-free_b) / 1024**3:.2f}G, "
                    f"allocated={alloc_b / 1024**3:.2f}G, "
                    f"reserved={reserved_b / 1024**3:.2f}G"
                )

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        st_T = time.time()
        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            negative_prompt,
            do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )
        if do_classifier_free_guidance:
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
        
        is_frame_cat_mode = 'frame_cat' in camera_movement
        is_frame_cat_compress_mode = 'frame_cat_compress' in camera_movement
        
        real_latent_length = (num_frames - 1) // self.vae.config.temporal_compression_ratio + 1
        if is_frame_cat_mode:
            if is_frame_cat_compress_mode:
                cond_latent_length = (num_selected_frames - 1) // self.vae_stride[0] + 1
            else:
                cond_latent_length = num_selected_frames

        st_T = time.time()

        # Keep the complete VAE on GPU for all input-side latent encoding.
        _vae_to_gpu()
        # _print_cuda_mem("before VAE latent preparation")

        # 5. Prepare latents.
        if input_image is not None or input_video is not None:
            # init video (list of images) [(H W C)] image (H W C)
            # used video (B, C, F, H, W) image (B C, 1, H, W)
            if input_video is not None:
                if reverse_width:
                    start_image = torch.flip(input_video[:,:,cond_index:(cond_index+1)], dims=[-1]) 
                else:
                    start_image = input_video[:,:,cond_index:(cond_index+1), :, :]
            else:
                if reverse_width:
                    start_image = torch.flip(input_image, dims=[-1])
                else:
                    start_image = input_image

            start_image = self.image_processor.preprocess(rearrange(start_image, "b c f h w -> (b f) c h w"), height=height, width=width) 
            start_image = start_image.to(dtype=torch.float32)
            start_image = rearrange(start_image, "(b f) c h w -> b c f h w", f=1)

            start_image_latent = self.prepare_video_latents(
                start_image,
                batch_size * num_videos_per_prompt,
                height,
                width,
                weight_dtype,
                device,
            )
        else:
            start_image_latent = None
        print("input video:", time.time()-st_T)

        latent_channels = self.vae.config.latent_channels # noise
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            latent_channels,
            num_frames,
            num_selected_frames,
            height,
            width,
            weight_dtype,
            device,
            generator,
            latents,
            is_frame_cat_compress_mode,
        )
        if comfyui_progressbar:
            pbar.update(1)

        st_T = time.time()
        # Prepare cond latent variables
        if is_frame_cat_mode:
            if cond_video is not None:
                cond_video_length = cond_video.shape[2]
                cond_video = self.image_processor.preprocess(rearrange(cond_video, "b c f h w -> (b f) c h w"), height=height, width=width) 
                cond_video = cond_video.to(dtype=torch.float32)
                cond_video = rearrange(cond_video, "(b f) c h w -> b c f h w", f=cond_video_length)

                if use_fixed_key_frames:
                    selected_indices = [index for i in range(num_selected_frames)]
                else:
                    if is_frame_cat_compress_mode:
                        selected_indices = get_continuous_selected_indices(cond_video_length, num_selected_frames)
                    else:
                        selected_indices = np.linspace(0, cond_video_length - 1, num_selected_frames, dtype=int)
                    selected_indices = [i for i in selected_indices]
                if is_frame_cat_compress_mode:
                    # cond_latent = self.prepare_video_latents(cond_video[:,:,selected_indices])
                    cond_latent = self.prepare_video_latents(
                        cond_video[:,:,selected_indices],
                        batch_size * num_videos_per_prompt,
                        height,
                        width,
                        weight_dtype,
                        device,
                    )
                    print(f'is_frame_cat_compress_mode {is_frame_cat_compress_mode}, noise.shape {noise.shape}, cond_latent.shape {cond_latent.shape}')
                    print(f'selected_indices {selected_indices}')
                else:
                    cond_latent = []
                    for i in selected_indiceds:
                        # cond_tmp = self.prepare_video_latents(cond_video[:,:,i:i+1])
                        cond_latent = self.prepare_video_latents(
                            cond_video[:,:,i:i+1],
                            batch_size * num_videos_per_prompt,
                            height,
                            width,
                            weight_dtype,
                            device,
                        )
                        cond_latent.append(cond_tmp)
                    cond_latent = torch.cat(cond_latent, dim=2)
            else:
                cond_latent = torch.cat([start_image_latent] * num_selected_frames, dim=2)

            
        print("cond video:", time.time()-st_T)

        # All VAE input-side encoding is finished. Reclaim VRAM before
        # Transformer denoising by moving the whole VAE to CPU.
        _vae_to_cpu()
        # _print_cuda_mem("after VAE encode offload")

        st_T = time.time()
        # Prepare mask latent variables
        if is_frame_cat_mode:
            if is_frame_cat_compress_mode:
                mask = torch.ones(
                    (1, 1, 
                    latents.shape[2],
                    latents.shape[3],
                    latents.shape[4]),
                    dtype=weight_dtype,
                    device=device,
                )
                mask[0, 0, 0].fill_(0.0)
                mask[0, -((num_selected_frames - 1) // self.vae.config.temporal_compression_ratio + 1):].fill_(0.0)
            else:
                mask = torch.ones(
                    (1, 1, 
                    latents.shape[2],
                    latents.shape[3],
                    latents.shape[4]),
                    dtype=weight_dtype,
                    device=device,
                )
                mask[0, 0, 0].fill_(0.0)
                mask[0, 0, -num_selected_frames:].fill_(0.0)

        latents[:, :, 0:1] = start_image_latent
        if is_frame_cat_mode:
            if is_frame_cat_compress_mode:
                latents[:, :, 0:1] = cond_latent[:, :, 0:1]
                latents[:, :, -((num_selected_frames - 1) // self.vae.config.temporal_compression_ratio + 1):] = cond_latent
            else:
                latents[:, :, -num_selected_frames:] = cond_latent
            
        print("mask generate:", time.time()-st_T)

        if comfyui_progressbar:
            pbar.update(1)

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        target_shape = (self.vae.config.latent_channels, latents.shape[2], width // self.vae.config.spatial_compression_ratio, height // self.vae.config.spatial_compression_ratio)
        seq_len = math.ceil((target_shape[2] * target_shape[3]) / (self.transformer.config.patch_size[1] * self.transformer.config.patch_size[2]) * target_shape[1]) 
        # print('### seq_len:', seq_len)
        # print('### latents:', latents.shape, latents.dtype, latents.min(), latents.max())
        # print('### mask:', mask.shape, mask.dtype, mask.min(), mask.max())
        # 7. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self.transformer.num_inference_steps = num_inference_steps
        print("latents preparation time:", time.time()-begin_T)

        st_T = time.time()
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                self.transformer.current_steps = i

                if self.interrupt:
                    continue

                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                if hasattr(self.scheduler, "scale_model_input"):
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                if self.vae.config.spatial_compression_ratio >= 16:
                    temp_ts = (mask[:,:,:,::2,::2] * t).flatten()
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
                    )

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                # if self.vae.config.spatial_compression_ratio >= 16 and not mask[:, :, 0, :, :].any():
                #     latents = (1 - mask) * masked_video_latents + mask * latents

                latents[:, :, 0:1] = start_image_latent
                if is_frame_cat_mode:
                    if is_frame_cat_compress_mode:
                        latents[:, :, 0:1] = cond_latent[:, :, 0:1]
                        latents[:, :, -cond_latent_length:] = cond_latent
                    else:
                        latents[:, :, -cond_latent_length:] = cond_latent

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
        latents = latents[:,:,:real_latent_length]

        # Transformer remains managed by sequential-offload hooks.
        # Clear cached allocator blocks before loading an offloaded VAE back.
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
            if offload_vae:
                torch.cuda.empty_cache()
        # _print_cuda_mem("after denoising / before VAE decode")

        st_T = time.time()
        if output_type == "latent":
            video = latents
        else:
            _vae_to_gpu()
            # _print_cuda_mem("VAE decode start")
            try:
                video = self.decode_latents_streaming(latents)
                if output_type != "numpy":
                    video = self.video_processor.postprocess_video(
                        video=video,
                        output_type=output_type,
                    )
            finally:
                # Do not offload individual VAE layers during streaming decode.
                # Move the whole VAE only after decode is completely finished.
                _vae_to_cpu()
                # _print_cuda_mem("after VAE decode offload")

        print("unpatchify:", time.time()-st_T)

        if not return_dict:
            video = torch.from_numpy(video)

        return WanPipelineOutput(videos=video)
