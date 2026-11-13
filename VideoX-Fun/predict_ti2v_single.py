import sys
import time
from pathlib import Path

import cv2
import torch
from loguru import logger
from omegaconf import OmegaConf

video_x_fun_root = str(Path(__file__).resolve().parent)
if video_x_fun_root not in sys.path:
    sys.path.insert(0, video_x_fun_root)

from videox_fun.models import AutoencoderKLWan3_8, AutoTokenizer, Wan2_2Transformer3DModel, WanT5EncoderModel
from videox_fun.pipeline.pipeline_wan2_2_ti2v_single import Wan2_2TI2VPipeline
from videox_fun.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from videox_fun.utils.lora_utils_new import merge_lora
from videox_fun.utils.utils_new import filter_kwargs, get_image_to_video_latent


class Wan_5b:
    """Single-clip TI2V predictor with the same worker interface as LivePhoto."""

    def __init__(self, args, device="cuda"):
        self.device = torch.device(device)
        weight_dtype = torch.bfloat16
        config = OmegaConf.load(args.config_path)
        transformer_kwargs = config["transformer_additional_kwargs"]
        if transformer_kwargs.get("transformer_combination_type", "single") != "single":
            raise ValueError("This predictor expects a single Wan2.2-TI2V-5B transformer")
        model_dir = Path(args.model_name)
        transformer = Wan2_2Transformer3DModel.from_pretrained(
            str(model_dir / transformer_kwargs.get("transformer_low_noise_model_subpath", "transformer")),
            transformer_additional_kwargs=OmegaConf.to_container(transformer_kwargs),
            low_cpu_mem_usage=True,
            torch_dtype=weight_dtype,
        )
        vae_kwargs = config["vae_kwargs"]
        vae = AutoencoderKLWan3_8.from_pretrained(
            str(model_dir / vae_kwargs.get("vae_subpath", "vae")),
            additional_kwargs=OmegaConf.to_container(vae_kwargs),
        ).to(weight_dtype)
        text_kwargs = config["text_encoder_kwargs"]
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir / text_kwargs.get("tokenizer_subpath", "tokenizer")))
        text_encoder = WanT5EncoderModel.from_pretrained(
            str(model_dir / text_kwargs.get("text_encoder_subpath", "text_encoder")),
            additional_kwargs=OmegaConf.to_container(text_kwargs),
            low_cpu_mem_usage=True,
            torch_dtype=weight_dtype,
        )
        scheduler_kwargs = OmegaConf.to_container(config["scheduler_kwargs"])
        scheduler_kwargs["shift"] = 1
        scheduler = FlowUniPCMultistepScheduler(
            **filter_kwargs(FlowUniPCMultistepScheduler, scheduler_kwargs)
        )
        pipeline = Wan2_2TI2VPipeline(
            transformer=transformer,
            transformer_2=None,
            vae=vae,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            scheduler=scheduler,
        )
        self.pipeline = merge_lora(
            pipeline, args.lora_path, args.lora_weight, device=self.device, dtype=weight_dtype
        )
        if args.model_cpu_offload:
            self.pipeline.enable_model_cpu_offload(device=self.device)
        else:
            self.pipeline.to(device=self.device)
        self.temporal_stride = vae.temporal_compression_ratio
        self.spatial_alignment = vae.spatial_compression_ratio * 2
        self.boundary = transformer_kwargs.get("boundary", 0.875)
        logger.info("TI2V model ready on {} (model CPU offload: {})", self.device, args.model_cpu_offload)

    @torch.inference_mode()
    def infer(self, args):
        started = time.perf_counter()
        if (args.video_length - 1) % self.temporal_stride:
            raise ValueError(f"TI2V video length must satisfy 1 + n * {self.temporal_stride}")
        width, height = args.img_size
        scale = min(args.short_edge / min(width, height), args.long_edge / max(width, height))
        alignment = self.spatial_alignment
        sample_size = [max(alignment, round(edge * scale / alignment) * alignment) for edge in (height, width)]
        logger.info("TI2V sample size: {} (H, W), {} frames", sample_size, args.video_length)
        input_video, input_mask, _ = get_image_to_video_latent(
            args.image_path, None, video_length=args.video_length, sample_size=sample_size
        )
        generator = torch.Generator(device=self.device).manual_seed(args.seed)
        prepared = time.perf_counter()
        sample = self.pipeline(
            args.prompt,
            num_frames=args.video_length,
            negative_prompt=args.negative_prompt,
            height=sample_size[0],
            width=sample_size[1],
            generator=generator,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            boundary=self.boundary,
            video=input_video,
            mask_video=input_mask,
            shift=args.shift,
        ).videos
        if sample.shape[2] != args.video_length:
            raise RuntimeError(f"TI2V returned {sample.shape[2]} frames, expected {args.video_length}")
        inferred = time.perf_counter()
        logger.info("TI2V prepare: {:.2f}s, infer: {:.2f}s", prepared - started, inferred - prepared)
        selected = set(args.frame_indices)
        output_dir = Path(args.save_path)
        height, width = sample.shape[-2:]
        writer = cv2.VideoWriter(
            str(output_dir / "ti2v.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
            args.video_fps, (width, height),
        )
        try:
            if not writer.isOpened():
                raise RuntimeError("Could not open TI2V MP4 writer")
            for index, frame in enumerate(sample[0].unbind(dim=1)):
                rgb = frame.permute(1, 2, 0).clamp(0, 1).mul(255).round().to(torch.uint8).cpu().numpy()
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                writer.write(bgr)
                if index in selected and not cv2.imwrite(str(output_dir / "image" / f"{index:05d}.png"), bgr):
                    raise RuntimeError(f"Could not save TI2V frame {index}")
        finally:
            writer.release()
        logger.info("TI2V save video and sampled images: {:.2f}s", time.perf_counter() - inferred)
