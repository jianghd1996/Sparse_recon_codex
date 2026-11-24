"""Dense BF16 Wan Control inference with native Wan2.2 VAE and LoRA."""

import os
import sys
import time

import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from loguru import logger
from omegaconf import OmegaConf

current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_file_dir)
for path in (current_file_dir, project_root):
    if path not in sys.path:
        sys.path.insert(0, path)

from resident_host_memory import release_free_host_memory, log_host_memory
from videox_fun.models.wan_transformer3d import Wan2_2Transformer3DModel
from videox_fun.models.wan_vae3_8 import AutoencoderKLWan3_8
from videox_fun.pipeline.pipeline_wan2_2_fun_control_infer_wjx_dense import Wan2_2FunControlPipeline_infer
from videox_fun.utils.lora_utils_new import merge_lora
from videox_fun.utils.utils import (
    filter_kwargs, get_image_latent, get_inpaint_video_latent,
    make_save_videos_grid, _write_video_opencv,
    save_selected_images_from_video_tensor,
)

save_videos_grid = make_save_videos_grid(
    _write_video_opencv, color_transfer_fn=globals().get("color_transfer"),
)


def group_video_segments(indices, segment_plan, frames, batch_size):
    # Native attention is validated one video per forward. GPU workers still
    # generate different segments in parallel, as on the source branch.
    for index in indices:
        yield [index]


class Wan_5b:
    manages_vae_placement = True

    def __init__(self, args, device="cuda"):
        self.device = torch.device(device)
        self.weight_dtype = torch.bfloat16
        self.video_batch_size = getattr(args, "video_batch_size", 1)
        if self.video_batch_size != 1:
            raise ValueError("Dense BF16 inference currently requires video_batch_size=1")
        if getattr(args, "transformer_path", None) is not None:
            raise ValueError("Dense baseline loads the base model plus LoRA; transformer_path must be None")
        model_offload = getattr(args, "offload_model", False)
        self.offload_vae = getattr(args, "offload_vae", True) or model_offload
        self.low_vram = model_offload
        config = OmegaConf.load(args.config_path)
        if config["transformer_additional_kwargs"].get("transformer_combination_type", "single") != "single":
            raise ValueError("This predictor targets the single-transformer Wan2.2 5B")
        transformer = Wan2_2Transformer3DModel.from_pretrained(
            os.path.join(
                args.model_name,
                config["transformer_additional_kwargs"].get("transformer_low_noise_model_subpath", "transformer"),
            ),
            transformer_additional_kwargs=OmegaConf.to_container(config["transformer_additional_kwargs"]),
            low_cpu_mem_usage=True,
            torch_dtype=self.weight_dtype,
        ).eval().requires_grad_(False)
        vae = AutoencoderKLWan3_8.from_pretrained(
            os.path.join(args.model_name, config["vae_kwargs"].get("vae_subpath", "Wan2.2_VAE.pth")),
            additional_kwargs=OmegaConf.to_container(config["vae_kwargs"]),
        ).eval().requires_grad_(False).to(dtype=self.weight_dtype)
        scheduler = FlowMatchEulerDiscreteScheduler(
            **filter_kwargs(
                FlowMatchEulerDiscreteScheduler,
                OmegaConf.to_container(config["scheduler_kwargs"]),
            ),
        )
        pipeline = Wan2_2FunControlPipeline_infer(
            transformer=transformer, transformer_2=None, vae=vae, scheduler=scheduler,
        )
        lora_path = getattr(args, "lora_path", None)
        if not lora_path or not os.path.isfile(lora_path):
            raise FileNotFoundError(f"Provide the dense Control LoRA: {lora_path}")
        pipeline = merge_lora(
            pipeline, lora_path, getattr(args, "lora_weight", 0.75),
            device="cpu", dtype=torch.float32, transformer_only=True,
        )
        self.uses_reference = bool(transformer.config.get("add_ref_conv", False))
        if not self.uses_reference:
            logger.info("Base model has no ref_conv; first/last image conditions remain enabled")
        if model_offload:
            pipeline.enable_model_cpu_offload(device=self.device)
        else:
            pipeline.to(device=self.device)
        self.vae = vae
        self.pipeline = pipeline
        self.generator = torch.Generator(device=self.device).manual_seed(43)
        self.boundary = config["transformer_additional_kwargs"].get("boundary", 0.875)
        self.shift = 5.0
        logger.info(
            f"Dense Control ready on {self.device}: BF16, full attention, native Wan2.2 VAE, "
            f"LoRA={lora_path}, weight={getattr(args, 'lora_weight', 0.75)}, "
            f"model offload={model_offload}, VAE offload={self.offload_vae}"
        )

    def warmup_video(self, frames, height, width, prompt_embeds, negative_prompt_embeds):
        """Warm the multi-image path without advancing the inference RNG."""
        ratio = self.vae.config.temporal_compression_ratio
        if (frames - 1) % ratio or height % 32 or width % 32 or min(height, width) < 32:
            raise ValueError(f"Invalid video warmup shape: {frames}x{height}x{width}")

        cpu_generator = torch.Generator(device="cpu").manual_seed(42)
        control = torch.rand(
            (self.video_batch_size, 3, frames, height, width),
            dtype=self.weight_dtype, generator=cpu_generator,
        )
        video = torch.zeros_like(control)
        video[:, :, 0] = control[:, :, 0]
        video[:, :, -1] = control[:, :, -1]
        mask = torch.full((1, 1, frames, height, width), 255, dtype=self.weight_dtype)
        mask[:, :, 0] = 0
        mask[:, :, -1] = 0
        ref_image = control[:, :, -1:].contiguous()

        self.pipeline(
            prompt=None,
            negative_prompt=None,
            prompt_embeds=prompt_embeds * self.video_batch_size,
            negative_prompt_embeds=negative_prompt_embeds,
            num_frames=frames,
            height=height,
            width=width,
            num_inference_steps=1,
            guidance_scale=0.0,
            generator=torch.Generator(device=self.device).manual_seed(42),
            video=video,
            mask_video=mask,
            mask_cache_video=mask,
            control_video=control,
            ref_image=ref_image if self.uses_reference else None,
            boundary=self.boundary,
            shift=self.shift,
            low_vram=self.low_vram,
            offload_vae=self.offload_vae,
            video_batch_size=self.video_batch_size,
        )

    def infer(self, args, device='cuda'):
        # New callers send a filename; older shared-tensor callers still work.
        # The mapping stays local to this invocation, never on self or args.
        mapped = None
        input_video_source = None
        try:
            input_video_path = getattr(args, 'input_video_path', None)
            if input_video_path:
                # Copy-on-write makes a writable NumPy view without eagerly
                # allocating a private full-video copy. Treat it as read-only.
                mapped = np.load(input_video_path, mmap_mode='c', allow_pickle=False)
                input_video_source = torch.from_numpy(mapped)
                if getattr(args, 'input_video_is_bf16', False):
                    input_video_source = input_video_source.view(torch.bfloat16)
            else:
                input_video_source = args.input_video
            log_host_memory("B worker before inference", logger)
            return self._infer_impl(args, device, input_video_source)
        finally:
            # _infer_impl has finished, so its slice/output locals can be freed.
            input_video_source = None
            mapped = None
            release_free_host_memory()
            log_host_memory("B worker task finished", logger)

    def _infer_impl(self, args, device, input_video_source):
        sample_size = [args.img_size[1], args.img_size[0]]
        print("sample size:", sample_size)
        vae = self.vae
        pipeline = self.pipeline
        prompt              = ""
        negative_prompt     = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
        generator           = self.generator
        guidance_scale      = args.guidance_scale
        num_inference_steps = args.num_inference_steps
        boundary            = self.boundary
        shift               = self.shift
        save_path           = args.save_path
        video_length        = args.video_length
        fps                 = args.video_fps

        # first-last in default
        # Preserve the old first/last-only cache mask. Build its encoder cache
        # inside pipeline(), AFTER encoder placement, including later cases
        # with a different resolution. This mask does not replace inpaint masks.
        mask_video = torch.full(
            (1, 1, video_length, sample_size[0], sample_size[1]),
            255.0, dtype=self.weight_dtype, device='cpu')
        mask_video[:, :, 0] = 0
        mask_video[:, :, video_length-1] = 0

        st_T = time.time()

        # with torch.no_grad():
        with torch.inference_mode():
            video_length = int((video_length - 1) // vae.config.temporal_compression_ratio * vae.config.temporal_compression_ratio) + 1 if video_length != 1 else 1
            latent_frames = (video_length - 1) // vae.config.temporal_compression_ratio + 1

            # input_video, input_video_mask, _, _ = get_video_to_video_latent(args.control_video, video_length=video_length, sample_size=sample_size, fps=fps, ref_image=None)
            control_camera_video = None


            prompt_embed_path = args.prompt_embed_path
            prompt_embeds = None
            negative_prompt_embeds = None
            if os.path.exists(prompt_embed_path):
                embeds = torch.load(prompt_embed_path)
                prompt_embeds, negative_prompt_embeds = [embeds["prompt"].to(self.device)], [embeds["negative"].to(self.device)]
                print('### pred load prompt embeds:', type(prompt_embeds), prompt_embeds[0].shape)
                print('### pred load negative prompt embeds:', type(negative_prompt_embeds), negative_prompt_embeds[0].shape)

            img_lst = sorted(os.listdir(args.image_path))
            pre_infer_time = time.time() - st_T
            print("loading video time:", pre_infer_time)
            infer_time = 0.0
            save_vid_time = 0.0
            # videos = []

            assigned = list(range(args.rank, len(args.index), args.num_gpus))
            for segment_ids in group_video_segments(assigned, args.index, video_length, self.video_batch_size):
                st_T = time.perf_counter()
                inpaint_videos, reference_images, control_videos = [], [], []
                for segment_id in segment_ids:
                    logger.info(f"gen vid: {segment_id + 1}/{len(args.index)}")
                    start_frame = segment_id * video_length
                    control_videos.append(
                        input_video_source[:, :, start_frame:start_frame + video_length].contiguous()
                    )
                    images = [os.path.join(args.image_path, img_lst[x[0]]) for x in args.index[segment_id]]
                    frame_ids = [x[1] for x in args.index[segment_id]]
                    video, inpaint_mask, _ = get_inpaint_video_latent(
                        images, frame_ids, video_length=video_length, sample_size=sample_size,
                    )
                    inpaint_videos.append(video)
                    reference_images.append(get_image_latent(images[-1], sample_size=sample_size))
                logger.info(f"load control video time: {time.perf_counter() - st_T:.3f}s")

                batch_size = len(segment_ids)
                if batch_size == 2:
                    input_video = torch.cat(control_videos)
                    inpaint_video = torch.cat(inpaint_videos)
                    ref_image = torch.cat(reference_images)
                    inpaint_mask = mask_video
                else:
                    input_video = control_videos[0]
                    inpaint_video = inpaint_videos[0]
                    ref_image = reference_images[0]
                del control_videos, inpaint_videos, reference_images, video, _

                st_T = time.perf_counter()
                samples = pipeline(
                    prompt=None,
                    num_frames=video_length,
                    negative_prompt=None,
                    prompt_embeds=prompt_embeds * batch_size if prompt_embeds is not None else None,
                    negative_prompt_embeds=negative_prompt_embeds,
                    height=sample_size[0],
                    width=sample_size[1],
                    generator=generator,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    video=inpaint_video,
                    mask_video=inpaint_mask,
                    mask_cache_video=mask_video,
                    control_video=input_video,
                    control_camera_video=control_camera_video,
                    ref_image=ref_image if self.uses_reference else None,
                    boundary=boundary,
                    shift=shift,
                    low_vram=self.low_vram,
                    offload_vae=self.offload_vae,
                    video_batch_size=batch_size,
                ).videos
                logger.info(f"infer_time: {time.perf_counter() - st_T:.3f}s "
                            f"(batch={batch_size}, segments={[i + 1 for i in segment_ids]})")
                del input_video, inpaint_video, inpaint_mask, ref_image

                os.makedirs(save_path, exist_ok=True)
                for position, segment_id in enumerate(segment_ids):
                    sample = samples[position:position + 1]
                    save_img_ids = args.save_img_ids[segment_id] if len(args.save_img_ids) == len(args.index) else None
                    st_T = time.perf_counter()
                    if save_img_ids is None or getattr(args, "save_video", False):
                        video_path = os.path.join(save_path, f"{segment_id:08d}.mp4")
                        save_videos_grid(sample, video_path, fps=fps)
                        logger.info(f"save gen video: {time.perf_counter() - st_T:.3f}s")
                    if save_img_ids is not None:
                        st_T = time.perf_counter()
                        save_selected_images_from_video_tensor(
                            videos=sample,
                            images_path=os.path.join(save_path, "image"),
                            index=segment_id,
                            save_image_ids=save_img_ids,
                            resize=args.img_size,
                            frames_per_segment=video_length,
                            rescale=False,
                            n_rows=6,
                            use_global_name=True,
                        )
                        logger.info(f"save gen images: {time.perf_counter() - st_T:.3f}s")
                    del sample
                del samples
                release_free_host_memory()

if __name__ == '__main__':
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Run predict_v2v_control")
    print(f'current_file_dir {current_file_dir}')
    parser.add_argument("--config_path", type=str, default=os.path.join(current_file_dir, "config/wan2.2/wan_civitai_5b.yaml"))
    parser.add_argument("--model_name", type=str, default=os.path.join(current_file_dir, "models/Diffusion_Transformer/Wan2.2-Fun-5B-Control/"))
    parser.add_argument("--transformer_path", type=list, 
                default= [os.path.join(current_file_dir, "models/transformer_w_skybox/diffusion_pytorch_model-00001-of-00002.safetensors"),
                os.path.join(current_file_dir, "models/transformer_w_skybox/diffusion_pytorch_model-00002-of-00002.safetensors")])
    parser.add_argument("--control_video", type=str, default=os.path.join(current_file_dir, "asset/test/gs_render.mp4"))
    parser.add_argument("--ref_image", type=str, default=os.path.join(current_file_dir, "asset/test/image_gs/0000000000000374.png"))
    parser.add_argument("--save_path", type=str, default=os.path.join(current_file_dir, "samples"))

    args = parser.parse_args()