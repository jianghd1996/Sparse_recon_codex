import os
import sys
import time
import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf
from PIL import Image
from transformers import AutoTokenizer
import argparse
from loguru import logger
import subprocess

current_file_path = os.path.abspath(__file__)
current_file_dir = os.path.dirname(current_file_path)
project_roots = [os.path.dirname(current_file_path), os.path.dirname(os.path.dirname(current_file_path)), os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None

from resident_host_memory import release_free_host_memory, log_host_memory

from videox_fun.dist import set_multi_gpus_devices, shard_model
from videox_fun.models import (AutoencoderKLWan, AutoencoderKLWan3_8, AutoTokenizer, CLIPModel,
                               WanT5EncoderModel, Wan2_2Transformer3DModel)
from videox_fun.data.dataset_image_video import process_pose_file
from videox_fun.models.cache_utils import get_teacache_coefficients
# from videox_fun.pipeline import Wan2_2FunControlPipeline_infer, WanPipeline

# from videox_fun.models.wan_vae3_8_stream_v2 import AutoencoderKLWan3_8Stream
from videox_fun.models.wan_vae_first_encoder_lightvae_decoder_light import AutoencoderKLWan3_8Stream
from videox_fun.pipeline.pipeline_wan2_2_fun_control_infer_wjx_stream_light import Wan2_2FunControlPipeline_infer

from videox_fun.utils.fp8_optimization import (convert_model_weight_to_float8,
                                               convert_weight_dtype_wrapper,
                                               replace_parameters_by_name)
from videox_fun.utils.lora_utils_new import merge_lora, unmerge_lora
from videox_fun.utils.utils import (filter_kwargs, get_image_latent, get_image_to_video_latent, get_inpaint_video_latent,
                                    get_video_to_video_latent,
                                    make_save_videos_grid, _write_video_opencv, save_selected_images_from_video_tensor)
from videox_fun.utils.fm_solvers import FlowDPMSolverMultistepScheduler
from videox_fun.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
import torch.distributed as dist

save_videos_grid = make_save_videos_grid(
    _write_video_opencv, color_transfer_fn=globals().get("color_transfer")
)

sla_roots = [os.path.join(current_file_dir, 'turbodiffusion')]
for project_root in sla_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None
from SLA import (
    SparseLinearAttention,
    SageSparseLinearAttention
)
from ops import FastLayerNorm, FastRMSNorm, Int8Linear

from videox_fun.models.wan_transformer3d_sla import Wan2_2Transformer3DModel_SLA
from videox_fun.models.wan_transformer3d_sla import WanSelfAttention, WanRMSNorm, WanLayerNorm


def replace_attention(
    model: torch.nn.Module,
    attention_type: str,
    sla_topk: float,
) -> torch.nn.Module:
    assert attention_type in ["sla", "sagesla"], "Invalid attention type."
    
    for module in model.modules():
        if type(module) is WanSelfAttention:
            if attention_type == "sla":
                module.attn_op.local_attn = SparseLinearAttention(head_dim=module.dim // module.num_heads, topk=sla_topk, BLKQ=128, BLKK=64)
            elif attention_type == "sagesla":
                module.attn_op.local_attn = SageSparseLinearAttention(head_dim=module.dim // module.num_heads, topk=sla_topk)
    return model

def replace_linear_norm(
    model: torch.nn.Module,
    replace_linear: bool = False,
    replace_norm: bool = False,
    quantize: bool = True,
    skip_layer: str = "proj_l"
) -> torch.nn.Module:
    replacements = {}
    for name, module in model.blocks.named_modules():
        if isinstance(module, torch.nn.Linear) and replace_linear:
            if skip_layer not in name:
                replacements[name] = Int8Linear.from_linear(module, quantize)
        
        if isinstance(module, WanRMSNorm) and replace_norm:
            replacements[name] = FastRMSNorm.from_rmsnorm(module)
        
        if isinstance(module, WanLayerNorm) and replace_norm:
            replacements[name] = FastLayerNorm.from_layernorm(module)

    for name, new_module in replacements.items():
        parent_module = model.blocks
        name_parts = name.split(".")
        for part in name_parts[:-1]:
            parent_module = getattr(parent_module, part)
        setattr(parent_module, name_parts[-1], new_module)
    return model

class Wan_5b(object):
    def __init__(self, args, device="cuda"):
        GPU_memory_mode = (
            "model_cpu_offload" if getattr(args, "offload_model", False) else "model_full_load"
        )
        self.offload_vae = getattr(args, "offload_vae", True) or GPU_memory_mode == "model_cpu_offload"
        logger.info(f"Wan memory mode: {GPU_memory_mode}, VAE offload: {self.offload_vae}")
        weight_dtype        = torch.bfloat16
        ulysses_degree      = 1
        ring_degree         = 1 # zj_test multi card
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.deterministic = False  # 仍然用默认高效算法
        
        # Use FSDP to save more GPU memory in multi gpus.
        fsdp_dit            = False
        fsdp_text_encoder   = False
        # Compile will give a speedup in fixed resolution and need a little GPU memory. 
        # The compile_dit is not compatible with the fsdp_dit and sequential_cpu_offload.
        compile_dit         = False

        # Support TeaCache.
        enable_teacache     = False
        teacache_threshold  = 0.00
        # The number of steps to skip TeaCache at the beginning of the inference process, which can
        # reduce the impact of TeaCache on generated video quality.
        num_skip_start_steps = 5
        # Whether to offload TeaCache tensors to cpu to save a little bit of GPU memory.
        teacache_offload    = False

        # Skip some cfg steps in inference
        # Recommended to be set between 0.00 and 0.25
        cfg_skip_ratio      = 0

        # Riflex config
        enable_riflex       = False
        # Index of intrinsic frequency
        riflex_k            = 6

        # Config and model path
        # config_path         = "config/wan2.2/wan_civitai_5b.yaml"
        config_path = args.config_path
        # model path
        # model_name          = "models/Diffusion_Transformer/Wan2.2-Fun-5B-Control/"
        model_name = args.model_name

        # Choose the sampler in "Flow", "Flow_Unipc", "Flow_DPM++"
        sampler_name        = "Flow"
        # [NOTE]: Noise schedule shift parameter. Affects temporal dynamics. 
        # Used when the sampler is in "Flow_Unipc", "Flow_DPM++".
        shift               = 5

        # Load pretrained model if need
        # The transformer_path is used for low noise model, the transformer_high_path is used for high noise model.
        # Since Wan2.2-5b consists of only one model, only transformer_path is used.
        # transformer_path        = ["/data_ssd/z00566689/dev/code/j00910438/VideoX-Fun/output_dir/ckpt-6200-backup/transformer/diffusion_pytorch_model-00001-of-00002.safetensors", "/data_ssd/z00566689/dev/code/j00910438/VideoX-Fun/output_dir/ckpt-6200-backup/transformer/diffusion_pytorch_model-00002-of-00002.safetensors"]
        transformer_path = args.transformer_path
        transformer_high_path   = None
        vae_path                = None
        # Load lora model if need
        # The lora_path is used for low noise model, the lora_high_path is used for high noise model.
        # Since Wan2.2-5b consists of only one model, only lora_path is used.
        if hasattr(args, "lora_path"):
            lora_path = args.lora_path
        else:
            lora_path               = None
        lora_high_path          = None

        # Other params
        
        # sample_size = [1280, 704]
        # logger.info(f"gen vid img size {args.img_size} sample size {sample_size}")
        # sample_size         = [1280, 704] # [1280, 960]
        

        # Use torch.float16 if GPU does not support torch.bfloat16
        # ome graphics cards, such as v100, 2080ti, do not support torch.bfloat16

        # control_video           = "asset/pose.mp4"
        # control_video           = "/data_ssd/z00566689/dev/code/j00910438/VideoX-Fun/asset/test/gs_render.mp4"

        # 使用更长的neg prompt如"模糊，突变，变形，失真，画面暗，文本字幕，画面固定，连环画，漫画，线稿，没有主体。"，可以增加稳定性
        # 在neg prompt中添加"安静，固定"等词语可以增加动态性。
        
        # Using longer neg prompt such as "Blurring, mutation, deformation, distortion, dark and solid, comics, text subtitles, line art." can increase stability
        # Adding words such as "quiet, solid" to the neg prompt can increase dynamism.
        # prompt                  = "A young woman with beautiful, clear eyes and blonde hair stands in the forest, wearing a white dress and a crown. Her expression is serene, reminiscent of a movie star, with fair and youthful skin. Her brown long hair flows in the wind. The video quality is very high, with a clear view. High quality, masterpiece, best quality, high resolution, ultra-fine, fantastical."
        # negative_prompt         = "Twisted body, limb deformities, text captions, comic, static, ugly, error, messy code."
        
        seed                    = 43
        
        # The lora_weight is used for low noise model, the lora_high_weight is used for high noise model.
        lora_weight             = 0.75
        lora_high_weight        = 0.75
        # save_path               = "samples/wan-videos-fun-control"
        

        # device = set_multi_gpus_devices(ulysses_degree, ring_degree)
        config = OmegaConf.load(config_path)
        boundary = config['transformer_additional_kwargs'].get('boundary', 0.875)

        if args.sla_topk > 1.0:
            wan22_transformer_module = Wan2_2Transformer3DModel
        else:
            wan22_transformer_module = Wan2_2Transformer3DModel_SLA
        transformer = wan22_transformer_module.from_pretrained(
            os.path.join(model_name, config['transformer_additional_kwargs'].get('transformer_low_noise_model_subpath', 'transformer')),
            transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
            low_cpu_mem_usage=True,
            torch_dtype=weight_dtype,
        )
        if config['transformer_additional_kwargs'].get('transformer_combination_type', 'single') == "moe":
            transformer_2 = wan22_transformer_module.from_pretrained(
                os.path.join(model_name, config['transformer_additional_kwargs'].get('transformer_high_noise_model_subpath', 'transformer')),
                transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
                low_cpu_mem_usage=True,
                torch_dtype=weight_dtype,
            )
        else:
            transformer_2 = None

        if transformer_path is not None:
            print(f"From checkpoint: {transformer_path}")
            all_state_dict = {}

            for path in transformer_path:
                if path.endswith("safetensors"):
                    from safetensors.torch import load_file, safe_open
                    state_dict = load_file(path)
                else:
                    state_dict = torch.load(path, map_location="cpu")
                state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict
                all_state_dict.update(state_dict)

            if args.sla_topk <= 0.9:
                logger.info(f"===>>>Replacing attention with SLA {args.sla_topk}")
                # transformer = replace_attention(transformer, args.attention_type, args.sla_topk) 
                transformer = replace_attention(transformer, "sla", args.sla_topk) 
                # transformer = replace_attention(transformer, "sagesla", args.sla_topk) 
                transformer = replace_linear_norm(transformer, replace_linear=True, replace_norm=True)                

            m, u = transformer.load_state_dict(all_state_dict, strict=False)
            print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

        if transformer_2 is not None:
            if transformer_high_path is not None:
                print(f"From checkpoint: {transformer_high_path}")
                if transformer_high_path.endswith("safetensors"):
                    from safetensors.torch import load_file, safe_open
                    state_dict = load_file(transformer_high_path)
                else:
                    state_dict = torch.load(transformer_high_path, map_location="cpu")
                state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

                m, u = transformer_2.load_state_dict(state_dict, strict=False)
                print(f"sla missing keys: {len(m)}, unexpected keys: {len(u)}")

        # Get Vae
        # Chosen_AutoencoderKL = {
        #     "AutoencoderKLWan": AutoencoderKLWan,
        #     "AutoencoderKLWan3_8": AutoencoderKLWan3_8
        # }[config['vae_kwargs'].get('vae_type', 'AutoencoderKLWan')]

        # vae = AutoencoderKLWan3_8Stream.from_pretrained(
        #     os.path.join(model_name, config['vae_kwargs'].get('vae_subpath', 'vae')),
        #     additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
        # ).to(weight_dtype)

        vae = AutoencoderKLWan3_8Stream.from_teacher_and_lightvae(
            teacher_vae_path=os.path.join(model_name, "Wan2.2_VAE.pth"),
            lightvae_decoder_path=args.vae_path,
            additional_kwargs={
                "latent_channels": 48,
                "c_dim": 160,
                "dim_mult": [1, 2, 4, 4],
                "temperal_downsample": [False, True, True],
            },
            # 通常可以省略，将根据 LightVAE decoder.conv1.weight 自动推断
            decoder_pruning_rate=None,
        )
        vae = vae.eval().requires_grad_(False).to(dtype=weight_dtype)

        # Get Tokenizer
        # tokenizer = AutoTokenizer.from_pretrained(
        #     os.path.join(model_name, config['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer')),
        # )

        # # Get Text encoder
        # text_encoder = WanT5EncoderModel.from_pretrained(
        #     os.path.join(model_name, config['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')),
        #     additional_kwargs=OmegaConf.to_container(config['text_encoder_kwargs']),
        #     low_cpu_mem_usage=True,
        #     torch_dtype=weight_dtype,
        # )
        # text_encoder = text_encoder.eval()

        # Get Scheduler
        Chosen_Scheduler = scheduler_dict = {
            "Flow": FlowMatchEulerDiscreteScheduler,
            "Flow_Unipc": FlowUniPCMultistepScheduler,
            "Flow_DPM++": FlowDPMSolverMultistepScheduler,
        }[sampler_name]
        if sampler_name == "Flow_Unipc" or sampler_name == "Flow_DPM++":
            config['scheduler_kwargs']['shift'] = 1
        scheduler = Chosen_Scheduler(
            **filter_kwargs(Chosen_Scheduler, OmegaConf.to_container(config['scheduler_kwargs']))
        )

        # Get Pipeline
        pipeline = Wan2_2FunControlPipeline_infer(
            transformer=transformer,
            transformer_2=transformer_2,
            vae=vae,
            # tokenizer=tokenizer,
            # text_encoder=text_encoder,
            scheduler=scheduler,
        )
        if ulysses_degree > 1 or ring_degree > 1:
            from functools import partial
            transformer.enable_multi_gpus_inference()
            if transformer_2 is not None:
                transformer_2.enable_multi_gpus_inference()
            if fsdp_dit:
                shard_fn = partial(shard_model, device_id=device, param_dtype=weight_dtype)
                pipeline.transformer = shard_fn(pipeline.transformer)
                if transformer_2 is not None:
                    pipeline.transformer_2 = shard_fn(pipeline.transformer_2)
                print("Add FSDP DIT")
            if fsdp_text_encoder:
                shard_fn = partial(shard_model, device_id=device, param_dtype=weight_dtype)
                pipeline.text_encoder = shard_fn(pipeline.text_encoder)
                print("Add FSDP TEXT ENCODER")

        if compile_dit:
            for i in range(len(pipeline.transformer.blocks)):
                pipeline.transformer.blocks[i] = torch.compile(pipeline.transformer.blocks[i])
            if transformer_2 is not None:
                for i in range(len(pipeline.transformer_2.blocks)):
                    pipeline.transformer_2.blocks[i] = torch.compile(pipeline.transformer_2.blocks[i])
            print("Add Compile")

        if GPU_memory_mode == "sequential_cpu_offload":
            replace_parameters_by_name(transformer, ["modulation",], device=device)
            # if hasattr(transformer, 'freqs'):
            #     transformer.freqs = transformer.freqs.to(device=device)
            if transformer_2 is not None:
                replace_parameters_by_name(transformer_2, ["modulation",], device=device)
                transformer_2.freqs = transformer_2.freqs.to(device=device)
            pipeline.enable_sequential_cpu_offload(device=device)
        elif GPU_memory_mode == "model_cpu_offload_and_qfloat8":
            convert_model_weight_to_float8(transformer, exclude_module_name=["modulation",], device=device)
            convert_weight_dtype_wrapper(transformer, weight_dtype)
            if transformer_2 is not None:
                convert_model_weight_to_float8(transformer_2, exclude_module_name=["modulation",], device=device)
                convert_weight_dtype_wrapper(transformer_2, weight_dtype)
            pipeline.enable_model_cpu_offload(device=device)
        elif GPU_memory_mode == "model_cpu_offload":
            pipeline.enable_model_cpu_offload(device=device)
        elif GPU_memory_mode == "model_full_load_and_qfloat8":
            convert_model_weight_to_float8(transformer, exclude_module_name=["modulation",], device=device)
            convert_weight_dtype_wrapper(transformer, weight_dtype)
            if transformer_2 is not None:
                convert_model_weight_to_float8(transformer_2, exclude_module_name=["modulation",], device=device)
                convert_weight_dtype_wrapper(transformer_2, weight_dtype)
            pipeline.to(device=device)
        else:
            pipeline.to(device=device)

        coefficients = get_teacache_coefficients(model_name) if enable_teacache else None
        if coefficients is not None:
            print(f"Enable TeaCache with threshold {teacache_threshold} and skip the first {num_skip_start_steps} steps.")
            pipeline.transformer.enable_teacache(
                coefficients, args.num_inference_steps, teacache_threshold, num_skip_start_steps=num_skip_start_steps, offload=teacache_offload
            )
            if transformer_2 is not None:
                pipeline.transformer_2.share_teacache(transformer=pipeline.transformer)

        if cfg_skip_ratio is not None:
            print(f"Enable cfg_skip_ratio {cfg_skip_ratio}.")
            pipeline.transformer.enable_cfg_skip(cfg_skip_ratio, args.num_inference_steps)
            if transformer_2 is not None:
                pipeline.transformer_2.share_cfg_skip(transformer=pipeline.transformer)

        generator = torch.Generator(device=device).manual_seed(seed)

        if lora_path is not None:
            pipeline = merge_lora(pipeline, lora_path, lora_weight, device=device, dtype=weight_dtype)
            if transformer_2 is not None:
                pipeline = merge_lora(pipeline, lora_high_path, lora_high_weight, device=device, dtype=weight_dtype, sub_transformer_name="transformer_2")

        if args.use_vae:
            vae.model.encode_first_step = torch.compile(
                vae.model.encode_first_step,
                mode="max-autotune", 
                fullgraph=True
            )
            vae.model.encode_next_step = torch.compile(
                vae.model.encode_next_step,
                mode="max-autotune", 
                fullgraph=True
            )
            vae.model.decode_first_step = torch.compile(
                vae.model.decode_first_step,
                mode="max-autotune", 
                fullgraph=True
            )
            vae.model.decode_warmup_step = torch.compile(
                vae.model.decode_warmup_step,
                mode="max-autotune", 
                fullgraph=True
            )
            vae.model.decode_steady_step = torch.compile(
                vae.model.decode_steady_step,
                mode="max-autotune", 
                fullgraph=True
            )

        video_length = args.video_length
        # mask_video = torch.full((1, 1, video_length, 1440, 1088), 255.0).to(device=device, dtype=weight_dtype)
        # mask_video[:, :, 0] = 0
        # mask_video[:, :, video_length-1] = 0
        # pipeline._ensure_mask_cache(
        #     mask_video, video_length, 1440, 1088,
        #     1, weight_dtype, device, 1
        # )

        self.vae = vae
        self.pipeline = pipeline
        self.generator = generator
        self.boundary = boundary
        self.shift    = shift
        self.device   = device
        self.weight_dtype   = weight_dtype

    def warmup_video(self, frames, height, width, prompt_embeds, negative_prompt_embeds):
        """Warm the multi-image path without advancing the inference RNG."""
        ratio = self.vae.config.temporal_compression_ratio
        if (frames - 1) % ratio or height % 32 or width % 32 or min(height, width) < 32:
            raise ValueError(f"Invalid video warmup shape: {frames}x{height}x{width}")

        cpu_generator = torch.Generator(device="cpu").manual_seed(42)
        control = torch.rand(
            (1, 3, frames, height, width), dtype=self.weight_dtype, generator=cpu_generator,
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
            prompt_embeds=prompt_embeds,
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
            ref_image=ref_image,
            boundary=self.boundary,
            shift=self.shift,
            low_vram=False,
            offload_vae=self.offload_vae,
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
        # ratio = min(args.img_size)/max(args.img_size)
        # if args.img_size[0] <= args.img_size[1]: # width <= height
        #     if ratio > 0.7:
        #         sample_size = [1440, 1088]
        #     elif ratio > 0.65:                                                                                                                                                                                           
        #         sample_size = [1600, 1088]
        #     else:
        #         sample_size = [1920, 1088] # [1664, 960] [1280, 704] # [1920, 1088]
        # else:
        #     if ratio > 0.7:
        #         sample_size = [1088, 1440] # [960, 1280] # 
        #     elif ratio > 0.65:
        #         sample_size = [1088, 1600]
        #     else:
        #         sample_size = [1088, 1920] # [960, 1664] [1088, 1920] 
        # if args.img_size[0] == args.img_size[1]:
        #     sample_size = [1280, 1280]
        # sample_size = [1440, 1088]
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

            for i in range(len(args.index)):
                if i % args.num_gpus != args.rank:
                    continue

                
                logger.info(f"gen vid: {i+1}/{len(args.index)}")
                input_video = input_video_source[:, :, i*video_length:(i+1)*video_length].contiguous()

                images = [os.path.join(args.image_path, img_lst[x[0]]) for x in args.index[i]]
                index = [x[1] for x in args.index[i]]
                ref_image = images[-1]

                if len(args.save_img_ids) == len(args.index):
                    save_img_ids = args.save_img_ids[i]
                else:
                    save_img_ids = None

                st_T = time.time()
                # inpaint_video, inpaint_video_mask, clip_image = get_image_to_video_latent(start_image, end_image, video_length=video_length, sample_size=sample_size)
                inpaint_video, inpaint_video_mask, clip_image = get_inpaint_video_latent(images, index, video_length=video_length, sample_size=sample_size)

                if ref_image is not None:
                    ref_image = get_image_latent(ref_image, sample_size=sample_size)
                end_T = time.time()
                logger.info(f"load control video time: {end_T-st_T}s ")

                st_T = time.time()
                sample = pipeline(
                    prompt=None, 
                    num_frames = video_length,
                    negative_prompt = None,
                    prompt_embeds = prompt_embeds,
                    negative_prompt_embeds = negative_prompt_embeds,
                    height      = sample_size[0],
                    width       = sample_size[1],
                    generator   = generator,
                    guidance_scale = guidance_scale,
                    num_inference_steps = num_inference_steps,

                    video      = inpaint_video,
                    mask_video   = inpaint_video_mask,
                    mask_cache_video = mask_video,
                    control_video = input_video,
                    control_camera_video = control_camera_video,
                    ref_image = ref_image,
                    boundary = boundary,
                    shift = shift,
                    low_vram = False,
                    offload_vae = self.offload_vae,
                ).videos
                infer_time = time.time()-st_T
                logger.info(f"infer_time: {infer_time}s ")
                # videos.append(sample)

            
                os.makedirs(save_path, exist_ok=True)

                st_T = time.time()
                if save_img_ids is None:
                    index = i
                    prefix = str(index).zfill(8)
                    video_path = os.path.join(save_path, prefix + ".mp4")
                    save_videos_grid(sample, video_path, fps=fps)
                    save_img_time = time.time()-st_T
                    logger.info(f"save gen video: {save_img_time}s")
                else:
                    # args.video_fps, args.img_size
                    image_save_dir = os.path.join(save_path, "image")
                    save_selected_images_from_video_tensor(
                        videos=sample,
                        images_path=image_save_dir,
                        index=i,
                        save_image_ids=save_img_ids,
                        resize=args.img_size,
                        frames_per_segment=video_length,
                        rescale=False,
                        n_rows=6,
                        use_global_name=True,
                    )
                    save_img_time = time.time()-st_T
                    logger.info(f"save gen images: {save_img_time}s")

                del sample, input_video, inpaint_video, inpaint_video_mask
                ref_image = clip_image = None
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
