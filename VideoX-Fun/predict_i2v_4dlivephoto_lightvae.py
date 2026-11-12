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

from videox_fun.dist import set_multi_gpus_devices, shard_model
from videox_fun.models import (AutoencoderKLWan, AutoencoderKLWan3_8, AutoTokenizer, CLIPModel,
                               WanT5EncoderModel, Wan2_2Transformer3DModel)
from videox_fun.data.dataset_image_video import process_pose_file
from videox_fun.models.cache_utils import get_teacache_coefficients
# from videox_fun.pipeline import Wan2_2FunControlPipeline_infer, WanPipeline

# from videox_fun.models.wan_vae3_8_stream_v2 import AutoencoderKLWan3_8Stream
# from videox_fun.models.wan_vae_first_encoder_lightvae_decoder import AutoencoderKLWan3_8Stream
# from videox_fun.pipeline.pipeline_wan2_2_fun_control_infer_wjx_stream import Wan2_2FunControlPipeline_infer

from videox_fun.models.wan_vae_first_encoder_lightvae_decoder_light import AutoencoderKLWan3_8Stream
from videox_fun.pipeline.pipeline_wan2_2_fun_control_infer_wjx_livephoto_light import Wan2_2LivePhotoPipeline_infer # TODO: Wan2_2LivePhotoPipeline_infer

from videox_fun.utils.fp8_optimization import (convert_model_weight_to_float8,
                                               convert_weight_dtype_wrapper,
                                               replace_parameters_by_name)
from videox_fun.utils.lora_utils import merge_lora, unmerge_lora
from videox_fun.utils.utils import (filter_kwargs, get_image_for_latent,
                                    make_save_videos_grid, _write_video_opencv, save_selected_images_from_video_tensor)
from videox_fun.utils.fm_solvers import FlowDPMSolverMultistepScheduler
from videox_fun.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
import torch.distributed as dist

save_videos_grid = make_save_videos_grid(
    _write_video_opencv, color_transfer_fn=globals().get("color_transfer")
)

cur_py_path = os.path.dirname(os.path.realpath(__file__))

class Wan_5b(object):
    # The pipeline places the complete VAE on GPU when encoding starts.
    manages_vae_placement = True

    def __init__(self, args, device="cuda"):
        self.offload_vae = getattr(args, "offload_vae", True)
        logger.info(f"LivePhoto VAE offload within generation: {self.offload_vae} ({device})")
        GPU_memory_mode = (
            "model_cpu_offload" if getattr(args, "model_cpu_offload", True) else "model_full_load"
        )
        logger.info(f"LivePhoto model memory mode: {GPU_memory_mode} ({device})")
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
        # config_path         = "VideoX-Fun/config/wan2.2/wan_civitai_4dlp_5b.yaml"
        config_path = args.config_path
        # model path
        # model_name          = "models/Diffusion_Transformer/Wan2.2-Fun-5B-Control/"
        model_name = args.model_name

        # Choose the sampler in "Flow", "Flow_Unipc", "Flow_DPM++"
        sampler_name        = "Flow"
        # [NOTE]: Noise schedule shift parameter. Affects temporal dynamics. 
        # Used when the sampler is in "Flow_Unipc", "Flow_DPM++".
        shift               = 3

        # Load pretrained model if need
        # The transformer_path is used for low noise model, the transformer_high_path is used for high noise model.
        # Since Wan2.2-5b consists of only one model, only transformer_path is used.
        # transformer_path        = ["/data_ssd/z00566689/dev/code/j00910438/VideoX-Fun/output_dir/ckpt-6200-backup/transformer/diffusion_pytorch_model-00001-of-00002.safetensors", "/data_ssd/z00566689/dev/code/j00910438/VideoX-Fun/output_dir/ckpt-6200-backup/transformer/diffusion_pytorch_model-00002-of-00002.safetensors"]
        # transformer_path = args.transformer_path
        transformer_path        = None
        transformer_high_path   = None
        vae_path                = None

        args.lora_path = os.path.join(args.model_name, 'lora_ckpts_4d_livephoto', 'epoch-1200-2402.safetensors')
        args.tdm_lora_path = os.path.join(args.model_name, 'lora_ckpts_4d_livephoto', 'checkpoints_lora_1101.safetensors')
        # Load lora model if need
        # The lora_path is used for low noise model, the lora_high_path is used for high noise model.
        # Since Wan2.2-5b consists of only one model, only lora_path is used.
        if hasattr(args, "lora_path"):
            lora_path = args.lora_path
        else:
            lora_path               = None
        # lora_high_path          = None
        if hasattr(args, "tdm_lora_path"):
            tdm_lora_path = args.tdm_lora_path
        else:
            tdm_lora_path               = None

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
        lora_weight            = 1
        tdm_lora_weight        = 1
        # save_path               = "samples/wan-videos-fun-control"
        

        # device = set_multi_gpus_devices(ulysses_degree, ring_degree)
        config = OmegaConf.load(config_path)
        boundary = config['transformer_additional_kwargs'].get('boundary', 0.875)

        Transformer3DModel = Wan2_2Transformer3DModel

        transformer = Transformer3DModel.from_pretrained(
            os.path.join(model_name, config['transformer_additional_kwargs'].get('transformer_low_noise_model_subpath', 'transformer')),
            transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
            low_cpu_mem_usage=True,
            torch_dtype=weight_dtype,
        )

        # Get Vae
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
        tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(model_name, config['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer')),
        )

        # # Get Text encoder
        text_encoder = WanT5EncoderModel.from_pretrained(
            os.path.join(model_name, config['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')),
            additional_kwargs=OmegaConf.to_container(config['text_encoder_kwargs']),
            low_cpu_mem_usage=True,
            torch_dtype=weight_dtype,
        )
        text_encoder = text_encoder.eval()

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
        pipeline = Wan2_2LivePhotoPipeline_infer(
            transformer=transformer,
            vae=vae,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            scheduler=scheduler,
        )
        if ulysses_degree > 1 or ring_degree > 1:
            from functools import partial
            transformer.enable_multi_gpus_inference()
            if fsdp_dit:
                shard_fn = partial(shard_model, device_id=device, param_dtype=weight_dtype)
                pipeline.transformer = shard_fn(pipeline.transformer)
                print("Add FSDP DIT")
            if fsdp_text_encoder:
                shard_fn = partial(shard_model, device_id=device, param_dtype=weight_dtype)
                pipeline.text_encoder = shard_fn(pipeline.text_encoder)
                print("Add FSDP TEXT ENCODER")

        if compile_dit:
            for i in range(len(pipeline.transformer.blocks)):
                pipeline.transformer.blocks[i] = torch.compile(pipeline.transformer.blocks[i])
            print("Add Compile")

        if GPU_memory_mode == "sequential_cpu_offload":
            replace_parameters_by_name(transformer, ["modulation",], device=device)
            transformer.freqs = transformer.freqs.to(device=device)
            pipeline.enable_sequential_cpu_offload(device=device)
        elif GPU_memory_mode == "model_cpu_offload_and_qfloat8":
            convert_model_weight_to_float8(transformer, exclude_module_name=["modulation",], device=device)
            convert_weight_dtype_wrapper(transformer, weight_dtype)
            pipeline.enable_model_cpu_offload(device=device)
        elif GPU_memory_mode == "model_cpu_offload":
            pipeline.enable_model_cpu_offload(device=device)
        elif GPU_memory_mode == "model_full_load_and_qfloat8":
            convert_model_weight_to_float8(transformer, exclude_module_name=["modulation",], device=device)
            convert_weight_dtype_wrapper(transformer, weight_dtype)
            pipeline.to(device=device)
        else:
            pipeline.to(device=device)

        coefficients = get_teacache_coefficients(model_name) if enable_teacache else None
        if coefficients is not None:
            print(f"Enable TeaCache with threshold {teacache_threshold} and skip the first {num_skip_start_steps} steps.")
            pipeline.transformer.enable_teacache(
                coefficients, args.num_inference_steps, teacache_threshold, num_skip_start_steps=num_skip_start_steps, offload=teacache_offload
            )

        generator = torch.Generator(device=device).manual_seed(seed)

        if args.lora_path is not None:
            pipeline = merge_lora(pipeline, lora_path, lora_weight, device=device, dtype=weight_dtype)
        if args.tdm_lora_path is not None:
            pipeline = merge_lora(pipeline, tdm_lora_path, tdm_lora_weight, device=device, dtype=weight_dtype)

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

        self.vae = vae
        self.pipeline = pipeline
        self.generator = generator
        self.boundary = boundary
        self.shift    = shift
        self.device   = device
        self.weight_dtype   = weight_dtype

    def infer(self, args, device='cuda'):
        width, height = args.img_size
        scale = min(1088 / min(width, height), 1664 / max(width, height))
        # The pipeline expects [height, width] on a 32-pixel grid.
        sample_size = [max(32, round(edge * scale / 32) * 32) for edge in (height, width)]
        print("sample size:", sample_size)
        vae = self.vae
        pipeline = self.pipeline
        prompt              = args.prompt
        negative_prompt     = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
        generator           = self.generator
        guidance_scale      = args.guidance_scale
        num_inference_steps = args.num_inference_steps
        boundary            = self.boundary
        shift               = self.shift
        save_path           = args.save_path
        video_length        = args.video_length
        fps                 = args.video_fps
        camera_movement     = args.camera_movement
        use_fixed_key_frames = args.use_fixed_key_frames
        num_selected_frames  = args.num_selected_frames
        denoising_strength = 1.0

        # first-last in default
        # mask_video = torch.full((1, 1, video_length, sample_size[0], sample_size[1]), 255.0).to(device=self.device, dtype=self.weight_dtype)
        # mask_video[:, :, 0] = 0
        # mask_video[:, :, video_length-1] = 0
        # self.pipeline._ensure_mask_cache(
        #     mask_video, video_length, sample_size[0], sample_size[1],
        #     1, self.weight_dtype, self.device, 1, video_length//4-1
        # )

        st_T = time.time()

        # with torch.no_grad():
        with torch.inference_mode():
            video_length = int((video_length - 1) // vae.config.temporal_compression_ratio * vae.config.temporal_compression_ratio) + 1 if video_length != 1 else 1
            latent_frames = (video_length - 1) // vae.config.temporal_compression_ratio + 1


            img_lst = sorted(os.listdir(args.image_path))
            assert len(img_lst) == 1
            image_file = os.path.join(args.image_path, img_lst[0])
            pre_infer_time = time.time() - st_T
            infer_time = 0.0
            save_vid_time = 0.0
            # videos = []

            # assume args.index == [0, 1]
            for i in range(len(args.index)):
                print('###:', i, args.num_gpus, args.rank)
                if i % args.num_gpus != args.rank:
                    continue
                
                logger.info(f"gen vid: {i+1}/{len(img_lst)}, on rank[{args.rank}] for N_GPU-[{args.num_gpus}]")
                # start_image = os.path.join(args.image_path, img_lst[args.index[i][0]])
                # end_image = os.path.join(args.image_path, img_lst[args.index[i][1]])

                # print(start_image, end_image, video_length, input_video[:, :, i*video_length:(i+1)*video_length].shape)
                reverse_width = args.index[i]==0            

                st_T = time.time()
                input_image = get_image_for_latent(image_file, sample_size=sample_size)
                # print('### input_image:', input_image.shape, input_image.dtype, input_image.max(), input_image.min())


                st_T = time.time()
                sample = pipeline(
                    prompt=prompt, 
                    num_frames = video_length,
                    negative_prompt = negative_prompt,
                    height      = sample_size[0],
                    width       = sample_size[1],
                    generator   = generator,
                    guidance_scale = guidance_scale,
                    num_inference_steps = num_inference_steps,

                    input_image = input_image,
                    input_video = None,
                    cond_video = None,
                    camera_movement = camera_movement,
                    num_selected_frames = num_selected_frames,
                    use_fixed_key_frames = use_fixed_key_frames,
                    reverse_width = reverse_width,
                    denoising_strength = denoising_strength,
                    boundary = boundary,
                    shift = shift,
                    offload_vae=self.offload_vae,
                ).videos
                infer_time = time.time()-st_T
                logger.info(f"infer_time: {infer_time}s ")
                # videos.append(sample)
                os.makedirs(save_path, exist_ok=True)
                if reverse_width:
                    prefix = 'left'
                else:
                    prefix = 'right'
                video_path = os.path.join(save_path, prefix + ".mp4")
                save_started = time.perf_counter()
                save_videos_grid(sample, video_path, fps=fps, reverse_width=reverse_width)
                save_vid_time = time.perf_counter() - save_started
                logger.info(f"save gen vid: {save_vid_time}s")

if __name__ == '__main__':
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Run predict_v2v_control")
    print(f'current_file_dir {current_file_dir}')
    parser.add_argument("--config_path", type=str, default=os.path.join(current_file_dir, "config/wan2.2/wan_civitai_4dlp_5b.yaml"))
    parser.add_argument("--model_name", type=str, default=os.path.join(current_file_dir, "models/Diffusion_Transformer/Wan2.2-TI2V-5B/"))
    parser.add_argument("--camera_movement", type=str, default="arc_right_frame_cat")
    parser.add_argument("--use_vae", action="store_true", default=False, help="Whether to use compile for vae.")
    parser.add_argument("--use_fixed_key_frames", action="store_true", default=False, help="Whether to use fixed frame.")
    parser.add_argument("--num_inference_steps", type=int, default=8)
    parser.add_argument("--video_fps", type=int, default=30)
    parser.add_argument("--video_length", type=int, default=81)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--num_selected_frames", type=int, default=5)
    parser.add_argument("--save_path", type=str, default=os.path.join(current_file_dir, "livephoto"))
    args = parser.parse_args()    
    # video_generator = Wan_5b(args)

    # args.index = [0, 1]
    # args.img_size = [2160, 3840]
    # args.num_gpus = 1
    # args.rank = 0
    
    # video_generator.infer(args)
