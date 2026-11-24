import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import atexit
import gc
import sys
import time
import sys
import json
from datetime import datetime
from pathlib import Path
import argparse
import shutil
import cv2
os.environ["MKL_THREADING_LAYER"] = "GNU"
import torch
import ffmpeg
from dataclasses import dataclass
from loguru import logger
import torch.multiprocessing as mp

from utils.extract_for_edit import copy_colmap_project

from utils.cv_tools import Segmentation, insert_img_in_src, merge_videos, images_to_video, extract_images_from_vid, crop_image, drop_img_icc, resize_img
from utils.cv_tools import align_images_resolution, get_crop_ratio, get_crop_ratio_w_face_track, remove_frames_by_index_opencv
from utils.face_detector_retina import FaceDetector
from utils.utils import calculate_md5, DictToObject
from utils.multiprocess_worker_pool import _cleanup_cuda, WorkerPool, PoolRecoveryError
from utils.video_reader_and_sample import seg_index_sample, get_video_to_video_latent_with_imgsize
from utils.reconstruction_sampling import prepare_reconstruction_images

logger.remove()
logger.add(sys.stderr, 
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{module}</cyan>:<cyan>{function}</cyan> - {message}")

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.cuda.amp.autocast.*is deprecated.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*spatial_compression_ratio.*")


class SparseRecon:
    '''
    功能说明：
    img_dir: str                    输入图片路径
        use_input_icc_drop: bool        测试图片进行icc处理，避免色差
    run_DA3: bool                   是否用DA3估计位姿
        use_enhance_crop: bool          画面中心增强
        use_view_extrapolation: bool    视角外插
    run_livephoto: bool             是否用livephoto进行视角扩充
    run_gen_video: bool             是否进行视频生成
        gen_vid_fps: int                视频生成保存fps
    run_3dgs_iter: bool             是否稠密重建
    '''

    def __init__(self, json_args):
        if type(json_args) != dict:
            json_args = json.loads(json_args)
        logger.info(f"args from json: {json_args}")
        logger.info(f"pipeline file: {Path(__file__).resolve()}")

        self.args = DictToObject(json_args)
        self.debug = getattr(self.args, "debug", False)
        self.save_necessary_only = getattr(self.args, "save_necessary_only", True)
        if type(self.save_necessary_only) is not bool:
            raise ValueError("save_necessary_only must be true or false")
        configured_max = getattr(self.args, "max_frames", None)
        if configured_max is not None and (type(configured_max) is not int or configured_max < 1):
            raise ValueError("max_frames must be a positive integer, or null for single=40/multi=80")
        self.reconstruction_only = (
            self.args.run_3dgs_iter and not self.args.run_DA3 and not self.args.run_gen_video
        )

        self.build_model()
    
    def build_model(self,):
        cur_py_path = os.path.dirname(os.path.realpath(__file__))

        if self.args.use_enhance_crop and not self.reconstruction_only:
            self.seg_module = Segmentation(os.path.join(cur_py_path, "Cloud_Models/Saliency-Face-DA3"), torch.device('cuda'))
            self.face_detector = FaceDetector(0.8, 50, os.path.join(cur_py_path, "Cloud_Models/Saliency-Face-DA3"), torch.device('cuda'))

        if self.args.run_DA3:
            from DA3.run_da3_full_crop_dev import run_DA3
            DA3_model_path = os.path.join(cur_py_path, "Cloud_Models/Saliency-Face-DA3/besthumanmodel_lr1e_6_ckpt2.pt")
            self.run_DA3 = run_DA3(DA3_model_path)

        if self.args.run_3dgs_iter:

            from Reconstruction_full import Reconstruction
            print("Load full Reconstruction version, support camera optimization")
            self.recon_iter_full = Reconstruction(preload=True, dense_init_scene=False)
                # except:
                #     project_root = str(Path(__file__).resolve().parent.parent)
                #     sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),"../../../dist_so/gaussian3d/third_party_envs"))
                #     if project_root not in sys.path:
                #         sys.path.append(project_root)
                #     from Reconstruction import Reconstruction

            from Reconstruction import Reconstruction
            print("Load vanilla version")
            self.recon_iter = Reconstruction(preload=True, dense_init_scene=False)

        
        self.world_size = self.args.world_size
        self.mp_ctx = mp.get_context("spawn")

        total_gpus = self.world_size
        if self.args.run_livephoto and not self.reconstruction_only:
            self.img_captioner = None
            if getattr(self.args, "livephoto_captioner", True):
                from qwen_vl3_images_caption_multigpu_offload import ImagesCaptioner

                self.img_captioner = ImagesCaptioner(
                    model_path=os.path.join(cur_py_path, "Cloud_Models/Qwen3-VL-8B-Instruct"),
                    num_gpus=min(self.world_size, 4),
                )

            # —— 模型 A：图片数 == 1 ——
            params_warmup_A = {
                "video_length": self.args.view_num,
                "sample_size_list": [],
                "prompt_embed_path": os.path.join(cur_py_path, "Cloud_Models/Wan2.2-Fun-5B-Control/prompt.pt"),
            }
            livephoto_model_offload = getattr(self.args, "livephoto_model_offload", True)
            livephoto_vae_offload = getattr(self.args, "livephoto_vae_offload", True)
            params_A = {
                "model_cpu_offload": livephoto_model_offload,
                "offload_vae": livephoto_vae_offload,
                "offload_model": livephoto_model_offload,
                "module_name": "VideoX-Fun.predict_i2v_4dlivephoto_standardvae",  
                "class_name": "Wan_5b",                                 
                "config_path": os.path.join(cur_py_path, "VideoX-Fun/config/wan2.2/wan_civitai_4dlp_5b.yaml"),
                "model_name": os.path.join(cur_py_path, "Cloud_Models/Wan2.2-TI2V-5B/"),
                "num_inference_steps": 8,
                "camera_movement": "arc_right_frame_cat",
                "num_selected_frames": 5,
                "use_fixed_key_frames": False,
                "use_vae": False,
                "video_length": self.args.view_num,
            }
            
            # pool_A 只用最后两张卡
            if total_gpus < 2:
                pool_A_device_ids = list(range(total_gpus))
                pool_A_world_size = total_gpus
            else:
                pool_A_device_ids = list(range(total_gpus - 2, total_gpus))
                pool_A_world_size = 2            
            logger.info(f"pool A device_ids: {pool_A_device_ids}")

            # pool_A 初始化时都建好并 warmup + offload
            self.pool_A = WorkerPool("A", pool_A_world_size, params_A, params_warmup_A, self.mp_ctx, device_ids=pool_A_device_ids)
            # self.pool_A.start()
            try:
                self.pool_A.start()
            except Exception:
                logger.error("pool A startup failed, cleaning up all workers...")
                self.shutdown_workers()   # 会遍历 pool_A / pool_B 逐个 shutdown
                raise

        self.prompt_embed_path = os.path.join(cur_py_path, "Cloud_Models/Wan2.2-Fun-5B-Control/prompt.pt")
        if self.args.run_gen_video:
            # —— 模型 B：图片数 >= 2 ——
            params_warmup_B = {
                "video_length": self.args.view_num,
                "sample_size_list": (
                    getattr(self.args, "video_warmup_sizes", [])
                    if getattr(self.args, "video_warmup", True) else []
                ),
                "prompt_embed_path": self.prompt_embed_path
            }
            params_B = {
                "module_name": "VideoX-Fun.predict_v2v_control_5b_dense",
                "offload_model": getattr(self.args, "video_offload", False),
                "offload_vae": getattr(self.args, "video_vae_offload", True),
                "video_batch_size": getattr(self.args, "video_batch_size", 1),
                "class_name": "Wan_5b",
                "config_path": os.path.join(cur_py_path, "VideoX-Fun/config/wan2.2/wan_civitai_5b.yaml"),
                "model_name": os.path.join(cur_py_path, "Cloud_Models/Wan2.2-Fun-5B-Control/"),
                "transformer_path": None,
                "lora_path": os.path.join(cur_py_path, "Cloud_Models/Wan2.2-Fun-5B-Control/12000_lora.safetensors"),
                "lora_weight": 0.75,
                "num_inference_steps": getattr(self.args, "gen_vid_infer_step", 8),
                "video_length": self.args.view_num,
            }
            # pool_B 正常使用全部卡
            if total_gpus < 6:
                pool_B_device_ids = list(range(total_gpus))
                pool_B_world_size = total_gpus
            else:
                pool_B_device_ids = list(range(6))
                pool_B_world_size = 6
            logger.info(f"pool B device_ids: {pool_B_device_ids}")

            # pool_B 初始化时都建好并 warmup + offload
            self.pool_B = WorkerPool("B", pool_B_world_size, params_B, params_warmup_B, self.mp_ctx, device_ids=pool_B_device_ids)
            startup_started = time.perf_counter()
            try:
                self.pool_B.start()
                logger.info(f"Pool B model startup (including warmup): {time.perf_counter() - startup_started:.2f}s")
            except Exception:
                logger.error("pool B startup failed, cleaning up all workers...")
                self.shutdown_workers()   # 会遍历 pool_A / pool_B 逐个 shutdown
                raise

        atexit.register(self.shutdown_workers)
        logger.info(f"[add shutdown] alloc={torch.cuda.memory_allocated()/1e9:.2f}G "
            f"reserved={torch.cuda.memory_reserved()/1e9:.2f}G")
        logger.info(f"-------------------recon_spatial模型初始化完成-------------------")
    
    def shutdown_workers(self):
        for name in ("pool_A", "pool_B"):
            pool = getattr(self, name, None)
            if pool is not None:
                pool.shutdown()

    def __del__(self):
        self.shutdown_workers()

    def __call__(self, work_dir:str, export_dir:str = None, mode:str='scene'):
        self.run_sparse_recon(work_dir, export_dir, mode)

    def xsj_sever_enter(self, imgs_list:list):
        logger.info(f"开始运行recon_spatial_xsj_Sever ......")
        logger.info(f"当前pipeline: {Path(__file__).name}")
        try:
            now = datetime.now()
            time_string = now.strftime("%Y-%m-%d_%H_%M_%S")
            if len(imgs_list) == 0:
                logger.error(f"no image received")
                return ['']
            work_dir = os.path.dirname(imgs_list[0])
            image_dir = os.path.join(work_dir, "image")
            logger.info(f"image_dir {image_dir}")
            os.makedirs(image_dir, exist_ok=True)
            for img_file in imgs_list:
                logger.info(f"img_file {img_file}")
                shutil.copy(img_file, image_dir)
            files_in_directory = []
            for entry in os.listdir(image_dir):
                full_path = os.path.join(image_dir, entry)
                if os.path.isfile(full_path):
                    files_in_directory.append(full_path)
            logger.info(files_in_directory)
            export_dir = os.path.join(work_dir, "export")
            self.run_sparse_recon(work_dir, export_dir, "scene")

            # output
            try:
                copy_colmap_project(work_dir , f"/workspace/z00566689/3DGS_result/{time_string}")
            except Exception as e:
                logger.error("copy colmap project failed!")
            ply_ori_path = os.path.join(export_dir, "point_cloud_final.ply")
            cam_ori_path = os.path.join(export_dir, "gs_camera_params_final.json")
            logger.info(f"ply_path {ply_ori_path} cam_jon_path {cam_ori_path}")
            if os.path.isfile(ply_ori_path) and os.path.isfile(cam_ori_path):
                logger.info(f"reconstruciton succeed")
                ply_path = os.path.join(export_dir, "point_cloud.ply")
                cam_path = os.path.join(export_dir, "gs_camera_params.json")
                os.rename(ply_ori_path, ply_path)
                os.rename(cam_ori_path, cam_path)
                file_list = [ply_path, cam_path]
                if os.path.isfile(ply_path) and os.path.isfile(cam_path):
                    md5_hash_ply = calculate_md5(ply_path)
                    md5_hash_cam_json = calculate_md5(cam_path)
                    logger.info(f"md5 of {ply_path} is {md5_hash_ply}, {cam_path} is {md5_hash_cam_json}")
                    data = {
                        "point_cloud_ply_md5":md5_hash_ply,
                        "gs_camera_params_json_md5":md5_hash_cam_json
                    }
                    json_file_path = os.path.join(export_dir, "md5.json")
                    with open(json_file_path, "w") as json_file:
                        json.dump(data, json_file, indent=4)
                    file_list.append(json_file_path)
                    logger.info(f"reconstruciton succeed return {file_list}")
                    return file_list
            logger.error("reconstruction failed: expected output files are missing")
            raise RuntimeError(f"reconstruciton failed")
        except PoolRecoveryError:
            # Preserve this type for the outer service/case loop. Recovery
            # failure must not be mistaken for a normal failed-but-ready case.
            logger.exception("worker recovery failed; service is not ready")
            raise
        except Exception as e:
            logger.exception("reconstruction failed for current request")
            raise RuntimeError(f"reconstruction failed: {e}") from e
        
    def _run_pool_A_livephoto(self, img_path, target_path, img_size):
        """单图前置：用 pool_A 生成 """
        pool = self.pool_A

        output_text = ""
        if self.img_captioner is not None:
            st_T = time.time()
            output_text = self.img_captioner.read_image_caption(img_path)
            print('### images caption:', output_text)
            _cleanup_cuda()
            calc_img_captioner_time = time.time() - st_T
            logger.info(f"calc_img_captioner_time time: {calc_img_captioner_time}s")

        if os.path.isdir(target_path):
            shutil.rmtree(target_path)
        os.makedirs(target_path, exist_ok=True)

        infer_params_dict = {
            "prompt": output_text,
            "image_path": img_path,
            "save_path": target_path,
            "video_length": self.args.view_num,
            "video_fps": 30,
            "num_inference_steps": 8,
            "index": [0, 1],
            "guidance_scale": 0.0,
            "img_size": img_size,
            "camera_movement": "arc_right_frame_cat",
            "use_fixed_key_frames": False,
            "num_selected_frames": 5,
        }

        st_T = time.time()
        try:
            _, failed = pool.broadcast_infer(infer_params_dict)
            if failed:
                raise RuntimeError(f"[pool A] current case failed: {failed}")
        finally:
            _, cleanup_failed = pool._offload_all_best_effort()
            if cleanup_failed:
                raise RuntimeError(f"[pool A] cleanup failed: {cleanup_failed}")
        cleanup_mode = "offload" if pool.model_params_dict["offload_model"] else "cache cleanup"
        logger.info(f"livephoto generation + {cleanup_mode}: {time.time() - st_T:.2f}s")
        return None  # Pool A cleanup completed synchronously.

    def _run_pool_B_gen_video(self, render_result, crop_image_path):
        """多图带引导信号：用 pool_B 生成 """
        pool = self.pool_B
        logger.info(f"multi images input -> 使用 pool {pool.name}")
        # if os.path.isdir(gen_vid_path):
        #     shutil.rmtree(gen_vid_path)
        os.makedirs(self.gen_vid_path, exist_ok=True)           

        gen_index = render_result.gen_index[0][:-1]
        frames = render_result.rendered_images
 
        input_video, _, _, _ = get_video_to_video_latent_with_imgsize(frames, img_size=self.img_size, fps=self.args.gen_vid_fps, ref_image=None)
        input_video = input_video.detach().cpu().contiguous().share_memory_()
        
        segment_index = []
        if self.save_necessary_only:
            st_T = time.time()
            segment_index = seg_index_sample(gen_index, self.colmap_path, num_frames=self.max_frames, video_length=self.args.view_num)
            colmap_sample_time = time.time() - st_T
            logger.info(f"colmap_camera_sampling took: {colmap_sample_time}s")

            img_extract_path = os.path.join(self.gen_vid_path, "image")
            if os.path.exists(img_extract_path):
                shutil.rmtree(img_extract_path)

        # Remove only generated video outputs before a new inference. Otherwise
        # a later images-only case could accidentally reuse an older debug video.
        for path in Path(self.gen_vid_path).glob("*.mp4"):
            if path.name == "result.mp4" or (len(path.stem) == 8 and path.stem.isdigit()):
                path.unlink()

        st_T = time.time()
        
        infer_params_dict = {
            "prompt": self.prompt,
            "image_path": crop_image_path,
            "save_path": self.gen_vid_path,
            "video_length": self.args.view_num,
            "video_fps": self.args.gen_vid_fps,
            "num_inference_steps": self.args.gen_vid_infer_step,
            "index": gen_index,
            "guidance_scale": 0.0,
            "img_size": self.img_size,
            "input_video": input_video,
            "save_img_ids": segment_index,
            "save_video": self.debug or not self.save_necessary_only,
            "prompt_embed_path": self.prompt_embed_path,
        }

        # Finish A's offload before B touches overlapping GPUs.
        if self.offload_A_future is not None:
            self.offload_A_future.result()
            self.offload_A_future = None

        st_T = time.time()
        try:
            _, failed = pool.broadcast_infer(infer_params_dict)
            if failed:
                # Never continue into COLMAP/3DGS using partial video outputs.
                raise RuntimeError(f"[pool B] current case failed: {failed}")
            logger.info(f"video generation: {time.time() - st_T:.2f}s")
        finally:
            cleanup_started = time.perf_counter()
            _, offload_failed = pool._offload_all_best_effort()
            if offload_failed:
                raise RuntimeError(f"[pool B] cleanup failed: {offload_failed}")
            cleanup_mode = "CPU offload" if getattr(self.args, "video_offload", False) else "cache cleanup"
            logger.info(f"Pool B {cleanup_mode}: {time.perf_counter() - cleanup_started:.2f}s")
        return None


    def _prepare_dense_inputs(self, gen_index, input_paths=None, prefer_video=False):
        started = time.perf_counter()
        count = prepare_reconstruction_images(
            gen_index[:-1], self.colmap_path, self.gen_vid_path,
            self.max_frames, self.args.view_num,
            input_paths=input_paths, prefer_video=prefer_video,
        )
        metadata_path = Path(self.gen_vid_path) / "reconstruction_selection.json"
        with metadata_path.open(encoding="utf-8") as file:
            metadata = json.load(file)
        logger.info(
            f"Reconstruction images: {self.max_frames} generated + {len(metadata['input_names'])} inputs"
            f" = {count}; sampled from {metadata['sampled_from']}; prepare took {time.perf_counter() - started:.2f}s"
        )
        return count

    def _run_dense_iter(self, frame_count, export_dir, mode, circle):
        started = time.perf_counter()
        self.recon_iter_full(
            work_dir=Path(self.gen_vid_path), export_dir=Path(export_dir),
            mode="scene" if self.is_livephoto else mode, circle=circle,
            max_frames=frame_count, use_exists_only=True,
        )
        # Every mode now supplies the exact selected images and their GT.
        # Never let the downstream video reader sample a second time.
        torch.cuda.empty_cache()
        return time.perf_counter() - started

    def init_recon(self, work_dir:str):
        self.img_path = os.path.join(work_dir, "image")
        self.list_imgs = [x for x in os.listdir(self.img_path) if x.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.list_imgs.sort()

        print(f"total num of img {len(self.list_imgs)}")
        if len(self.list_imgs) < 1:
            logger.error(f"not enough images")
            return -1

        self.gs_render_path = os.path.join(work_dir, "3dgs_render")
        # self.gs_render_vid_path =  os.path.join(self.gs_render_path, "gs_render.mp4")
        self.gen_vid_path = os.path.join(work_dir, "gen_video")
        self.vid_path = f"{self.gen_vid_path}/result.mp4"
        self.livephoto_vid_path = f"{self.gen_vid_path}/livephoto.mp4"
        self.colmap_path = os.path.join(self.gen_vid_path, "sparse", "0")
        self.live_photo_video = os.path.join(work_dir, "livephoto_gen")
        self.live_photo_image = os.path.join(work_dir, "livephoto_gen_image")

        os.makedirs(self.gs_render_path, exist_ok=True)
        os.makedirs(self.gen_vid_path, exist_ok=True)
        os.makedirs(self.colmap_path, exist_ok=True)
        os.makedirs(self.export_dir, exist_ok=True)

        if len(self.list_imgs) == 1:
            self.is_livephoto = True
        else:
            self.is_livephoto = False

    def run_sparse_recon(self, work_dir:str, export_dir:str = None, mode:str='auto-scene'):
        ffrecon_time = 0.0
        gen_vid_time = 0.0
        iter_3dgs_time = 0.0
        run_livephoto_time = 0.0
        
        work_dir = os.path.abspath(work_dir)
        self.export_dir = export_dir
        print(f"sparse recon pipline in {work_dir}")

        logger.remove()
        logger.add(sys.stderr, 
            format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{module}</cyan>:<cyan>{function}</cyan> - {message}")
        log_path = os.path.join(work_dir, "pipeline.log")
        logger.add(log_path, format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {module}:{function} - {message}")

        self.prompt = 'static scene with a people or an obejct'
        st_T_total = time.time()

        cur_py_path = os.path.dirname(os.path.realpath(__file__))
        print(f"sparse pipeline at: {cur_py_path}")

        # 预处理信息
        self.init_recon(work_dir)
        configured_max = getattr(self.args, "max_frames", None)
        self.max_frames = configured_max if configured_max is not None else (40 if self.is_livephoto else 80)
        self.offload_A_future = self.offload_B_future = None
        if self.reconstruction_only:
            plan_path = Path(self.gs_render_path) / "gen_index.json"
            with plan_path.open(encoding="utf-8") as file:
                gen_index = json.load(file)
            count = self._prepare_dense_inputs(gen_index, prefer_video=True)
            elapsed = self._run_dense_iter(count, export_dir, mode, gen_index[-1])
            logger.info(f"iter 3dgs took: {elapsed}s")
            logger.info(f"sparse_recon took: {time.time() - st_T_total}s in total (reconstruction only)")
            return export_dir

        logger.info(align_images_resolution(self.img_path, os.path.join(work_dir, "image_ori")))
                
        img1st = cv2.imread(os.path.join(self.img_path, self.list_imgs[0]))
        source_height, source_width, _ = img1st.shape

        target_short_edge   = 1088
        align_to             = 32

        if source_height >= source_width:
            scale = target_short_edge / source_width
            new_width = target_short_edge
            new_height = int(round(source_height * scale))
        else:
            scale = target_short_edge / source_height
            new_height = target_short_edge
            new_width = int(round(source_width * scale))

        # Wan VAE/DiT spatial dimensions are kept on a 32-pixel grid.
        height = max(align_to, int(round(new_height / align_to) * align_to))
        width = max(align_to, int(round(new_width / align_to) * align_to))

        self.img_size = (width, height)

        # 处理图像ICC drop的问题，在最开始就去掉，默认后面都无需管了
        # Warning! 用Image.save()可能会保留图片原本的分辨率，导致分辨率不对齐。
        if self.args.use_input_icc_drop and not self.args.use_resize_drop:
            img_ori_path = os.path.join(work_dir, "image_ori")
            if not os.path.exists(img_ori_path):
                drop_img_icc(self.img_path, img_ori_path)

        # 这里把逻辑变了，原图文件夹保持不变，以免影响其他的算法执行
        if self.args.use_resize_drop:
            print(f"resize images for matching")
            img_resize_path = os.path.join(work_dir, "image_resize")
            # if not os.path.exists(img_resize_path):
            resize_img(
                self.img_path, img_resize_path, self.img_size,
                drop_icc=self.args.use_input_icc_drop,
            )
            self.img_path = img_resize_path

        end_T = time.time()
        logger.info(f"initialization took time: {end_T-st_T_total}s")

        # Preserve the real inputs before LivePhoto replaces self.img_path.
        gt_image_paths = [os.path.join(self.img_path, name) for name in self.list_imgs]
        self.offload_A_future = None
        if self.is_livephoto:
            # 用livephoto补充新视角
            if self.args.run_livephoto:
                st_T = time.time()
                logger.info("run livephoto to augment images first")
                self.offload_A_future = self._run_pool_A_livephoto(self.img_path, self.live_photo_video, self.img_size)
                _cleanup_cuda()

                # 清空残留
                os.makedirs(self.live_photo_image, exist_ok=True)
                os.system(f"rm -rf {self.live_photo_image}/*")
                # 保存新视角
                insert_img_in_src(self.live_photo_video, self.live_photo_image, self.img_size[0], self.img_size[1], self.livephoto_vid_path, st_index = 61)

                run_livephoto_time = time.time() - st_T
                logger.info(f"livephoto infer took: {run_livephoto_time}s")

            # os.makedirs(self.live_photo_image, exist_ok=True)
            # os.system(f"rm -rf {self.live_photo_image}/*")
            # insert_img_in_src(self.live_photo_video, self.live_photo_image, self.img_size[0], self.img_size[1], self.livephoto_vid_path, st_index = 41)

            # 当前默认采用livephoto跑完切到多图    
            # TODO: 纯单图+单图几何重建生成
            if not os.path.exists(self.live_photo_image) or len(os.listdir(self.live_photo_image)) == 0:
                logger.error("单图输入，但没有进行扩图")
            else:
                self.img_path = self.live_photo_image

            self.list_imgs = [x for x in os.listdir(self.img_path) if x.lower().endswith(('.png', '.jpg', '.jpeg'))]
            self.list_imgs.sort()
            if self.img_path == self.live_photo_image:
                # The center view must be the actual input, not a re-encoded
                # generated frame. DA3 estimates its pose together with both sides.
                if len(self.list_imgs) != 3:
                    raise ValueError("Expected three LivePhoto views: left, input, right")
                center_path = Path(self.img_path) / self.list_imgs[1]
                input_path = center_path.with_suffix(".png")
                input_image = cv2.imread(gt_image_paths[0], cv2.IMREAD_COLOR)
                if input_image is None:
                    raise ValueError(f"Cannot read input image: {gt_image_paths[0]}")
                if (input_image.shape[1], input_image.shape[0]) != self.img_size:
                    input_image = cv2.resize(input_image, self.img_size, interpolation=cv2.INTER_AREA)
                if not cv2.imwrite(str(input_path), input_image):
                    raise RuntimeError(f"Cannot save input image: {input_path}")
                if input_path != center_path:
                    center_path.unlink()
                gt_image_paths = [str(input_path)]
                self.list_imgs = sorted(
                    name for name in os.listdir(self.img_path)
                    if name.lower().endswith(('.png', '.jpg', '.jpeg'))
                )


        # 进行画面裁剪，实现局部增强
        # 这里涉及两次，
        #   第一次得到crop ratio给到前馈式用于轨迹生成
        #   第二次对图像进行裁切，用于视频生成局部增强
        st_T = time.time()
        if self.args.use_enhance_crop and not self.is_livephoto:
            crop_mask_path = os.path.join(work_dir, "crop_image_mask")
            os.makedirs(crop_mask_path, exist_ok=True)
            try:
                crop_ratio = get_crop_ratio_w_face_track(
                    self.img_path,
                    crop_mask_path,
                    self.seg_module,
                    self.face_detector,
                )
            finally:
                self.seg_module.offload()
                self.face_detector.offload()

                gc.collect()  # 文件顶部增加 import gc

                if torch.cuda.is_initialized():
                    for device in {
                        self.seg_module.device,
                        self.face_detector.device,
                    }:
                        if device.type == "cuda":
                            with torch.cuda.device(device):
                                torch.cuda.empty_cache()
            if crop_ratio > 0.9:
                crop_ratio = 0.7
            
            # for video gen
            crop_image_path = os.path.join(work_dir, "crop_image")
            os.makedirs(crop_image_path, exist_ok=True)
            
            for i in range(len(self.list_imgs)):
                os.system(f"cp {self.img_path}/{self.list_imgs[i]} {crop_image_path}/{i:05d}.jpg")
                if crop_ratio < 1:
                    crop_image(f"{self.img_path}/{self.list_imgs[i]}", f"{crop_image_path}/{(i+len(self.list_imgs)):05d}.jpg", crop_ratio)
        else:
            crop_ratio = 1
            crop_image_path = self.img_path

        end_T = time.time()
        logger.info(f"images cropping took time: {end_T-st_T}s")        

        logger.info("start estimate camera views...")
        st_T = time.time()
        # run DA3
        if self.args.run_DA3:
            # os.makedirs(self.gs_render_path, exist_ok=True)
            @dataclass
            class DA3Params:
                data_path: str
                colmap_path: str
                gs_render_path: str
                view_num: int
                crop_ratio: float
                extra_view: list

            render_result = self.run_DA3.infer(args=DA3Params(
                data_path=self.img_path,
                colmap_path=self.colmap_path,
                gs_render_path=self.gs_render_path,
                view_num=self.args.view_num,
                crop_ratio=crop_ratio,
                extra_view=[self.args.use_view_extrapolation, self.is_livephoto] # 默认只有单图才走livephoto模式 
            ))
            if self.debug:
                debug_video_path = os.path.join(self.gs_render_path, "gs_render.mp4")
                frames = render_result.rendered_images
                first_frame = torch.as_tensor(frames[0])
                if first_frame.ndim != 3:
                    raise ValueError(f"Expected a DA3 RGB frame, got {first_frame.shape}")
                channel_first = first_frame.shape[-1] != 3 and first_frame.shape[0] == 3
                height, width = first_frame.shape[1:] if channel_first else first_frame.shape[:2]
                writer = cv2.VideoWriter(
                    debug_video_path, cv2.VideoWriter_fourcc(*"mp4v"),
                    self.args.gen_vid_fps, (width, height),
                )
                if not writer.isOpened():
                    writer.release()
                    raise RuntimeError(f"Cannot open DA3 debug video: {debug_video_path}")
                try:
                    for frame in frames:
                        frame = torch.as_tensor(frame).detach().cpu()
                        if channel_first:
                            frame = frame.permute(1, 2, 0)
                        if frame.is_floating_point():
                            frame = frame.clamp(0, 1).mul(255)
                        frame = frame.to(torch.uint8).contiguous().numpy()
                        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                finally:
                    writer.release()
                if os.path.getsize(debug_video_path) == 0:
                    raise RuntimeError(f"DA3 debug video is empty: {debug_video_path}")
                logger.info(f"Saved DA3 debug video: {debug_video_path}")
            gen_index, enhance = render_result.gen_index
            logger.info(f"gen_index: {gen_index}")

            # 如果进行画面裁剪，则修改对应图片下标（对应裁剪后图片）
            if self.args.use_enhance_crop and not self.is_livephoto:
                for i in range(len(enhance)):
                    if enhance[i] == 1:
                        for j in range(len(gen_index[i])):
                            gen_index[i][j][0] += len(self.list_imgs)

            # 持久化轨迹计划（含末尾circle标记），供 run_DA3=false 的分阶段重跑读取
            with open(os.path.join(self.gs_render_path, "gen_index.json"), "w", encoding="utf-8") as f:
                json.dump(gen_index, f, default=int)
            circle = gen_index[-1]

            _cleanup_cuda()

        elif self.args.run_gen_video or self.args.run_3dgs_iter:
            # 分阶段重跑：从磁盘读取上次 DA3 保存的轨迹计划
            _gen_index_path = os.path.join(self.gs_render_path, "gen_index.json")
            if not os.path.isfile(_gen_index_path):
                raise FileNotFoundError(
                    f"run_DA3=false 但找不到 {_gen_index_path}；"
                    f"请先完整跑一次 run_DA3=true 生成轨迹文件")
            with open(_gen_index_path, "r", encoding="utf-8") as f:
                gen_index = json.load(f)
            circle = gen_index[-1]

        ffrecon_time = time.time()-st_T
        logger.info(f"ffrecon took: {ffrecon_time}s")

        img_extract_path = os.path.join(self.gen_vid_path, "image")
        # vid_colmap_path = os.path.join(self.gen_vid_path, "sparse", "0")

        self.offload_B_future = None
        prepared_frame_count = None
        if self.args.run_gen_video:
            logger.info("run video gen to denoise gs_render video")
            st_T = time.time()     
            self.offload_B_future = self._run_pool_B_gen_video(render_result, crop_image_path)

            if self.debug and self.save_necessary_only:
                writer = None
                try:
                    for segment_id in range(len(render_result.gen_index[0]) - 1):
                        segment_path = os.path.join(self.gen_vid_path, f"{segment_id:08d}.mp4")
                        capture = cv2.VideoCapture(segment_path)
                        try:
                            ok, frame = capture.read()
                            if not ok:
                                raise RuntimeError(f"Cannot read generated video: {segment_path}")
                            if writer is None:
                                frame_size = (frame.shape[1], frame.shape[0])
                                writer = cv2.VideoWriter(
                                    self.vid_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                    self.args.gen_vid_fps, frame_size,
                                )
                                if not writer.isOpened():
                                    raise RuntimeError(f"Cannot open merged video: {self.vid_path}")
                            while ok:
                                if (frame.shape[1], frame.shape[0]) != frame_size:
                                    raise ValueError(f"Generated video size mismatch: {segment_path}")
                                writer.write(frame)
                                ok, frame = capture.read()
                        finally:
                            capture.release()
                    if writer is None:
                        raise RuntimeError("No generated video segments to merge")
                finally:
                    if writer is not None:
                        writer.release()
                logger.info(f"Saved merged debug video: {self.vid_path}")

            if not self.save_necessary_only:
                merge_videos(self.gen_vid_path, self.vid_path)
            prepared_frame_count = self._prepare_dense_inputs(
                gen_index, gt_image_paths, prefer_video=not self.save_necessary_only,
            )
            
            gen_vid_time = time.time() - st_T
            
        # os.makedirs(vid_colmap_path, exist_ok=True)
        # shutil.copytree(self.colmap_path, vid_colmap_path, dirs_exist_ok=True)
        
        if self.args.run_3dgs_iter:
            if prepared_frame_count is None:
                prepared_frame_count = self._prepare_dense_inputs(
                    gen_index, gt_image_paths, prefer_video=True,
                )
            iter_3dgs_time = self._run_dense_iter(
                prepared_frame_count, export_dir, mode, circle,
            )
            logger.info(f"iter3dgs videos took: {iter_3dgs_time}s")


        logger.info(f"\n-------total elapsed time-------")
        logger.info(f"4dlivephoto took: {run_livephoto_time}s")
        logger.info(f"ffrecon took: {ffrecon_time}s")
        logger.info(f"gen vid took: {gen_vid_time}s")
        logger.info(f"iter 3dgs took: {iter_3dgs_time}s")
        logger.info(f"sparse_recon took: {time.time()-st_T_total}s in total")

        if self.offload_A_future is not None:
            self.offload_A_future.result()
        if self.offload_B_future is not None:
            self.offload_B_future.result()

        return export_dir
