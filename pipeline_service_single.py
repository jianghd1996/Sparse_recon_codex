import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["MKL_THREADING_LAYER"] = "GNU"

import atexit
import json
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from tempfile import mkdtemp

import cv2
import torch
import torch.multiprocessing as mp
from loguru import logger

from utils.multiprocess_worker_pool import _cleanup_cuda, WorkerPool
from utils.utils import DictToObject


@dataclass
class DA3Params:
    data_path: str
    colmap_path: str
    gs_render_path: str
    view_num: int
    crop_ratio: float
    extra_view: list


def sample_frame_indices(video_length, sample_count):
    if not 2 <= sample_count <= video_length:
        raise ValueError("Expected 2 <= sample_count <= video_length")
    return [round(index * (video_length - 1) / (sample_count - 1)) for index in range(sample_count)]


def _colmap_image_names(colmap_path):
    binary_path = colmap_path / "images.bin"
    if binary_path.is_file():
        names = set()
        with binary_path.open("rb") as model:
            image_count, = struct.unpack("<Q", model.read(8))
            for _ in range(image_count):
                model.seek(64, os.SEEK_CUR)  # image ID, quaternion, translation, camera ID
                name = bytearray()
                while True:
                    character = model.read(1)
                    if not character:
                        raise ValueError(f"Truncated COLMAP image record in {binary_path}")
                    if character == b"\0":
                        break
                    name.extend(character)
                names.add(name.decode("utf-8"))
                point_count, = struct.unpack("<Q", model.read(8))
                model.seek(point_count * 24, os.SEEK_CUR)
        return names

    with (colmap_path / "images.txt").open(encoding="utf-8") as model:
        records = (line for line in model if not line.lstrip().startswith("#"))
        names = set()
        for line in records:
            if line.strip():
                names.add(line.split(maxsplit=9)[9].strip())
                next(records)  # POINTS2D row, which may be empty
    return names


DEFAULT_TI2V_CONFIG = {
    # Set this to the checkpoint for the 121-frame TI2V model.
    "lora_path": "Cloud_Models/Wan2.2-TI2V-5B/lora_ckpts_single8/checkpoint-9600.safetensors",
    "lora_weight": 0.55,
    "model_cpu_offload": True,
    "video_length": 121,
    "video_fps": 24,
    "sample_count": 40,
    "num_inference_steps": 8,
    "guidance_scale": 6.0,
    "shift": 5.0,
    "seed": 43,
    "short_edge": 704,
    "long_edge": 1280,
    "prompt": "",
    "negative_prompt": "",
}


class SparseRecon:
    """One input image, one TI2V clip, DA3 and iterative 3DGS."""

    def __init__(self, json_args):
        if type(json_args) != dict:
            json_args = json.loads(json_args)
        self.args = DictToObject({**DEFAULT_TI2V_CONFIG, **json_args})
        self.frame_indices = sample_frame_indices(self.args.video_length, self.args.sample_count)
        if (self.args.video_length - 1) % 4:
            raise ValueError("Wan2.2 TI2V requires a video length of 1 + n * 4")
        if self.args.run_3dgs_iter and not self.args.run_DA3:
            raise ValueError("This entry point requires DA3 for iterative 3DGS")
        if not Path(self.args.lora_path).is_file():
            raise FileNotFoundError(f"TI2V LoRA not found: {self.args.lora_path}")
        self.build_model()

    def build_model(self):
        cur_py_path = Path(__file__).resolve().parent
        if self.args.run_DA3:
            from DA3.run_da3_full_crop_dev import run_DA3
            DA3_model_path = cur_py_path / "Cloud_Models/Saliency-Face-DA3/besthumanmodel_lr1e_6_ckpt2.pt"
            self.run_DA3 = run_DA3(str(DA3_model_path))

        if self.args.run_3dgs_iter:
            from Reconstruction_full import Reconstruction
            self.recon_iter_full = Reconstruction(preload=True, dense_init_scene=False)

        model_params = {
            "module_name": "VideoX-Fun.predict_ti2v_single",
            "class_name": "Wan_5b",
            "config_path": getattr(self.args, "config_path", str(
                cur_py_path / "VideoX-Fun/config/wan2.2/wan_ti2v_single_5b.yaml"
            )),
            "model_name": getattr(self.args, "model_name", str(
                cur_py_path / "Cloud_Models/Wan2.2-TI2V-5B"
            )),
            "lora_path": self.args.lora_path,
            "lora_weight": self.args.lora_weight,
            "model_cpu_offload": self.args.model_cpu_offload,
            "offload_model": self.args.model_cpu_offload,
        }
        self.pool_A = WorkerPool(
            "TI2V", 1, model_params, {"sample_size_list": []}, mp.get_context("spawn"),
            device_ids=[getattr(self.args, "device_id", 0)],
        )
        try:
            self.pool_A.start()
        except Exception:
            self.shutdown_workers()
            raise
        atexit.register(self.shutdown_workers)

    def shutdown_workers(self):
        pool = getattr(self, "pool_A", None)
        if pool is not None:
            self.pool_A = None
            pool.shutdown()

    def __call__(self, work_dir: str, export_dir: str = None, mode: str = "scene"):
        return self.run_sparse_recon(work_dir, export_dir, mode)

    def run_sparse_recon(self, work_dir: str, export_dir: str = None, mode: str = "scene"):
        work_dir = Path(work_dir).resolve()
        sources = sorted(
            path for path in (work_dir / "image").iterdir()
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )
        if len(sources) != 1:
            raise ValueError(f"Expected exactly one input image in {work_dir / 'image'}, got {len(sources)}")
        source = cv2.imread(str(sources[0]))
        if source is None:
            raise ValueError(f"Could not read {sources[0]}")
        height, width = source.shape[:2]
        del source

        run_root = work_dir / "single_ti2v"
        run_root.mkdir(exist_ok=True)
        run_dir = Path(mkdtemp(prefix="run_", dir=run_root))
        gen_vid_path = run_dir / "gen_video"
        image_path = gen_vid_path / "image"
        colmap_path = gen_vid_path / "sparse" / "0"
        gs_render_path = run_dir / "3dgs_render"
        export_dir = Path(export_dir).resolve() if export_dir is not None else run_dir / "export"
        image_path.mkdir(parents=True)
        logger.info("Single TI2V work directory: {}", run_dir)
        run_dir_log_id = logger.add(run_dir / "pipeline.log")
        started = time.perf_counter()
        try:
            self._run_ti2v(sources[0], gen_vid_path, (width, height))
            expected_names = {f"{index:05d}.png" for index in self.frame_indices}
            if {path.name for path in image_path.iterdir()} != expected_names:
                raise RuntimeError(f"TI2V did not save the expected sampled frames in {image_path}")
            if self.args.run_DA3:
                circle = self._run_da3(image_path, colmap_path, gs_render_path, expected_names)
            if self.args.run_3dgs_iter:
                stage_started = time.perf_counter()
                export_dir.mkdir(parents=True, exist_ok=True)
                self.recon_iter_full(
                    work_dir=gen_vid_path, export_dir=export_dir, mode=mode, circle=circle,
                    max_frames=len(self.frame_indices), use_exists_only=True,
                )
                for filename in ("point_cloud_final.ply", "gs_camera_params_final.json"):
                    if not (export_dir / filename).is_file():
                        raise RuntimeError(f"Reconstruction did not export {export_dir / filename}")
                torch.cuda.empty_cache()
                logger.info("iter 3dgs took: {:.2f}s", time.perf_counter() - stage_started)
            logger.info("sparse_recon took: {:.2f}s in total", time.perf_counter() - started)
            return export_dir if self.args.run_3dgs_iter else run_dir
        finally:
            logger.remove(run_dir_log_id)

    def _run_ti2v(self, image_path, target_path, img_size):
        infer_params = {
            "prompt": self.args.prompt,
            "negative_prompt": self.args.negative_prompt,
            "image_path": str(image_path),
            "save_path": str(target_path),
            "img_size": img_size,
            "video_length": self.args.video_length,
            "video_fps": self.args.video_fps,
            "num_inference_steps": self.args.num_inference_steps,
            "guidance_scale": self.args.guidance_scale,
            "shift": self.args.shift,
            "seed": self.args.seed,
            "short_edge": self.args.short_edge,
            "long_edge": self.args.long_edge,
            "frame_indices": self.frame_indices,
        }
        started = time.perf_counter()
        try:
            _, failed = self.pool_A.broadcast_infer(infer_params)
            if failed:
                raise RuntimeError(f"TI2V worker failed: {failed}")
        finally:
            _, cleanup_failed = self.pool_A._offload_all_best_effort()
            if cleanup_failed:
                logger.error("TI2V worker cleanup failed: {}", cleanup_failed)
        if cleanup_failed:
            raise RuntimeError(f"TI2V worker cleanup failed: {cleanup_failed}")
        logger.info("TI2V generation and cleanup took: {:.2f}s", time.perf_counter() - started)

    def _run_da3(self, image_path, colmap_path, gs_render_path, expected_names):
        colmap_path.mkdir(parents=True)
        gs_render_path.mkdir()
        started = time.perf_counter()
        render_result = self.run_DA3.infer(args=DA3Params(
            data_path=str(image_path), colmap_path=str(colmap_path),
            # This controls the legacy render plan, not the number of input images.
            gs_render_path=str(gs_render_path), view_num=81,
            crop_ratio=1.0, extra_view=[False, False],
        ))
        if self.args.run_3dgs_iter:
            gen_index, _ = render_result.gen_index
            if len(gen_index) == 0:
                raise RuntimeError("DA3 did not return the circle flag required by Reconstruction_full")
            circle = gen_index[-1]
        else:
            circle = None
        del render_result
        _cleanup_cuda()
        logger.info("ffrecon took: {:.2f}s", time.perf_counter() - started)

        for stem in ("cameras", "images", "points3D"):
            if not any((colmap_path / f"{stem}.{ext}").is_file() for ext in ("bin", "txt")):
                raise RuntimeError(f"DA3 did not export COLMAP {stem} to {colmap_path}")
        missing_names = expected_names - _colmap_image_names(colmap_path)
        if missing_names:
            raise RuntimeError(
                f"DA3 COLMAP export lacks {len(missing_names)} sampled images "
                f"(first: {', '.join(sorted(missing_names)[:5])}). "
                "The local DA3 wrapper must export cameras for the sampled input filenames."
            )
        return circle
