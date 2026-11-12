"""Qwen3-VL captioning: GPU sharding during generate, whole-model CPU offload between calls.

Run this captioner in ONE process with access to all selected GPUs. It is not
tensor parallelism and must not be replicated by every torchrun rank.
This implementation targets ordinary FP16/BF16 Transformers checkpoints,
not quantized models, DDP/FSDP wrappers, torch.compile, or disk-offloaded models.
"""

import argparse
import gc
import json
import warnings
from collections import Counter
from pathlib import Path

import torch
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.hooks import remove_hook_from_module
from accelerate.utils import get_balanced_memory
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForImageTextToText, AutoProcessor


def parse_gib(memory_str: str) -> int:
    value = memory_str.strip().lower()
    for suffix, multiplier in (("gib", 1024 ** 3), ("gb", 1000 ** 3)):
        if value.endswith(suffix):
            result = int(float(value[:-len(suffix)]) * multiplier)
            if result <= 0:
                raise ValueError("Memory budget must be positive")
            return result
    raise ValueError(f"Unsupported memory format: {memory_str!r}; use '32GiB' or '32GB'")


def build_max_memory_from_total(
    num_gpus=2, total_gpu_memory="32GiB", cpu_memory="64GiB", gpu_ids=None
):
    visible = torch.cuda.device_count()
    gpu_ids = list(range(num_gpus)) if gpu_ids is None else list(gpu_ids)
    if num_gpus < 1 or len(gpu_ids) != num_gpus or len(set(gpu_ids)) != num_gpus:
        raise ValueError("num_gpus must be positive and match the unique gpu_ids")
    if any(i < 0 or i >= visible for i in gpu_ids):
        raise ValueError(f"gpu_ids={gpu_ids}, but only {visible} CUDA devices are visible")
    per_gpu = parse_gib(total_gpu_memory) // num_gpus
    # Explicitly exclude visible GPUs not selected by this captioner.
    budgets = {i: (per_gpu if i in gpu_ids else 0) for i in range(visible)}
    if cpu_memory is not None:
        budgets["cpu"] = parse_gib(cpu_memory)
    return budgets


def resize_image(img, max_side=None, max_pixels=None):
    w, h = img.size
    scale = 1.0
    if max_side is not None:
        if max_side <= 0:
            raise ValueError("max_side must be positive")
        scale = min(scale, max_side / max(w, h))
    if max_pixels is not None:
        if max_pixels <= 0:
            raise ValueError("max_pixels must be positive")
        scale = min(scale, (max_pixels / (w * h)) ** 0.5)
    if scale < 1:
        return img.resize(
            (max(1, round(w * scale)), max(1, round(h * scale))),
            Image.Resampling.LANCZOS,
        )
    return img


def read_all_images_from(folder_path, min_images=1, max_images=8,
                         max_side=None, max_pixels=None):
    folder = Path(folder_path)
    if not folder.is_dir():
        raise ValueError(f"Image folder does not exist: {folder}")
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    paths = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in extensions)
    if len(paths) < min_images:
        raise ValueError(f"Need at least {min_images} images, found {len(paths)} in {folder}")
    images = []
    try:
        for path in paths[:max_images]:
            with Image.open(path) as image:
                rgb = image.convert("RGB")
            resized = resize_image(rgb, max_side, max_pixels)
            if resized is not rgb:
                rgb.close()
            images.append(resized)
        return images
    except BaseException:
        for image in images:
            image.close()
        raise


def build_messages(image_list):
    content = [{"type": "image", "image": image} for image in image_list]
    content.append({"type": "text", "text": (
        "The images show the same static scene captured from nearby views. "
        "Generate one caption that preserves important visual details from all images. "
        "Include objects, spatial layout, materials, colors, and background elements visible in any image, "
        "while describing them as part of one coherent scene. "
        "Do not say that anything changes, moves, appears, disappears, is added, or is removed. "
        "Do not mention the camera, viewpoint, frame, angle, or perspective. "
        "Do not infer actions or events. "
        "Output a single natural sentence in detail. "
        "Start with 'This video shows'."
    )})
    return [{"role": "user", "content": content}]


class ImagesCaptioner:
    def __init__(
        self,
        model_path=None,
        model_name="Qwen3-VL-8B-Instruct",
        max_side=960,
        max_pixels=None,
        num_gpus=2,
        total_gpu_memory="32GiB",
        cpu_memory="64GiB",
        device_map="balanced",
        attn_implementation="sdpa",
        *,
        gpu_ids=None,
        auto_offload=True,
        allow_cpu_layers=False,
        verbose=False,
    ):
        """Preserves the original constructor's positional argument order.

        Initially loads into CPU RAM only. onload() dispatches to the saved map;
        offload() detaches hooks then moves all live weights/buffers back to CPU.
        read_image_caption() calls both automatically unless auto_offload=False.

        allow_cpu_layers=False ensures generation weights stay on GPUs: no slow
        per-token CPU weight transfers. If the map exceeds the GPU budgets,
        initialization reports the issue instead of silently degrading to CPU.
        Set True to allow the original GPU+CPU spill behavior; that is slower.
        cpu_memory is a mapping budget, NOT a limit on actual host RAM use.
        """
        if device_map not in {"auto", "balanced", "balanced_low_0", "sequential"}:
            raise ValueError("device_map must be auto/balanced/balanced_low_0/sequential")
        self.model_path = model_path or model_name
        self.model_name = model_name
        self.dtype = torch.bfloat16
        self.max_side = max_side
        self.max_pixels = max_pixels
        self.num_gpus = num_gpus
        self.gpu_ids = list(range(num_gpus)) if gpu_ids is None else list(gpu_ids)
        self.auto_offload = auto_offload
        self.verbose = verbose
        self._state = "cpu"
        self.max_memory = build_max_memory_from_total(
            num_gpus, total_gpu_memory, cpu_memory, self.gpu_ids
        )

        # CPU-only loading prevents a GPU memory spike during initialization.
        # Do not apply cpu_offload(): weights should remain GPU-resident while
        # generating a caption, distributed over the selected GPUs.
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_path,
            dtype=self.dtype,
            device_map={"": "cpu"},
            attn_implementation=attn_implementation,
            low_cpu_mem_usage=True,
        ).eval()
        if getattr(self.model, "is_loaded_in_4bit", False) or getattr(self.model, "is_loaded_in_8bit", False):
            raise ValueError("This onload/offload implementation requires a non-quantized checkpoint")
        self.model.requires_grad_(False)
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.assert_offloaded()

        # Keep decoder and vision residual blocks intact, as in HF auto dispatch.
        no_split = sorted({
            name
            for module in self.model.modules()
            for name in (getattr(module, "_no_split_modules", None) or [])
        })
        budgets = dict(self.max_memory)
        if device_map != "sequential":
            budgets = get_balanced_memory(
                self.model,
                max_memory=budgets,
                no_split_module_classes=no_split,
                dtype=self.dtype,
                low_zero=(device_map == "balanced_low_0"),
            )
        self.active_device_map = dict(infer_auto_device_map(
            self.model,
            max_memory=budgets,
            no_split_module_classes=no_split,
            dtype=self.dtype,
        ))
        targets = set(self.active_device_map.values())
        if "disk" in targets:
            raise ValueError(
                "The computed map requires disk offload, which this script does not support. "
                "Increase the GPU/CPU budgets or use a smaller checkpoint."
            )
        if "cpu" in targets and not allow_cpu_layers:
            raise ValueError(
                "Some layers do not fit in the GPU mapping budget. Increase total_gpu_memory "
                "only if physical free VRAM permits it, use balanced rather than balanced_low_0, "
                "or set allow_cpu_layers=True for slower GPU+CPU generation."
            )
        self.used_gpu_ids = sorted({
            int(target) for target in targets if isinstance(target, int)
        })
        if not self.used_gpu_ids:
            raise ValueError("No model layers were assigned to a GPU; increase the GPU budget")
        if any(i not in self.gpu_ids for i in self.used_gpu_ids):
            raise RuntimeError("The inferred map assigned a layer to an unselected GPU")
        self.input_device = torch.device(f"cuda:{self.used_gpu_ids[0]}")
        self._skip_keys = getattr(self.model, "_skip_keys_device_placement", None)
        if num_gpus > 1 and len(self.used_gpu_ids) < num_gpus:
            warnings.warn(
                f"Requested {num_gpus} GPUs, but the inferred map uses {self.used_gpu_ids}. "
                "balanced_low_0 can put the whole model on another card; try device_map='balanced'."
            )
        if verbose:
            print("Caption GPU ids (logical):", self.gpu_ids)
            print("Caption saved inference device map:")
            print(json.dumps(self.active_device_map, indent=2))
            print("Caption starts offloaded on CPU; actual GPU placement is printed after onload().")

    def _clear_cuda_cache(self):
        gc.collect()
        for gpu in self.gpu_ids:
            with torch.cuda.device(gpu):
                torch.cuda.empty_cache()

    def assert_offloaded(self):
        """Verifies model tensors, rather than relying on nvidia-smi/cache size."""
        bad = [
            f"{name}={tensor.device}"
            for name, tensor in list(self.model.named_parameters()) + list(self.model.named_buffers())
            if tensor.device.type != "cpu"
        ]
        if bad:
            raise RuntimeError("Caption is not fully on CPU: " + ", ".join(bad[:8]))

    def offload(self):
        """Move the existing model from ALL mapped GPUs back to CPU; no disk reload."""
        if self._state == "cpu":
            self.assert_offloaded()
            return self
        # Complete pending computation before dismantling cross-device hooks.
        for gpu in self.used_gpu_ids:
            torch.cuda.synchronize(gpu)
        remove_hook_from_module(self.model, recurse=True)
        # Older Accelerate versions may retain root movement-warning wrappers.
        # This captioner is not quantized and had no custom instance overrides.
        self.model.__dict__.pop("to", None)
        self.model.__dict__.pop("cuda", None)
        # With optional CPU-spill maps, hook removal materializes the CPU-backed
        # parameters before .to('cpu'). Never call .to() on a still-hooked model.
        meta = [n for n, p in self.model.named_parameters() if p.device.type == "meta"]
        if meta:
            raise RuntimeError("Unresolved meta weights after hook removal: " + ", ".join(meta[:8]))
        self.model.to("cpu")
        self.model.tie_weights()
        self.model.hf_device_map = {"": "cpu"}
        self._state = "cpu"
        self.assert_offloaded()
        self._clear_cuda_cache()
        if self.verbose:
            print("Caption offloaded: all model parameters and buffers are on CPU.")
        return self

    def onload(self):
        """Restore the SAME saved GPU map, never gathering all weights onto GPU 0."""
        if self._state == "gpu":
            return self
        if self._state != "cpu":
            raise RuntimeError(f"Captioner is in state {self._state!r}; call offload() to recover")
        self.assert_offloaded()
        self.model.tie_weights()
        self._state = "dispatching"
        try:
            self.model = dispatch_model(
                self.model,
                device_map=dict(self.active_device_map),
                main_device=self.input_device,
                skip_keys=self._skip_keys,
                offload_buffers=False,
            )
            self._state = "gpu"
            if self.verbose:
                sizes = Counter()
                for parameter in self.model.parameters():
                    if parameter.device.type == "cuda":
                        sizes[str(parameter.device)] += parameter.numel() * parameter.element_size()
                print("Caption GPU parameter GiB:", {
                    device: round(size / 1024 ** 3, 3) for device, size in sorted(sizes.items())
                })
        except BaseException:
            try:
                self.offload()
            except Exception as cleanup_error:
                warnings.warn(f"Caption dispatch failed and cleanup also failed: {cleanup_error}")
            raise
        return self

    def read_image_caption(self, image_folder, max_side=None, max_pixels=None,
                           *, max_new_tokens=1024, auto_offload=None):
        """Automatically restore GPU shards, generate, then offload by default."""
        unload = self.auto_offload if auto_offload is None else auto_offload
        self.onload()
        try:
            # Separate frame ensures generation tensors are cleared BEFORE
            # offload/cache cleanup. _generate also clears them on exceptions.
            return self._generate(image_folder, max_side, max_pixels, max_new_tokens)
        finally:
            if unload:
                self.offload()

    def _generate(self, image_folder, max_side, max_pixels, max_new_tokens):
        inputs = generated_ids = trimmed = None
        image_list = []
        try:
            image_list = read_all_images_from(
                image_folder,
                max_side=self.max_side if max_side is None else max_side,
                max_pixels=self.max_pixels if max_pixels is None else max_pixels,
            )
            messages = build_messages(image_list)
            images, videos, video_kwargs = process_vision_info(
                messages,
                image_patch_size=16,
                return_video_kwargs=True,
                return_video_metadata=True,
            )
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            video_metadatas = None
            if videos is not None:
                videos, video_metadatas = zip(*videos)
                videos, video_metadatas = list(videos), list(video_metadatas)
            inputs = self.processor(
                text=text, images=images, videos=videos,
                video_metadata=video_metadatas,
                return_tensors="pt", do_resize=False, **video_kwargs,
            ).to(self.input_device)
            if self.verbose:
                print(f"{image_folder} caption input:", len(image_list), [im.size for im in image_list])
            with torch.inference_mode(), torch.amp.autocast("cuda", dtype=self.dtype):
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    return_dict_in_generate=False,
                    output_scores=False,
                    output_attentions=False,
                    output_hidden_states=False,
                )
            trimmed = generated_ids[:, inputs.input_ids.shape[1]:].detach().cpu()
            return self.processor.batch_decode(
                trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
        finally:
            # A traceback can retain this frame after OOM: explicitly drop CUDA
            # input/output references before the outer finally offloads weights.
            inputs = generated_ids = trimmed = None
            for image in image_list:
                image.close()

    def memory_report(self):
        return {
            f"cuda:{gpu}": {
                "allocated_GiB": round(torch.cuda.memory_allocated(gpu) / 1024 ** 3, 3),
                "reserved_GiB": round(torch.cuda.memory_reserved(gpu) / 1024 ** 3, 3),
            }
            for gpu in self.gpu_ids
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--image_folder", required=True)
    parser.add_argument("--num_gpus", type=int, default=2)
    parser.add_argument("--gpu_ids", type=int, nargs="+", default=None,
                        help="Logical indices AFTER CUDA_VISIBLE_DEVICES filtering")
    parser.add_argument("--total_gpu_memory", default="32GiB",
                        help="Combined mapping budget across selected GPUs, not per-GPU")
    parser.add_argument("--cpu_memory", default="64GiB")
    parser.add_argument("--device_map", default="balanced_low_0",
                        choices=["auto", "balanced", "balanced_low_0", "sequential"])
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--max_side", type=int, default=1280)
    parser.add_argument("--max_pixels", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--allow_cpu_layers", action="store_true")
    parser.add_argument("--keep_loaded", action="store_true",
                        help="Keep GPU shards between repeated calls; unload at process end")
    parser.add_argument("--repeat", type=int, default=1,
                        help="Use 2 to check repeated CPU -> GPU shards -> CPU transitions")
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be positive")
    captioner = ImagesCaptioner(
        model_path=args.model_path,
        max_side=args.max_side,
        max_pixels=args.max_pixels,
        num_gpus=args.num_gpus,
        gpu_ids=args.gpu_ids,
        total_gpu_memory=args.total_gpu_memory,
        cpu_memory=args.cpu_memory,
        device_map=args.device_map,
        attn_implementation=args.attn_implementation,
        auto_offload=not args.keep_loaded,
        allow_cpu_layers=args.allow_cpu_layers,
    )
    print("Before caption:", captioner.memory_report())
    try:
        for index in range(args.repeat):
            text = captioner.read_image_caption(
                args.image_folder, max_new_tokens=args.max_new_tokens
            )
            print(f"Caption {index + 1}:", text)
            print("After caption:", captioner.memory_report())
    finally:
        captioner.offload()
    captioner.assert_offloaded()


if __name__ == "__main__":
    main()
