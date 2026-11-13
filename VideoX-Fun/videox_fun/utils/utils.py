import gc
import inspect
import os
import shutil
import time

import cv2
import numpy as np
import torch
import torchvision
from einops import rearrange
from PIL import Image

import re
import math
import copy
import struct
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

from .save_videos_grid_low_ram import make_save_videos_grid



def filter_kwargs(cls, kwargs):
    sig = inspect.signature(cls.__init__)
    valid_params = set(sig.parameters.keys()) - {'self', 'cls'}
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
    return filtered_kwargs

def get_width_and_height_from_image_and_base_resolution(image, base_resolution):
    target_pixels = int(base_resolution) * int(base_resolution)
    original_width, original_height = Image.open(image).size
    ratio = (target_pixels / (original_width * original_height)) ** 0.5
    width_slider = round(original_width * ratio)
    height_slider = round(original_height * ratio)
    return height_slider, width_slider

def color_transfer(sc, dc):
    """
    Transfer color distribution from of sc, referred to dc.

    Args:
        sc (numpy.ndarray): input image to be transfered.
        dc (numpy.ndarray): reference image

    Returns:
        numpy.ndarray: Transferred color distribution on the sc.
    """

    def get_mean_and_std(img):
        x_mean, x_std = cv2.meanStdDev(img)
        x_mean = np.hstack(np.around(x_mean, 2))
        x_std = np.hstack(np.around(x_std, 2))
        return x_mean, x_std

    sc = cv2.cvtColor(sc, cv2.COLOR_RGB2LAB)
    s_mean, s_std = get_mean_and_std(sc)
    dc = cv2.cvtColor(dc, cv2.COLOR_RGB2LAB)
    t_mean, t_std = get_mean_and_std(dc)
    img_n = ((sc - s_mean) * (t_std / s_std)) + t_mean
    np.putmask(img_n, img_n > 255, 255)
    np.putmask(img_n, img_n < 0, 0)
    dst = cv2.cvtColor(cv2.convertScaleAbs(img_n), cv2.COLOR_LAB2RGB)
    return dst

def _open_video_writer(path: str, fps: float, frame_size):
    """Create an OpenCV VideoWriter without relying on a system multimedia executable."""
    width, height = map(int, frame_size)
    ext = os.path.splitext(path)[1].lower()

    if ext == ".avi":
        codecs = ("XVID", "MJPG")
    else:
        # mp4v is the most portable choice for opencv-python on Windows.
        # avc1 is kept as a fallback for environments that expose an H.264 encoder.
        codecs = ("mp4v", "avc1")

    last_error = None
    for codec in codecs:
        writer = cv2.VideoWriter(
            path,
            cv2.VideoWriter_fourcc(*codec),
            float(fps),
            (width, height),
        )
        if writer.isOpened():
            return writer
        writer.release()
        last_error = codec

    raise RuntimeError(
        f"OpenCV could not create a video writer for {path!r}. "
        f"Tried codecs: {codecs}. Last codec: {last_error}."
    )


def _write_video_opencv(frames_rgb: np.ndarray, path: str, fps: float):
    """Write RGB uint8 frames with OpenCV only."""
    if frames_rgb.ndim != 4 or frames_rgb.shape[-1] != 3:
        raise ValueError(f"Expected frames [T,H,W,3], got {frames_rgb.shape}")
    if len(frames_rgb) == 0:
        raise ValueError("Cannot write an empty video.")

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    height, width = frames_rgb.shape[1:3]
    writer = _open_video_writer(path, fps, (width, height))

    try:
        for frame_rgb in frames_rgb:
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)
    finally:
        writer.release()

    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        raise RuntimeError(f"Video writing failed: {path}")


# def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=6, fps=12,
#                      imageio_backend=True, color_transfer_post_process=False, reverse_width=False):
#     """
#     Save a video tensor without invoking any system multimedia executable.

#     `imageio_backend` is retained only for backward API compatibility. MP4/AVI/MOV
#     output is written by OpenCV; GIF output is written by Pillow.
#     """
#     del imageio_backend  # backward-compatible argument, intentionally unused

#     videos = rearrange(videos, "b c t h w -> t b c h w")

#     grids = torch.stack([torchvision.utils.make_grid(x, nrow=n_rows) for x in videos])
#     grids = grids.permute(0, 2, 3, 1)  # [T,H,W,C]
#     if rescale:
#         grids = (grids + 1.0) / 2.0

#     frames_np = (
#         grids.clamp(0, 1)
#         .mul(255)
#         .to(torch.uint8)
#         .contiguous()
#         .cpu()
#         .numpy()
#     )

#     if color_transfer_post_process:
#         ref = frames_np[0]
#         for i in range(1, len(frames_np)):
#             frames_np[i] = color_transfer(frames_np[i], ref)

#     if reverse_width:
#         frames_np = np.ascontiguousarray(frames_np[:, :, ::-1, :])

#     parent = os.path.dirname(os.path.abspath(path))
#     os.makedirs(parent, exist_ok=True)

#     ext = os.path.splitext(path)[1].lower()
#     if ext in {".mp4", ".avi", ".mov", ".mkv"}:
#         _write_video_opencv(frames_np, path, fps)
#     else:
#         # GIF / image animation path, also no external executable required.
#         gif_path = path if ext == ".gif" else os.path.splitext(path)[0] + ".gif"
#         imgs = [Image.fromarray(f) for f in frames_np]
#         duration_ms = max(1, int(round(1000.0 / float(fps))))
#         imgs[0].save(
#             gif_path,
#             format="GIF",
#             append_images=imgs[1:],
#             save_all=True,
#             duration=duration_ms,
#             loop=0,
#         )

def save_selected_images_from_video_tensor(
    videos,                      # torch.Tensor 或 np.ndarray, [B, C, T, H, W]
    images_path: str,
    index: int,
    save_image_ids,
    resize=None,
    frames_per_segment=81,
    rescale=False,
    n_rows=6,
    pad_width=3,
    max_workers=None,
    use_global_name=True,
):
    """
    只保存当前段视频中指定帧序号的图片。
 
    优化点:
    - 一次性批量做 GPU→CPU、clamp、转 uint8,避免逐帧搬运
    - B=1 时跳过 make_grid
    - RGB→BGR 用 flip(零拷贝),避免 cv2.cvtColor
    - 去掉 copy.deepcopy
    - PNG 压缩等级降到 1
    - 合理 worker 数
    """
    if save_image_ids is None or len(save_image_ids) == 0:
        return []
 
    os.makedirs(images_path, exist_ok=True)
 
    if max_workers is None:
        max_workers = min(16, os.cpu_count() or 4)
 
    # ---------- 1) 统一成 torch.Tensor,且尽快只保留要的帧 ----------
    if isinstance(videos, np.ndarray):
        videos = torch.from_numpy(videos)
 
    with torch.no_grad():
        videos = videos.detach()
 
        # [B, C, T, H, W] -> [T, B, C, H, W]
        videos_t = rearrange(videos, "b c t h w -> t b c h w")
        total_t = videos_t.shape[0]
 
        save_image_ids = sorted({int(x) for x in save_image_ids})
        valid_ids = [i for i in save_image_ids if 0 <= i < total_t]
        invalid_ids = [i for i in save_image_ids if not (0 <= i < total_t)]
 
        if invalid_ids:
            print(
                f"[Warning] T={total_t}, 越界并跳过的 save_image_ids: {invalid_ids}",
                flush=True,
            )
        if not valid_ids:
            return []
 
        # ---------- 2) 一次性切出所有目标帧,后续操作全部批量化 ----------
        # selected: [N, B, C, H, W]
        selected = videos_t[valid_ids]
        N, B, C, H, W = selected.shape
 
        # rescale: [-1, 1] -> [0, 1]
        if rescale:
            selected = (selected + 1.0) * 0.5
 
        # 批量 clamp + 转 uint8,只触发一次同步(如果还在 GPU 上)
        selected = selected.clamp_(0, 1).mul_(255).to(torch.uint8)
 
        # ---------- 3) B=1 跳过 make_grid;B>1 才走 grid ----------
        if B == 1:
            grids = selected.squeeze(1)  # [N, C, H, W]
        else:
            # 仅在确实需要时才逐帧做 grid
            # import torchvision
            grids = torch.stack([
                torchvision.utils.make_grid(selected[k], nrow=n_rows)
                for k in range(N)
            ], dim=0)  # [N, C, H', W']
 
        # ---------- 4) RGB -> BGR 用 flip(零拷贝 view) ----------
        # [N, C, H, W] -> [N, H, W, C]
        grids = grids.flip(1).permute(0, 2, 3, 1).contiguous()
 
        # ---------- 5) 一次性下 CPU + 转 numpy ----------
        grids_np = grids.cpu().numpy()  # [N, H, W, 3], uint8, BGR
 
    # ---------- 6) 准备保存任务(resize 一并放进线程池并行) ----------
    save_items = []
    for k, local_idx in enumerate(valid_ids):
        img = grids_np[k]  # 已经是 contiguous BGR
 
        save_idx = (
            index * frames_per_segment + local_idx if use_global_name else local_idx
        )
        out = os.path.join(images_path, f"{save_idx:0>{pad_width}}.png")
        save_items.append((out, img))
 
    saved_paths = [item[0] for item in save_items]
 
    # ---------- 7) 多线程 resize + 写 PNG(OpenCV 两步都释放 GIL) ----------
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(executor.map(lambda x: _resize_and_save(x[0], x[1], resize), save_items))
 
    # logger.info(f"images num {len(saved_paths)}")
    return saved_paths


def _media_duration_seconds(container, stream):
    """Best-effort media duration in seconds for PyAV containers/streams."""
    if getattr(stream, "duration", None) is not None and getattr(stream, "time_base", None) is not None:
        return float(stream.duration * stream.time_base)
    if getattr(container, "duration", None) is not None:
        try:
            import av
            return float(container.duration / av.time_base)
        except Exception:
            pass
    return None


def merge_video_audio(video_path: str, audio_path: str):
    """
    Merge video and audio using PyAV only; no system multimedia executable is used.

    PyAV Windows wheels bundle the required multimedia libraries. If this function
    is needed, install once with: `python -m pip install av`.
    The output duration is limited to the shorter available stream when durations
    are known. Video is re-encoded (H.264 when available, otherwise MPEG-4) and
    audio is encoded as AAC for MP4 compatibility.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"video file {video_path} does not exist")
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"audio file {audio_path} does not exist")

    try:
        import av
    except ImportError as exc:
        raise RuntimeError(
            "merge_video_audio requires the Python package 'av'. "
            "Install it with: python -m pip install av"
        ) from exc

    base, ext = os.path.splitext(video_path)
    if not ext:
        ext = ".mp4"
    temp_output = f"{base}_temp{ext}"

    video_in = audio_in = output = None
    try:
        print("Start merging video and audio with PyAV...")
        video_in = av.open(video_path)
        audio_in = av.open(audio_path)

        video_stream = next((s for s in video_in.streams if s.type == "video"), None)
        audio_stream = next((s for s in audio_in.streams if s.type == "audio"), None)
        if video_stream is None:
            raise RuntimeError(f"No video stream found in {video_path}")
        if audio_stream is None:
            raise RuntimeError(f"No audio stream found in {audio_path}")

        video_duration = _media_duration_seconds(video_in, video_stream)
        audio_duration = _media_duration_seconds(audio_in, audio_stream)
        durations = [x for x in (video_duration, audio_duration) if x is not None and x > 0]
        max_duration = min(durations) if durations else None

        output = av.open(temp_output, mode="w")

        fps = video_stream.average_rate or video_stream.base_rate or video_stream.guessed_rate or 30
        try:
            out_video = output.add_stream("libx264", rate=fps)
        except Exception:
            out_video = output.add_stream("mpeg4", rate=fps)

        out_video.width = int(video_stream.codec_context.width)
        out_video.height = int(video_stream.codec_context.height)
        out_video.pix_fmt = "yuv420p"
        try:
            out_video.options = {"preset": "veryfast", "crf": "23"}
        except Exception:
            pass

        sample_rate = (
            getattr(audio_stream.codec_context, "sample_rate", None)
            or getattr(audio_stream, "rate", None)
            or 44100
        )
        layout_obj = getattr(audio_stream.codec_context, "layout", None)
        layout_name = getattr(layout_obj, "name", None) or "stereo"

        out_audio = output.add_stream("aac", rate=int(sample_rate))
        out_audio.bit_rate = 192000
        try:
            out_audio.layout = layout_name
        except Exception:
            pass

        # Encode video frames.
        for frame in video_in.decode(video_stream):
            if max_duration is not None and frame.pts is not None and frame.time_base is not None:
                if float(frame.pts * frame.time_base) >= max_duration:
                    break
            for packet in out_video.encode(frame):
                output.mux(packet)
        for packet in out_video.encode(None):
            output.mux(packet)

        # Resample/encode audio frames.
        resampler = av.audio.resampler.AudioResampler(
            format="fltp",
            layout=layout_name,
            rate=int(sample_rate),
        )
        for frame in audio_in.decode(audio_stream):
            if max_duration is not None and frame.pts is not None and frame.time_base is not None:
                if float(frame.pts * frame.time_base) >= max_duration:
                    break
            converted = resampler.resample(frame)
            if converted is None:
                continue
            if not isinstance(converted, (list, tuple)):
                converted = [converted]
            for audio_frame in converted:
                for packet in out_audio.encode(audio_frame):
                    output.mux(packet)

        flushed = resampler.resample(None)
        if flushed is not None:
            if not isinstance(flushed, (list, tuple)):
                flushed = [flushed]
            for audio_frame in flushed:
                for packet in out_audio.encode(audio_frame):
                    output.mux(packet)
        for packet in out_audio.encode(None):
            output.mux(packet)

        output.close()
        output = None
        video_in.close()
        video_in = None
        audio_in.close()
        audio_in = None

        os.replace(temp_output, video_path)
        print(f"Merge completed, saved to {video_path}")

    except Exception:
        for obj in (output, video_in, audio_in):
            if obj is not None:
                try:
                    obj.close()
                except Exception:
                    pass
        if os.path.exists(temp_output):
            try:
                os.remove(temp_output)
            except OSError:
                pass
        raise

def get_image_for_latent(image_file, sample_size):
    if type(image_file) is str and os.path.isfile(image_file):
        image_start = Image.open(image_file).convert("RGB")
        image_start = image_start.resize([sample_size[1], sample_size[0]])
    else:
        image_start = image_file
        image_start = [_image_start.resize([sample_size[1], sample_size[0]]) for _image_start in image_start]
    if type(image_start) is list:
        input_image = torch.cat(
                [torch.from_numpy(np.array(_image_start)).permute(2, 0, 1).unsqueeze(1).unsqueeze(0) for _image_start in image_start], 
                dim=2
            )
    else:
        input_image = torch.from_numpy(np.array(image_start)).permute(2, 0, 1).unsqueeze(1).unsqueeze(0)
    input_image = input_image / 255
    return  input_image

def get_image_to_video_latent(validation_image_start, validation_image_end, video_length, sample_size):
    if validation_image_start is not None and validation_image_end is not None:
        if type(validation_image_start) is str and os.path.isfile(validation_image_start):
            image_start = clip_image = Image.open(validation_image_start).convert("RGB")
            image_start = image_start.resize([sample_size[1], sample_size[0]])
            clip_image = clip_image.resize([sample_size[1], sample_size[0]])
        else:
            image_start = clip_image = validation_image_start
            image_start = [_image_start.resize([sample_size[1], sample_size[0]]) for _image_start in image_start]
            clip_image = [_clip_image.resize([sample_size[1], sample_size[0]]) for _clip_image in clip_image]

        if type(validation_image_end) is str and os.path.isfile(validation_image_end):
            image_end = Image.open(validation_image_end).convert("RGB")
            image_end = image_end.resize([sample_size[1], sample_size[0]])
        else:
            image_end = validation_image_end
            image_end = [_image_end.resize([sample_size[1], sample_size[0]]) for _image_end in image_end]

        if type(image_start) is list:
            clip_image = clip_image[0]
            start_video = torch.cat(
                [torch.from_numpy(np.array(_image_start)).permute(2, 0, 1).unsqueeze(1).unsqueeze(0) for _image_start in image_start], 
                dim=2
            )
            input_video = torch.tile(start_video[:, :, :1], [1, 1, video_length, 1, 1])
            input_video[:, :, :len(image_start)] = start_video
            
            input_video_mask = torch.zeros_like(input_video[:, :1])
            input_video_mask[:, :, len(image_start):] = 255
        else:
            input_video = torch.tile(
                torch.from_numpy(np.array(image_start)).permute(2, 0, 1).unsqueeze(1).unsqueeze(0), 
                [1, 1, video_length, 1, 1]
            )
            input_video_mask = torch.zeros_like(input_video[:, :1])
            input_video_mask[:, :, 1:] = 255

        if type(image_end) is list:
            image_end = [_image_end.resize(image_start[0].size if type(image_start) is list else image_start.size) for _image_end in image_end]
            end_video = torch.cat(
                [torch.from_numpy(np.array(_image_end)).permute(2, 0, 1).unsqueeze(1).unsqueeze(0) for _image_end in image_end], 
                dim=2
            )
            input_video[:, :, -len(end_video):] = end_video
            
            input_video_mask[:, :, -len(image_end):] = 0
        else:
            image_end = image_end.resize(image_start[0].size if type(image_start) is list else image_start.size)
            input_video[:, :, -1:] = torch.from_numpy(np.array(image_end)).permute(2, 0, 1).unsqueeze(1).unsqueeze(0)
            input_video_mask[:, :, -1:] = 0

        input_video = input_video / 255

    elif validation_image_start is not None:
        if type(validation_image_start) is str and os.path.isfile(validation_image_start):
            image_start = clip_image = Image.open(validation_image_start).convert("RGB")
            image_start = image_start.resize([sample_size[1], sample_size[0]])
            clip_image = clip_image.resize([sample_size[1], sample_size[0]])
        else:
            image_start = clip_image = validation_image_start
            image_start = [_image_start.resize([sample_size[1], sample_size[0]]) for _image_start in image_start]
            clip_image = [_clip_image.resize([sample_size[1], sample_size[0]]) for _clip_image in clip_image]
        image_end = None
        
        if type(image_start) is list:
            clip_image = clip_image[0]
            start_video = torch.cat(
                [torch.from_numpy(np.array(_image_start)).permute(2, 0, 1).unsqueeze(1).unsqueeze(0) for _image_start in image_start], 
                dim=2
            )
            input_video = torch.tile(start_video[:, :, :1], [1, 1, video_length, 1, 1])
            input_video[:, :, :len(image_start)] = start_video
            input_video = input_video / 255
            
            input_video_mask = torch.zeros_like(input_video[:, :1])
            input_video_mask[:, :, len(image_start):] = 255
        else:
            input_video = torch.tile(
                torch.from_numpy(np.array(image_start)).permute(2, 0, 1).unsqueeze(1).unsqueeze(0), 
                [1, 1, video_length, 1, 1]
            ) / 255
            input_video_mask = torch.zeros_like(input_video[:, :1])
            input_video_mask[:, :, 1:, ] = 255
    else:
        image_start = None
        image_end = None
        input_video = torch.zeros([1, 3, video_length, sample_size[0], sample_size[1]])
        input_video_mask = torch.ones([1, 1, video_length, sample_size[0], sample_size[1]]) * 255
        clip_image = None

    del image_start
    del image_end
    gc.collect()

    return  input_video, input_video_mask, clip_image

def _decode_video_opencv(path, w, h, fps=None):
    """
    Decode, optionally downsample in time, resize, and return RGB uint8 frames
    using OpenCV only. Returns shape [N,H,W,3].
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV failed to open video: {path}")

    try:
        original_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps is not None and fps > 0 and original_fps > 0:
            frame_skip = max(1, int(original_fps // float(fps)))
        else:
            frame_skip = 1

        frames = []
        frame_idx = 0
        target_w, target_h = int(w), int(h)

        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            if frame_idx % frame_skip == 0:
                src_h, src_w = frame_bgr.shape[:2]
                if src_w != target_w or src_h != target_h:
                    interpolation = (
                        cv2.INTER_AREA
                        if target_w * target_h < src_w * src_h
                        else cv2.INTER_LINEAR
                    )
                    frame_bgr = cv2.resize(
                        frame_bgr,
                        (target_w, target_h),
                        interpolation=interpolation,
                    )
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)

            frame_idx += 1
    finally:
        cap.release()

    if not frames:
        return np.empty((0, int(h), int(w), 3), dtype=np.uint8)
    return np.stack(frames, axis=0).astype(np.uint8, copy=False)


def resize_bgr_like_video(img_bgr: np.ndarray, resize):
    """
    Resize a BGR image with the same width/height semantics used by this module.

    Args:
        img_bgr: uint8 BGR image [H,W,3]
        resize: None or (w, h); w/h may be -1 to preserve aspect ratio.
    """
    if resize is None:
        return img_bgr

    height, width = img_bgr.shape[:2]
    w, h = resize

    if w != -1 and h != -1:
        if math.copysign(1, w - h) * math.copysign(1, width - height) < 0:
            w, h = h, w

    if w == -1 and h == -1:
        return img_bgr
    elif w == -1:
        scale = h / height
        w = int(round(width * scale))
    elif h == -1:
        scale = w / width
        h = int(round(height * scale))

    w, h = int(w), int(h)
    if w == width and h == height:
        return img_bgr

    interpolation = cv2.INTER_AREA if w * h < width * height else cv2.INTER_LINEAR
    return cv2.resize(img_bgr, (w, h), interpolation=interpolation)

def _resize_and_save(out, img, resize):
    if resize is not None:
        img = resize_bgr_like_video(img, resize)
    ok = cv2.imwrite(out, img, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    if not ok:
        raise RuntimeError(f"Failed to save image: {out}")

def save_selected_images_from_video_tensor(
    videos,                      # torch.Tensor 或 np.ndarray, [B, C, T, H, W]
    images_path: str,
    index: int,
    save_image_ids,
    resize=None,
    frames_per_segment=81,
    rescale=False,
    n_rows=6,
    pad_width=3,
    max_workers=None,
    use_global_name=True,
):
    if save_image_ids is None or len(save_image_ids) == 0:
        return []
 
    os.makedirs(images_path, exist_ok=True)
 
    if max_workers is None:
        max_workers = min(16, os.cpu_count() or 4)
 
    if isinstance(videos, np.ndarray):
        videos = torch.from_numpy(videos)
 
    with torch.no_grad():
        videos = videos.detach()
 
        videos_t = rearrange(videos, "b c t h w -> t b c h w")
        total_t = videos_t.shape[0]
 
        save_image_ids = sorted({int(x) for x in save_image_ids})
        valid_ids = [i for i in save_image_ids if 0 <= i < total_t]
        invalid_ids = [i for i in save_image_ids if not (0 <= i < total_t)]
 
        if invalid_ids:
            print(
                f"[Warning] T={total_t}, 越界并跳过的 save_image_ids: {invalid_ids}",
                flush=True,
            )
        if not valid_ids:
            return []
 
        selected = videos_t[valid_ids]
        N, B, C, H, W = selected.shape
        if rescale:
            selected = (selected + 1.0) * 0.5
 
        selected = selected.clamp_(0, 1).mul_(255).to(torch.uint8)
 
        if B == 1:
            grids = selected.squeeze(1)  # [N, C, H, W]
        else:
            grids = torch.stack([
                torchvision.utils.make_grid(selected[k], nrow=n_rows)
                for k in range(N)
            ], dim=0)  # [N, C, H', W']
 
        # [N, C, H, W] -> [N, H, W, C]
        grids = grids.flip(1).permute(0, 2, 3, 1).contiguous()
 
        grids_np = grids.cpu().numpy()  # [N, H, W, 3], uint8, BGR
 
    save_items = []
    for k, local_idx in enumerate(valid_ids):
        img = grids_np[k]  # 已经是 contiguous BGR
 
        save_idx = (
            index * frames_per_segment + local_idx if use_global_name else local_idx
        )
        out = os.path.join(images_path, f"{save_idx:0>{pad_width}}.png")
        save_items.append((out, img))
 
    saved_paths = [item[0] for item in save_items]
 
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(executor.map(lambda x: _resize_and_save(x[0], x[1], resize), save_items))
 
    return saved_paths

def parse_frame_name(name: str) -> int:
    base = os.path.basename(name)
    m = re.match(r"^(\d+)\.png$", base)
    if m is None:
        raise ValueError(f"无法解析帧名: {name}")
    return int(m.group(1))

def get_unique_original_frame_names_prefer_first(
    gen_index,
    frames_per_video=81,
    pad_width=3,
    ext=".png",
):
    first_seen_order = []
    best_record = {}

    for seg_id, pair_group in enumerate(gen_index):
        base_frame_idx = seg_id * frames_per_video

        for origin_img_id, local_frame_idx in pair_group:
            global_frame_idx = base_frame_idx + local_frame_idx

            # 记录原始图片第一次出现的顺序
            if origin_img_id not in best_record:
                first_seen_order.append(origin_img_id)
                best_record[origin_img_id] = {
                    "global_frame_idx": global_frame_idx,
                    "local_frame_idx": local_frame_idx,
                    "seg_id": seg_id,
                }
                continue

            old_local_idx = best_record[origin_img_id]["local_frame_idx"]

            # 如果旧记录不是首帧，而当前是首帧，则替换成当前首帧
            if old_local_idx != 0 and local_frame_idx == 0:
                best_record[origin_img_id] = {
                    "global_frame_idx": global_frame_idx,
                    "local_frame_idx": local_frame_idx,
                    "seg_id": seg_id,
                }

    frame_names = [
        f"{best_record[origin_img_id]['global_frame_idx']:0{pad_width}d}{ext}"
        for origin_img_id in first_seen_order
    ]

    return frame_names

def map_selected_to_segment_local_indices(
    selected_names,
    gen_index,
    frames_per_segment=81,
):
    segment_num = len(gen_index)

    segment_map = {
        seg_id: {
            "gen_pair": gen_index[seg_id],
            "global_range": [
                seg_id * frames_per_segment,
                seg_id * frames_per_segment + frames_per_segment - 1,
            ],
            "global_frame_names": [],
            "global_frame_indices": [],
            "local_frame_indices": [],
        }
        for seg_id in range(segment_num)
    }

    for name in selected_names:
        global_idx = parse_frame_name(name)

        seg_id = global_idx // frames_per_segment
        local_idx = global_idx - seg_id * frames_per_segment

        
        if seg_id < 0 or seg_id >= segment_num:
            print(
                f"[Warning] {name} 对应 seg_id={seg_id}，"
                f"超出 gen_index 段数 {segment_num}，已跳过。"
            )
            continue
        

        segment_map[seg_id]["global_frame_names"].append(f"{global_idx:03d}.png")
        segment_map[seg_id]["global_frame_indices"].append(global_idx)
        segment_map[seg_id]["local_frame_indices"].append(local_idx)

    return segment_map

def get_colmap_name(colmap_image):
    if isinstance(colmap_image, dict):
        return os.path.basename(colmap_image["name"])
    return os.path.basename(colmap_image.name)


def get_colmap_c2w(colmap_image):
    if isinstance(colmap_image, dict):
        return colmap_image["c2w"]
    return colmap_image.c2w


def parse_frame_name_to_idx(name: str):
    """
    '083.png' -> 83
    不能解析则返回 None，例如 4k_xxx.png
    """
    base = os.path.basename(name)
    m = re.match(r"^(\d+)\.png$", base)
    if m is None:
        return None
    return int(m.group(1))


def is_bad_local_frame(
    name: str,
    frames_per_segment=81,
    bad_local_indices=(1, 2, 3, 4),
):
    """
    判断某张 xxx.png 是否属于坏帧。
    - 无法解析为 “纯数字.png”（例如临时文件、4k_xxx.jpg）→ 视为坏帧，过滤掉
    - 每段视频 local 1~4 → 坏帧
    - local 0 → 正常帧
    """
    idx = parse_frame_name_to_idx(name)

    # 非 “纯数字.png” 格式，直接当坏帧丢弃
    if idx is None:
        return True

    local_idx = idx % frames_per_segment
    return local_idx in set(bad_local_indices)


def farthest_point_sampling_with_init(points: np.ndarray, selected_num: int, init_indices=None):
    num_total = len(points)

    if selected_num <= 0:
        return []

    if selected_num >= num_total:
        return list(range(num_total))

    if init_indices is None:
        init_indices = []

    init_indices = list(dict.fromkeys(init_indices))
    init_indices = [i for i in init_indices if 0 <= i < num_total]

    if len(init_indices) > selected_num:
        raise ValueError(
            f"init_indices 数量 {len(init_indices)} 超过 selected_num={selected_num}"
        )

    if len(init_indices) == 0:
        selected_index = [0]
    else:
        selected_index = init_indices.copy()

    selected_points = points[selected_index]
    distance = np.min(
        np.sum((points[:, None, :] - selected_points[None, :, :]) ** 2, axis=2),
        axis=1,
    )
    distance[selected_index] = -1

    while len(selected_index) < selected_num:
        new_index = int(np.argmax(distance))
        selected_index.append(new_index)

        new_distance = np.sum((points - points[new_index]) ** 2, axis=1)
        distance = np.minimum(distance, new_distance)
        distance[selected_index] = -1

    return selected_index

def farthest_colmap_camera_sampling_idx_force_init_avoid_bad_frames(
    colmap_images: dict,
    selected_num=80,
    init_image_names=None,
    keep_4k=True,
    frames_per_segment=81,
    bad_local_indices=(1, 2, 3, 4),
):
    """
    强制包含 init_image_names，同时避开每段视频 local frame 1~4 的低质量帧。
    local frame 0 不会被过滤。

    逻辑：
    1. init_image_names 强制保留；
    2. 普通候选帧中排除每段 local 1~4；
    3. 以 init frames 作为 FPS 初始点继续最远点采样；
    4. 追加 4k。
    """
    if init_image_names is None:
        init_image_names = []

    init_image_names = [os.path.basename(x) for x in init_image_names]
    init_name_set = set(init_image_names)

    name_to_key = {}
    keys_4k = []
    normal_keys = []

    # 1. 建立映射，同时构建候选集
    for key, img in colmap_images.items():
        name = get_colmap_name(img)
        name_to_key[name] = key

        if keep_4k and name.startswith("4k"):
            keys_4k.append(key)
            continue

        # init 强制保留，不受坏帧过滤影响
        if name in init_name_set:
            normal_keys.append(key)
            continue

        # 非 init 的普通帧：避开每段 local 1~4
        if is_bad_local_frame(
            name,
            frames_per_segment=frames_per_segment,
            bad_local_indices=bad_local_indices,
        ):
            continue

        normal_keys.append(key)

    # 2. 找 required keys
    required_keys = []
    missing_names = []

    for name in init_image_names:
        if name not in name_to_key:
            missing_names.append(name)
        else:
            key = name_to_key[name]
            if key not in required_keys:
                required_keys.append(key)

    if missing_names:
        raise ValueError(f"这些 init_image_names 没有在 colmap_images 中找到: {missing_names}")

    if len(required_keys) > selected_num:
        raise ValueError(
            f"required_keys 数量 {len(required_keys)} 超过 selected_num={selected_num}"
        )

    # 3. required key 必须在 normal_keys 中，防止因为过滤逻辑被排除
    for key in required_keys:
        if key not in normal_keys:
            normal_keys.insert(0, key)

    key_to_local_idx = {key: i for i, key in enumerate(normal_keys)}
    init_local_indices = [key_to_local_idx[key] for key in required_keys]

    # 4. 构造 pose
    poses = []
    for key in normal_keys:
        c2w = get_colmap_c2w(colmap_images[key])
        poses.append(c2w[:3, 3].reshape(-1))

    poses = np.asarray(poses)

    selected_local_indices = farthest_point_sampling_with_init(
        points=poses,
        selected_num=selected_num,
        init_indices=init_local_indices,
    )

    final_keys = [normal_keys[i] for i in selected_local_indices]

    # 5. 追加 4k
    if keep_4k:
        for key in keys_4k:
            if key not in final_keys:
                final_keys.append(key)

    final_names = [get_colmap_name(colmap_images[key]) for key in final_keys]

    # 6. 强校验：init 必须存在
    for name in init_image_names:
        if name not in final_names:
            raise RuntimeError(f"{name} 应该被强制保留，但最终结果里没有。")

    # 7. 检查是否仍有非 init 的坏帧
    bad_selected = []
    for name in final_names:
        if name in init_name_set:
            continue
        if is_bad_local_frame(
            name,
            frames_per_segment=frames_per_segment,
            bad_local_indices=bad_local_indices,
        ):
            bad_selected.append(name)

    if bad_selected:
        raise RuntimeError(f"仍然选中了非 init 的坏帧: {bad_selected}")

    return {key: colmap_images[key] for key in final_keys}


def get_video_to_video_latent(input_video_path, video_length, sample_size,
                               fps=None, validation_video_mask=None, ref_image=None):
    if input_video_path is not None:
        if isinstance(input_video_path, str):
            # 1. OpenCV 完成解码、resize、抽帧和色彩转换
            frames = _decode_video_opencv(
                input_video_path, sample_size[1], sample_size[0], fps
            )
            # (N,H,W,C) -> (1,C,N,H,W) float, 无中间 list
            input_video = (torch.from_numpy(frames.copy())   # copy 因为 frombuffer 不可写
                           .permute(3, 0, 1, 2)
                           .unsqueeze(0)
                           .float() / 255)
        else:
            input_video = input_video_path
            if isinstance(input_video, list):
                input_video = torch.from_numpy(np.stack(input_video))
                input_video = input_video.permute(3, 0, 1, 2).unsqueeze(0).float() / 255

        # 2. mask
        if validation_video_mask is not None:
            mask_img = Image.open(validation_video_mask).convert('L').resize(
                (sample_size[1], sample_size[0])
            )
            mask_np = np.array(mask_img)
            mask_2d = torch.from_numpy(np.where(mask_np < 240, 0, 255).astype(np.uint8))
            # expand 是 view 操作，不分配内存
            input_video_mask = (mask_2d[None, None, None, :, :]
                                .expand(1, 1, input_video.shape[2], -1, -1)
                                .to(input_video.dtype))
        else:
            input_video_mask = torch.full_like(input_video[:, :1], 255.0)
    else:
        input_video, input_video_mask = None, None

    # 3. ref_image 只读一次，复用给 clip_image
    clip_image = None
    if ref_image is not None:
        if isinstance(ref_image, str):
            ref_pil = Image.open(ref_image).convert("RGB")
        else:
            ref_pil = Image.fromarray(np.array(ref_image, np.uint8))

        clip_image = ref_pil  # clip_image 直接复用

        ref_resized = ref_pil.resize((sample_size[1], sample_size[0]))
        ref_image = (torch.from_numpy(np.array(ref_resized))
                     .permute(2, 0, 1)[None, None]
                     .float() / 255)

    return input_video, input_video_mask, ref_image, clip_image

# def get_video_to_video_latent(input_video_path, video_length, sample_size, fps=None, validation_video_mask=None, ref_image=None):
#     if input_video_path is not None:
#         if isinstance(input_video_path, str):
#             cap = cv2.VideoCapture(input_video_path)
#             input_video = []

#             original_fps = cap.get(cv2.CAP_PROP_FPS)
#             frame_skip = 1 if fps is None else max(1,int(original_fps // fps))

#             frame_count = 0

#             while True:
#                 ret, frame = cap.read()
#                 if not ret:
#                     break

#                 if frame_count % frame_skip == 0:
#                     frame = cv2.resize(frame, (sample_size[1], sample_size[0]))
#                     input_video.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

#                 frame_count += 1

#             cap.release()
#         else:
#             input_video = input_video_path

#         input_video = torch.from_numpy(np.array(input_video))
#         input_video = input_video.permute([3, 0, 1, 2]).unsqueeze(0) / 255

#         if validation_video_mask is not None:
#             validation_video_mask = Image.open(validation_video_mask).convert('L').resize((sample_size[1], sample_size[0]))
#             input_video_mask = np.where(np.array(validation_video_mask) < 240, 0, 255)
            
#             input_video_mask = torch.from_numpy(np.array(input_video_mask)).unsqueeze(0).unsqueeze(-1).permute([3, 0, 1, 2]).unsqueeze(0)
#             input_video_mask = torch.tile(input_video_mask, [1, 1, input_video.size()[2], 1, 1])
#             input_video_mask = input_video_mask.to(input_video.device, input_video.dtype)
#         else:
#             input_video_mask = torch.zeros_like(input_video[:, :1])
#             input_video_mask[:, :, :] = 255
#     else:
#         input_video, input_video_mask = None, None

#     if ref_image is not None:
#         if isinstance(ref_image, str):
#             clip_image = Image.open(ref_image).convert("RGB")
#         else:
#             clip_image = Image.fromarray(np.array(ref_image, np.uint8))
#     else:
#         clip_image = None

#     if ref_image is not None:
#         if isinstance(ref_image, str):
#             ref_image = Image.open(ref_image).convert("RGB")
#             ref_image = ref_image.resize((sample_size[1], sample_size[0]))
#             ref_image = torch.from_numpy(np.array(ref_image))
#             ref_image = ref_image.unsqueeze(0).permute([3, 0, 1, 2]).unsqueeze(0) / 255
#         else:
#             ref_image = torch.from_numpy(np.array(ref_image))
#             ref_image = ref_image.unsqueeze(0).permute([3, 0, 1, 2]).unsqueeze(0) / 255
#     return input_video, input_video_mask, ref_image, clip_image

# def get_video_to_video_latent_inp(input_video_path, video_length, sample_size, fps=None, input_video_mask=None, ref_image=None):
#     if input_video_path is not None:
#         if isinstance(input_video_path, str):
#             cap = cv2.VideoCapture(input_video_path)
#             input_video = []

#             original_fps = cap.get(cv2.CAP_PROP_FPS)
#             frame_skip = 1 if fps is None else max(1,int(original_fps // fps))

#             frame_count = 0

#             while True:
#                 ret, frame = cap.read()
#                 if not ret:
#                     break

#                 if frame_count % frame_skip == 0:
#                     frame = cv2.resize(frame, (sample_size[1], sample_size[0]))
#                     input_video.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

#                 frame_count += 1

#             cap.release()
#         else:
#             input_video = input_video_path

#         input_video = torch.from_numpy(np.array(input_video))
#         input_video = input_video.permute([3, 0, 1, 2]).unsqueeze(0) / 255

#         if input_video_mask is not None:
#             cap = cv2.VideoCapture(input_video_mask)
#             input_video_mask = []

#             original_fps = cap.get(cv2.CAP_PROP_FPS)
#             frame_skip = 1 if fps is None else max(1,int(original_fps // fps))

#             frame_count = 0

#             while True:
#                 ret, frame = cap.read()
#                 if not ret:
#                     break

#                 if frame_count % frame_skip == 0:
#                     frame = cv2.resize(frame, (sample_size[1], sample_size[0]))
#                     input_video_mask.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

#                 frame_count += 1

#             cap.release()

#             input_video_mask = torch.from_numpy(np.array(input_video_mask))
#             input_video_mask = input_video_mask.permute([3, 0, 1, 2]).unsqueeze(0)[:, :1]
#         else:
#             input_video_mask = torch.ones_like(input_video[:, :1]) * 255.0
#     else:
#         input_video, input_video_mask = None, None

#     if ref_image is not None:
#         if isinstance(ref_image, str):
#             clip_image = Image.open(ref_image).convert("RGB")
#         else:
#             clip_image = Image.fromarray(np.array(ref_image, np.uint8))
#     else:
#         clip_image = None

#     if ref_image is not None:
#         if isinstance(ref_image, str):
#             ref_image = Image.open(ref_image).convert("RGB")
#             ref_image = ref_image.resize((sample_size[1], sample_size[0]))
#             ref_image = torch.from_numpy(np.array(ref_image))
#             ref_image = ref_image.unsqueeze(0).permute([3, 0, 1, 2]).unsqueeze(0) / 255
#         else:
#             ref_image = torch.from_numpy(np.array(ref_image))
#             ref_image = ref_image.unsqueeze(0).permute([3, 0, 1, 2]).unsqueeze(0) / 255
#     return input_video, 255-input_video_mask, ref_image, clip_image


def padding_image(images, new_width, new_height):
    new_image = Image.new('RGB', (new_width, new_height), (255, 255, 255))

    aspect_ratio = images.width / images.height
    if new_width / new_height > 1:
        if aspect_ratio > new_width / new_height:
            new_img_width = new_width
            new_img_height = int(new_img_width / aspect_ratio)
        else:
            new_img_height = new_height
            new_img_width = int(new_img_height * aspect_ratio)
    else:
        if aspect_ratio > new_width / new_height:
            new_img_width = new_width
            new_img_height = int(new_img_width / aspect_ratio)
        else:
            new_img_height = new_height
            new_img_width = int(new_img_height * aspect_ratio)

    resized_img = images.resize((new_img_width, new_img_height))

    paste_x = (new_width - new_img_width) // 2
    paste_y = (new_height - new_img_height) // 2

    new_image.paste(resized_img, (paste_x, paste_y))

    return new_image

def _load_and_resize(path, size):
    """size = (H, W)"""
    img = Image.open(path)
    img = img.resize((size[1], size[0]), Image.BILINEAR).convert("RGB")
    return np.array(img, dtype=np.uint8)

def get_inpaint_video_latent(images, index, video_length, sample_size):
    input_video = torch.zeros(1, 3, video_length, sample_size[0], sample_size[1])
    input_video_mask = torch.full((1, 1, video_length, sample_size[0], sample_size[1]), 255.0)

    # 1. 多线程并行读取所有图片（I/O bound，线程足够）
    with ThreadPoolExecutor(max_workers=min(8, len(images))) as pool:
        arrays = list(pool.map(lambda p: _load_and_resize(p, sample_size), images))

    # 2. 批量写入 tensor
    for i, arr in enumerate(arrays):
        input_video[:, :, index[i]] = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float() / 255
        input_video_mask[:, :, index[i]] = 0

    # 3. clip_image 直接复用已读取的第一张，不重复 open
    clip_image = Image.fromarray(arrays[0])

    return input_video, input_video_mask, clip_image

# def get_inpaint_video_latent(images, index, video_length, sample_size):
#     input_video = torch.zeros([1, 3, video_length, sample_size[0], sample_size[1]]).float()
#     input_video_mask = torch.ones([1, 1, video_length, sample_size[0], sample_size[1]]).float() * 255

#     for i in range(len(images)):
#         img = Image.open(images[i]).resize([sample_size[1], sample_size[0]]).convert("RGB")
#         input_video[:, :, index[i]] = torch.from_numpy(np.array(img)).permute(2, 0, 1).unsqueeze(0)  / 255
#         input_video_mask[:, :, index[i]] = 0

#     clip_image = Image.open(images[0]).resize([sample_size[1], sample_size[0]]).convert("RGB")
#     clip_image = clip_image.resize([sample_size[1], sample_size[0]])

#     return input_video, input_video_mask, clip_image

def get_image_latent(ref_image=None, sample_size=None, padding=False):
    if ref_image is not None:
        if isinstance(ref_image, str):
            ref_image = Image.open(ref_image).convert("RGB")
            if padding:
                ref_image = padding_image(ref_image, sample_size[1], sample_size[0])
            ref_image = ref_image.resize((sample_size[1], sample_size[0]))
            ref_image = torch.from_numpy(np.array(ref_image))
            ref_image = ref_image.unsqueeze(0).permute([3, 0, 1, 2]).unsqueeze(0) / 255
        elif isinstance(ref_image, Image.Image):
            ref_image = ref_image.convert("RGB")
            if padding:
                ref_image = padding_image(ref_image, sample_size[1], sample_size[0])
            ref_image = ref_image.resize((sample_size[1], sample_size[0]))
            ref_image = torch.from_numpy(np.array(ref_image))
            ref_image = ref_image.unsqueeze(0).permute([3, 0, 1, 2]).unsqueeze(0) / 255
        else:
            ref_image = torch.from_numpy(np.array(ref_image))
            ref_image = ref_image.unsqueeze(0).permute([3, 0, 1, 2]).unsqueeze(0) / 255

    return ref_image

def get_image(ref_image=None):
    if ref_image is not None:
        if isinstance(ref_image, str):
            ref_image = Image.open(ref_image).convert("RGB")
        elif isinstance(ref_image, Image.Image):
            ref_image = ref_image.convert("RGB")

    return ref_image

def timer(func):
    def wrapper(*args, **kwargs):
        start_time  = time.time()
        result      = func(*args, **kwargs)
        end_time    = time.time()
        print(f"function {func.__name__} running for {end_time - start_time} seconds")
        return result
    return wrapper

def timer_record(model_name=""):
    def decorator(func):
        def wrapper(*args, **kwargs):
            torch.cuda.synchronize()
            start_time = time.time()
            result      = func(*args, **kwargs)
            torch.cuda.synchronize()
            end_time = time.time()
            import torch.distributed as dist
            if dist.is_initialized():
                if dist.get_rank() == 0:
                    time_sum  = end_time - start_time
                    print('# --------------------------------------------------------- #')
                    print(f'#   {model_name} time: {time_sum}s')
                    print('# --------------------------------------------------------- #')
                    _write_to_excel(model_name, time_sum)
            else:
                time_sum  = end_time - start_time
                print('# --------------------------------------------------------- #')
                print(f'#   {model_name} time: {time_sum}s')
                print('# --------------------------------------------------------- #')
                _write_to_excel(model_name, time_sum)
            return result
        return wrapper
    return decorator

def _write_to_excel(model_name, time_sum):
    import os

    import pandas as pd

    row_env = os.environ.get(f"{model_name}_EXCEL_ROW", "1")  # 默认第1行
    col_env = os.environ.get(f"{model_name}_EXCEL_COL", "1")  # 默认第A列
    file_path = os.environ.get("EXCEL_FILE", "timing_records.xlsx")  # 默认文件名

    try:
        df = pd.read_excel(file_path, sheet_name="Sheet1", header=None)
    except FileNotFoundError:
        df = pd.DataFrame()

    row_idx = int(row_env)
    col_idx = int(col_env)

    if row_idx >= len(df):
        df = pd.concat([df, pd.DataFrame([ [None] * (len(df.columns) if not df.empty else 0) ] * (row_idx - len(df) + 1))], ignore_index=True)

    if col_idx >= len(df.columns):
        df = pd.concat([df, pd.DataFrame(columns=range(len(df.columns), col_idx + 1))], axis=1)

    df.iloc[row_idx, col_idx] = time_sum

    df.to_excel(file_path, index=False, header=False, sheet_name="Sheet1")
