import gc
import inspect
import os
import shutil
import subprocess
import time

import cv2
import imageio
import numpy as np
import torch
import torchvision
from einops import rearrange
from PIL import Image

import math
import copy
import struct
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor


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

def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=6, fps=12,
                     imageio_backend=True, color_transfer_post_process=False, reverse_width=False):
    videos = rearrange(videos, "b c t h w -> t b c h w")

    # 1. 仍用 make_grid 保证布局完全一致，但跳过 PIL
    grids = torch.stack([torchvision.utils.make_grid(x, nrow=n_rows) for x in videos])
    # grids: (T, C, grid_H, grid_W)

    # 2. 批量转换：rescale + to uint8，一次完成
    grids = grids.permute(0, 2, 3, 1)  # (T, H, W, C)
    if rescale:
        grids = (grids + 1.0) / 2.0
    frames_np = grids.clamp(0, 1).mul(255).byte().contiguous().numpy()

    # 3. color transfer（按需）
    if color_transfer_post_process:
        ref = frames_np[0]
        for i in range(1, len(frames_np)):
            frames_np[i] = color_transfer(frames_np[i], ref)
    if reverse_width:
        frames_np = frames_np[:,:,::-1]

    # 4. 写视频
    os.makedirs(os.path.dirname(path), exist_ok=True)
    h, w = frames_np.shape[1], frames_np.shape[2]

    if imageio_backend:
        if path.endswith("mp4"):
            # ffmpeg 管道写入，比 imageio.mimsave 快很多
            # -c:v libx264 "-threads", str(ffmpeg_threads), or -c:v h264_nvenc
            cmd = [
                'ffmpeg', '-y', '-loglevel', 'error',
                '-f', 'rawvideo', '-vcodec', 'rawvideo',
                '-s', f'{w}x{h}', '-pix_fmt', 'rgb24',
                '-r', str(fps),
                '-i', 'pipe:0',
                '-c:v', 'libx264',
                '-preset', 'ultrafast',
                '-crf', '23',
                '-pix_fmt', 'yuv420p',
                '-threads', '1',
                path
            ]
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )

            try:
                proc.stdin.write(frames_np.tobytes())
                proc.stdin.close()

                err = proc.stderr.read()
                ret = proc.wait()

                if ret != 0:
                    raise RuntimeError(
                        f"ffmpeg failed with return code {ret}:\n"
                        f"{err.decode('utf-8', errors='ignore')}"
                    )

            except BrokenPipeError:
                err = proc.stderr.read()
                ret = proc.wait()
                raise RuntimeError(
                    f"ffmpeg broken pipe, return code {ret}:\n"
                    f"{err.decode('utf-8', errors='ignore')}"
                )
        else:
            imageio.mimsave(path, list(frames_np), duration=int(1000 / fps))
    else:
        if path.endswith("mp4"):
            path = path.replace('.mp4', '.gif')
        imgs = [Image.fromarray(f) for f in frames_np]
        imgs[0].save(path, format='GIF', append_images=imgs[1:], save_all=True, duration=100, loop=0)

# def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=6, fps=12, imageio_backend=True, color_transfer_post_process=False):
#     videos = rearrange(videos, "b c t h w -> t b c h w")
#     outputs = []
#     for x in videos:
#         x = torchvision.utils.make_grid(x, nrow=n_rows)
#         x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
#         if rescale:
#             x = (x + 1.0) / 2.0  # -1,1 -> 0,1
#         x = (x * 255).numpy().astype(np.uint8)
#         outputs.append(Image.fromarray(x))

#     if color_transfer_post_process:
#         for i in range(1, len(outputs)):
#             outputs[i] = Image.fromarray(color_transfer(np.uint8(outputs[i]), np.uint8(outputs[0])))

#     os.makedirs(os.path.dirname(path), exist_ok=True)
#     if imageio_backend:
#         if path.endswith("mp4"):
#             # imageio.mimsave(path, outputs, fps=fps)
#             imageio.mimsave(path, outputs)
#         else:
#             imageio.mimsave(path, outputs, duration=(1000 * 1/fps))
#     else:
#         if path.endswith("mp4"):
#             path = path.replace('.mp4', '.gif')
#         outputs[0].save(path, format='GIF', append_images=outputs, save_all=True, duration=100, loop=0)
def tensor_frame_to_bgr_image(
    frame_tensor: torch.Tensor,
    rescale=False,
    n_rows=6,
):
    """
    将一帧 tensor 转成 BGR uint8 ndarray，格式对齐 extract_frames() 的输出。

    Args:
        frame_tensor: [B, C, H, W]
        rescale: 如果 tensor 范围是 [-1, 1]，设为 True
        n_rows: make_grid 的 nrow

    Returns:
        img_bgr: np.ndarray, uint8, BGR, shape [H, W, 3]
    """
    grid = torchvision.utils.make_grid(frame_tensor, nrow=n_rows)  # [C, H, W]

    if rescale:
        grid = (grid + 1.0) / 2.0

    img_rgb = (
        grid.clamp(0, 1)
        .mul(255)
        .byte()
        .permute(1, 2, 0)
        .cpu()
        .contiguous()
        .numpy()
    )

    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    return img_bgr

def resize_bgr_like_ffmpeg(img_bgr: np.ndarray, resize):
    """
    模拟 start_ffmpeg_process(video_path, resize) 中的 resize 逻辑。

    Args:
        img_bgr: np.ndarray, uint8, BGR, shape [H, W, 3]
        resize: None 或 (w, h)，允许 w/h 为 -1

    Returns:
        resized BGR image
    """
    if resize is None:
        return img_bgr

    height, width = img_bgr.shape[:2]
    w, h = resize

    # 对齐你原来的方向判断逻辑
    if w != -1 and h != -1:
        if math.copysign(1, w - h) * math.copysign(1, width - height) < 0:
            w, h = h, w

    # 模拟 ffmpeg scale 中 -1 的行为：按比例保持宽高比
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
    
    # 根据放大/缩小选 interpolation
    if w * h < width * height:
        interp = cv2.INTER_AREA       # 缩小最佳
    else:
        interp = cv2.INTER_LINEAR     # 放大用 LINEAR,速度和质量平衡
        # 追求质量用 cv2.INTER_CUBIC,但慢约 2-3 倍

    return cv2.resize(img_bgr, (w, h), interpolation=interp)

def _resize_and_save(out, img, resize):
    """resize + 写 PNG 合并到一个 task,两步都释放 GIL,可并行。"""
    if resize is not None:
        img = resize_bgr_like_ffmpeg(img, resize)
    # PNG 压缩等级从默认 3 降到 1,通常能快 2-3 倍
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



def merge_video_audio(video_path: str, audio_path: str):
    """
    Merge the video and audio into a new video, with the duration set to the shorter of the two,
    and overwrite the original video file.

    Parameters:
    video_path (str): Path to the original video file
    audio_path (str): Path to the audio file
    """
    # check
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"video file {video_path} does not exist")
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"audio file {audio_path} does not exist")

    base, ext = os.path.splitext(video_path)
    temp_output = f"{base}_temp{ext}"

    try:
        # create ffmpeg command
        command = [
            'ffmpeg',
            '-y',  # overwrite
            '-i',
            video_path,
            '-i',
            audio_path,
            '-c:v',
            'copy',  # copy video stream
            '-c:a',
            'aac',  # use AAC audio encoder
            '-b:a',
            '192k',  # set audio bitrate (optional)
            '-map',
            '0:v:0',  # select the first video stream
            '-map',
            '1:a:0',  # select the first audio stream
            '-shortest',  # choose the shortest duration
            temp_output
        ]

        # execute the command
        print("Start merging video and audio...")
        result = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # check result
        if result.returncode != 0:
            error_msg = f"FFmpeg execute failed: {result.stderr}"
            print(error_msg)
            raise RuntimeError(error_msg)

        shutil.move(temp_output, video_path)
        print(f"Merge completed, saved to {video_path}")

    except Exception as e:
        if os.path.exists(temp_output):
            os.remove(temp_output)
        print(f"merge_video_audio failed with error: {e}")

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

def _decode_video_ffmpeg(path, w, h, fps=None):
    """用 ffmpeg 一次性解码 + resize + 抽帧，返回 (N, H, W, 3) uint8 numpy"""
    # 先探测原始 fps
    if fps is not None:
        probe = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
             '-show_entries', 'stream=r_frame_rate', '-of', 'csv=p=0', path],
            capture_output=True, text=True
        )
        num, den = map(int, probe.stdout.strip().split('/'))
        original_fps = num / den
        frame_skip = max(1, int(original_fps // fps))
        # ffmpeg select 过滤器：比 Python 循环 skip 快得多
        vf = f"select=not(mod(n\\,{frame_skip})),scale={w}:{h}"
    else:
        vf = f"scale={w}:{h}"

    cmd = [
        'ffmpeg', '-loglevel', 'error',
        '-i', path,
        '-vf', vf,
        '-vsync', 'vfr',
        '-f', 'rawvideo', '-pix_fmt', 'rgb24',
        'pipe:1'
    ]
    proc = subprocess.run(cmd, capture_output=True)
    raw = proc.stdout
    frame_bytes = h * w * 3
    n_frames = len(raw) // frame_bytes
    # 零拷贝：直接从 buffer 构建 numpy
    frames = np.frombuffer(raw, dtype=np.uint8).reshape(n_frames, h, w, 3)
    return frames


def get_video_to_video_latent(input_video_path, video_length, sample_size,
                               fps=None, validation_video_mask=None, ref_image=None):
    if input_video_path is not None:
        if isinstance(input_video_path, str):
            # 1. ffmpeg 一步完成 解码+resize+抽帧+色彩转换
            frames = _decode_video_ffmpeg(
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

def get_video_to_video_latent_with_mask(input_video_path, video_length, sample_size, fps=None, input_video_mask=None, ref_image=None):
    if input_video_path is not None:
        if isinstance(input_video_path, str):
            cap = cv2.VideoCapture(input_video_path)
            input_video = []

            original_fps = cap.get(cv2.CAP_PROP_FPS)
            frame_skip = 1 if fps is None else max(1,int(original_fps // fps))

            frame_count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % frame_skip == 0:
                    frame = cv2.resize(frame, (sample_size[1], sample_size[0]))
                    input_video.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                frame_count += 1

            cap.release()
        else:
            input_video = input_video_path

        input_video = torch.from_numpy(np.array(input_video))
        input_video = input_video.permute([3, 0, 1, 2]).unsqueeze(0) / 255

        if input_video_mask is not None:
            cap = cv2.VideoCapture(input_video_mask)
            input_video_mask = []

            original_fps = cap.get(cv2.CAP_PROP_FPS)
            frame_skip = 1 if fps is None else max(1,int(original_fps // fps))

            frame_count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % frame_skip == 0:
                    frame = cv2.resize(frame, (sample_size[1], sample_size[0]))
                    input_video_mask.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                frame_count += 1

            cap.release()

            input_video_mask = torch.from_numpy(np.array(input_video_mask))
            input_video_mask = input_video_mask.permute([3, 0, 1, 2]).unsqueeze(0)[:, :1]
        else:
            input_video_mask = torch.ones_like(input_video[:, :1]) * 255.0
    else:
        input_video, input_video_mask = None, None

    if ref_image is not None:
        if isinstance(ref_image, str):
            clip_image = Image.open(ref_image).convert("RGB")
        else:
            clip_image = Image.fromarray(np.array(ref_image, np.uint8))
    else:
        clip_image = None

    if ref_image is not None:
        if isinstance(ref_image, str):
            ref_image = Image.open(ref_image).convert("RGB")
            ref_image = ref_image.resize((sample_size[1], sample_size[0]))
            ref_image = torch.from_numpy(np.array(ref_image))
            ref_image = ref_image.unsqueeze(0).permute([3, 0, 1, 2]).unsqueeze(0) / 255
        else:
            ref_image = torch.from_numpy(np.array(ref_image))
            ref_image = ref_image.unsqueeze(0).permute([3, 0, 1, 2]).unsqueeze(0) / 255
    return input_video, 255-input_video_mask, ref_image, clip_image

def _decode_video_ffmpeg_segment(path, w, h, fps=None,
                                  start_frame=0, num_frames=None):
    """只解码抽帧后 [start_frame, start_frame+num_frames) 的片段"""
    # 决定目标 fps；如果没指定，就用原始 fps
    if fps is None:
        probe = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
             '-show_entries', 'stream=r_frame_rate', '-of', 'csv=p=0', path],
            capture_output=True, text=True
        )
        num, den = map(int, probe.stdout.strip().split('/'))
        target_fps = num / den
    else:
        target_fps = fps

    # 在"目标 fps 时间轴"上 seek
    start_time = start_frame / target_fps

    # 用 fps filter 抽帧，干净且不依赖帧计数器
    vf_parts = []
    if fps is not None:
        vf_parts.append(f"fps={fps}")
    vf_parts.append(f"scale={w}:{h}")
    vf = ",".join(vf_parts)

    cmd = ['ffmpeg', '-loglevel', 'error',
           '-ss', f'{start_time:.6f}',   # input seek，快
           '-i', path,
           '-vf', vf,
           '-vsync', 'vfr']
    if num_frames is not None:
        cmd += ['-frames:v', str(num_frames)]
    cmd += ['-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1']

    proc = subprocess.run(cmd, capture_output=True)
    frame_bytes = h * w * 3
    n = len(proc.stdout) // frame_bytes  # 实际解码出的帧数

    frames = np.frombuffer(proc.stdout, dtype=np.uint8).reshape(n, h, w, 3)
    # frames.setflags(write=True)
    return frames


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
