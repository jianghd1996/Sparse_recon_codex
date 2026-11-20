import os
import sys
import torch
import torch.multiprocessing as mp
import time
import re
import numpy as np
import imageio_ffmpeg
from PIL import Image

def _decode_video_imageio_ffmpeg(path, w, h, fps=None):
    """
    使用 imageio-ffmpeg 解码视频，不依赖系统安装的 ffmpeg.exe / ffprobe.exe。

    功能保持与原实现一致：
    1. 解码为 RGB24；
    2. resize 到 (w, h)；
    3. 若指定 fps，则按照原来的 frame_skip 规则抽帧；
    4. 返回 shape=(N, H, W, 3)、dtype=uint8 的 numpy 数组。
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Video file does not exist: {path}")

    # imageio-ffmpeg 自带/管理 ffmpeg binary，这里不调用系统 ffmpeg/ffprobe。
    # scale 直接放在 imageio-ffmpeg 的输出参数中，避免 Python 端逐帧 resize。
    reader = imageio_ffmpeg.read_frames(
        path,
        pix_fmt="rgb24",
        output_params=["-vf", f"scale={int(w)}:{int(h)}"],
    )

    try:
        meta = next(reader)
    except StopIteration as e:
        raise RuntimeError(f"Failed to read video metadata: {path}") from e

    original_fps = meta.get("fps", None)

    if fps is not None:
        if original_fps is None or original_fps <= 0:
            raise RuntimeError(
                f"imageio-ffmpeg did not return a valid FPS for {path}: "
                f"{original_fps}"
            )
        # 保持原代码的抽帧规则不变。
        frame_skip = max(1, int(float(original_fps) // float(fps)))
    else:
        frame_skip = 1

    expected_bytes = int(h) * int(w) * 3
    frames = []

    try:
        for frame_idx, frame_bytes in enumerate(reader):
            if frame_idx % frame_skip != 0:
                continue

            if len(frame_bytes) != expected_bytes:
                raise RuntimeError(
                    f"Decoded frame size mismatch at frame {frame_idx}: "
                    f"got {len(frame_bytes)} bytes, expected {expected_bytes} "
                    f"for {w}x{h} RGB24."
                )

            # frombuffer 本身零拷贝；np.stack 时再统一生成连续数组。
            frame = np.frombuffer(
                frame_bytes,
                dtype=np.uint8,
            ).reshape(int(h), int(w), 3)

            frames.append(frame)

    finally:
        # generator.close() 会让 imageio-ffmpeg 清理其内部进程/管道。
        close_fn = getattr(reader, "close", None)
        if callable(close_fn):
            close_fn()

    if not frames:
        return np.empty((0, int(h), int(w), 3), dtype=np.uint8)

    return np.stack(frames, axis=0)

def get_video_to_video_latent_with_imgsize(frames, img_size,
                               fps=None, validation_video_mask=None, ref_image=None):
    
    ratio = min(img_size)/max(img_size)
    if img_size[0] <= img_size[1]: # width <= height
        if ratio > 0.7:
            sample_size = [1440, 1088]
        elif ratio > 0.65:                                                                                                                                                                                           
            sample_size = [1600, 1088]
        else:
            sample_size = [1440, 832] # [1280, 704] # 1664, 960
    else:
        if ratio > 0.7:
            sample_size = [1088, 1440] # [960, 1280] # 
        elif ratio > 0.65:
            sample_size = [1088, 1600]
        else:
            sample_size = [832, 1440] # 1664, 960
    if img_size[0] == img_size[1]:
        sample_size = [1280, 1280]

    if frames is not None:
        # frames: np.ndarray # N, H, W, 3
        input_video = torch.from_numpy(frames).permute(3, 0, 1, 2).unsqueeze(0).float() / 255

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


def seg_index_sample(gen_index, colmap_path, num_frames=80, video_length=81):
    segment_index = []
    init_image_names = get_unique_original_frame_names_prefer_first(
        gen_index, frames_per_video=video_length,
    )
    from Reconstruction.src.utils.colmap_loader import read_colmap_model
    # from Reconstruction.src.module.cameras import farthest_colmap_camera_sampling_idx_force_init_avoid_bad_frames

    colmap_cameras, colmap_images, colmap_points = read_colmap_model(colmap_path)
    ordered_images = sorted(list(colmap_images.values()), key=lambda i: i.name)
    colmap_images_dict = {image.id: image for image in ordered_images}
    sampled_colmap_images = farthest_colmap_camera_sampling_idx_force_init_avoid_bad_frames(
        colmap_images=colmap_images_dict,
        selected_num=num_frames,
        init_image_names=init_image_names,
        keep_4k=False,  # Input images are copied separately, outside this budget.
        frames_per_segment=video_length,
    )
    selected_list = []
    for key, value in sampled_colmap_images.items():
        selected_list.append(value.name)
    selected_list.sort()
    # print('### selected:', selected_list)
    segment_map = map_selected_to_segment_local_indices(
        selected_names=selected_list,
        gen_index=gen_index,
        frames_per_segment=video_length,
    )
    # print('### select key frames:', len(sampled_colmap_images))
    for seg_id, info in segment_map.items():
        # print(f"\nsegment {seg_id}")
        # print("gen_pair:", info["gen_pair"])
        # print("global_range:", info["global_range"])
        # print("global_frame_names:", info["global_frame_names"])
        # print("local_frame_indices:", info["local_frame_indices"])
        segment_index.append(info["local_frame_indices"])
    selected_count = sum(len(indices) for indices in segment_index)
    if selected_count != num_frames:
        raise ValueError(f"Requested {num_frames} generated frames, but only {selected_count} valid views are available")
    return segment_index


def append_input_images(input_paths, image_dir, colmap_path):
    """Copy true inputs under their DA3 COLMAP names; return total image count."""
    import shutil
    from pathlib import Path
    from Reconstruction.src.utils.colmap_loader import read_colmap_model

    image_dir = Path(image_dir)
    cameras, images, _ = read_colmap_model(str(colmap_path))
    by_name = {image.name: image for image in images.values()}
    sources = [(Path(path), "input_" + Path(path).name) for path in input_paths]
    if not sources or len({name for _, name in sources}) != len(sources):
        raise ValueError("Expected nonempty input images with unique names")
    # Validate all inputs before copying any files. Never borrow a generated pose.
    for source, name in sources:
        if not source.is_file():
            raise FileNotFoundError(source)
        if name not in by_name:
            raise ValueError(f"Missing input pose {name}; rerun DA3 with the updated code")
        camera = cameras[by_name[name].camera_id]
        with Image.open(source) as image:
            if image.size != (camera.width, camera.height):
                raise ValueError(f"Input dimensions do not match COLMAP: {source}")
    image_dir.mkdir(parents=True, exist_ok=True)
    for source, name in sources:
        shutil.copy2(source, image_dir / name)
    names = {
        path.name for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    }
    missing = names - by_name.keys()
    if missing:
        raise ValueError(f"Reconstruction images without COLMAP poses: {sorted(missing)}")
    return len(names)
