import cv2
import os
import tempfile
from io import BytesIO
import numpy as np
from glob import glob
from tqdm import tqdm
import torch
from torchvision import transforms
from torch import Tensor
from torch.nn import functional as F
import re
import math
import copy
from pathlib import Path
import pycolmap
import shutil
import concurrent.futures
from PIL import Image, ImageCms, ImageOps
import sys
from .face_match import FaceMatcher

'''
主体分割检测模型
'''
class Segmentation:
    name = 'BiRefNet'
    tensor2pil = transforms.ToPILImage()

    def __init__(self, path, device: str = "cuda"):
        torch._C._jit_set_profiling_mode(False)

        # 这里只记录推理设备，不分配 CUDA 张量
        self.device = torch.device(device)
        self._onloaded = False

        load_path = os.path.join(
            path, "BiRefNet-general-20250610_fp16.sbin"
        )
        self.model = torch.jit.load(
            load_path, map_location="cpu"
        ).eval()

        self.normalize_opt = transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        )

    def onload(self):
        if not self._onloaded:
            self.model.to(self.device)
            self._onloaded = True
        return self

    def offload(self):
        self.model.to("cpu")
        self._onloaded = False
        return self

    def reset(self):
        pass

    @torch.no_grad()
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        :param image: (h, w, c) in rgb mode, dtype is uint8
        :return: - **mask**. (h, w) uint8
        """
        image = torch.tensor(image, device=self.device).div(255.0).permute(2, 0, 1)
        mask = self.process_tensor(image)
        return mask.cpu().numpy()

    @torch.no_grad()
    def process_tensor(self, image: Tensor) -> Tensor:
        self.onload()

        raw_shape = image.shape[-2:]
        image = image.to(self.device).half()

        image_resized = F.interpolate(
            image[None],
            size=[512, 512],
            mode="bilinear",
            align_corners=True,
        )
        image_normalized = self.normalize_opt(image_resized)
        mask_resized = self.model(image_normalized.half())

        mask = F.interpolate(
            mask_resized,
            size=raw_shape,
            mode="bilinear",
            align_corners=True,
        )
        return mask.clip(0, 1)[0, 0].mul(255).to(torch.uint8)
        

def _open_video_writer(output_path: str, fps: float, frame_size):
    """Create an MP4 writer without invoking any external executable."""
    output_path = str(output_path)
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    fps = float(fps) if fps and fps > 0 else 30.0
    width, height = map(int, frame_size)
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid video frame size: {(width, height)}")

    # mp4v is the most portable codec exposed by opencv-python on Windows.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(
            f"OpenCV VideoWriter could not open output file: {output_path}. "
            "Please verify that your OpenCV build supports MP4 writing."
        )
    return writer


def _video_meta(cap: cv2.VideoCapture, fallback_fps: float = 30.0):
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 1e-6:
        fps = float(fallback_fps)
    width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    frame_count = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    return fps, width, height, frame_count


def _fit_frame_to_size(frame: np.ndarray, size):
    width, height = map(int, size)
    if frame.shape[1] == width and frame.shape[0] == height:
        return frame
    interpolation = cv2.INTER_AREA if width * height < frame.shape[1] * frame.shape[0] else cv2.INTER_LINEAR
    return cv2.resize(frame, (width, height), interpolation=interpolation)


def insert_img_in_src(folder_path, target_path, img_width, img_height, vid_path, st_index=81, trans=False):
    """
    Concatenate reversed right.mp4 + left.mp4 using OpenCV only, then extract
    the two boundary frames used by the original pipeline.

    The public signature and output layout are kept unchanged.
    """
    left_path = os.path.join(folder_path, "left.mp4")
    right_path = os.path.join(folder_path, "right.mp4")

    if not os.path.isfile(left_path):
        raise FileNotFoundError(f"left video not found: {left_path}")
    if not os.path.isfile(right_path):
        raise FileNotFoundError(f"right video not found: {right_path}")

    left_cap = cv2.VideoCapture(left_path)
    right_cap = cv2.VideoCapture(right_path)
    if not left_cap.isOpened():
        raise RuntimeError(f"OpenCV cannot open video: {left_path}")
    if not right_cap.isOpened():
        left_cap.release()
        raise RuntimeError(f"OpenCV cannot open video: {right_path}")

    try:
        left_fps, left_w, left_h, left_count = _video_meta(left_cap)
        right_fps, right_w, right_h, right_count = _video_meta(right_cap, fallback_fps=left_fps)

        if left_w <= 0 or left_h <= 0:
            raise RuntimeError(f"Invalid left video dimensions: {left_w}x{left_h}")
        if right_w <= 0 or right_h <= 0:
            raise RuntimeError(f"Invalid right video dimensions: {right_w}x{right_h}")

        if trans:
            output_size = (left_h, left_w)
        else:
            output_size = (left_w, left_h)

        writer = _open_video_writer(vid_path, left_fps, output_size)
        written = 0
        try:
            # Original behavior: reverse the right video first.
            for frame_idx in range(right_count - 1, -1, -1):
                right_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ok, frame = right_cap.read()
                if not ok:
                    continue
                if trans:
                    # Equivalent to a 90-degree counter-clockwise transpose.
                    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                frame = _fit_frame_to_size(frame, output_size)
                writer.write(frame)
                written += 1

            # Then append left video in the original order.
            left_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            while True:
                ok, frame = left_cap.read()
                if not ok:
                    break
                if trans:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                frame = _fit_frame_to_size(frame, output_size)
                writer.write(frame)
                written += 1
        finally:
            writer.release()

    finally:
        left_cap.release()
        right_cap.release()

    if written <= 0:
        raise RuntimeError(f"No frames were written to merged video: {vid_path}")

    os.makedirs(target_path, exist_ok=True)
    cap = cv2.VideoCapture(vid_path)
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV cannot reopen generated video: {vid_path}")

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        length = total_frames // 2
        st_index = min(int(st_index), length)

        target_frames = [length - st_index, length, length + st_index - 1]
        target_frames = [i for i in target_frames if 0 <= i < total_frames]

        for frame_idx in target_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            success, frame = cap.read()
            if not success or frame is None:
                raise RuntimeError(f"Failed to read frame {frame_idx} from {vid_path}")
            frame = cv2.resize(frame, (int(img_width), int(img_height)), interpolation=cv2.INTER_AREA)
            output_image = os.path.join(target_path, f"livephoto_{frame_idx:03d}.jpg")
            if not cv2.imwrite(output_image, frame):
                raise RuntimeError(f"Failed to save image: {output_image}")
    finally:
        cap.release()

    view_num = total_frames // 2
    print("livephoto view numbers:", view_num, "resize to:", img_width, img_height)


def merge_videos(folder_path, output_name="merged.mp4"):
    """Concatenate numbered MP4 segments with OpenCV and delete the segments."""
    videos = sorted([
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f))
        and re.match(r'^\d{8}\.mp4$', f)
    ])

    if len(videos) < 1:
        print("needs at least 1 video")
        return

    output_path = os.path.join(folder_path, output_name)

    if len(videos) == 1:
        if os.path.abspath(videos[0]) != os.path.abspath(output_path):
            if os.path.exists(output_path):
                os.remove(output_path)
            os.replace(videos[0], output_path)
        return

    print(f"concat order: {[os.path.basename(v) for v in videos]}")

    first_cap = cv2.VideoCapture(videos[0])
    if not first_cap.isOpened():
        raise RuntimeError(f"OpenCV cannot open video: {videos[0]}")
    fps, width, height, _ = _video_meta(first_cap)
    first_cap.release()

    if width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid video size for {videos[0]}: {width}x{height}")

    base, ext = os.path.splitext(output_path)
    temp_output = base + "_opencv_tmp" + (ext or ".mp4")
    if os.path.exists(temp_output):
        os.remove(temp_output)

    writer = _open_video_writer(temp_output, fps, (width, height))
    total_written = 0
    try:
        for video_path in videos:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise RuntimeError(f"OpenCV cannot open video: {video_path}")
            try:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    frame = _fit_frame_to_size(frame, (width, height))
                    writer.write(frame)
                    total_written += 1
            finally:
                cap.release()
    finally:
        writer.release()

    if total_written <= 0:
        if os.path.exists(temp_output):
            os.remove(temp_output)
        raise RuntimeError("No frames were written while merging videos")

    if os.path.exists(output_path):
        os.remove(output_path)
    os.replace(temp_output, output_path)

    for video_path in videos:
        if os.path.exists(video_path):
            os.remove(video_path)
            print(f"deleted: {os.path.basename(video_path)}")


def images_to_video(image_dir, output_path, framerate=24):
    """Create a video from *_img_pred.jpg images using OpenCV only."""
    pattern = re.compile(r"^(\d+)_img_pred\.jpg$")

    numbered_images = []
    for file_name in os.listdir(image_dir):
        match = pattern.match(file_name)
        if match:
            numbered_images.append((int(match.group(1)), os.path.join(image_dir, file_name)))

    if not numbered_images:
        print(f"❌ 错误：在 '{image_dir}' 中没有找到符合 'XXX_img_pred.jpg' 格式的图片！")
        return

    numbered_images.sort(key=lambda x: x[0])
    image_numbers = [x[0] for x in numbered_images]
    start_number = image_numbers[0]
    print(f"🎬 发现 {len(numbered_images)} 张图片，起始编号为: {start_number}")

    expected_count = image_numbers[-1] - start_number + 1
    if len(image_numbers) != expected_count:
        print("⚠️ 警告：图片编号存在缺失（不连续），将跳过缺失编号并按现有图片顺序写入。")

    first_frame = None
    first_path = None
    for _, path in numbered_images:
        frame = cv2.imread(path, cv2.IMREAD_COLOR)
        if frame is not None:
            first_frame = frame
            first_path = path
            break

    if first_frame is None:
        raise RuntimeError(f"No readable images found in: {image_dir}")

    height, width = first_frame.shape[:2]
    writer = _open_video_writer(output_path, framerate, (width, height))
    written = 0
    try:
        for _, image_path in numbered_images:
            frame = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if frame is None:
                print(f"⚠️ 无法读取，已跳过: {image_path}")
                continue
            frame = _fit_frame_to_size(frame, (width, height))
            writer.write(frame)
            written += 1
    finally:
        writer.release()

    if written <= 0:
        raise RuntimeError(f"No frames were written to: {output_path}")

    print(f"🎉 视频合成成功！保存在: {output_path}")


def sharp_score(image):
    return cv2.mean(cv2.Laplacian(image, cv2.CV_8UC1, ksize=3))[0]

def _resolve_video_resize(width: int, height: int, resize=None):
    """Resolve (w, h), supporting -1 as aspect-ratio-preserving dimension."""
    width, height = int(width), int(height)
    if resize is None:
        return width, height

    w, h = int(resize[0]), int(resize[1])

    if w != -1 and h != -1:
        # Preserve the orientation-adaptation behavior of the original code.
        if math.copysign(1, w - h) * math.copysign(1, width - height) < 0:
            w, h = h, w

    if w == -1 and h == -1:
        return width, height
    if w == -1:
        if h <= 0:
            raise ValueError(f"Invalid resize: {resize}")
        w = max(1, int(round(width * (h / height))))
    elif h == -1:
        if w <= 0:
            raise ValueError(f"Invalid resize: {resize}")
        h = max(1, int(round(height * (w / width))))

    if w <= 0 or h <= 0:
        raise ValueError(f"Invalid resize: {resize} -> {(w, h)}")
    return int(w), int(h)


def start_opencv_process(video_path: str, resize=None):
    """
    Open a video with OpenCV and return (capture, target_height, target_width, fps).
    This replaces the former external-process decoder.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV cannot open video: {video_path}")

    fps, width, height, _ = _video_meta(cap)
    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError(f"Invalid video dimensions for {video_path}: {width}x{height}")

    target_width, target_height = _resolve_video_resize(width, height, resize)
    print(f"raw w: {width} h: {height}")
    print("rotation: 0")
    print(f"decode resize: {target_width}x{target_height}")
    return cap, target_height, target_width, fps


def extract_frames(video_path: str, desired_fps: float | None, resize):
    """
    Decode frames with OpenCV, optionally resize them, and preserve the original
    sharpness-based frame selection strategy for target FPS sampling.
    """
    cap, height, width, fps = start_opencv_process(video_path, resize)
    print(f"zj_test fps {fps} desired_fps {desired_fps}")

    if desired_fps:
        fps_ratio = float(desired_fps) / float(fps)
    else:
        fps_ratio = 1.0

    portion = 0.0
    images = []
    shape_scores = []
    offset = -1000

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            if frame.shape[1] != width or frame.shape[0] != height:
                interpolation = cv2.INTER_AREA if width * height < frame.shape[1] * frame.shape[0] else cv2.INTER_LINEAR
                frame = cv2.resize(frame, (width, height), interpolation=interpolation)

            portion += fps_ratio
            images.append(frame)
            shape_scores.append(sharp_score(frame))

            if portion >= 1.0:
                portion -= 1.0
                len_ = len(shape_scores)
                mid = len_ // 2
                start = max(0, offset + mid)
                search_scores = shape_scores[start:]
                if not search_scores:
                    start = 0
                    search_scores = shape_scores
                best_local = int(np.argmax(search_scores))
                chosen_one = start + best_local
                offset = chosen_one - len_
                yield copy.deepcopy(images[chosen_one])
                images.clear()
                shape_scores.clear()
    finally:
        cap.release()


def _save_image(sc_idx: int, out_path: str, img):
    cv2.imwrite(out_path, img)
    return sc_idx

def copy_4k_images_rename(img_path, img_extract_path, colmap_path):
    source_path = Path(img_path)
    dest_path = Path(img_extract_path)

    # Create destination directory if it doesn't exist
    dest_path.mkdir(parents=True, exist_ok=True)

    reconstruction = pycolmap.Reconstruction(colmap_path)
    image_list = []
    for image_id, image in reconstruction.images.items():
        if "4k_" in image.name:
            image_list.append(image.name)
    if len(image_list) == 0:
        shutil.copytree(img_path, img_extract_path, dirs_exist_ok=True)
        return
    print(f"4k image_list {image_list}")
    # enumerate(path_list, start=1) provides the index 'i' starting from 1
    # Use *.[jJ][pP][gG] to catch .jpg and .JPG in Python 3.9
    jpg_files = list(source_path.glob("*.[jJ][pP][gG]"))

    for i, file_path in enumerate(jpg_files, start=1):
        if i > len(image_list):
            break
        # Create a new name like 'photo_001.jpg' using f-string padding
        # :03d ensures at least 3 digits (001, 002...)
        matches = [s for s in image_list if file_path.name in s]
        if len(matches) == 0:
            continue
        target_path = dest_path / matches[0]

        # Copy file and preserve metadata
        shutil.copy2(file_path, target_path)
        print(f"Index: {file_path.name} -> {matches}")

def extract_images_from_vid(video_file, images_path, desired_fps, resize, start_idx=0):
    tasks = []
    count_list = []
    os.makedirs(images_path, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        for idx, img in enumerate(extract_frames(video_file, desired_fps, resize)):
            img_idx = start_idx + idx
            out = os.path.join(images_path, f"{img_idx:0>3}.png")
            tasks.append(executor.submit(_save_image, idx, out, img))
        for future in concurrent.futures.as_completed(tasks):
            i = future.result()
            count_list.append(i)
    count_list.sort()

def instance_to_dict(obj):
    """类实例自动转为字典（包含所有实例变量）"""
    if not hasattr(obj, '__dict__'):
        raise TypeError(f"对象 {type(obj)} 没有 __dict__ 属性，无法自动转为字典")
    return obj.__dict__.copy()  # 返回副本，避免修改原实例属性

def crop_image(input_path, output_path, ratio):
    # 1. 打开图片
    with Image.open(input_path) as img:
        # 获取原图宽高
        width, height = img.size

        # 2. 计算中心 1/2 区域的坐标
        # 新宽高 = 原图的 1/2
        new_w = int(width * ratio)
        new_h = int(height * ratio)

        # 中心区域左上角坐标
        left = (width - new_w) // 2
        top = (height - new_h) // 2

        # 中心区域右下角坐标
        right = left + new_w
        bottom = top + new_h

        # 3. 裁剪图片
        cropped_img = img.crop((left, top, right, bottom))

        # 4. 保存
        cropped_img.save(output_path)
        print(f"裁剪完成！已保存到：{output_path}")


def read_image(input_path):
    image = cv2.imread(input_path)    
    img = image.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1) # Shape: [3, H, W]
    return image, torch.from_numpy(img)

def convert_image_to_srgb_values(input_image_path, output_image_path):
    """
    Converts an image's RGB values from its embedded ICC profile
    to standard sRGB values and saves the resulting untagged image.
    """
    try:
        with Image.open(input_image_path) as img:
            # 1. Check if the image has an ICC profile
            icc_profile_data = img.info.get("icc_profile")

            if icc_profile_data:
                # Use a temporary file to load the embedded profile data
                # This is a common workaround for Pillow-CMS
                with tempfile.NamedTemporaryFile(delete=False, suffix='.icc') as f:
                    f.write(icc_profile_data)
                    temp_profile_path = f.name

                # 2. Define the source and destination profiles
                source_profile = ImageCms.ImageCmsProfile(temp_profile_path)
                # Create a standard sRGB profile
                dest_profile = ImageCms.createProfile('sRGB')

                # 3. Apply the color transformation
                # The pixel values are internally converted to match the sRGB color space
                converted_img = ImageCms.profileToProfile(
                    img,
                    source_profile,
                    dest_profile,
                    renderingIntent=0,  # Perceptual rendering intent
                    outputMode='RGB'  # Ensure output is RGB
                )

                # Clean up the temporary file
                os.remove(temp_profile_path)
                print(f"Successfully converted image from embedded profile to sRGB values.")

            else:
                # If no profile is found, assume it's already sRGB and ensure RGB mode
                if img.mode != 'RGB':
                    converted_img = img.convert('RGB')
                else:
                    converted_img = img
                print(f"No embedded ICC profile found. Image saved as RGB.")

            # 4. Save the new image data without embedding a profile
            # The 'icc_profile=None' parameter drops the metadata
            converted_img.save(output_image_path, quality=90, icc_profile=None)
            print(f"Saved the untagged sRGB image to {output_image_path}.")

    except IOError as e:
        print(f"Error processing image file: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def drop_img_icc(img_path, img_ori_path):
        if not os.path.exists(img_ori_path):
            os.rename(img_path, img_ori_path)
        os.makedirs(img_path, exist_ok=True)
        for root, _, files in os.walk(img_ori_path):
            for file_name in files:
                if '.png' in file_name or '.jpg' in file_name:
                    file_path = os.path.join(root, file_name)
                    dest_path = os.path.join(img_path, file_name)
                    print(f"drop icc {file_path}")
                    convert_image_to_srgb_values(file_path, dest_path)

def resize_img(img_path, img_resize_path, img_size, drop_icc=True):
    """修正 EXIF 方向、缩放图片，并按需转为无 ICC 的 sRGB 图片。"""
    src_dir = Path(img_path).resolve()
    dst_dir = Path(img_resize_path).resolve()
    if not src_dir.is_dir():
        raise ValueError(f"源图片目录不存在: {src_dir}")
    if src_dir == dst_dir or src_dir in dst_dir.parents or dst_dir in src_dir.parents:
        raise ValueError("源目录和输出目录必须相互独立")

    target_size = (img_size, img_size) if isinstance(img_size, int) else tuple(img_size)
    if len(target_size) != 2 or any(not isinstance(size, int) or size <= 0 for size in target_size):
        raise ValueError(f"无效的目标尺寸: {img_size}")
    valid_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
    image_paths = sorted(
        path for path in src_dir.iterdir()
        if path.is_file() and path.suffix.lower() in valid_extensions
    )
    if not image_paths:
        print(f"源目录中没有支持的图片: {src_dir}")
        return

    dst_dir.mkdir(parents=True, exist_ok=True)
    for path in dst_dir.iterdir():
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()

    success_count = 0
    for path in image_paths:
        try:
            with Image.open(path) as source:
                image = ImageOps.exif_transpose(source)
                icc_profile = image.info.get("icc_profile")
                exif = image.getexif().tobytes()
                if drop_icc and icc_profile:
                    if image.mode not in {"RGB", "CMYK", "LAB"}:
                        image = image.convert("RGB")
                    image = ImageCms.profileToProfile(
                        image,
                        ImageCms.ImageCmsProfile(BytesIO(icc_profile)),
                        ImageCms.createProfile("sRGB"),
                        renderingIntent=0,
                        outputMode="RGB",
                    )
                elif image.mode != "RGB":
                    image = image.convert("RGB")

                image = image.resize(target_size, Image.Resampling.LANCZOS)
                image.save(
                    dst_dir / path.name,
                    quality=95,
                    icc_profile=None if drop_icc else icc_profile,
                    exif=exif,
                )
                success_count += 1
        except Exception as error:
            print(f"处理图片失败 {path.name}: {error}")

    print(f"Resize 完成: {success_count}/{len(image_paths)}, 尺寸 {target_size}, ICC drop={drop_icc}")
'''
def resize_img(img_path, img_ori_path, max_long_edge=1920, multiple=16):
    """
    检查 img_path 文件夹下的图片：
    若存在需要缩放的图片（> max_long_edge），则将全部原图备份移动至 img_ori_path；
    随后在 img_path 生成处理后的图片（超限的做缩放，未超限的复制原图）。
    若全部未超过尺寸，则不做任何操作。
    """
    valid_extensions = ('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff')
    
    if not os.path.exists(img_path):
        print(f"[跳过] 源路径不存在: {img_path}")
        return

    img_files = [f for f in os.listdir(img_path) if f.lower().endswith(valid_extensions)]
    if not img_files:
        print(f"[跳过] 文件夹内无有效图片: {img_path}")
        return

    # --- 阶段 1：预检查，判断是否存在超限图片 ---
    need_resize_count = 0
    for file_name in img_files:
        src_path = os.path.join(img_path, file_name)
        try:
            with Image.open(src_path) as img:
                w, h = img.size
                if max(w, h) > max_long_edge:
                    need_resize_count += 1
        except Exception as e:
            print(f"[警告] 无法读取图片尺寸 {src_path}: {e}")

    # 若没有任何图片需要缩放，则不做任何操作
    if need_resize_count == 0:
        print(f"[不做操作] {img_path} 内所有图片尺寸均未超过 {max_long_edge}")
        return

    # --- 阶段 2：执行全部原图备份 ---
    print(f"[发现 {need_resize_count} 张图片超限] 开始移动全部原图至备份目录: {img_ori_path}")
    os.makedirs(img_ori_path, exist_ok=True)
    
    for file_name in img_files:
        src_file_path = os.path.join(img_path, file_name)
        backup_file_path = os.path.join(img_ori_path, file_name)
        shutil.move(src_file_path, backup_file_path)

    # --- 阶段 3：从备份文件夹读取图片并处理输出到 img_path ---
    os.makedirs(img_path, exist_ok=True)

    for file_name in img_files:
        backup_file_path = os.path.join(img_ori_path, file_name)
        dst_file_path = os.path.join(img_path, file_name)

        try:
            with Image.open(backup_file_path) as img:
                w, h = img.size
                long_edge = max(w, h)

                # 未超过尺寸的图：直接完整复制回 img_path 保持原样
                if long_edge <= max_long_edge:
                    shutil.copy2(backup_file_path, dst_file_path)
                    print(f"[保持原样] {file_name} ({w}x{h})")
                    continue

                # 超过尺寸的图：计算并缩放
                scale = max_long_edge / float(long_edge)
                target_w = int(w * scale)
                target_h = int(h * scale)

                final_w = max((target_w // multiple) * multiple, multiple)
                final_h = max((target_h // multiple) * multiple, multiple)

                # RGBA 转 RGB 模式兼顾 JPG 保存
                if img.mode in ("RGBA", "P") and file_name.lower().endswith(('.jpg', '.jpeg')):
                    img = img.convert("RGB")

                resized_img = img.resize((final_w, final_h), Image.Resampling.LANCZOS)
                resized_img.save(dst_file_path)
                print(f"[缩放处理] {file_name}: {w}x{h} -> {final_w}x{final_h}")

        except Exception as e:
            print(f"[错误] 处理备份图片失败 {backup_file_path}: {e}")
'''

def get_crop_ratio(img_path, crop_mask_path, seg_module, face_detector):
    x_min, y_min, x_max, y_max = -1, -1, -1, -1
    for img in os.listdir(img_path):
        image = np.asarray(Image.open(os.path.join(img_path, img)))
        mask = seg_module(image)
        Image.fromarray(mask, mode='L').save(os.path.join(crop_mask_path, img))

        if x_min == -1:
            x_min, x_max, y_min, y_max = mask.shape[1], 0, mask.shape[0], 0
            print(mask.shape)

        rows, cols = np.where(mask == 255)

        if rows.size > 0:
            y_min = min(y_min, np.min(rows))
            y_max = max(y_max, np.max(rows))
            x_min = min(x_min, np.min(cols))
            x_max = max(x_max, np.max(cols))
        else:
            x_min, x_max, y_min, y_max = 0, mask.shape[1], 0, mask.shape[0]

    crop_ratio = max(1 - x_min / mask.shape[1], x_max / mask.shape[1], 1 - y_min / mask.shape[0], y_max / mask.shape[0])
    crop_ratio_sig = 1-(1-crop_ratio)*2
    

    x_min, y_min, x_max, y_max = -1, -1, -1, -1
    for img in os.listdir(img_path):
        raw_image, image = read_image(os.path.join(img_path, img))

        if x_min == -1:
            x_min, x_max, y_min, y_max = mask.shape[1], 0, mask.shape[0], 0

        pred_result = face_detector.predict(image)
        for pred in pred_result:
            x1, y1, x2, y2 = pred[:4]
            if (x2-x1) / mask.shape[1] > 0.1 or (y2-y1) / mask.shape[0] > 0.1:
                x_min = min(x_min, int(x1))
                x_max = max(x_max, int(x2))
                y_min = min(y_min, int(y1))
                y_max = max(y_max, int(y2))
                cv2.rectangle(raw_image, (int(x1), int(y1)), (int(x2), int(y2)), 255, 5)

        cv2.imwrite(os.path.join(crop_mask_path, "face_"+img), raw_image)

    crop_ratio = max(1 - x_min / mask.shape[1], x_max / mask.shape[1], 1 - y_min / mask.shape[0], y_max / mask.shape[0])
    if crop_ratio == 0:
        crop_ratio = 1.0
    crop_ratio_face = 1-(1-crop_ratio)*2

    crop_ratio = min(crop_ratio_sig, crop_ratio_face)
    print(f"cropping ratio, ratio: {crop_ratio}, face: {crop_ratio_face}, signi: {crop_ratio_sig}")

    return crop_ratio


def get_crop_ratio_w_face_track(img_path, crop_mask_path, seg_module, face_detector):
    x_min, y_min, x_max, y_max = -1, -1, -1, -1
    for img in os.listdir(img_path):
        image = np.asarray(Image.open(os.path.join(img_path, img)))
        mask = seg_module(image)
        Image.fromarray(mask, mode='L').save(os.path.join(crop_mask_path, img))

        if x_min == -1:
            x_min, x_max, y_min, y_max = mask.shape[1], 0, mask.shape[0], 0
            print(mask.shape)

        rows, cols = np.where(mask == 255)

        if rows.size > 0:
            y_min = min(y_min, np.min(rows))
            y_max = max(y_max, np.max(rows))
            x_min = min(x_min, np.min(cols))
            x_max = max(x_max, np.max(cols))
        else:
            x_min, x_max, y_min, y_max = 0, mask.shape[1], 0, mask.shape[0]

    crop_ratio = max(1 - x_min / mask.shape[1], x_max / mask.shape[1], 1 - y_min / mask.shape[0], y_max / mask.shape[0])
    crop_ratio_sig = 1-(1-crop_ratio)*2
    

    tracked_faces_lst = []
    tracked_faces_conf_lst = []
    tracked_faces_lm5_lst = []

    image_name_lst = os.listdir(img_path)

    x_min, y_min, x_max, y_max = -1, -1, -1, -1
    image_lst = []
    for img in image_name_lst:
        raw_image, image = read_image(os.path.join(img_path, img))
        image_lst.append(raw_image)
        boxes_conf_landms = face_detector.predict(image)
        tracked_faces = boxes_conf_landms[:, :4].long().tolist()
        tracked_faces_conf = boxes_conf_landms[:, 4].tolist()
        tracked_faces_lm5 = list(boxes_conf_landms[:, 5:].view(-1, 5, 2).cpu().numpy())
        tracked_faces_lst.append(tracked_faces)
        tracked_faces_conf_lst.append(tracked_faces_conf)
        tracked_faces_lm5_lst.append(tracked_faces_lm5)

    bboxes_list = tracked_faces_lst
    confs_list = tracked_faces_conf_lst
    landmarks_list = tracked_faces_lm5_lst

    m = FaceMatcher(min_frames=len(os.listdir(img_path)))
    result = m.match(bboxes_list, confs_list, landmarks_list)

    import random

    for gid, apps in result.items():
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        for fi, li, conf in apps:
            bbox = bboxes_list[fi][li]
            print(f"    frame {fi}, face_idx={li}, conf={conf:.4f}, bbox={bbox}")
            x1, y1, x2, y2 = bbox
            if x_min == -1:
                x_min, x_max, y_min, y_max = mask.shape[1], 0, mask.shape[0], 0
            x_min = min(x_min, int(x1))
            x_max = max(x_max, int(x2))
            y_min = min(y_min, int(y1))
            y_max = max(y_max, int(y2))
            
            cv2.rectangle(image_lst[fi], (int(x1), int(y1)), (int(x2), int(y2)), color, 5)

    for i in range(len(image_name_lst)):
        cv2.imwrite(os.path.join(crop_mask_path, "face_"+image_name_lst[i]), image_lst[i])


    crop_ratio = max(1 - x_min / mask.shape[1], x_max / mask.shape[1], 1 - y_min / mask.shape[0], y_max / mask.shape[0])
    if crop_ratio == 0:
        crop_ratio = 1.0
    crop_ratio_face = 1-(1-crop_ratio)*2

    if crop_ratio_face < 0.95:
        crop_ratio = crop_ratio_face
    else:
        crop_ratio = crop_ratio_sig
    print(f"cropping ratio, ratio: {crop_ratio}, face: {crop_ratio_face}, signi: {crop_ratio_sig}")

    return crop_ratio


def remove_frames_by_index_opencv(video_path, frames_to_remove=162):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("无法打开视频文件")
        return

    # 自动获取视频原有的 FPS、宽高和编码格式
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    dir_name, file_name = os.path.split(video_path)
    name, ext = os.path.splitext(file_name)
    temp_output = os.path.join(dir_name, f"{name}_processed{ext}")
    
    out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))
    
    # 采用直接快速定位到第 162 帧（如果底层解码器支持快进）
    # 为了保险，结合逐帧读取计数
    cap.set(cv2.CAP_PROP_POS_FRAMES, frames_to_remove)
    
    frame_idx = frames_to_remove
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        frame_idx += 1
        
    cap.release()
    out.release()
    
    if frame_idx > frames_to_remove:
        os.remove(video_path)
        os.rename(temp_output, video_path)
        print(f"处理完成，已覆盖原文件。共保留了 {frame_idx - frames_to_remove} 帧。")
    else:
        print("视频总帧数不足，未作修改。")
        if os.path.exists(temp_output):
            os.remove(temp_output)

def align_images_resolution(img_path: str, img_ori_path: str) -> str:
    """
    检查目录中图片分辨率，如果不一致，则备份原图并强行对齐到最低分辨率。
    
    :param img_path: 待处理图片所在文件夹路径
    :param img_ori_path: 原图备份文件夹路径
    :return: 状态说明字符串
    """
    valid_exts = {'.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tif', '.tiff'}
    
    if not os.path.exists(img_path):
        return "输入的图片路径不存在"
        
    img_files = [f for f in os.listdir(img_path) if os.path.splitext(f)[1].lower() in valid_exts]
    
    if not img_files:
        return "目录中没有找到图片"
    if len(img_files) == 1:
        return "分辨率一致" # 仅有一张图视为一致

    shapes = [] # 存储格式: (width, height, full_path, filename)
    for f in img_files:
        p = os.path.join(img_path, f)
        img = cv2.imread(p)
        if img is not None:
            h, w = img.shape[:2]
            shapes.append((w, h, p, f))
            
    if not shapes:
        return "无法读取目录中的图片"

    widths = [s[0] for s in shapes]
    heights = [s[1] for s in shapes]

    # 1. 判断是否分辨率完全一致
    is_resolution_same = (len(set(widths)) == 1) and (len(set(heights)) == 1)
    if is_resolution_same:
        return "分辨率一致"

    # 2. 判断是否尺寸（宽高比）一致
    aspect_ratios = [w / h for w, h in zip(widths, heights)]
    # 考虑极小的浮点误差（例如 4096/3072 和 5760/4320）
    is_aspect_ratio_same = (max(aspect_ratios) - min(aspect_ratios)) < 1e-3

    # 3. 确定“最低分辨率” (基于最小面积 w * h)
    # 你也可以根据需求改为 target_w, target_h = min(widths), min(heights)
    target_w, target_h = min(shapes, key=lambda x: x[0] * x[1])[:2]

    # 备份并执行强行对齐
    os.makedirs(img_ori_path, exist_ok=True)
    
    for w, h, p, f in shapes:
        # 备份原图 (保留元数据)
        ori_p = os.path.join(img_ori_path, f)
        shutil.copy2(p, ori_p)
        
        # 如果当前图片不是目标分辨率，则重采样并覆盖
        if w != target_w or h != target_h:
            img = cv2.imread(p)
            # 强行对齐到目标分辨率，无视长宽比变形
            resized_img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
            cv2.imwrite(p, resized_img)

    if is_aspect_ratio_same:
        return "尺寸一致但分辨率不同"
    else:
        return "尺寸不同"