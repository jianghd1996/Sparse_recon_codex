"""Prepare an exact generated-frame budget plus GT for 3DGS, including reruns."""

import json
import shutil
import tempfile
from pathlib import Path

import cv2
from PIL import Image

from .video_reader_and_sample import seg_index_sample


def _save_frame(frame, destination, camera):
    size = (camera.width, camera.height)
    if frame is None:
        raise ValueError(f"Cannot read frame for {destination.name}")
    if (frame.shape[1], frame.shape[0]) != size:
        frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    if not cv2.imwrite(str(destination), frame):
        raise RuntimeError(f"Cannot write reconstruction frame: {destination}")


def _extract_frames(path, requests, stage, by_name, cameras, expected_frames):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Cannot open saved generated video: {path}")
    written = set()
    count = 0
    try:
        # Sequential decode preserves exact frame IDs; do not resample by FPS
        # or pick sharp frames, since each name already identifies a COLMAP pose.
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if count in requests:
                name = requests[count]
                _save_frame(frame, stage / name, cameras[by_name[name].camera_id])
                written.add(count)
            count += 1
    finally:
        capture.release()
    if count != expected_frames:
        raise ValueError(
            f"Saved video {path} has {count} frames, expected {expected_frames}. "
            "Keep view_num and the saved trajectory plan unchanged when rerunning 3DGS."
        )
    missing = requests.keys() - written
    if missing:
        raise ValueError(f"Missing frames in {path}: {sorted(missing)}")


def prepare_reconstruction_images(
    gen_index, colmap_path, video_dir, max_frames, video_length,
    input_paths=None, prefer_video=False,
):
    """Select max_frames generated views, append GT, then replace image/ safely.

    input_paths contains original/preprocessed inputs on a fresh pipeline run.
    Reconstruction-only reruns recover the GT copies from image/ instead.
    """
    from Reconstruction.src.utils.colmap_loader import read_colmap_model

    if type(max_frames) is not int or max_frames < 1:
        raise ValueError("max_frames must be a positive integer")
    if type(video_length) is not int or video_length < 1 or not gen_index:
        raise ValueError("A nonempty trajectory plan and positive view_num are required")
    video_dir = Path(video_dir)
    image_dir = video_dir / "image"
    if image_dir.is_symlink():
        raise ValueError(f"Reconstruction image directory cannot be a symlink: {image_dir}")
    cameras, images, _ = read_colmap_model(str(colmap_path))
    by_name = {image.name: image for image in images.values()}
    previous = {}
    manifest_path = video_dir / "reconstruction_selection.json"
    if input_paths is None and manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous["video_length"] != video_length or previous["gen_index"] != gen_index:
            raise ValueError("Saved sampling metadata differs from view_num/trajectory plan; use the original run settings")

    if input_paths is not None:
        sources = [(Path(path), "input_" + Path(path).name) for path in input_paths]
    else:
        # Older cases predate sampling metadata but already contain input_* GT.
        names = previous.get("input_names")
        if names is None:
            names = sorted(
                path.name for path in image_dir.iterdir()
                if path.is_file() and path.name.startswith("input_") and path.name in by_name
            ) if image_dir.is_dir() else []
        sources = [(image_dir / name, name) for name in names]
    if not sources or len({name for _, name in sources}) != len(sources):
        raise ValueError("No unique cached GT images found; run the full pipeline once to save input_* images")
    for source, name in sources:
        if not source.is_file():
            raise FileNotFoundError(f"Missing GT image: {source}")
        if name not in by_name:
            raise ValueError(f"Missing COLMAP pose for GT {name}; rerun DA3")
        camera = cameras[by_name[name].camera_id]
        with Image.open(source) as image:
            if image.size != (camera.width, camera.height):
                raise ValueError(f"GT image dimensions differ from COLMAP: {source}")

    selected = seg_index_sample(
        gen_index, str(colmap_path), num_frames=max_frames, video_length=video_length,
        colmap_images=images,
    )
    selected_names = [
        f"{segment * video_length + frame:03d}.png"
        for segment, indices in enumerate(selected) for frame in indices
    ]
    if len(set(selected_names)) != max_frames:
        raise ValueError("Selected generated frame count does not match max_frames")
    if any(name not in by_name for name in selected_names):
        raise ValueError("Selected generated frames are missing COLMAP poses")
    if set(selected_names) & {name for _, name in sources}:
        raise ValueError("Generated frame names collide with GT names")

    needed_segments = [i for i, frames in enumerate(selected) if frames]
    segment_videos = [video_dir / f"{i:08d}.mp4" for i in needed_segments]
    merged_video = video_dir / "result.mp4"
    has_segments = all(path.is_file() for path in segment_videos)
    has_video = has_segments or merged_video.is_file()
    missing_images = [name for name in selected_names if not (image_dir / name).is_file()]
    use_video = has_video and (prefer_video or bool(missing_images))
    if missing_images and not use_video:
        raise FileNotFoundError(
            f"New selection needs {len(missing_images)} unavailable generated images "
            f"(e.g. {missing_images[:5]}). No complete saved gen_video videos were found. "
            "Increasing max_frames requires saved debug/full videos or rerunning gen_video."
        )

    video_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".recon_sample_", dir=video_dir) as temporary:
        stage = Path(temporary) / "image"
        stage.mkdir()
        if use_video:
            if has_segments:
                for segment, path in zip(needed_segments, segment_videos):
                    requests = {frame: f"{segment * video_length + frame:03d}.png" for frame in selected[segment]}
                    _extract_frames(path, requests, stage, by_name, cameras, video_length)
            else:
                requests = {segment * video_length + frame: f"{segment * video_length + frame:03d}.png"
                            for segment, indices in enumerate(selected) for frame in indices}
                _extract_frames(merged_video, requests, stage, by_name, cameras, len(gen_index) * video_length)
        else:
            for name in selected_names:
                source = image_dir / name
                camera = cameras[by_name[name].camera_id]
                with Image.open(source) as image:
                    same_size = image.size == (camera.width, camera.height)
                if same_size:
                    shutil.copy2(source, stage / name)
                else:
                    _save_frame(cv2.imread(str(source)), stage / name, camera)
        for source, name in sources:
            shutil.copy2(source, stage / name)
        metadata = dict(
            max_frames=max_frames, input_names=[name for _, name in sources],
            generated_names=selected_names, video_length=video_length,
            gen_index=gen_index, sampled_from="video" if use_video else "images",
        )
        stage_manifest = Path(temporary) / "selection.json"
        stage_manifest.write_text(json.dumps(metadata, indent=2, default=int), encoding="utf-8")
        actual = {path.name for path in stage.iterdir()}
        expected = set(selected_names) | {name for _, name in sources}
        if actual != expected:
            raise RuntimeError("Prepared image set differs from the requested selection")
        # Keep the old image directory intact until every frame and GT succeeded.
        backup = Path(temporary) / "previous_image"
        had_images = image_dir.exists()
        if had_images:
            image_dir.rename(backup)
        installed = False
        try:
            stage.rename(image_dir)
            installed = True
            stage_manifest.replace(manifest_path)
        except BaseException:
            if installed:
                image_dir.rename(stage)
            if had_images:
                backup.rename(image_dir)
            raise
    return max_frames + len(sources)
