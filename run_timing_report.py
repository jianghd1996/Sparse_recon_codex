import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
SECONDS = r"\s*([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)\s*s?"
TOTALS = {
    "initialization took time:": "input",
    "images cropping took time:": "crop",
    "ffrecon took:": "da3",
    "4dlivephoto took:": "livephoto",
    "gen vid took:": "gen_video",
    "iter 3dgs took:": "dense_3dgs",
}
SUBSTAGES = {
    "livephoto": {
        "calc_img_captioner_time time:": "captioner_wall_s",
        "latents preparation time:": "prepare_s",
        "unpatchify:": "decode_s",
        r"livephoto generation \+ (?:offload|cache cleanup):": "workers_and_cleanup_wall_s",
    },
    "da3": {
        "load DA3 to gpu. Time:": "onload_s",
        "Model Forward Pass Done. Time:": "forward_s",
        "Conversion to Prediction Done. Time:": "convert_prediction_s",
        "Forward and Rendering DONE. Time:": "forward_and_render_s",
        "Export Results Done. Time:": "export_s",
        "Saving colmap info DONE. Time:": "save_colmap_s",
        "offload DA3 to cpu. Time:": "offload_s",
        "Clear cache DONE. Time:": "cleanup_s",
    },
    "gen_video": {
        "colmap_camera_sampling took:": "colmap_sampling_s",
        "load control video time:": "input_prepare_s",
        "total preparation:": "prepare_s",
        "input video:": "input_video_s",
        "control video:": "control_video_s",
        "images:": "reference_images_s",
        "(?:numpy|pt) decode:": "decode_s",
        "video generation:": "workers_wall_s",
        "Pool B (?:CPU offload|cache cleanup):": "cleanup_s",
    },
}
VIDEO_SUBSTAGES = {
    "infer_time:": "inference_s",
    "denoising:": "denoise_s",
    "save gen (?:vid|video):": "save_mp4_s",
    "save gen images:": "save_jpg_s",
}


class TimingReport:
    def __init__(self, work_dir):
        self.work_dir = work_dir
        self.started = time.perf_counter()
        self.stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.report = {"status": "running", "startup": {}, "cases": []}
        self.case = None
        self.stage = None

    def ingest(self, line):
        line = ANSI.sub("", line).strip()
        if line.startswith("process "):
            path = Path(line.removeprefix("process "))
            if path.is_dir():
                if self.case is None:
                    self.work_dir = self.work_dir or path.parent
                    self.report["startup"]["wall_to_first_case_s"] = time.perf_counter() - self.started
                self.case = {"case": path.name, "status": "incomplete", "stages": {}}
                self.report["cases"].append(self.case)
                self.stage = None
                return

        if self.case is None:
            match = re.search(r"Pool B model startup \(including warmup\):" + SECONDS, line)
            if match:
                self.report["startup"]["pool_b_startup_s"] = float(match[1])
            match = re.search(r"\[video warmup\] (cuda:\d+) (\d+) frames (\d+x\d+):" + SECONDS, line)
            if match:
                self.report["startup"].setdefault("warmup", []).append({
                    "device": match[1], "frames": int(match[2]), "size_hw": match[3], "seconds": float(match[4]),
                })
            return

        stages = self.case["stages"]
        match = re.search(r"sparse_recon took:" + SECONDS, line)
        if match:
            self.case.update(total_s=float(match[1]), status="ok")
            self.write()
            return
        for label, stage in TOTALS.items():
            match = re.search(re.escape(label) + SECONDS, line)
            if match:
                stages.setdefault(stage, {})["total_s"] = float(match[1])
                return

        for marker, stage in (
            ("run livephoto to augment images first", "livephoto"),
            ("start estimate camera views", "da3"),
            ("run video gen to denoise gs_render video", "gen_video"),
            ("开始运行3d高斯", "dense_3dgs"),
        ):
            if marker in line:
                self.stage = stage
                return
        if self.stage is None:
            return
        record = stages.setdefault(self.stage, {})
        if self.stage == "livephoto" and line.startswith("[caption timing] "):
            try:
                record["captioner"] = json.loads(line.removeprefix("[caption timing] "))
            except json.JSONDecodeError:
                pass
            return
        if self.stage == "dense_3dgs":
            match = re.search(r"训练耗时:" + SECONDS + r"\s*分钟", line)
            if match:
                record["training_s"] = float(match[1]) * 60
            return
        patterns = SUBSTAGES[self.stage]
        if self.stage in ("livephoto", "gen_video"):
            patterns = {**patterns, **VIDEO_SUBSTAGES}
            match = re.search(r"\[worker pid=\d+ (cuda:\d+)\].*peak_alloc=([0-9.]+) GiB", line)
            if match:
                record.setdefault("peak_alloc_gib", {})[match[1]] = float(match[2])
                return
        for pattern, name in patterns.items():
            match = re.search(pattern + SECONDS, line)
            if match:
                record.setdefault("samples", {}).setdefault(name, []).append(float(match[1]))
                return

    def write(self):
        if self.work_dir is None:
            return None
        self.report["batch_wall_s"] = time.perf_counter() - self.started
        totals = {}
        for case in self.report["cases"]:
            if case["status"] == "ok":
                totals.setdefault("case", []).append(case["total_s"])
                for stage, metrics in case["stages"].items():
                    if "total_s" in metrics:
                        totals.setdefault(stage, []).append(metrics["total_s"])
        self.report["stage_summary"] = {
            stage: {"count": len(values), "sum_s": sum(values), "mean_s": sum(values) / len(values)}
            for stage, values in totals.items()
        }
        path = self.work_dir / f"timing_summary_{self.stamp}.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
        return path


def main():
    parser = argparse.ArgumentParser(description="Run service tests and summarize their timing logs.")
    parser.add_argument("--work-dir", type=Path, help="Report destination. Defaults to the test root inferred from the first case.")
    args = parser.parse_args()
    report = TimingReport(args.work_dir)
    runner = Path(__file__).with_name("run_test_service.py")
    with subprocess.Popen(
        [sys.executable, "-u", str(runner)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    ) as process:
        for raw_line in process.stdout:
            line = raw_line.decode("utf-8", errors="replace")
            print(line, end="", flush=True)
            for entry in line.split("\r"):
                report.ingest(entry)
        code = process.wait()
    report.report.update(status="ok" if code == 0 else "failed", returncode=code)
    path = report.write()
    if path is not None:
        print(f"Timing summary saved: {path}")
    return code


if __name__ == "__main__":
    sys.exit(main())
