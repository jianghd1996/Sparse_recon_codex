import os

from pipeline_service_debug import SparseRecon
# from pipeline_jiang3_debug import SparseRecon

import json
import time
from datetime import datetime
from pathlib import Path

# pose_method = pi3 or vggt or data_verify
# CUDA_VISIBLE_DEVICES=4,5,6,7 python run_vid_gen_single_zj_test_new_V008.py 

'''
    功能说明：
    img_dir: str                    输入图片路径
        use_input_icc_drop: bool        测试图片进行icc处理，避免色差
    run_DA3: bool                   是否用DA3估计位姿
        use_enhance_crop: bool          画面中心增强
        use_view_extrapolation: bool    视角外插
    run_livephoto: bool             是否用livephoto进行视角扩充
        livephoto_model_offload: bool  是否将 LivePhoto T5 和 DiT 卸载到 CPU
        livephoto_vae_offload: bool    是否在单次生成内将 VAE 卸载到 CPU
    run_gen_video: bool             是否进行视频生成
        video_offload: bool            多图 Wan 推理启用 CPU offload，默认关闭
        video_vae_offload: bool        多图 VAE 搬到 CPU，video_offload 开启时强制卸载
        video_warmup_sizes: list       多图预热尺寸列表，顺序为 [高, 宽]，空列表关闭
        gen_vid_fps: int                视频生成保存fps
    run_3dgs_iter: bool             是否稠密重建
'''

def write_timing_summary(path, report):
    completed = [case["timings"] for case in report["cases"] if case["status"] == "ok"]
    totals = {}
    for case in completed:
        for stage, value in case.items():
            elapsed = value.get("total_s") if isinstance(value, dict) else value if stage == "total_s" else None
            if elapsed is not None:
                totals.setdefault(stage, []).append(elapsed)
    report["completed_cases"] = len(completed)
    report["stage_summary"] = {
        stage: {"count": len(values), "sum_s": sum(values), "mean_s": sum(values) / len(values)}
        for stage, values in totals.items()
    }
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


if __name__ == "__main__":
    json_str="""{
                "use_input_icc_drop":false,
                "use_resize_drop":true,
                
                "run_DA3":true,
                    "use_enhance_crop":false,
                    "use_view_extrapolation":true,

                "run_livephoto":false,
                    "livephoto_model_offload":false,
                    "livephoto_vae_offload":false,
                
                "run_gen_video":true,
                    "view_num":81,
                    "gen_vid_infer_step":6,
                    "gen_vid_fps":30,
                    "video_offload":false,
                    "video_vae_offload":false,
                    "video_batch_size":1,
                    "video_warmup_sizes":[[1920,1088]],
                "world_size":2,

                "run_3dgs_iter":true
                }"""
    
    if os.path.exists(json_str):
        with open(json_str, "r") as json_file:
            json_args = json.load(json_file)
            print(f"args from json: {json_args}")
    else:
        json_args = json.loads(json_str)
        print(f"args from json: {json_args}")

    skip_list = []
    # work_dir = "/mnt/DataPart/PrivateSpace/wujinxiao/00_data/00_0829_synchaisi_image"
    work_dir = "/home/z00566689/dev/mnt/jiang_test/00_1110_multi_res_time_test/multi_image"
    run_list = os.listdir(work_dir)
    run_list = ["2160x3840"]

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    report_path = Path(work_dir) / f"timing_summary_{stamp}.json"
    batch_started = time.perf_counter()
    report = {
        "status": "running",
        "config": json_args,
        "startup": {},
        "cases": [],
    }
    print(f"Timing summary: {report_path}")
    try:
        recon = SparseRecon(json_args)
        report["startup"] = recon.startup_timings
        for item in sorted(os.listdir(work_dir)):
            if run_list and item not in run_list:
                continue
            if item in skip_list or "." in item:
                continue

            data_dir = os.path.join(work_dir, item)
            if not os.path.isdir(data_dir):
                continue
            print(f"process {data_dir}")
            export_dir = os.path.join(data_dir, "export")
            case = {"case": item, "status": "running"}
            report["cases"].append(case)
            started = time.perf_counter()
            try:
                recon(work_dir=data_dir, export_dir=export_dir, mode="scene")
            except BaseException as exc:
                case["status"] = "failed"
                case["error"] = f"{type(exc).__name__}: {exc}"
                raise
            else:
                case["status"] = "ok"
            finally:
                case["wall_s"] = time.perf_counter() - started
                case["timings"] = recon.last_timings
                report["batch_wall_s"] = time.perf_counter() - batch_started
                write_timing_summary(report_path, report)
        report["status"] = "ok"
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["batch_wall_s"] = time.perf_counter() - batch_started
        write_timing_summary(report_path, report)
        print(f"Timing summary saved: {report_path}")
