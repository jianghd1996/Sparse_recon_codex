import argparse
from pathlib import Path


def parse_args():
    repo_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Run single-image TI2V reconstruction")
    parser.add_argument("work_dir", type=Path, help="Case directory with image/, or a directory of cases")
    parser.add_argument("--case", action="append", help="Run only this case when given a directory of cases")
    parser.add_argument("--lora", dest="lora_path", type=Path, required=True)
    parser.add_argument("--lora-weight", type=float, default=0.55)
    parser.add_argument("--model-name", type=Path, default=repo_root / "Cloud_Models/Wan2.2-TI2V-5B")
    parser.add_argument("--config-path", type=Path, default=repo_root / "VideoX-Fun/config/wan2.2/wan_ti2v_single_5b.yaml")
    parser.add_argument("--device-id", type=int, default=0, help="TI2V device index after CUDA_VISIBLE_DEVICES")
    parser.add_argument("--model-cpu-offload", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--guidance-scale", type=float, default=6.0)
    parser.add_argument("--steps", dest="num_inference_steps", type=int, default=50)
    parser.add_argument("--shift", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--short-edge", type=int, default=704)
    parser.add_argument("--long-edge", type=int, default=1280)
    parser.add_argument("--video-length", type=int, default=121)
    parser.add_argument("--video-fps", type=int, default=24)
    parser.add_argument("--sample-count", type=int, default=40)
    parser.add_argument("--stage", choices=("video", "pose", "full"), default="full")
    args = parser.parse_args()

    if not args.lora_path.is_file():
        parser.error(f"LoRA checkpoint does not exist: {args.lora_path}")
    if not args.config_path.is_file():
        parser.error(f"TI2V config does not exist: {args.config_path}")
    if not args.model_name.is_dir():
        parser.error(f"TI2V model directory does not exist: {args.model_name}")
    if not (2 <= args.sample_count <= args.video_length and (args.video_length - 1) % 4 == 0):
        parser.error("Expected 2 <= sample count <= video length and a video length of 1 + n * 4")
    if not 0 < args.short_edge <= args.long_edge or min(args.video_fps, args.num_inference_steps) < 1:
        parser.error("Expected positive sizes, fps and inference steps, with short edge <= long edge")
    return args


def main():
    args = parse_args()
    work_dir = args.work_dir.resolve()
    if (work_dir / "image").is_dir():
        if args.case:
            raise SystemExit("--case applies only to a directory of cases")
        cases = [work_dir]
    elif args.case:
        cases = [work_dir / case for case in args.case]
    elif work_dir.is_dir():
        cases = sorted(path for path in work_dir.iterdir() if (path / "image").is_dir())
    else:
        cases = []
    if not cases or any(not (case / "image").is_dir() for case in cases):
        raise SystemExit(f"Could not find the requested case directories with image/ in {work_dir}")

    from pipeline_service_single import SparseRecon

    config = vars(args).copy()
    for key in ("model_name", "lora_path", "config_path"):
        config[key] = str(config[key].resolve())
    for key in ("case", "work_dir", "stage"):
        del config[key]
    config["run_DA3"] = args.stage in {"pose", "full"}
    config["run_3dgs_iter"] = args.stage == "full"
    recon = SparseRecon(config)
    try:
        for case in cases:
            print(f"process {case}")
            print(f"result: {recon(work_dir=case, mode='scene')}")
    finally:
        recon.shutdown_workers()


if __name__ == "__main__":
    main()
