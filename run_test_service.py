import os

from pipeline_service_single import SparseRecon

import json
'''
    work_dir 下每个 case 的 image/ 放一张输入图。
    在 pipeline_service_single.py 中设置 121 帧 TI2V 的 lora_path。
    run_DA3 和 run_3dgs_iter 都设为 false 时只生成视频和采样帧。
'''

if __name__ == "__main__":
    json_str="""{
                "run_DA3":true,
                "run_3dgs_iter":true
                }"""

    if os.path.exists(json_str):
        with open(json_str, "r") as json_file:
            json_args = json.load(json_file)
            print(f"args from json: {json_args}")
    else:
        json_args = json.loads(json_str)
        print(f"args from json: {json_args}")
    recon = SparseRecon(json_args)

    skip_list = []
    work_dir = "/path/to/single_image_cases"
    run_list = []  # 空列表跑全部 case，例如 ["2160x3840"] 只跑一个

    for item in sorted(os.listdir(work_dir)):
        if len(run_list) != 0 and item not in run_list:
            continue
        if item in skip_list:
            continue
        if '.' in item:
            continue

        data_dir = os.path.join(work_dir, item)
        if os.path.isdir(data_dir):
            print(f"process {data_dir}")
            export_dir=os.path.join(data_dir,"export")
            recon(work_dir=data_dir, export_dir=export_dir, mode="scene")
