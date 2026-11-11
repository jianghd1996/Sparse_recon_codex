import os

from pipeline_service_debug import SparseRecon
# from pipeline_jiang3_debug import SparseRecon

import json
import time

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
    run_gen_video: bool             是否进行视频生成
        gen_vid_fps: int                视频生成保存fps
    run_3dgs_iter: bool             是否稠密重建
'''

if __name__ == "__main__":
    json_str="""{
                "use_input_icc_drop":false,
                "use_resize_drop":true,
                
                "run_DA3":true,
                    "use_enhance_crop":false,
                    "use_view_extrapolation":true,

                "run_livephoto":false,
                
                "run_gen_video":true,
                    "view_num":81,
                    "gen_vid_infer_step":6,
                    "gen_vid_fps":30,
                "world_size":2,

                "run_3dgs_iter":false
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
    # work_dir = "/mnt/DataPart/PrivateSpace/wujinxiao/00_data/00_0829_synchaisi_image"
    work_dir = "/home/z00566689/dev/mnt/jiang_test/00_1110_multi_res_time_test/multi_image"
    run_list = os.listdir(work_dir)
    run_list = ["2160x3840"]

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
