import glob, os, torch
from depth_anything_3.api import DepthAnything3
from pathlib import Path
from PIL import Image
import numpy as np
from loguru import logger
import pycolmap
import gc
import trimesh
import time


def _build_pycolmap_intri(fidx, intrinsics, camera_type, extra_params=None):
    """
    Helper function to get camera parameters based on camera type.

    Args:
        fidx: Frame index
        intrinsics: Camera intrinsic parameters
        camera_type: Type of camera model
        extra_params: Additional parameters for certain camera types

    Returns:
        pycolmap_intri: NumPy array of camera parameters
    """
    if camera_type == "PINHOLE":
        pycolmap_intri = np.array(
            [intrinsics[fidx][0, 0], intrinsics[fidx][1, 1], intrinsics[fidx][0, 2], intrinsics[fidx][1, 2]]
        )
    elif camera_type == "SIMPLE_PINHOLE":
        focal = (intrinsics[fidx][0, 0] + intrinsics[fidx][1, 1]) / 2
        pycolmap_intri = np.array([focal, intrinsics[fidx][0, 2], intrinsics[fidx][1, 2]])
    elif camera_type == "SIMPLE_RADIAL":
        raise NotImplementedError("SIMPLE_RADIAL is not supported yet")
        focal = (intrinsics[fidx][0, 0] + intrinsics[fidx][1, 1]) / 2
        pycolmap_intri = np.array([focal, intrinsics[fidx][0, 2], intrinsics[fidx][1, 2], extra_params[fidx][0]])
    else:
        raise ValueError(f"Camera type {camera_type} is not supported yet")

    return pycolmap_intri

def interpolate_intrinsics(matrices, m):
    """
    将n个3×3矩阵线性插值为m个3×3矩阵
    
    参数:
        matrices: 形状为(n, 3, 3)的numpy数组，包含n个原始矩阵
        m: 插值后的矩阵数量
        
    返回:
        形状为(m, 3, 3)的numpy数组，包含m个插值后的矩阵
    """
    n = len(matrices)
    if n <= 1:
        raise ValueError("至少需要2个矩阵进行插值")
    
    # 生成原始序列和插值序列的位置参数（0到1之间）
    original_pos = np.linspace(0, 1, n)
    interp_pos = np.linspace(0, 1, m)
    
    # 初始化插值结果
    interpolated = np.zeros((m, 3, 3))
    
    # 对每个位置进行插值
    for i, pos in enumerate(interp_pos):
        # 找到当前位置所在的区间
        idx = np.searchsorted(original_pos, pos) - 1
        idx = np.clip(idx, 0, n-2)  # 确保索引在有效范围内
        
        # 获取区间的两个端点
        t0, t1 = original_pos[idx], original_pos[idx+1]
        m0, m1 = matrices[idx], matrices[idx+1]
        
        # 计算插值比例
        if t1 == t0:
            alpha = 0.0
        else:
            alpha = (pos - t0) / (t1 - t0)
        
        # 线性插值：m = (1-alpha)*m0 + alpha*m1
        interpolated[i] = (1 - alpha) * m0 + alpha * m1
    
    return interpolated

def interpolate_intrinsics_sparse_view(matrices, gen_index, view_num):
    """
    将n个3×3矩阵线性插值为m个3×3矩阵
    
    参数:
        matrices: 形状为(n, 3, 3)的numpy数组，包含n个原始矩阵
        m: 插值后的矩阵数量
        
    返回:
        形状为(m, 3, 3)的numpy数组，包含m个插值后的矩阵
    """
    # n = len(matrices)
    # if n <= 1:
    #     raise ValueError("至少需要2个矩阵进行插值")
    
    interpolated = np.zeros((len(gen_index) * view_num, 3, 3))

    for i in range(len(gen_index)):
        t0, t1 = gen_index[i][0][0], gen_index[i][-1][0]
        m0, m1 = matrices[t0], matrices[t1]

        for j in range(view_num):
            alpha = 1.0 * j / (view_num-1)
            interpolated[i*view_num + j] = (1 - alpha) * m0 + alpha * m1
    
    return interpolated



def batch_np_matrix_to_pycolmap_wo_track(
    points3d,
    points_xyf,
    points_rgb,
    extrinsics,
    intrinsics,
    image_ori_size,
    image_names,
    use_new_view=True,
    shared_camera=False,
    camera_type="PINHOLE",
):
    """
    Convert Batched NumPy Arrays to PyCOLMAP

    Different from batch_np_matrix_to_pycolmap, this function does not use tracks.

    It saves points3d to colmap reconstruction format only to serve as init for Gaussians or other nvs methods.

    Do NOT use this for BA.
    """
    # points3d: Px3
    # points_xyf: Px3, with x, y coordinates and frame indices
    # points_rgb: Px3, rgb colors
    # extrinsics: Nx3x4
    # intrinsics: Nx3x3
    # image_size: 2, assume all the frames have been padded to the same size
    # where N is the number of frames and P is the number of tracks

    N = len(extrinsics)
    P = len(points3d)

    # Reconstruction object, following the format of PyCOLMAP/COLMAP
    reconstruction = pycolmap.Reconstruction()

    for vidx in range(P):
        reconstruction.add_point3D(points3d[vidx], pycolmap.Track(), points_rgb[vidx] * 255)

    camera = None
    # frame idx
    for fidx in range(N):
        # set camera
        if camera is None or (not shared_camera):
            pycolmap_intri = _build_pycolmap_intri(fidx, intrinsics, camera_type)

            camera = pycolmap.Camera(
                model=camera_type, width=image_ori_size[0], height=image_ori_size[1], params=pycolmap_intri, camera_id=fidx + 1
            )

            # add camera
            reconstruction.add_camera(camera)

            # add rig (pycolmap>=3.10 需要 Rig/Frame 架构承载位姿)
            rig = pycolmap.Rig()
            rig.rig_id = camera.camera_id
            rig.add_ref_sensor(camera.sensor_id)
            reconstruction.add_rig(rig)

        # set image
        cam_from_world = pycolmap.Rigid3d(
            pycolmap.Rotation3d(extrinsics[fidx][:3, :3]), extrinsics[fidx][:3, 3]
        )  # Rot and Trans

        if use_new_view:
            if fidx < len(image_names):
                extent = image_names[fidx][-4:]
                img_name = image_names[fidx]
            else:
                extent = '.png'
                img_name = str(fidx - len(image_names)).zfill(3) + extent
        else: 
            img_name = image_names[fidx]

        # pycolmap>=3.10: Image 构造不再接受 id/cam_from_world，位姿通过 Frame(Rig) 设置
        image = pycolmap.Image(
            image_id=fidx + 1, name=img_name, camera_id=camera.camera_id
        )

        frame = pycolmap.Frame()
        frame.frame_id = fidx + 1
        frame.rig_id = camera.camera_id
        frame.add_data_id(image.data_id)
        frame.rig_from_world = cam_from_world
        reconstruction.add_frame(frame)

        image.frame_id = frame.frame_id
        reconstruction.add_image(image)

    return reconstruction


def create_pixel_coordinate_grid(num_frames, height, width):
    """
    Creates a grid of pixel coordinates and frame indices for all frames.
    Returns:
        tuple: A tuple containing:
            - points_xyf (numpy.ndarray): Array of shape (num_frames, height, width, 3)
                                            with x, y coordinates and frame indices
            - y_coords (numpy.ndarray): Array of y coordinates for all frames
            - x_coords (numpy.ndarray): Array of x coordinates for all frames
            - f_coords (numpy.ndarray): Array of frame indices for all frames
    """
    # Create coordinate grids for a single frame
    y_grid, x_grid = np.indices((height, width), dtype=np.float32)
    x_grid = x_grid[np.newaxis, :, :]
    y_grid = y_grid[np.newaxis, :, :]

    # Broadcast to all frames
    x_coords = np.broadcast_to(x_grid, (num_frames, height, width))
    y_coords = np.broadcast_to(y_grid, (num_frames, height, width))

    # Create frame indices and broadcast
    f_idx = np.arange(num_frames, dtype=np.float32)[:, np.newaxis, np.newaxis]
    f_coords = np.broadcast_to(f_idx, (num_frames, height, width))

    # Stack coordinates and frame indices
    points_xyf = np.stack((x_coords, y_coords, f_coords), axis=-1)

    return points_xyf


def save_sparserecon(prediction, render_result, images_paths, colmap_path, view_num, crop_ratio):
    processed_images = prediction.processed_images # np.array
    points_3d_conf = prediction.conf # np.array
    extrinsics = prediction.extrinsics # np.array
    intrinsics = prediction.intrinsics # np.array
    gaussians = prediction.gaussians # torch.tensor
    plys = gaussians.means # torch.tensor
    # opacitys = gaussians.opacities

    if prediction.is_metric:
        scale_factor = prediction.scale_factor
        if scale_factor is not None:
            extrinsics[:, :3, 3] /= scale_factor

    # target size: N * H * W * 3
    N, H, W, _ = processed_images.shape

    plys = plys.reshape(N, H, W, 3)
    # opacitys = opacitys.reshape(N, H, W, 1)

    # export colors
    image_names = []
    ori_rgbs = []
    for ip in images_paths:
        image_names.append(os.path.basename(ip))
        ori_rgbs.append(np.array(Image.open(ip)))
        
    ori_rgbs = np.array(ori_rgbs)

    ori_resolution = (ori_rgbs.shape[2], ori_rgbs.shape[1]) # (w,h)

    num_frames, height, width, _ = plys.shape

    default_intrinsic = intrinsics
    # convert the intrinsic to the original size
    scale = ori_resolution[0] / width
    ori_intrinsic = default_intrinsic * scale
    ori_intrinsic[:, 2, 2] = 1

    extrinsic_cp = extrinsics
    intrinsic_cp = np.zeros((num_frames, 3, 3))
    ori_intrinsic_cp = np.zeros((num_frames, 3, 3))
    for idx in range(intrinsic_cp.shape[0]):
        intrinsic_cp[idx] = default_intrinsic[idx]
        ori_intrinsic_cp[idx] = ori_intrinsic[idx]

    extr = extrinsic_cp
    intr = ori_intrinsic_cp
    if render_result is not None:
        extrinsic_render = render_result.extrinsics
        gen_index, enhance = render_result.gen_index
        intrinsic_render = interpolate_intrinsics_sparse_view(ori_intrinsic_cp, gen_index[:-1], view_num)
        for i in range(len(enhance)):
            if enhance[i] == 1:
                st = i * view_num
                ed = (i+1) * view_num
                for k in range(st, ed):
                    intrinsic_render[k][0][0] /= crop_ratio
                    intrinsic_render[k][1][1] /= crop_ratio
        extr = np.concatenate((extrinsic_cp, extrinsic_render), 0)
        intr = np.concatenate((ori_intrinsic_cp, intrinsic_render), 0)

    points_xyf = create_pixel_coordinate_grid(num_frames, height, width)

    # 4. process mask
    thres = np.percentile(points_3d_conf, 20)
    # print(f"filter conf_thres: {thres}")
    masks = points_3d_conf > thres 
    masks = masks

    points_3d = plys.cpu().numpy()[masks]
    points_xyf = points_xyf[masks]
    points_rgb = processed_images[masks]

    # print(points_3d.shape, points_rgb.shape, extr.shape, intr.shape)
    # print(points_3d[0], points_rgb[0], extr[0], intr[0])

    # logger.info("Converting to COLMAP format")
    reconstruction = batch_np_matrix_to_pycolmap_wo_track(points_3d, points_xyf, points_rgb, extr, intr,
                                                          (ori_resolution[0], ori_resolution[1]), image_names, use_new_view=render_result is not None,\
                                                           shared_camera=False, camera_type="PINHOLE")

    # logger.info(f"Saving reconstruction to {colmap_path}")
    sparse_reconstruction_dir = colmap_path
    os.makedirs(sparse_reconstruction_dir, exist_ok=True)
    reconstruction.write(sparse_reconstruction_dir)
    # reconstruction.write_text(sparse_reconstruction_dir)
    # # Save point cloud for fast visualization
    trimesh.PointCloud(points_3d, colors=points_rgb).export(os.path.join(colmap_path, "points3D.ply"))

def report_cuda(tag, device):
    device = torch.device(device)
    if device.type != "cuda":
        return

    torch.cuda.synchronize(device)
    print(
        f"[DA3][pid={os.getpid()}][{device}][{tag}] "
        f"allocated={torch.cuda.memory_allocated(device) / 1024**3:.3f} GiB, "
        f"reserved={torch.cuda.memory_reserved(device) / 1024**3:.3f} GiB",
        flush=True,
    )

class run_DA3(object):
    def __init__(self, model_path, device = torch.device("cuda")):
        self.model = DepthAnything3(model_name="da3nested-giant-large")
        # state_dicts = torch.load(model_path)
        state_dicts = torch.load(model_path, map_location="cpu")
        self.model.load_state_dict(state_dicts["model"], strict=False)
        del state_dicts
        self.model.to('cpu')
        self.device = device
    
    def infer(self, args):

        colmap_path = args.colmap_path
        img_path =  args.data_path
        gs_render_path = args.gs_render_path
        extra_view = args.extra_view

        images = sorted(str(f) for f in Path(img_path).iterdir() if f.suffix.lower() in {".png", ".jpg", ".jpeg"})

        start_time = time.time()
        self.model.to(device=self.device)
        end_time = time.time()
        logger.info(f"load DA3 to gpu. Time: {end_time - start_time} seconds")

        start_time = time.time()
        if gs_render_path is None:
            result = self.model.inference(
                images, export_dir=None, infer_gs=True, use_ray_pose=True, process_res=540,
            )
            # The local service and upstream DA3 differ in their return shape.
            prediction = result[0] if isinstance(result, tuple) else result
            render_result = None
            del result
        else:
            prediction, render_result = self.model.inference(
                images, export_dir = gs_render_path, infer_gs=True, export_format='gs_video', use_ray_pose=True, process_res=540,
                crop_ratio=args.crop_ratio, view_num=args.view_num, extra_view = extra_view)
        end_time = time.time()
        stage = "Model Forward Pass Done" if gs_render_path is None else "Forward and Rendering DONE"
        logger.info(f"{stage}. Time: {end_time - start_time} seconds")

        start_time = time.time()
        save_sparserecon(prediction, render_result, images, colmap_path, args.view_num, args.crop_ratio)
        end_time = time.time()
        logger.info(f"Saving colmap info DONE. Time: {end_time - start_time} seconds")

        start_time = time.time()
        self.model.to('cpu')
        end_time = time.time()
        logger.info(f"offload DA3 to cpu. Time: {end_time - start_time} seconds")

        prediction = None
        render_infos = None
        
        start_time = time.time()
        gc.collect()
        torch.cuda.empty_cache()
        report_cuda("F: after model infer", self.device)
        end_time = time.time()
        logger.info(f"Clear cache DONE. Time: {end_time - start_time} seconds")

        return render_result
