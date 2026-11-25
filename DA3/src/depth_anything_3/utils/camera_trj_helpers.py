# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from einops import einsum, rearrange, reduce
import math
import copy
from loguru import logger

try:
    from scipy.spatial.transform import Rotation as R
except ImportError:
    from depth_anything_3.utils.logger import logger

    logger.warn("Dependency 'scipy' not found. Required for interpolating camera trajectory.")

from depth_anything_3.utils.geometry import as_homogeneous



def get_raw_pitch_yaw_limitations(pitches, yaws):

    yaws = sorted(yaws)

    yaw_min, yaw_max = yaws[0], yaws[-1]

    for i in range(1, len(yaws)):
        if yaws[i-1]+360-yaws[i] < yaw_max-yaw_min:
            yaw_min = yaws[i]
            yaw_max = yaws[i-1]+360

    return [min(pitches), max(pitches), yaw_min, yaw_max]


def get_yaw_loop(loop, yaw, last_yaw):
    if yaw + loop < last_yaw - 180:
        loop += 360
    else:
        if last_yaw < yaw + loop - 180:
            loop -= 360
    return loop

def get_norm_x_z(coord, z):
    z_norm = safe_normalize(z)

    row1 = coord[1, :]
    row1_norm = safe_normalize(row1)

    cross_result = torch.cross(z_norm, row1_norm)
    x_norm = safe_normalize(cross_result)
    return x_norm, z_norm


def get_rot_by_xz(x_norm, z_norm):
    x_norm = safe_normalize(x_norm)
    z_norm = safe_normalize(z_norm)
    y_norm = torch.cross(z_norm, x_norm)
    x_norm = torch.cross(y_norm, z_norm)

    rot = torch.zeros((3, 3))
    rot[0, :3] = safe_normalize(x_norm)
    rot[1, :3] = safe_normalize(y_norm)
    rot[2, :3] = z_norm
    return rot


def estimate_down_vec(w2c_r_list):
    """
    Args:
        w2c_rot_mat: 相机位姿
    Returns:
        利用下向量平均值估计重力
    """
    right_vecs = w2c_r_list[:, 0, :]  # n, 3
    cov_mat = right_vecs.T @ right_vecs  # 向量中心视为0 不用去中心化
    U, sigma, V = torch.svd(cov_mat)
    pred_down_vec = V.mT[2]
    mean_down_vec = torch.mean(w2c_r_list[:, 1, :], dim=0)  # n, 3
    if torch.dot(pred_down_vec, mean_down_vec) < 0:
        pred_down_vec = - pred_down_vec
    return pred_down_vec


def estimate_rotation(w2c_rot_mat):
    pred_down_vec = estimate_down_vec(w2c_rot_mat)  # 利用和右向量垂直估计重力方向
    guess_right_vec = w2c_rot_mat[0, 0, :]  # first frame
    # 由于pred_down_vec和guess_right_vec夹角不一定为90度，可能导致模长不为1
    pred_lookat_vec = torch.linalg.cross(guess_right_vec, pred_down_vec)
    pred_lookat_vec = F.normalize(pred_lookat_vec, dim=0)
    pred_right_vec = torch.linalg.cross(pred_down_vec, pred_lookat_vec)
    pred_right_vec = F.normalize(pred_right_vec, dim=0)
    pred_rot = torch.stack([pred_right_vec, pred_down_vec, pred_lookat_vec], dim=0)
    return pred_rot.T

def compute_sight_center(c2w_t_list, sight_dir_list):
    device = c2w_t_list.device
    num = len(c2w_t_list)
    A1 = torch.eye(3).to(device).repeat(num, 1)
    A2 = torch.stack([torch.diag(sight_dir_list[:, 0]).to(device),
                      torch.diag(sight_dir_list[:, 1]).to(device),
                      torch.diag(sight_dir_list[:, 2]).to(device)], dim=1)
    A = torch.cat([A1, -A2.reshape(-1, num)], dim=1).double()  # n*3, n+3
    b = c2w_t_list.flatten().double()
    m = torch.inverse(A.mT @ A) @ (A.mT @ b)
    center = m[:3].float().contiguous()
    return center


def safe_normalize(v):
    norm = torch.norm(v)
    return v / (norm + 1e-10) if norm > 1e-10 else v * 0

def calculate_camera_angle(cam_loc, is_reverse=False):
    center = torch.tensor([0.0, 0.0, 0.0])
    direction_vector = (center - cam_loc)
    normalized_vector = safe_normalize(direction_vector)
    vx = -normalized_vector[0]
    vy = normalized_vector[1]
    vz = normalized_vector[2]
    if is_reverse:
        yaw = torch.atan2(vx, vz)
    else:
        yaw = torch.atan2(-vx, -vz)
    pitch = torch.arcsin(-vy)
    yaw_deg = torch.rad2deg(yaw)
    pitch_deg = torch.rad2deg(pitch)

    return pitch_deg, yaw_deg




def get_cam_radius_tensor(views):
    cam_pos = torch.stack(views)
    pos_mean = torch.mean(cam_pos, dim=0)
    dist_mean = torch.mean(torch.norm(cam_pos, dim=1), dim=0)
    return dist_mean, pos_mean, cam_pos

def get_loc_by_pitch_yaw(pitch, yaw, radius):
    vy = math.sin(pitch * math.pi / 180) * radius
    theta = math.cos(pitch * math.pi / 180) * radius
    vx = math.sin(yaw * math.pi / 180) * theta
    vz = math.cos(yaw * math.pi / 180) * theta
    loc = torch.tensor([-vx, vy, vz])
    return loc

from itertools import combinations
def solve_max_union(intervals_data, M):
    """
    intervals_data: 列表，每个元素为 [左, 右, 编号1, 编号2]
    M: 最多选择的区间数量
    """
    n = len(intervals_data)
    # 如果实际区间数少于 M，则全部选择
    actual_m = min(n, M)
    
    best_coverage = -1
    best_combination = []

    # 1. 离散化：将所有坐标点提取出来，切分成不重叠的小段
    # 这样我们可以通过计算这些“小段”是否被覆盖来计算并集长度
    coords = sorted(list(set([x[0] for x in intervals_data] + [x[1] for x in intervals_data])))
    # segments 存储为 (左, 右, 长度)
    segments = []
    for i in range(len(coords) - 1):
        segments.append((coords[i], coords[i+1], coords[i+1] - coords[i]))

    # 2. 预处理：每个原始区间覆盖了哪些小段索引
    interval_masks = []
    for item in intervals_data:
        l, r = item[0], item[1]
        mask = set()
        for idx, (sl, sr, length) in enumerate(segments):
            if l <= sl and sr <= r:
                mask.add(idx)
        interval_masks.append(mask)

    # 3. 组合搜索：从 n 个区间里选 m 个
    # 因为 M 很小，直接遍历组合是最稳妥的
    for m in range(1, actual_m + 1):
        for indices in combinations(range(n), m):
            # 计算当前组合覆盖的所有小段索引的并集
            current_union_indices = set()
            for idx in indices:
                current_union_indices.update(interval_masks[idx])
            
            # 计算总长度
            current_total_len = sum(segments[i][2] for i in current_union_indices)
            
            # 如果当前组合更好，更新记录
            if current_total_len > best_coverage:
                best_coverage = current_total_len
                # 记录对应的 [编号1, 编号2]
                best_combination = [ [intervals_data[i][2], intervals_data[i][3]] for i in indices ]

    return best_combination

class Camera_grid(object):
    def __init__(self, cameras, enhance=False, extra=[False, False], center=None, pred_rot=None):
        self.cameras = torch.tensor(cameras).float()  
        self.load_info()
        if len(self.cameras) == 1:
            print("Only single image, use simple mode")
            return

        self.pose_adjust(pred_rot, center)
        self.get_yaw_pitch()
        self.radius = self.get_radius() * 1.0
        if type(extra)==bool:
            self.extra = extra
            self.is_livephoto = False
        else:
            self.extra = extra[0]
            self.is_livephoto = extra[1]

        self.enhance = enhance

        self.pitch_min, self.pitch_max, self.yaw_min, self.yaw_max, self.circle = self.get_limitations()
        if self.extra and self.pitch_max-self.pitch_min < 45 and self.is_livephoto:
            if self.pitch_min > -15:
                self.pitch_min = max(-15, self.pitch_min-5)
            if self.pitch_max < 90:
                self.pitch_max = min(90, self.pitch_max+5)
            print("using extrapolation, extend pitch to ", self.pitch_min, self.pitch_max)
        print("yaw range: ", self.yaw_min, self.yaw_max, self.yaws)
        print("pitch range:", self.pitch_min, self.pitch_max, self.pitches)
        print("circle:", self.circle)

    def load_info(self):
        self.w2c_R = []
        self.w2c_T = []
        for i in range(len(self.cameras)):
            self.w2c_R.append(self.cameras[i][:3, :3])
            self.w2c_T.append(self.cameras[i][:3, 3])

    def c2w_R(self, w2c_R, w2c_T):
        return w2c_R.T

    def c2w_T(self, w2c_R, w2c_T):
        return -w2c_R.T @ w2c_T

    def pose_adjust(self, pred_rot=None, center=None):
        if center == None:
            w2c_r_list = []
            c2w_t_list = []
            sight_dir_list = []
            for i in range(len(self.cameras)):
                w2c_r_list.append(self.w2c_R[i])
                c2w_t_list.append(self.c2w_T(self.w2c_R[i], self.w2c_T[i]))
                sight_dir = self.w2c_R[i][2]
                sight_dir_list.append(sight_dir)

            w2c_r_list = torch.stack(w2c_r_list)
            c2w_t_list = torch.stack(c2w_t_list)
            sight_dir_list = torch.stack(sight_dir_list)
            center = compute_sight_center(c2w_t_list, sight_dir_list)
            
            pred_rot = estimate_rotation(w2c_r_list)
            pred_rot = pred_rot @ torch.diag(pred_rot.new_tensor([1, -1, -1]))
            
            self.extent = torch.median(torch.norm(c2w_t_list - center, dim=1)).item()

        scale = 1.0

        self.new_w2c_R = []
        self.new_w2c_T = []
        self.new_cameras = []
        self.center = center
        self.pred_rot = pred_rot

        print("pred_rot!!!", pred_rot)

        for i in range(len(self.cameras)):
            self.new_w2c_T.append(self.w2c_T[i] + self.w2c_R[i] @ center)
            self.new_w2c_R.append(self.w2c_R[i] @ pred_rot)
            self.new_cameras.append(torch.cat([self.new_w2c_R[-1], self.new_w2c_T[-1].unsqueeze(1)], 1))

            # tmp_w2c_R = self.new_w2c_R[i] @ self.pred_rot.T
            # tmp_w2c_T = self.new_w2c_T[i] - tmp_w2c_R @ self.center

            # print(i, torch.cat([tmp_w2c_R, tmp_w2c_T.unsqueeze(1)], 1))
            # print(i, self.cameras[i])

    def get_yaw_pitch(self):
        self.yaws = []
        self.pitches = []

        for i in range(len(self.cameras)):
            pitch, yaw = calculate_camera_angle(self.c2w_T(self.new_w2c_R[i], self.new_w2c_T[i]))
            self.yaws.append(yaw)
            self.pitches.append(pitch)

        print("yaw list:", self.yaws)
        print("pitch list:", self.pitches)

    def get_radius(self):
        radius, _, _ = get_cam_radius_tensor([self.c2w_T(self.new_w2c_R[i], self.new_w2c_T[i]) for i in range(len(self.cameras))])
        return radius

    def order_interpolate(self, M, idx, circle):
        # M * 2 trajectory

        ref = []
        for i in range(len(idx)):
            ref.append([self.yaws[idx[i]], idx[i]])
        ref.sort()
        if circle == 1:
            ref.append(copy.deepcopy(ref[0]))
            ref[-1][0] += 360
        
        delta_min, delta_max = 0, ref[-1][0]-ref[0][0]

        final_trajectories = []
        final_index = []

        while (delta_max - delta_min > 1e-2) or len(final_trajectories) == 0:
            delta_yaw = (delta_max+delta_min) / 2

            trajectories = []
            trajectory = []
            index = []
            for yaw, idx in ref:
                trajectory.append([yaw, idx])
                if trajectory[-1][0] - trajectory[0][0] > delta_yaw:
                    trajectories.append(trajectory)
                    trajectory = [trajectory[-1]]
                    index.append(idx)
            if len(trajectory) > 1:
                trajectories.append(trajectory)
            if len(trajectories) <= M:
                final_trajectories = trajectories
                final_index = index
                delta_max = delta_yaw
            else:
                delta_min = delta_yaw

        return final_trajectories, final_index

    def separate_match(self, idx, circle, yaw_th=15, pitch_th=2.5):
        # 基本原理：分上、中、下三条路线串联
        # yaws_th: 水平阈值，水平相邻两个视角优先在该范围内检索
        # pitch_th: 垂直阈值，在考虑候选时，选择大于pitch_max-pitch_th且yaw最大的

        ref = []
        for i in range(len(idx)):
            ref.append([self.yaws[idx[i]], idx[i]])
        ref.sort()
        if circle == 1:
            ref.append(copy.deepcopy(ref[0]))
            ref[-1][0] += 360
        
        
        results = dict()
        for cate in ['up', 'down', 'direct']:
            current = 0
            trajectories = []
            while True:
                candidate = []
                # 寻找所有yaw[current] + yaw_th范围内的选择
                for i in range(current+1, len(ref)):
                    if ref[i][0]-ref[current][0] <= yaw_th:
                        candidate.append(i)
                    else:
                        break
                if len(candidate) == 0 and current != len(ref)-1:
                    candidate.append(current+1)
                if len(candidate) == 0:
                    break
                # 从candidate里找到pitch范围
                if cate == 'up':
                    pitch_std = max([self.pitches[ref[x][1]] for x in candidate])
                elif cate == 'down':
                    pitch_std = min([self.pitches[ref[x][1]] for x in candidate])
                else:
                    pitch_std = self.pitches[ref[current][1]]
                    pitch_th = min([abs(pitch_std-self.pitches[ref[x][1]]) for x in candidate]) + 0.1
                
                # 选择所有满足pitch范围的轨迹
                candidate = [x for x in candidate if abs(pitch_std-self.pitches[ref[x][1]]) <= pitch_th]
                # 最后选择yaws最大的
                trajectories.append([ref[current], ref[max(candidate)]])
                current = max(candidate)
            results[cate] = trajectories
        return results

    def cross_match(self, up_index, down_index):
        if len(up_index) > len(down_index):
            up_index, down_index = down_index, up_index

        trajectories = []
        for idx in down_index:
            min_idx = up_index[0]
            for u_idx in up_index:
                if abs(self.yaws[u_idx]-self.yaws[idx]) < abs(self.yaws[min_idx]-self.yaws[idx]):
                    min_idx = u_idx
            trajectories.append([[self.yaws[min_idx], min_idx], [self.yaws[idx], idx]])
        
        return trajectories

    def cross_match2(self, up_index, down_index, M):
        # print(up_index, down_index)

        i = 0
        j = 0
        intervals = []
        for idx in down_index:
            min_idx = up_index[0]
            for u_idx in up_index:
                if abs(self.yaws[u_idx]-self.yaws[idx]) < abs(self.yaws[min_idx]-self.yaws[idx]):
                    min_idx = u_idx
            intervals.append([self.yaws[idx], self.yaws[min_idx], idx, min_idx])
        for idx in up_index:
            min_idx = down_index[0]
            for u_idx in down_index:
                if abs(self.yaws[u_idx]-self.yaws[idx]) < abs(self.yaws[min_idx]-self.yaws[idx]):
                    min_idx = u_idx
            intervals.append([self.yaws[idx], self.yaws[min_idx], idx, min_idx])

        results = solve_max_union(intervals, M)

        trajectories = []
        for p, q in results:
            trajectories.append([[self.yaws[p], p], [self.yaws[q], q]])
        
        # print(trajectories)

        return trajectories

    def get_yaw_pitch_sphere_view(self, pitch, yaw):
        target = torch.zeros(3)
        loc = get_loc_by_pitch_yaw(pitch, yaw, self.radius)
        x_norm, z_norm = get_norm_x_z(torch.eye(3), target - loc)
        new_w2c_R = get_rot_by_xz(x_norm, z_norm)
        new_w2c_T = torch.linalg.pinv(-new_w2c_R.T) @ loc

        # w2c_R = new_w2c_R @ self.pred_rot.T
        # w2c_T = new_w2c_T - w2c_R @ self.center

        return new_w2c_R, new_w2c_T

    def sequence_gen(self, trajectories, view_num, mode='up'):
        # mode in ['up', 'down', 'direct']
        index = []
        results = np.zeros((len(trajectories) * view_num, 3, 4))
        lines = []

        for i in range(len(trajectories)):
            # print([x[1] for x in trajectories[i]])

            line = []
            
            idx = [x[1] for x in trajectories[i]]
            st = idx[0]
            ed = idx[-1]

            extr = [self.new_cameras[st]]
            extr_idx = [st]
            line.append([self.yaws[st], self.pitches[st]])

            '''
            # extra
            if self.circle == 0 and self.extra == 1 and mode in ['up', 'down']:
                if self.yaws[st] in [self.yaw_min, self.yaw_max]:
                    delta = -15 if self.yaws[st] == self.yaw_min else 15

                    w2c_R, w2c_T = self.get_yaw_pitch_sphere_view(self.pitches[st], self.yaws[st]+delta)
                    extr.append(torch.cat([w2c_R, w2c_T.unsqueeze(1)], 1))
                    extr_idx.append(-1)
                    line.append([self.yaws[st]+delta, self.pitches[st]])

                    if mode == 'up':
                        w2c_R, w2c_T = self.get_yaw_pitch_sphere_view(self.pitch_max, self.yaws[st]+delta)
                        line.append([self.yaws[st]+delta, self.pitch_max])
                    else:
                        w2c_R, w2c_T = self.get_yaw_pitch_sphere_view(self.pitch_min, self.yaws[st]+delta)
                        line.append([self.yaws[st]+delta, self.pitch_min])
                    extr.append(torch.cat([w2c_R, w2c_T.unsqueeze(1)], 1))
                    extr_idx.append(-1)
            '''

            
            # go down
            if mode == 'up' and self.pitch_max-self.pitches[st] >= 0:
                w2c_R, w2c_T = self.get_yaw_pitch_sphere_view(self.pitch_max, self.yaws[st])
                extr.append(torch.cat([w2c_R, w2c_T.unsqueeze(1)], 1))
                extr_idx.append(-1)
                line.append([self.yaws[st], self.pitch_max])
            elif mode == 'down' and self.pitches[st]-self.pitch_min >= 0:
                w2c_R, w2c_T = self.get_yaw_pitch_sphere_view(self.pitch_min, self.yaws[st])
                extr.append(torch.cat([w2c_R, w2c_T.unsqueeze(1)], 1))
                extr_idx.append(-1)
                line.append([self.yaws[st], self.pitch_min])
            else:
                pass

            if self.yaws[ed]-self.yaws[st] > 180:
                w2c_R, w2c_T = self.get_yaw_pitch_sphere_view(line[-1][1], (self.yaws[ed]+self.yaws[st]) / 2)
                extr.append(torch.cat([w2c_R, w2c_T.unsqueeze(1)], 1))
                extr_idx.append(-1)
                line.append([(self.yaws[ed]+self.yaws[st]) / 2, line[-1][1]])

            # for j in idx[1:-1]:
            #     extr.append(self.new_cameras[j])
            #     extr_idx.append(j)
            #     line.append([self.yaws[j], self.pitches[j]])

            if mode == 'up' and self.pitch_max-self.pitches[ed] >= 0:
                w2c_R, w2c_T = self.get_yaw_pitch_sphere_view(self.pitch_max, self.yaws[ed])
                extr.append(torch.cat([w2c_R, w2c_T.unsqueeze(1)], 1))
                extr_idx.append(-1)
                line.append([self.yaws[ed], self.pitch_max])
            elif mode == 'down' and self.pitches[ed]-self.pitch_min >= 0:
                w2c_R, w2c_T = self.get_yaw_pitch_sphere_view(self.pitch_min, self.yaws[ed])
                extr.append(torch.cat([w2c_R, w2c_T.unsqueeze(1)], 1))
                extr_idx.append(-1)
                line.append([self.yaws[ed], self.pitch_min])
            # elif mode == 'direct':
            #     w2c_R, w2c_T = self.get_yaw_pitch_sphere_view(self.pitches[st], self.yaws[ed])
            #     extr.append(torch.cat([w2c_R, w2c_T.unsqueeze(1)], 1))
            #     extr_idx.append(-1)
            #     line.append([self.yaws[ed], self.pitches[st]])
            else:
                pass
            '''
            if self.circle == 0 and self.extra == 1 and mode in ['up', 'down']:
                if self.yaws[ed] in [self.yaw_min, self.yaw_max]:
                    delta = -15 if self.yaws[ed] == self.yaw_min else 15

                    if mode == 'up':
                        w2c_R, w2c_T = self.get_yaw_pitch_sphere_view(self.pitch_max, self.yaws[ed]+delta)
                        line.append([self.yaws[ed]+delta, self.pitch_max])
                    else:
                        w2c_R, w2c_T = self.get_yaw_pitch_sphere_view(self.pitch_min, self.yaws[ed]+delta)
                        line.append([self.yaws[ed]+delta, self.pitch_min])
                    extr.append(torch.cat([w2c_R, w2c_T.unsqueeze(1)], 1))
                    extr_idx.append(-1)


                    w2c_R, w2c_T = self.get_yaw_pitch_sphere_view(self.pitches[ed], self.yaws[ed]+delta)
                    extr.append(torch.cat([w2c_R, w2c_T.unsqueeze(1)], 1))
                    extr_idx.append(-1)
                    line.append([self.yaws[ed]+delta, self.pitches[ed]])
            '''

            extr.append(self.new_cameras[ed])
            extr_idx.append(ed)
            line.append([self.yaws[ed], self.pitches[ed]])

            # if len(extr) <= 5 and mode == 'direct':
            # results[i*view_num:(i+1)*view_num], pos_idx = interpolate_extrinsics_sparseview_Bezier(torch.stack(extr).numpy(), extr_idx, view_num)
            # else:
            results[i*view_num:(i+1)*view_num], pos_idx = interpolate_extrinsics_sparseview(torch.stack(extr).numpy(), extr_idx, view_num)
            index.append(pos_idx)
            lines.append(line)

        return results, index, lines

    def get_trajectory_sequence(self, view_num, video_only=False):
        results = []
        indexes = []
        enhance = []

        if video_only:
            trajectories, _ = self.order_interpolate(5, np.arange(len(self.cameras)), 0)

            result, index, line = self.sequence_gen(trajectories, view_num, 'video_only')
            results.append(result)
            indexes += index
            enhance += [0] * len(index)

            results = torch.tensor(np.concatenate(results, 0)).float()

            for i in range(len(results)):
                new_w2c_R = results[i][:3, :3]
                new_w2c_T = results[i][:3, 3]

                w2c_R = new_w2c_R @ self.pred_rot.T
                w2c_T = new_w2c_T - w2c_R @ self.center

                results[i][:3, :3] = w2c_R
                results[i][:3, 3] = w2c_T
            
            indexes.append(0)

            return results.numpy(), (indexes, enhance)

        trajectories = self.separate_match(np.arange(len(self.cameras)), self.circle)

        print(trajectories)
        print("??????")

        # if self.is_livephoto:
        #     result, index, line = self.sequence_gen(trajectories['direct'], view_num, 'direct')
        #     results.append(result)
        #     indexes += index
        #     enhance += [0] * len(index)
        
        # else:
        if self.extra:
            result, index, line = self.sequence_gen(trajectories['up'], view_num, 'up')
            results.append(result)
            indexes += index
            enhance += [0] * len(index)

            result, index, line = self.sequence_gen(trajectories['down'], view_num, 'down')
            results.append(result)
            indexes += index
            enhance += [0] * len(index)
        else:
            result, index, line = self.sequence_gen(trajectories['direct'], view_num, 'direct')
            results.append(result)
            indexes += index
            enhance += [0] * len(index)

        if self.enhance and not self.is_livephoto:
            result, index, line = self.sequence_gen(trajectories['direct'], view_num, 'direct')
            results.append(result)
            indexes += index
            enhance += [1] * len(index)

        '''
        self.extra = True
        if self.pitch_min > -30:
            self.pitch_min = max(-30, self.pitch_min-30)
        if self.pitch_max < 90:
            self.pitch_max = min(90, self.pitch_max+30)

        result, index, line = self.sequence_gen(trajectories['direct'], view_num, 'up')
        results.append(result)
        indexes += index
        enhance += [0] * len(index)

        result, index, line = self.sequence_gen(trajectories['direct'], view_num, 'down')
        results.append(result)
        indexes += index
        enhance += [0] * len(index)
        '''


        '''
        if self.pitch_max - self.pitch_min < 25:
        # if True:
            print("pitch delta: ", self.pitch_max - self.pitch_min, "use camera mode 1")
            trajectories, _ = self.order_interpolate(5, np.arange(len(self.cameras)), self.circle)

            if self.pitch_max - self.pitch_min > 5 or self.extra:
                result, index, line = self.sequence_gen(trajectories, view_num, 'up')
                results.append(result)
                indexes += index
                enhance += [0] * len(index)

                result, index, line = self.sequence_gen(trajectories, view_num, 'down')
                results.append(result)
                indexes += index
                enhance += [0] * len(index)
            else:
                result, index, line = self.sequence_gen(trajectories, view_num, 'direct')
                results.append(result)
                indexes += index
                enhance += [0] * len(index)
            
            if self.enhance:
                result, index, line = self.sequence_gen(trajectories, view_num, 'direct')
                results.append(result)
                indexes += index
                enhance += [1] * len(index)

        else:
            print("pitch delta: ", self.pitch_max - self.pitch_min, "use camera mode 2")
            up = []
            down = []
            st, ed = 0, 0
            for i in range(len(self.cameras)):
                if self.yaws[i] == self.yaw_min:
                    st = i
                elif self.yaws[i] == self.yaw_max:
                    ed = i
                elif self.pitches[i] > (self.pitch_max+self.pitch_min) / 2:
                    up.append(i)
                else:
                    down.append(i)
            total_M = 0
            # up
            trajectories, up_index = self.order_interpolate(3, np.array([st, ed] + up), 0)
            result, index, line = self.sequence_gen(trajectories, view_num, 'up')
            results.append(result)
            indexes += index
            total_M += len(trajectories)
            enhance += [0] * len(index)
            # down
            trajectories, down_index = self.order_interpolate(5, np.array([st, ed] + down), self.circle)
            result, index, line = self.sequence_gen(trajectories, view_num, 'down')
            results.append(result)
            indexes += index
            # cross TODO
            total_M += len(trajectories)
            enhance += [0] * len(index)

            sample_num = min(min(len(up), len(down)), 10-total_M)

            if sample_num != 0:
                trajectories = self.cross_match2(up, down, sample_num)
                result, index, line = self.sequence_gen(trajectories, view_num, 'direct')
                results.append(result)
                indexes += index
                # param.lines += line
                enhance += [1] * len(index)
        '''

        results = torch.tensor(np.concatenate(results, 0)).float()

        # TEMP YAW DEBUG: both pose sets use the planning center and axes.
        yaw_mid = (self.yaw_min + self.yaw_max) / 2
        for label, poses in [('input', torch.stack(self.new_cameras)),
                             ('planned', results)]:
            poses_np = poses.detach().cpu().numpy()
            centers = -np.einsum('nji,nj->ni', poses_np[:, :3, :3], poses_np[:, :3, 3])
            yaws = np.degrees(np.arctan2(-centers[:, 0], centers[:, 2]))
            yaws = yaw_mid + (yaws - yaw_mid + 180) % 360 - 180
            print(f'[YAW DEBUG] {label}: count={len(yaws)}, '
                  f'min={yaws.min():.6f}, max={yaws.max():.6f}, '
                  f'span={np.ptp(yaws):.6f} deg', flush=True)
        # END TEMP YAW DEBUG

        for i in range(len(results)):
            new_w2c_R = results[i][:3, :3]
            new_w2c_T = results[i][:3, 3]

            w2c_R = new_w2c_R @ self.pred_rot.T
            w2c_T = new_w2c_T - w2c_R @ self.center

            results[i][:3, :3] = w2c_R
            results[i][:3, 3] = w2c_T

        indexes.append(self.circle)

        # exit()

        return results.numpy(), (indexes, enhance)

    def check_position(self):
        for i in range(len(self.cameras)):
            print(i, get_loc_by_pitch_yaw(self.pitches[i], self.yaws[i], self.radius))
            print(i, self.c2w_T(self.new_w2c_R[i], self.new_w2c_T[i]))

    def get_limitations(self):
        limitations = get_raw_pitch_yaw_limitations(self.pitches, self.yaws)
        pitch_min, pitch_max, yaw_min, yaw_max = limitations
        pitch_min = float(pitch_min)
        pitch_max = float(pitch_max)
        yaw_min = float(yaw_min)
        yaw_max = float(yaw_max)

        for i in range(len(self.cameras)):
            if self.yaws[i] < yaw_min:
                self.yaws[i] += 360
            if self.yaws[i] > yaw_max:
                self.yaws[i] -= 360

        circle = 0
        if yaw_max - yaw_min > 255:
            circle = 1
        # print(self.yaws)
        logger.info(f"yaw_max {yaw_max} yaw_min {yaw_min} circle {circle}")
        # elif yaw_max - yaw_min > 20:
        #     yaw_max = max(yaw_max - 10, yaw_init)
        #     yaw_min = min(yaw_min + 10, yaw_init)
        # if pitch_max - pitch_min > 10:
        #     pitch_max = max(pitch_max - 5, pitch_init)
        #     pitch_min = min(pitch_min + 5, pitch_init)

        return pitch_min, pitch_max, yaw_min, yaw_max, circle


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
        # raise ValueError("至少需要2个矩阵进行插值")
    
    print("interp: ", gen_index)

    interpolated = np.zeros((len(gen_index) * view_num, 3, 3))

    for i in range(len(gen_index)):
        t0, t1 = gen_index[i][0][0], gen_index[i][-1][0]
        m0, m1 = matrices[t0], matrices[t1]

        # print(i*view_num, (i+1)*view_num, t0, t1)

        for j in range(view_num):
            alpha = 1.0 * j / (view_num-1)
            interpolated[i*view_num + j] = (1 - alpha) * m0 + alpha * m1
    
    return interpolated


def interpolate_intrinsics_sparse_view_bk(matrices, m):
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


def render_sparse_view_gen(extrinsic_cp, intrinsic_cp, view_num, crop_ratio=1, extra_view=False):
    cam_grid = Camera_grid(extrinsic_cp, enhance=crop_ratio!=1, extra=extra_view)
    # extrinsic_render, gen_index = cam_grid.get_trajectory_sequence(args.view_num)
    '''
    for i in range(3):
        extrinsic_render, gen_index = cam_grid.get_trajectory_sequence(view_num)
        # extrinsic_render = extrinsic_render.astype(np.float32)
        cam_grid = Camera_grid(extrinsic_render)
        cam_grid = Camera_grid(extrinsic_cp, cam_grid.center, cam_grid.pred_rot)
    '''
    extrinsic_render, gen_index = cam_grid.get_trajectory_sequence(view_num)
    gen_index, enhance = gen_index
    intrinsic_render = interpolate_intrinsics_sparse_view(intrinsic_cp, gen_index[:-1], view_num)

    for i in range(len(enhance)):
        if enhance[i] == 1:
            st = i * view_num
            ed = (i+1) * view_num
            for k in range(st, ed):
                intrinsic_render[k][0][0] /= crop_ratio
                intrinsic_render[k][1][1] /= crop_ratio
    # intrinsic_render = interpolate_intrinsics_sparse_view(intrinsic_cp, len(extrinsic_render))

    extrinsic_render = torch.tensor(extrinsic_render).unsqueeze(0)
    intrinsic_render = torch.tensor(intrinsic_render).unsqueeze(0)
    
    return extrinsic_render, intrinsic_render, (gen_index, enhance)


@torch.no_grad()
def render_stabilization_path(poses, k_size=45):
    """Rendering stabilized camera path.
    poses: [batch, 4, 4] or [batch, 3, 4],
    return:
        smooth path: [batch 4 4]"""
    num_frames = poses.shape[0]
    device = poses.device
    dtype = poses.dtype

    # Early exit for trivial cases
    if num_frames <= 1:
        return as_homogeneous(poses)

    # Make k_size safe: positive odd and not larger than num_frames
    # 1) Ensure odd
    if k_size < 1:
        k_size = 1
    if k_size % 2 == 0:
        k_size += 1
    # 2) Cap to num_frames (keep odd)
    max_odd = num_frames if (num_frames % 2 == 1) else (num_frames - 1)
    if max_odd < 1:
        max_odd = 1  # covers num_frames == 0 theoretically
    k_size = min(k_size, max_odd)
    # 3) enforce a minimum of 3 when possible (for better smoothing)
    if num_frames >= 3 and k_size < 3:
        k_size = 3

    input_poses = []
    for i in range(num_frames):
        input_poses.append(
            torch.cat([poses[i, :3, 0:1], poses[i, :3, 1:2], poses[i, :3, 3:4]], dim=-1)
        )
    input_poses = torch.stack(input_poses)  # (num_frames, 3, 3)

    # Prepare Gaussian kernel
    gaussian_kernel = cv2.getGaussianKernel(ksize=k_size, sigma=-1).astype(np.float32).squeeze()
    gaussian_kernel = torch.tensor(gaussian_kernel, dtype=dtype, device=device).view(1, 1, -1)
    pad = k_size // 2

    output_vectors = []
    for idx in range(3):  # For r1, r2, t
        vec = (
            input_poses[:, :, idx].T.unsqueeze(0).unsqueeze(0)
        )  # (1, 1, 3, num_frames) -> (1, 1, 3, num_frames)
        # But actually, we want (batch=3, channel=1, width=num_frames)
        # So:
        vec = input_poses[:, :, idx].T.unsqueeze(1)  # (3, 1, num_frames)
        vec_padded = F.pad(vec, (pad, pad), mode="reflect")
        filtered = F.conv1d(vec_padded, gaussian_kernel)
        output_vectors.append(filtered.squeeze(1).T)  # (num_frames, 3)

    output_r1, output_r2, output_t = output_vectors  # Each is (num_frames, 3)

    # Normalize r1 and r2
    output_r1 = output_r1 / output_r1.norm(dim=-1, keepdim=True)
    output_r2 = output_r2 / output_r2.norm(dim=-1, keepdim=True)

    output_poses = []
    for i in range(num_frames):
        output_r3 = torch.linalg.cross(output_r1[i], output_r2[i])
        render_pose = torch.cat(
            [
                output_r1[i].unsqueeze(-1),
                output_r2[i].unsqueeze(-1),
                output_r3.unsqueeze(-1),
                output_t[i].unsqueeze(-1),
            ],
            dim=-1,
        )
        output_poses.append(render_pose[:3, :])
    output_poses = as_homogeneous(torch.stack(output_poses, dim=0))

    return output_poses


@torch.no_grad()
def render_wander_path(
    cam2world: torch.Tensor,
    intrinsic: torch.Tensor,
    h: int,
    w: int,
    num_frames: int = 120,
    max_disp: float = 48.0,
):
    device, dtype = cam2world.device, cam2world.dtype
    fx = intrinsic[0, 0] * w
    r = max_disp / fx
    th = torch.linspace(0, 2.0 * torch.pi, steps=num_frames, device=device, dtype=dtype)
    x = r * torch.sin(th)
    yz = r * torch.cos(th) / 3.0
    T = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).repeat(num_frames, 1, 1)
    T[:, :3, 3] = torch.stack([x, yz, yz], dim=-1) * -1.0
    c2ws = cam2world.unsqueeze(0) @ T
    # Start at reference pose and end back at reference pose
    c2ws = torch.cat([cam2world.unsqueeze(0), c2ws, cam2world.unsqueeze(0)], dim=0)
    Ks = intrinsic.unsqueeze(0).repeat(c2ws.shape[0], 1, 1)
    return c2ws, Ks


@torch.no_grad()
def render_dolly_zoom_path(
    cam2world: torch.Tensor,
    intrinsic: torch.Tensor,
    h: int,
    w: int,
    num_frames: int = 120,
    max_disp: float = 0.1,
    D_focus: float = 10.0,
):
    device, dtype = cam2world.device, cam2world.dtype
    fx0, fy0 = intrinsic[0, 0] * w, intrinsic[1, 1] * h
    t = torch.linspace(0.0, 2.0, steps=num_frames, device=device, dtype=dtype)
    z = 0.5 * (1.0 - torch.cos(torch.pi * t)) * max_disp
    T = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).repeat(num_frames, 1, 1)
    T[:, 2, 3] = -z
    c2ws = cam2world.unsqueeze(0) @ T
    Df = torch.as_tensor(D_focus, device=device, dtype=dtype)
    scale = (Df / (Df + z)).clamp(min=1e-6)
    Ks = intrinsic.unsqueeze(0).repeat(num_frames, 1, 1)
    Ks[:, 0, 0] = (fx0 * scale) / w
    Ks[:, 1, 1] = (fy0 * scale) / h
    return c2ws, Ks


@torch.no_grad()
def interpolate_intrinsics(
    initial: torch.Tensor,  # "*#batch 3 3"
    final: torch.Tensor,  # "*#batch 3 3"
    t: torch.Tensor,  # " time_step"
) -> torch.Tensor:  # "*batch time_step 3 3"
    initial = rearrange(initial, "... i j -> ... () i j")
    final = rearrange(final, "... i j -> ... () i j")
    t = rearrange(t, "t -> t () ()")
    return initial + (final - initial) * t


def intersect_rays(
    a_origins: torch.Tensor,  # "*#batch dim"
    a_directions: torch.Tensor,  # "*#batch dim"
    b_origins: torch.Tensor,  # "*#batch dim"
    b_directions: torch.Tensor,  # "*#batch dim"
) -> torch.Tensor:  # "*batch dim"
    """Compute the least-squares intersection of rays. Uses the math from here:
    https://math.stackexchange.com/a/1762491/286022
    """

    # Broadcast and stack the tensors.
    a_origins, a_directions, b_origins, b_directions = torch.broadcast_tensors(
        a_origins, a_directions, b_origins, b_directions
    )
    origins = torch.stack((a_origins, b_origins), dim=-2)
    directions = torch.stack((a_directions, b_directions), dim=-2)

    # Compute n_i * n_i^T - eye(3) from the equation.
    n = einsum(directions, directions, "... n i, ... n j -> ... n i j")
    n = n - torch.eye(3, dtype=origins.dtype, device=origins.device)

    # Compute the left-hand side of the equation.
    lhs = reduce(n, "... n i j -> ... i j", "sum")

    # Compute the right-hand side of the equation.
    rhs = einsum(n, origins, "... n i j, ... n j -> ... n i")
    rhs = reduce(rhs, "... n i -> ... i", "sum")

    # Left-matrix-multiply both sides by the inverse of lhs to find p.
    return torch.linalg.lstsq(lhs, rhs).solution


def normalize(a: torch.Tensor) -> torch.Tensor:  # "*#batch dim" -> "*#batch dim"
    return a / a.norm(dim=-1, keepdim=True)


def generate_coordinate_frame(
    y: torch.Tensor,  # "*#batch 3"
    z: torch.Tensor,  # "*#batch 3"
) -> torch.Tensor:  # "*batch 3 3"
    """Generate a coordinate frame given perpendicular, unit-length Y and Z vectors."""
    y, z = torch.broadcast_tensors(y, z)
    return torch.stack([y.cross(z, dim=-1), y, z], dim=-1)


def generate_rotation_coordinate_frame(
    a: torch.Tensor,  # "*#batch 3"
    b: torch.Tensor,  # "*#batch 3"
    eps: float = 1e-4,
) -> torch.Tensor:  # "*batch 3 3"
    """Generate a coordinate frame where the Y direction is normal to the plane defined
    by unit vectors a and b. The other axes are arbitrary."""
    device = a.device

    # Replace every entry in b that's parallel to the corresponding entry in a with an
    # arbitrary vector.
    b = b.detach().clone()
    parallel = (einsum(a, b, "... i, ... i -> ...").abs() - 1).abs() < eps
    b[parallel] = torch.tensor([0, 0, 1], dtype=b.dtype, device=device)
    parallel = (einsum(a, b, "... i, ... i -> ...").abs() - 1).abs() < eps
    b[parallel] = torch.tensor([0, 1, 0], dtype=b.dtype, device=device)

    # Generate the coordinate frame. The initial cross product defines the plane.
    return generate_coordinate_frame(normalize(torch.linalg.cross(a, b)), a)


def matrix_to_euler(
    rotations: torch.Tensor,  # "*batch 3 3"
    pattern: str,
) -> torch.Tensor:  # "*batch 3"
    *batch, _, _ = rotations.shape
    rotations = rotations.reshape(-1, 3, 3)
    angles_np = R.from_matrix(rotations.detach().cpu().numpy()).as_euler(pattern)
    rotations = torch.tensor(angles_np, dtype=rotations.dtype, device=rotations.device)
    return rotations.reshape(*batch, 3)


def euler_to_matrix(
    rotations: torch.Tensor,  # "*batch 3"
    pattern: str,
) -> torch.Tensor:  # "*batch 3 3"
    *batch, _ = rotations.shape
    rotations = rotations.reshape(-1, 3)
    matrix_np = R.from_euler(pattern, rotations.detach().cpu().numpy()).as_matrix()
    rotations = torch.tensor(matrix_np, dtype=rotations.dtype, device=rotations.device)
    return rotations.reshape(*batch, 3, 3)


def extrinsics_to_pivot_parameters(
    extrinsics: torch.Tensor,  # "*#batch 4 4"
    pivot_coordinate_frame: torch.Tensor,  # "*#batch 3 3"
    pivot_point: torch.Tensor,  # "*#batch 3"
) -> torch.Tensor:  # "*batch 5"
    """Convert the extrinsics to a representation with 5 degrees of freedom:
    1. Distance from pivot point in the "X" (look cross pivot axis) direction.
    2. Distance from pivot point in the "Y" (pivot axis) direction.
    3. Distance from pivot point in the Z (look) direction
    4. Angle in plane
    5. Twist (rotation not in plane)
    """

    # The pivot coordinate frame's Z axis is normal to the plane.
    pivot_axis = pivot_coordinate_frame[..., :, 1]

    # Compute the translation elements of the pivot parametrization.
    translation_frame = generate_coordinate_frame(pivot_axis, extrinsics[..., :3, 2])
    origin = extrinsics[..., :3, 3]
    delta = pivot_point - origin
    translation = einsum(translation_frame, delta, "... i j, ... i -> ... j")

    # Add the rotation elements of the pivot parametrization.
    inverted = pivot_coordinate_frame.inverse() @ extrinsics[..., :3, :3]
    y, _, z = matrix_to_euler(inverted, "YXZ").unbind(dim=-1)

    return torch.cat([translation, y[..., None], z[..., None]], dim=-1)


def pivot_parameters_to_extrinsics(
    parameters: torch.Tensor,  # "*#batch 5"
    pivot_coordinate_frame: torch.Tensor,  # "*#batch 3 3"
    pivot_point: torch.Tensor,  # "*#batch 3"
) -> torch.Tensor:  # "*batch 4 4"
    translation, y, z = parameters.split((3, 1, 1), dim=-1)

    euler = torch.cat((y, torch.zeros_like(y), z), dim=-1)
    rotation = pivot_coordinate_frame @ euler_to_matrix(euler, "YXZ")

    # The pivot coordinate frame's Z axis is normal to the plane.
    pivot_axis = pivot_coordinate_frame[..., :, 1]

    translation_frame = generate_coordinate_frame(pivot_axis, rotation[..., :3, 2])
    delta = einsum(translation_frame, translation, "... i j, ... j -> ... i")
    origin = pivot_point - delta

    *batch, _ = origin.shape
    extrinsics = torch.eye(4, dtype=parameters.dtype, device=parameters.device)
    extrinsics = extrinsics.broadcast_to((*batch, 4, 4)).clone()
    extrinsics[..., 3, 3] = 1
    extrinsics[..., :3, :3] = rotation
    extrinsics[..., :3, 3] = origin
    return extrinsics


def interpolate_circular(
    a: torch.Tensor,  # "*#batch"
    b: torch.Tensor,  # "*#batch"
    t: torch.Tensor,  # "*#batch"
) -> torch.Tensor:  # " *batch"
    a, b, t = torch.broadcast_tensors(a, b, t)

    tau = 2 * torch.pi
    a = a % tau
    b = b % tau

    # Consider piecewise edge cases.
    d = (b - a).abs()
    a_left = a - tau
    d_left = (b - a_left).abs()
    a_right = a + tau
    d_right = (b - a_right).abs()
    use_d = (d < d_left) & (d < d_right)
    use_d_left = (d_left < d_right) & (~use_d)
    use_d_right = (~use_d) & (~use_d_left)

    result = a + (b - a) * t
    result[use_d_left] = (a_left + (b - a_left) * t)[use_d_left]
    result[use_d_right] = (a_right + (b - a_right) * t)[use_d_right]

    return result


def interpolate_pivot_parameters(
    initial: torch.Tensor,  # "*#batch 5"
    final: torch.Tensor,  # "*#batch 5"
    t: torch.Tensor,  # " time_step"
) -> torch.Tensor:  # "*batch time_step 5"
    initial = rearrange(initial, "... d -> ... () d")
    final = rearrange(final, "... d -> ... () d")
    t = rearrange(t, "t -> t ()")
    ti, ri = initial.split((3, 2), dim=-1)
    tf, rf = final.split((3, 2), dim=-1)

    t_lerp = ti + (tf - ti) * t
    r_lerp = interpolate_circular(ri, rf, t)

    return torch.cat((t_lerp, r_lerp), dim=-1)


def get_camera_trajectory_relative_positions(
    extrinsics,
    alpha: float = 0.8,
):
    N = len(extrinsics)
    assert N >= 2, "至少需要2个相机外参"
    
    trans_norm = np.zeros(N-1)
    theta_norm = np.zeros(N-1)
    for i in range(N - 1):
        w2c_curr = extrinsics[i]
        w2c_next = extrinsics[i + 1]
        trans_norm[i], theta_norm[i] = camera_distance(w2c_curr, w2c_next)
    
    cumulative_trans_dist = np.zeros(N)
    cumulative_theta_dist = np.zeros(N)

    for i in range(1, N):
        cumulative_trans_dist[i] = cumulative_trans_dist[i - 1] + trans_norm[i - 1]
        cumulative_theta_dist[i] = cumulative_theta_dist[i - 1] + theta_norm[i - 1]
    
    max_trans_dist = (cumulative_trans_dist.max() + 1e-8)
    max_theta_dist = (cumulative_theta_dist.max() + 1e-8)

    relative_pos = alpha * cumulative_trans_dist / max_trans_dist + (1 - alpha) * cumulative_theta_dist / max_theta_dist

    print('trans', cumulative_trans_dist / max_trans_dist)
    print('rota', cumulative_theta_dist / max_theta_dist)

    def linear_squeeze(x, alpha=0.5):
        """
        alpha: 挤压权重 (0 到 1 之间)
        alpha=0: 保持原样
        alpha=1: 全部变成 0.5
        """
        # 将 x 向 0.5 插值： (1-alpha)*x + alpha*0.5
        # 但为了保持边界 0 和 1 不动，我们需要根据距离 0.5 的远近动态调整 alpha
        # 这里用一个简单的二次方权重来保证边界不动
        weight = alpha * 4 * x * (1 - x) 
        return (1 - weight) * x + weight * 0.5

    return linear_squeeze(relative_pos)

def camera_distance(
    w2c1,
    w2c2,
) -> float:
    
    R1, t1 = w2c1[:3, :3], w2c1[:3, 3]
    R2, t2 = w2c2[:3, :3], w2c2[:3, 3]
    
    C1 = -R1.T @ t1
    C2 = -R2.T @ t2
    d_trans_norm = np.linalg.norm(C1 - C2)
    
    R_rel = R1 @ R2.T
    tr_rel = np.trace(R_rel)
    cos_theta = np.clip((tr_rel - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    theta_norm = theta / np.pi 

    return d_trans_norm, theta_norm


def slerp(q0, q1, t):
    """
    四元数球面线性插值
    
    参数:
        q0: 起始四元数
        q1: 结束四元数
        t: 插值参数，0<=t<=1
        
    返回:
        插值后的四元数
    """
    # 确保四元数是单位四元数
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    
    # 计算点积
    dot = np.dot(q0, q1)
    
    # 如果点积为负，取反以选择最短路径
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    
    # 确保数值稳定性
    dot = np.clip(dot, -1.0, 1.0)
    
    # 计算角度
    theta_0 = np.arccos(dot)
    theta = theta_0 * t
    
    # 计算插值四元数
    q2 = (q1 - q0 * dot) / np.linalg.norm(q1 - q0 * dot)
    result = q0 * np.cos(theta) + q2 * np.sin(theta)
    
    return result


def interpolate_extrinsics_sparseview(extrinsics, extr_idx, m):
    """
    将n个3×4的相机外参矩阵插值为m个
    
    参数:
        extrinsics: 形状为(n, 3, 4)的numpy数组，包含n个外参矩阵
        m: 插值后的外参矩阵数量
        
    返回:
        形状为(m, 3, 4)的numpy数组，包含m个插值后的外参矩阵
    """
    n = len(extrinsics)
    if n <= 1:
        raise ValueError("至少需要2个外参矩阵进行插值")
    
    print(extr_idx)

    # 分解外参矩阵为旋转矩阵和平移向量
    rotations = []
    translations = []
    for extrinsic in extrinsics:
        # 提取旋转矩阵 (3x3)
        rot = extrinsic[:, :3]
        # 提取平移向量 (3x1)
        trans = extrinsic[:, 3]
        
        rotations.append(rot)
        translations.append(trans)
    
    # 生成插值的时间点
    original_times = get_camera_trajectory_relative_positions(extrinsics)
    # original_times = np.linspace(0, 1, n)
    interp_times = np.linspace(0, 1, m)

    index = []
    for i in range(n):
        if extr_idx[i] != -1:
            min_idx = np.argmin(np.abs(interp_times-original_times[i]))
            original_times[i] = interp_times[min_idx]
            index.append([extr_idx[i], min_idx])

    # print("原始位置", original_times)
    # print("插值位置", interp_times)

    # 对平移向量进行线性插值
    interp_translations = np.zeros((m, 3))
    for i in range(3):
        interp_translations[:, i] = np.interp(interp_times, original_times, 
                                             [t[i] for t in translations])
    
    # 将旋转矩阵转换为四元数
    rotations = R.from_matrix(rotations)
    quaternions = rotations.as_quat()  # 形状为(n, 4)
    
    # 对四元数进行球面线性插值
    interp_quaternions = []
    for t in interp_times:
        # 找到当前时间点所在的区间
        idx = np.searchsorted(original_times, t) - 1
        idx = np.clip(idx, 0, n-2)
        
        # 获取区间端点
        t0, t1 = original_times[idx], original_times[idx+1]
        q0, q1 = quaternions[idx], quaternions[idx+1]
        
        # 计算插值比例
        if t1 == t0:
            alpha = 0.0
        else:
            alpha = (t - t0) / (t1 - t0)
        
        # 四元数球面线性插值
        interp_q = slerp(q0, q1, alpha)
        interp_quaternions.append(interp_q)
    
    # 将四元数转换回旋转矩阵
    interp_rotations = R.from_quat(interp_quaternions).as_matrix()  # 形状为(m, 3, 3)
    
    # 组合旋转矩阵和平移向量为外参矩阵
    result = np.zeros((m, 3, 4))
    for i in range(m):
        result[i, :, :3] = interp_rotations[i]
        result[i, :, 3] = interp_translations[i]
    
    return result, index


def get_cubic_bezier_trajectory(control_points, m):
    """
    使用标准三阶贝塞尔公式生成相机平移轨迹。
    control_points: list 或 array, 包含 2-4 个 (3,) 的控制点坐标。
    m: 插值后的总帧数。
    返回: (m, 3) 的插值轨迹数组。
    """
    n = len(control_points)
    
    # 将控制点转换为 numpy 数组 (n, 3)
    P = np.array(control_points)
    
    if n < 2 or n > 5:
        raise ValueError("控制点数量必须在 2 到 5 之间。")
    
    # 生成时间采样点 [0, 1]，并重塑以便广播计算 (m, 1)
    t = np.linspace(0, 1, m).reshape(-1, 1)
    
    # 核心公式：根据不同的控制点数量动态适应
    if n == 2:
        # 特殊情况：2个点其实是线性插值
        # B(t) = (1-t)P0 + tP1
        path = (1-t) * P[0] + t * P[1]
        
    elif n == 3:
        # 特殊情况：3个点是二阶贝塞尔
        # B(t) = (1-t)^2 P0 + 2t(1-t)P1 + t^2 P2
        path = (1-t)**2 * P[0] + 2*(1-t)*t * P[1] + t**2 * P[2]
        
    elif n == 4:
        # 标准三阶贝塞尔
        # B(t) = (1-t)^3 P0 + 3(1-t)^2t P1 + 3(1-t)t^2 P2 + t^3 P3
        path = (1-t)**3 * P[0] + 3*(1-t)**2 * t * P[1] + 3*(1-t) * t**2 * P[2] + t**3 * P[3]
    
    elif n == 5:
        path = (1-t)**4 * P[0] + 4*(1-t)**3 *t * P[1] + 6 *(1-t)**2 *t **2 * P[2] + 4 * (1-t) * t**3 * P[3] + t**4 * P[4]

    return path

def get_bezier_rotation_trajectory(rot_matrices, m):
    """
    使用递归 SLERP 实现旋转的贝塞尔平滑
    """
    n = len(rot_matrices)
    quats = R.from_matrix(rot_matrices).as_quat()
    
    # 统一四元数符号，防止旋转“绕远路”
    for i in range(1, n):
        if np.dot(quats[i], quats[0]) < 0:
            quats[i] = -quats[i]

    t_steps = np.linspace(0, 1, m)
    interp_quats = []

    for t in t_steps:
        # 这里就是你喜欢的“贝塞尔公式”在旋转空间的递归体现
        if n == 2:
            res_q = slerp(quats[0], quats[1], t)
        elif n == 3:
            q01 = slerp(quats[0], quats[1], t)
            q12 = slerp(quats[1], quats[2], t)
            res_q = slerp(q01, q12, t)
        elif n == 4:
            # 每一层递归本质上都是在做一次“加权”
            q01 = slerp(quats[0], quats[1], t)
            q11 = slerp(quats[1], quats[2], t)
            q21 = slerp(quats[2], quats[3], t)
            
            q02 = slerp(q01, q11, t)
            q12 = slerp(q11, q21, t)
            
            res_q = slerp(q02, q12, t)
        elif n == 5:
            q01 = slerp(quats[0], quats[1], t)
            q11 = slerp(quats[1], quats[2], t)
            q21 = slerp(quats[2], quats[3], t)
            q31 = slerp(quats[3], quats[4], t)

            q02 = slerp(q01, q11, t)
            q12 = slerp(q11, q21, t)
            q22 = slerp(q21, q31, t)
            
            q03 = slerp(q02, q12, t)
            q13 = slerp(q12, q22, t)

            res_q = slerp(q03, q13, t)
        
        interp_quats.append(res_q)

    return R.from_quat(interp_quats).as_matrix()

def interpolate_extrinsics_sparseview_Bezier(extrinsics, extr_idx, m):
    """
    将n个3×4的相机外参矩阵插值为m个
    
    参数:
        extrinsics: 形状为(n, 3, 4)的numpy数组，包含n个外参矩阵
        m: 插值后的外参矩阵数量
        
    返回:
        形状为(m, 3, 4)的numpy数组，包含m个插值后的外参矩阵
    """
    n = len(extrinsics)
    if n <= 1:
        raise ValueError("至少需要2个外参矩阵进行插值")
    
    print(extr_idx)

    # 分解外参矩阵为旋转矩阵和平移向量
    rotations = []
    translations = []
    for extrinsic in extrinsics:
        # 提取旋转矩阵 (3x3)
        rot = extrinsic[:, :3]
        # 提取平移向量 (3x1)
        trans = extrinsic[:, 3]
        
        rotations.append(rot)
        translations.append(trans)
    

    index = [[extr_idx[0], 0], [extr_idx[-1], m-1]]

    # 对平移向量进行线性插值

    interp_translations = get_cubic_bezier_trajectory(translations, m)
    interp_rotations = get_bezier_rotation_trajectory(rotations, m)

    
    # 组合旋转矩阵和平移向量为外参矩阵
    result = np.zeros((m, 3, 4))
    for i in range(m):
        result[i, :, :3] = interp_rotations[i]
        result[i, :, 3] = interp_translations[i]

    return result, index



@torch.no_grad()
def interpolate_extrinsics(
    initial: torch.Tensor,  # "*#batch 4 4"
    final: torch.Tensor,  # "*#batch 4 4"
    t: torch.Tensor,  # " time_step"
    eps: float = 1e-4,
) -> torch.Tensor:  # "*batch time_step 4 4"
    """Interpolate extrinsics by rotating around their "focus point," which is the
    least-squares intersection between the look vectors of the initial and final
    extrinsics.
    """

    initial = initial.type(torch.float64)
    final = final.type(torch.float64)
    t = t.type(torch.float64)

    # Based on the dot product between the look vectors, pick from one of two cases:
    # 1. Look vectors are parallel: interpolate about their origins' midpoint.
    # 3. Look vectors aren't parallel: interpolate about their focus point.
    initial_look = initial[..., :3, 2]
    final_look = final[..., :3, 2]
    dot_products = einsum(initial_look, final_look, "... i, ... i -> ...")
    parallel_mask = (dot_products.abs() - 1).abs() < eps

    # Pick focus points.
    initial_origin = initial[..., :3, 3]
    final_origin = final[..., :3, 3]
    pivot_point = 0.5 * (initial_origin + final_origin)
    pivot_point[~parallel_mask] = intersect_rays(
        initial_origin[~parallel_mask],
        initial_look[~parallel_mask],
        final_origin[~parallel_mask],
        final_look[~parallel_mask],
    )

    # Convert to pivot parameters.
    pivot_frame = generate_rotation_coordinate_frame(initial_look, final_look, eps=eps)
    initial_params = extrinsics_to_pivot_parameters(initial, pivot_frame, pivot_point)
    final_params = extrinsics_to_pivot_parameters(final, pivot_frame, pivot_point)

    # Interpolate the pivot parameters.
    interpolated_params = interpolate_pivot_parameters(initial_params, final_params, t)

    # Convert back.
    return pivot_parameters_to_extrinsics(
        interpolated_params.type(torch.float32),
        rearrange(pivot_frame, "... i j -> ... () i j").type(torch.float32),
        rearrange(pivot_point, "... xyz -> ... () xyz").type(torch.float32),
    )


@torch.no_grad()
def generate_wobble_transformation(
    radius: torch.Tensor,  # "*#batch"
    t: torch.Tensor,  # " time_step"
    num_rotations: int = 1,
    scale_radius_with_t: bool = True,
) -> torch.Tensor:  # "*batch time_step 4 4"]:
    # Generate a translation in the image plane.
    tf = torch.eye(4, dtype=torch.float32, device=t.device)
    tf = tf.broadcast_to((*radius.shape, t.shape[0], 4, 4)).clone()
    radius = radius[..., None]
    if scale_radius_with_t:
        radius = radius * t
    tf[..., 0, 3] = torch.sin(2 * torch.pi * num_rotations * t) * radius
    tf[..., 1, 3] = -torch.cos(2 * torch.pi * num_rotations * t) * radius
    return tf


@torch.no_grad()
def render_wobble_inter_path(
    cam2world: torch.Tensor, intr_normed: torch.Tensor, inter_len: int, n_skip: int = 3
):
    """
    cam2world: [batch, 4, 4],
    intr_normed: [batch, 3, 3]
    """
    frame_per_round = n_skip * inter_len
    num_rotations = 1

    t = torch.linspace(0, 1, frame_per_round, dtype=torch.float32, device=cam2world.device)
    # t = (torch.cos(torch.pi * (t + 1)) + 1) / 2
    tgt_c2w_b = []
    tgt_intr_b = []
    for b_idx in range(cam2world.shape[0]):
        tgt_c2w = []
        tgt_intr = []
        for cur_idx in range(0, cam2world.shape[1] - n_skip, n_skip):
            origin_a = cam2world[b_idx, cur_idx, :3, 3]
            origin_b = cam2world[b_idx, cur_idx + n_skip, :3, 3]
            delta = (origin_a - origin_b).norm(dim=-1)
            if cur_idx == 0:
                delta_prev = delta
            else:
                delta = (delta_prev + delta) / 2
                delta_prev = delta
            tf = generate_wobble_transformation(
                radius=delta * 0.5,
                t=t,
                num_rotations=num_rotations,
                scale_radius_with_t=False,
            )
            cur_extrs = (
                interpolate_extrinsics(
                    cam2world[b_idx, cur_idx],
                    cam2world[b_idx, cur_idx + n_skip],
                    t,
                )
                @ tf
            )
            tgt_c2w.append(cur_extrs[(0 if cur_idx == 0 else 1) :])
            tgt_intr.append(
                interpolate_intrinsics(
                    intr_normed[b_idx, cur_idx],
                    intr_normed[b_idx, cur_idx + n_skip],
                    t,
                )[(0 if cur_idx == 0 else 1) :]
            )
        tgt_c2w_b.append(torch.cat(tgt_c2w))
        tgt_intr_b.append(torch.cat(tgt_intr))
    tgt_c2w = torch.stack(tgt_c2w_b)  # b v 4 4
    tgt_intr = torch.stack(tgt_intr_b)  # b v 3 3
    return tgt_c2w, tgt_intr
