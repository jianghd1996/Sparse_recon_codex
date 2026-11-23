#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

from itertools import product as product
from math import ceil
from pathlib import Path
import os
import cv2
import torch
import torch.nn.functional as F
from torch import Tensor
from torchvision.ops import nms


class FaceDetector:
    image_size = 1024

    nms_thres = 0.3

    min_sizes = [[16, 32], [64, 128], [256, 512]]
    steps = [8, 16, 32]
    variances = [0.1, 0.2]

    dtype = torch.float32

    @torch.no_grad()
    def __init__(
        self, conf_thres, box_thres, path,
        device=torch.device("cuda"),
    ):
        torch._C._jit_set_profiling_mode(False)

        self.device = torch.device(device)
        self.conf_thres = conf_thres
        self.box_thres = box_thres
        self._onloaded = False

        self.model_path = os.path.join(
            path, "face_detector_1024.bin"
        )

        # 初始化只加载 CPU 版本
        self._cpu_model = torch.jit.load(
            self.model_path, map_location="cpu"
        ).eval()
        self.model = self._cpu_model

        self._cpu_anchors = self.get_anchors(
            self.image_size
        ).to(device="cpu", dtype=self.dtype)

        self._cpu_offset = torch.tensor(
            [104, 117, 123],
            dtype=self.dtype,
            device="cpu",
        )

        self.anchors = self._cpu_anchors
        self.offset = self._cpu_offset


    def onload(self):
        if self._onloaded:
            return self

        if self.device.type != "cuda":
            raise ValueError(
                "该 RetinaFace 含 cuDNN 融合算子，推理设备需要设为 CUDA"
            )

        # 首次正式调用时才解析当前 GPU
        if self.device.index is None:
            self.device = torch.device(
                "cuda", torch.cuda.current_device()
            )

        try:
            # 关键：重新加载，才能迁移图里的常量权重
            self.model = torch.jit.load(
                self.model_path,
                map_location=self.device,
            ).eval()

            self.anchors = self._cpu_anchors.to(self.device)
            self.offset = self._cpu_offset.to(self.device)
            self._onloaded = True

        except Exception:
            self.offload()
            raise

        return self


    def offload(self):
        # 关键：释放 GPU 模型引用，而不是调用 model.cpu()
        self.model = self._cpu_model
        self.anchors = self._cpu_anchors
        self.offset = self._cpu_offset
        self._onloaded = False
        return self
    
    @staticmethod
    def get_anchors(image_size: int) -> Tensor:
        feature_maps = [[ceil(image_size / step), ceil(image_size / step)] for step in FaceDetector.steps]
        anchors = []
        for k, f in enumerate(feature_maps):
            min_sizes = FaceDetector.min_sizes[k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / image_size
                    s_ky = min_size / image_size
                    dense_cx = [x * FaceDetector.steps[k] / image_size for x in [j + 0.5]]
                    dense_cy = [y * FaceDetector.steps[k] / image_size for y in [i + 0.5]]
                    anchors += [[cx, cy, s_kx, s_ky] for cy, cx in product(dense_cy, dense_cx)]
        
        return torch.tensor(anchors, dtype=FaceDetector.dtype, device="cpu")

    @staticmethod
    def decode(loc: Tensor, priors: Tensor) -> Tensor:
        boxes = torch.cat((priors[:, :2] + loc[:, :2] * FaceDetector.variances[0] * priors[:, 2:],
                           priors[:, 2:] * torch.exp(loc[:, 2:] * FaceDetector.variances[1])), 1)
        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]
        return boxes

    @staticmethod
    def decode_landm(pre, priors):
        landms = torch.cat((priors[:, :2] + pre[:, :2] * FaceDetector.variances[0] * priors[:, 2:],
                            priors[:, :2] + pre[:, 2:4] * FaceDetector.variances[0] * priors[:, 2:],
                            priors[:, :2] + pre[:, 4:6] * FaceDetector.variances[0] * priors[:, 2:],
                            priors[:, :2] + pre[:, 6:8] * FaceDetector.variances[0] * priors[:, 2:],
                            priors[:, :2] + pre[:, 8:10] * FaceDetector.variances[0] * priors[:, 2:]), dim=1)
        return landms

    @staticmethod
    def non_max_suppression(detection, conf_thres):
        mask = detection[:, 4] >= conf_thres
        detection = detection[mask]
        if len(detection) <= 0:
            return detection
        keep = nms(detection[:, :4], detection[:, 4], FaceDetector.nms_thres)
        best_box = detection[keep]
        return best_box

    @staticmethod
    def visualize(image: Tensor, boxes_conf_landms: Tensor, viz_path):
        image_np = image.flip(0).mul(255).to(torch.uint8).permute(1, 2, 0).contiguous().cpu().numpy()
        for bfl in boxes_conf_landms:
            box = bfl[:4].long().tolist()
            cv2.rectangle(image_np, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
            cv2.putText(image_np, "{:.4f}".format(bfl[4].item()), (box[0], box[1] + 12),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

            lmk = bfl[5:].long().tolist()
            cv2.circle(image_np, (lmk[0], lmk[1]), 1, (0, 0, 255), 4)
            cv2.circle(image_np, (lmk[2], lmk[3]), 1, (0, 255, 255), 4)
            cv2.circle(image_np, (lmk[4], lmk[5]), 1, (255, 0, 255), 4)
            cv2.circle(image_np, (lmk[6], lmk[7]), 1, (0, 255, 0), 4)
            cv2.circle(image_np, (lmk[8], lmk[9]), 1, (255, 0, 0), 4)
        cv2.imwrite(viz_path, image_np)

    @torch.no_grad()
    def predict(self, image: Tensor) -> Tensor:
        self.onload()
        image = image.to(device=self.device, dtype=self.dtype).mul(255).unsqueeze(0)

        h, w = image.shape[-2:]
        input_size = max(h, w)
        image = F.pad(image, (0, input_size - w, 0, input_size - h), "constant", 0)
        image = F.interpolate(image, size=(self.image_size, self.image_size), mode="area")

        boxes, conf, landms = self.model(image.sub(self.offset[:, None, None]))

        boxes = self.decode(boxes.squeeze(0), self.anchors)
        conf = conf.squeeze(0)[:, 1:2]
        landms = self.decode_landm(landms.squeeze(0), self.anchors)

        bcls: Tensor = torch.cat([boxes, conf, landms], -1)
        bcls = self.non_max_suppression(bcls, self.conf_thres)

        bcls[:, :4].mul_(input_size)
        bcls[:, 5:].mul_(input_size)

        bcls[:, :4].clamp_min_(0)
        bcls[:, 0].clamp_max_(w - 1)
        bcls[:, 1].clamp_max_(h - 1)
        bcls[:, 2].clamp_max_(w - 1)
        bcls[:, 3].clamp_max_(h - 1)

        boxes_size = torch.maximum(bcls[:, 2].sub(bcls[:, 0]), bcls[:, 3].sub(bcls[:, 1]))
        valid_mask = boxes_size.greater_equal(self.box_thres)

        bcls = bcls[valid_mask]
        return bcls
