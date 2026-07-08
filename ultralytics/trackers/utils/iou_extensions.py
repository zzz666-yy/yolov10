"""
IOU Extension Module for BYTETrack
支持多种 IOU 变体，可通过配置文件灵活切换
"""

import numpy as np
import math
from typing import Union, List, Optional, Dict, Any

try:
    from numba import jit, prange

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


    def jit(*args, **kwargs):
        def decorator(func):
            return func

        return decorator if args and callable(args[0]) else decorator


    prange = range

# 尝试导入 shapely 用于精确旋转框计算
try:
    from shapely.geometry import Polygon
    from shapely.affinity import rotate

    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False


class IOUConfig:
    """IOU 配置类"""

    # 支持的 IOU 类型
    SUPPORTED_TYPES = {
        'iou': 'Standard Intersection over Union',
        'giou': 'Generalized IoU - Handles non-overlapping boxes',
        'diou': 'Distance IoU - Considers center point distance',
        'ciou': 'Complete IoU - Comprehensive metric with aspect ratio',
        'eiou': 'Enhanced IoU - Separate width and height penalties',
        'siou': 'Scylla IoU - Incorporates angle consideration',
        'focal_iou': 'Focal IoU - Focuses on hard samples',
        'wise_iou': 'Wise IoU - Dynamic adjustment',
        'alpha_iou': 'Alpha IoU - Power transformation for better discrimination',
        'rotated_iou': 'Rotated IoU - Precise calculation for rotated boxes',
        'adaptive_iou': 'Adaptive IoU - Dynamic weighting based on IoU value'
    }

    # 默认参数
    DEFAULT_PARAMS = {
        'giou': {'use_giou': True},
        'diou': {},
        'ciou': {},
        'eiou': {},
        'siou': {'theta': 4},
        'focal_iou': {'gamma': 0.5},
        'wise_iou': {'beta': 1.0},
        'alpha_iou': {'alpha': 2},
        'rotated_iou': {},
        'adaptive_iou': {'iou_threshold': 0.5}
    }

    def __init__(self, iou_type: str = 'iou', **kwargs):
        """
        初始化 IOU 配置

        Args:
            iou_type: IOU 类型
            **kwargs: 特定 IOU 类型的参数
        """
        self.iou_type = iou_type
        self.params = self.DEFAULT_PARAMS.get(iou_type, {}).copy()
        self.params.update(kwargs)

    def validate(self) -> bool:
        """验证配置是否有效"""
        if self.iou_type not in self.SUPPORTED_TYPES:
            raise ValueError(f"Unsupported IOU type: {self.iou_type}. "
                             f"Supported types: {list(self.SUPPORTED_TYPES.keys())}")

        if self.iou_type == 'rotated_iou' and not SHAPELY_AVAILABLE:
            print("Warning: shapely not available, falling back to standard IoU")
            self.iou_type = 'iou'

        return True


# ==================== 基础 IOU 计算函数 ====================

@jit(nopython=True, parallel=True, cache=True)
def compute_iou_matrix_fast(atlbrs, btlbrs, iou_thresh=0.0):
    """
    使用 Numba 加速的 IOU 矩阵计算
    """
    m, n = len(atlbrs), len(btlbrs)
    ious = np.zeros((m, n), dtype=np.float32)

    is_rotated = len(atlbrs[0]) == 5 and len(btlbrs[0]) == 5

    if is_rotated:
        for i in prange(m):
            for j in prange(n):
                iou = compute_rotated_iou_approx(atlbrs[i], btlbrs[j])
                if iou > iou_thresh:
                    ious[i, j] = iou
    else:
        for i in prange(m):
            box1 = atlbrs[i]
            area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])

            for j in prange(n):
                box2 = btlbrs[j]

                if (box1[2] <= box2[0] or box2[2] <= box1[0] or
                        box1[3] <= box2[1] or box2[3] <= box1[1]):
                    continue

                x1 = max(box1[0], box2[0])
                y1 = max(box1[1], box2[1])
                x2 = min(box1[2], box2[2])
                y2 = min(box1[3], box2[3])

                inter_area = max(0, x2 - x1) * max(0, y2 - y1)
                area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
                union_area = area1 + area2 - inter_area

                iou = inter_area / union_area if union_area > 0 else 0
                if iou > iou_thresh:
                    ious[i, j] = iou

    return ious


@jit(nopython=True, cache=True)
def compute_rotated_iou_approx(box1, box2):
    """
    快速近似计算旋转框 IOU
    """

    def expand_box(box):
        cx, cy, w, h, angle = box
        cos_a = abs(np.cos(angle))
        sin_a = abs(np.sin(angle))
        new_w = w * cos_a + h * sin_a
        new_h = w * sin_a + h * cos_a
        return np.array([cx - new_w / 2, cy - new_h / 2, cx + new_w / 2, cy + new_h / 2])

    hbox1 = expand_box(box1)
    hbox2 = expand_box(box2)

    x1 = max(hbox1[0], hbox2[0])
    y1 = max(hbox1[1], hbox2[1])
    x2 = min(hbox1[2], hbox2[2])
    y2 = min(hbox1[3], hbox2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (hbox1[2] - hbox1[0]) * (hbox1[3] - hbox1[1])
    area2 = (hbox2[2] - hbox2[0]) * (hbox2[3] - hbox2[1])
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0


def extract_boxes(atracks, btracks):
    """提取边界框"""
    if atracks and isinstance(atracks[0], np.ndarray) or btracks and isinstance(btracks[0], np.ndarray):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.xywha if track.angle is not None else track.xyxy for track in atracks]
        btlbrs = [track.xywha if track.angle is not None else track.xyxy for track in btracks]
    return atlbrs, btlbrs


# ==================== IOU 变体实现 ====================

class IOUCalculator:
    """IOU 计算器基类"""

    def __init__(self, config: IOUConfig):
        self.config = config
        self.params = config.params

    def calculate(self, atracks, btracks) -> np.ndarray:
        """计算距离矩阵"""
        raise NotImplementedError


class StandardIOU(IOUCalculator):
    """标准 IoU"""

    def calculate(self, atracks, btracks, use_fast=True, iou_thresh=0.01) -> np.ndarray:
        atlbrs, btlbrs = extract_boxes(atracks, btracks)

        if len(atlbrs) == 0 or len(btlbrs) == 0:
            return np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float32)

        if use_fast and NUMBA_AVAILABLE:
            ious = compute_iou_matrix_fast(
                np.ascontiguousarray(atlbrs, dtype=np.float32),
                np.ascontiguousarray(btlbrs, dtype=np.float32),
                iou_thresh
            )
        else:
            ious = self._compute_iou_slow(atlbrs, btlbrs)

        return 1 - ious

    def _compute_iou_slow(self, atlbrs, btlbrs):
        """慢速但兼容性好的 IoU 计算"""
        m, n = len(atlbrs), len(btlbrs)
        ious = np.zeros((m, n), dtype=np.float32)

        for i in range(m):
            box1 = atlbrs[i]
            area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])

            for j in range(n):
                box2 = btlbrs[j]
                area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

                x1 = max(box1[0], box2[0])
                y1 = max(box1[1], box2[1])
                x2 = min(box1[2], box2[2])
                y2 = min(box1[3], box2[3])
                inter_area = max(0, x2 - x1) * max(0, y2 - y1)
                union_area = area1 + area2 - inter_area

                ious[i, j] = inter_area / union_area if union_area > 0 else 0

        return ious


class GeneralizedIOU(IOUCalculator):
    """GIoU - Generalized IoU"""

    def calculate(self, atracks, btracks) -> np.ndarray:
        atlbrs, btlbrs = extract_boxes(atracks, btracks)

        if len(atlbrs) == 0 or len(btlbrs) == 0:
            return np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float32)

        m, n = len(atlbrs), len(btlbrs)
        gious = np.zeros((m, n), dtype=np.float32)

        for i in range(m):
            box1 = atlbrs[i]
            area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])

            for j in range(n):
                box2 = btlbrs[j]
                area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

                # 计算 IoU
                x1 = max(box1[0], box2[0])
                y1 = max(box1[1], box2[1])
                x2 = min(box1[2], box2[2])
                y2 = min(box1[3], box2[3])
                inter_area = max(0, x2 - x1) * max(0, y2 - y1)
                union_area = area1 + area2 - inter_area
                iou = inter_area / union_area if union_area > 0 else 0

                # 计算最小外接矩形
                enclose_x1 = min(box1[0], box2[0])
                enclose_y1 = min(box1[1], box2[1])
                enclose_x2 = max(box1[2], box2[2])
                enclose_y2 = max(box1[3], box2[3])
                enclose_area = (enclose_x2 - enclose_x1) * (enclose_y2 - enclose_y1)

                giou = iou - (enclose_area - union_area) / enclose_area if enclose_area > 0 else iou
                gious[i, j] = 1 - giou

        return gious


class DistanceIOU(IOUCalculator):
    """DIoU - Distance IoU"""

    def calculate(self, atracks, btracks) -> np.ndarray:
        atlbrs, btlbrs = extract_boxes(atracks, btracks)

        if len(atlbrs) == 0 or len(btlbrs) == 0:
            return np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float32)

        m, n = len(atlbrs), len(btlbrs)
        dious = np.zeros((m, n), dtype=np.float32)

        for i in range(m):
            box1 = atlbrs[i]
            center1 = np.array([(box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2])
            area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])

            for j in range(n):
                box2 = btlbrs[j]
                center2 = np.array([(box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2])
                area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

                # 计算 IoU
                x1 = max(box1[0], box2[0])
                y1 = max(box1[1], box2[1])
                x2 = min(box1[2], box2[2])
                y2 = min(box1[3], box2[3])
                inter_area = max(0, x2 - x1) * max(0, y2 - y1)
                union_area = area1 + area2 - inter_area
                iou = inter_area / union_area if union_area > 0 else 0

                # 计算中心点距离平方
                center_dist_sq = np.sum((center1 - center2) ** 2)

                # 计算最小外接矩形对角线距离平方
                enclose_x1 = min(box1[0], box2[0])
                enclose_y1 = min(box1[1], box2[1])
                enclose_x2 = max(box1[2], box2[2])
                enclose_y2 = max(box1[3], box2[3])
                enclose_diag_sq = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2

                diou = iou - (center_dist_sq / (enclose_diag_sq + 1e-7))
                dious[i, j] = 1 - np.clip(diou, -1, 1)

        return dious


class CompleteIOU(IOUCalculator):
    """CIoU - Complete IoU"""

    def calculate(self, atracks, btracks) -> np.ndarray:
        atlbrs, btlbrs = extract_boxes(atracks, btracks)

        if len(atlbrs) == 0 or len(btlbrs) == 0:
            return np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float32)

        m, n = len(atlbrs), len(btlbrs)
        cious = np.zeros((m, n), dtype=np.float32)

        for i in range(m):
            box1 = atlbrs[i]
            center1_x = (box1[0] + box1[2]) / 2
            center1_y = (box1[1] + box1[3]) / 2
            width1 = box1[2] - box1[0]
            height1 = box1[3] - box1[1]
            area1 = width1 * height1

            for j in range(n):
                box2 = btlbrs[j]
                center2_x = (box2[0] + box2[2]) / 2
                center2_y = (box2[1] + box2[3]) / 2
                width2 = box2[2] - box2[0]
                height2 = box2[3] - box2[1]
                area2 = width2 * height2

                # 计算 IoU
                x1 = max(box1[0], box2[0])
                y1 = max(box1[1], box2[1])
                x2 = min(box1[2], box2[2])
                y2 = min(box1[3], box2[3])
                inter_area = max(0, x2 - x1) * max(0, y2 - y1)
                union_area = area1 + area2 - inter_area
                iou = inter_area / union_area if union_area > 0 else 0

                # 计算中心点距离平方
                center_dist_sq = (center1_x - center2_x) ** 2 + (center1_y - center2_y) ** 2

                # 计算最小外接矩形对角线距离平方
                enclose_x1 = min(box1[0], box2[0])
                enclose_y1 = min(box1[1], box2[1])
                enclose_x2 = max(box1[2], box2[2])
                enclose_y2 = max(box1[3], box2[3])
                enclose_diag_sq = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2

                # 计算宽高比惩罚项
                v = (4 / (math.pi ** 2)) * ((math.atan(width2 / max(height2, 1e-7)) -
                                             math.atan(width1 / max(height1, 1e-7))) ** 2)
                alpha = v / ((1 - iou) + v + 1e-7)

                ciou = iou - (center_dist_sq / (enclose_diag_sq + 1e-7)) - (alpha * v)
                cious[i, j] = 1 - np.clip(ciou, -1, 1)

        return cious


class EnhancedIOU(IOUCalculator):
    """EIoU - Enhanced IoU"""

    def calculate(self, atracks, btracks) -> np.ndarray:
        atlbrs, btlbrs = extract_boxes(atracks, btracks)

        if len(atlbrs) == 0 or len(btlbrs) == 0:
            return np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float32)

        m, n = len(atlbrs), len(btlbrs)
        eious = np.zeros((m, n), dtype=np.float32)

        for i in range(m):
            box1 = atlbrs[i]
            center1_x = (box1[0] + box1[2]) / 2
            center1_y = (box1[1] + box1[3]) / 2
            width1 = box1[2] - box1[0]
            height1 = box1[3] - box1[1]
            area1 = width1 * height1

            for j in range(n):
                box2 = btlbrs[j]
                center2_x = (box2[0] + box2[2]) / 2
                center2_y = (box2[1] + box2[3]) / 2
                width2 = box2[2] - box2[0]
                height2 = box2[3] - box2[1]
                area2 = width2 * height2

                # 计算 IoU
                x1 = max(box1[0], box2[0])
                y1 = max(box1[1], box2[1])
                x2 = min(box1[2], box2[2])
                y2 = min(box1[3], box2[3])
                inter_area = max(0, x2 - x1) * max(0, y2 - y1)
                union_area = area1 + area2 - inter_area
                iou = inter_area / union_area if union_area > 0 else 0

                # 计算中心点距离平方
                center_dist_sq = (center1_x - center2_x) ** 2 + (center1_y - center2_y) ** 2

                # 计算最小外接矩形
                enclose_x1 = min(box1[0], box2[0])
                enclose_y1 = min(box1[1], box2[1])
                enclose_x2 = max(box1[2], box2[2])
                enclose_y2 = max(box1[3], box2[3])
                enclose_w = enclose_x2 - enclose_x1
                enclose_h = enclose_y2 - enclose_y1
                enclose_diag_sq = enclose_w ** 2 + enclose_h ** 2

                # 计算宽度和高度的差异惩罚
                width_diff_sq = (width1 - width2) ** 2
                height_diff_sq = (height1 - height2) ** 2
                width_penalty = width_diff_sq / (enclose_w ** 2 + 1e-7)
                height_penalty = height_diff_sq / (enclose_h ** 2 + 1e-7)

                eiou = iou - (center_dist_sq / (enclose_diag_sq + 1e-7)) - width_penalty - height_penalty
                eious[i, j] = 1 - np.clip(eiou, -1, 1)

        return eious


class ScyllaIOU(IOUCalculator):
    """SIoU - Scylla IoU with angle consideration"""

    def calculate(self, atracks, btracks) -> np.ndarray:
        atlbrs, btlbrs = extract_boxes(atracks, btracks)

        if len(atlbrs) == 0 or len(btlbrs) == 0:
            return np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float32)

        theta = self.params.get('theta', 4)
        m, n = len(atlbrs), len(btlbrs)
        sious = np.zeros((m, n), dtype=np.float32)

        for i in range(m):
            box1 = atlbrs[i]
            center1_x = (box1[0] + box1[2]) / 2
            center1_y = (box1[1] + box1[3]) / 2
            width1 = box1[2] - box1[0]
            height1 = box1[3] - box1[1]
            area1 = width1 * height1

            for j in range(n):
                box2 = btlbrs[j]
                center2_x = (box2[0] + box2[2]) / 2
                center2_y = (box2[1] + box2[3]) / 2
                width2 = box2[2] - box2[0]
                height2 = box2[3] - box2[1]
                area2 = width2 * height2

                # 计算 IoU
                x1 = max(box1[0], box2[0])
                y1 = max(box1[1], box2[1])
                x2 = min(box1[2], box2[2])
                y2 = min(box1[3], box2[3])
                inter_area = max(0, x2 - x1) * max(0, y2 - y1)
                union_area = area1 + area2 - inter_area
                iou = inter_area / union_area if union_area > 0 else 0

                # 计算角度惩罚
                dx = center1_x - center2_x
                dy = center1_y - center2_y
                sigma = np.sqrt(dx ** 2 + dy ** 2)
                sin_alpha = abs(dy) / (sigma + 1e-7)
                angle_cost = 1 - 2 * sin_alpha ** 2

                # 计算距离惩罚
                enclose_x1 = min(box1[0], box2[0])
                enclose_y1 = min(box1[1], box2[1])
                enclose_x2 = max(box1[2], box2[2])
                enclose_y2 = max(box1[3], box2[3])
                enclose_w = enclose_x2 - enclose_x1
                enclose_h = enclose_y2 - enclose_y1

                rho_x = (dx / (enclose_w + 1e-7)) ** 2
                rho_y = (dy / (enclose_h + 1e-7)) ** 2
                distance_cost = 2 - np.exp(-angle_cost * rho_x) - np.exp(-angle_cost * rho_y)

                # 计算形状惩罚
                w_diff = abs(width1 - width2) / max(width1, width2, 1e-7)
                h_diff = abs(height1 - height2) / max(height1, height2, 1e-7)
                shape_cost = (1 - np.exp(-w_diff)) ** theta + (1 - np.exp(-h_diff)) ** theta

                siou = iou - (distance_cost + shape_cost) / 2
                sious[i, j] = 1 - np.clip(siou, -1, 1)

        return sious


class FocalIOU(IOUCalculator):
    """Focal IoU - Focus on hard samples"""

    def calculate(self, atracks, btracks) -> np.ndarray:
        gamma = self.params.get('gamma', 0.5)

        # 先计算标准 IoU
        standard_iou = StandardIOU(self.config)
        dists = standard_iou.calculate(atracks, btracks, use_fast=True)

        # 转换为 IoU
        iou_sim = 1 - dists

        # 应用 Focal 变换
        focal_iou = (iou_sim ** gamma) * (1 - iou_sim)

        return 1 - focal_iou


class WiseIOU(IOUCalculator):
    """Wise IoU - Dynamic adjustment"""

    def calculate(self, atracks, btracks) -> np.ndarray:
        beta = self.params.get('beta', 1.0)

        # 先计算标准 IoU
        standard_iou = StandardIOU(self.config)
        dists = standard_iou.calculate(atracks, btracks, use_fast=True)

        # 转换为 IoU
        iou_sim = 1 - dists

        # 应用 Wise 变换
        wise_iou = iou_sim * (1 + beta * (1 - iou_sim))

        return 1 - wise_iou


class AlphaIOU(IOUCalculator):
    """Alpha IoU - Power transformation"""

    def calculate(self, atracks, btracks) -> np.ndarray:
        alpha = self.params.get('alpha', 2)

        # 先计算标准 IoU
        standard_iou = StandardIOU(self.config)
        dists = standard_iou.calculate(atracks, btracks, use_fast=True)

        # 转换为 IoU 并应用幂变换
        iou_sim = 1 - dists
        alpha_iou = iou_sim ** alpha

        return 1 - alpha_iou


class RotatedIOU(IOUCalculator):
    """Precise rotated IoU using shapely"""

    def calculate(self, atracks, btracks) -> np.ndarray:
        if not SHAPELY_AVAILABLE:
            print("Warning: shapely not available, using approximated rotated IoU")
            standard_iou = StandardIOU(self.config)
            return standard_iou.calculate(atracks, btracks)

        atlbrs, btlbrs = extract_boxes(atracks, btracks)

        if len(atlbrs) == 0 or len(btlbrs) == 0:
            return np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float32)

        # 检查是否是旋转框
        is_rotated = len(atlbrs[0]) == 5 and len(btlbrs[0]) == 5

        if not is_rotated:
            standard_iou = StandardIOU(self.config)
            return standard_iou.calculate(atracks, btracks)

        m, n = len(atlbrs), len(btlbrs)
        ious = np.zeros((m, n), dtype=np.float32)

        for i in range(m):
            box1 = atlbrs[i]
            cx1, cy1, w1, h1, angle1 = box1

            # 创建旋转矩形多边形
            rect1 = Polygon([(cx1 - w1 / 2, cy1 - h1 / 2),
                             (cx1 + w1 / 2, cy1 - h1 / 2),
                             (cx1 + w1 / 2, cy1 + h1 / 2),
                             (cx1 - w1 / 2, cy1 + h1 / 2)])
            rect1 = rotate(rect1, angle1, origin=(cx1, cy1), use_radians=True)

            for j in range(n):
                box2 = btlbrs[j]
                cx2, cy2, w2, h2, angle2 = box2

                rect2 = Polygon([(cx2 - w2 / 2, cy2 - h2 / 2),
                                 (cx2 + w2 / 2, cy2 - h2 / 2),
                                 (cx2 + w2 / 2, cy2 + h2 / 2),
                                 (cx2 - w2 / 2, cy2 + h2 / 2)])
                rect2 = rotate(rect2, angle2, origin=(cx2, cy2), use_radians=True)

                if rect1.intersects(rect2):
                    inter_area = rect1.intersection(rect2).area
                    union_area = rect1.area + rect2.area - inter_area
                    iou = inter_area / union_area if union_area > 0 else 0
                else:
                    iou = 0

                ious[i, j] = 1 - iou

        return ious


class AdaptiveIOU(IOUCalculator):
    """Adaptive IoU - Dynamic weighting based on IoU value"""

    def calculate(self, atracks, btracks) -> np.ndarray:
        iou_threshold = self.params.get('iou_threshold', 0.5)

        # 先计算标准 IoU
        standard_iou = StandardIOU(self.config)
        dists = standard_iou.calculate(atracks, btracks, use_fast=True)

        # 转换为相似度并应用自适应权重
        iou_sim = 1 - dists
        adaptive_weights = 1 / (1 + np.exp(-10 * (iou_sim - iou_threshold)))
        adjusted_sim = iou_sim * adaptive_weights

        return 1 - adjusted_sim


# ==================== IOU 工厂类 ====================

class IOUFactory:
    """IOU 计算器工厂"""

    _calculators = {
        'iou': StandardIOU,
        'giou': GeneralizedIOU,
        'diou': DistanceIOU,
        'ciou': CompleteIOU,
        'eiou': EnhancedIOU,
        'siou': ScyllaIOU,
        'focal_iou': FocalIOU,
        'wise_iou': WiseIOU,
        'alpha_iou': AlphaIOU,
        'rotated_iou': RotatedIOU,
        'adaptive_iou': AdaptiveIOU,
    }

    @classmethod
    def create(cls, iou_type: str, **kwargs) -> IOUCalculator:
        """
        创建 IOU 计算器

        Args:
            iou_type: IOU 类型
            **kwargs: 特定 IOU 类型的参数

        Returns:
            IOUCalculator 实例
        """
        config = IOUConfig(iou_type, **kwargs)
        config.validate()

        calculator_class = cls._calculators.get(iou_type)
        if calculator_class is None:
            raise ValueError(f"Unknown IOU type: {iou_type}")

        return calculator_class(config)

    @classmethod
    def get_available_types(cls) -> List[str]:
        """获取所有可用的 IOU 类型"""
        return list(cls._calculators.keys())

    @classmethod
    def get_info(cls, iou_type: str) -> Optional[str]:
        """获取 IOU 类型的说明信息"""
        return IOUConfig.SUPPORTED_TYPES.get(iou_type)


# ==================== 便捷函数 ====================

def compute_iou_distance(atracks, btracks, iou_type: str = 'iou', **kwargs) -> np.ndarray:
    """
    计算 IOU 距离矩阵的便捷函数

    Args:
        atracks: 轨迹列表或边界框数组
        btracks: 轨迹列表或边界框数组
        iou_type: IOU 类型
        **kwargs: 特定 IOU 类型的参数

    Returns:
        距离矩阵
    """
    calculator = IOUFactory.create(iou_type, **kwargs)
    return calculator.calculate(atracks, btracks)