# Ultralytics YOLO 🚀, AGPL-3.0 license
# 修改版 - 集成 IOU 扩展模块

import numpy as np
import scipy
from scipy.spatial.distance import cdist

from ultralytics.utils.metrics import batch_probiou, bbox_ioa

# 导入 IOU 扩展模块
try:
    from .iou_extensions import IOUFactory, compute_iou_distance

    IOU_EXTENSIONS_AVAILABLE = True
except ImportError:
    IOU_EXTENSIONS_AVAILABLE = False
    print("Warning: iou_extensions module not found, using standard IoU only")

try:
    import lap

    assert lap.__version__
except (ImportError, AssertionError, AttributeError):
    from ultralytics.utils.checks import check_requirements

    check_requirements("lapx>=0.5.2")
    import lap


def linear_assignment(cost_matrix: np.ndarray, thresh: float, use_lap: bool = True) -> tuple:
    """执行线性分配"""
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))

    if use_lap:
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
        matches = [[ix, mx] for ix, mx in enumerate(x) if mx >= 0]
        unmatched_a = np.where(x < 0)[0]
        unmatched_b = np.where(y < 0)[0]
    else:
        x, y = scipy.optimize.linear_sum_assignment(cost_matrix)
        matches = [[x[i], y[i]] for i in range(len(x)) if cost_matrix[x[i], y[i]] <= thresh]
        matched_rows = [m[0] for m in matches]
        matched_cols = [m[1] for m in matches]
        unmatched_a = [i for i in range(cost_matrix.shape[0]) if i not in matched_rows]
        unmatched_b = [j for j in range(cost_matrix.shape[1]) if j not in matched_cols]

    return matches, unmatched_a, unmatched_b


def iou_distance(atracks: list, btracks: list,
                 iou_type: str = 'iou',
                 use_fast: bool = False,
                 **kwargs) -> np.ndarray:
    """
    计算基于 IOU 的距离矩阵（支持多种 IOU 变体）

    Args:
        atracks: 轨迹列表或边界框数组
        btracks: 轨迹列表或边界框数组
        iou_type: IOU 类型 ('iou', 'giou', 'diou', 'ciou', 'eiou', 'siou',
                           'focal_iou', 'wise_iou', 'alpha_iou', 'rotated_iou', 'adaptive_iou')
        use_fast: 是否使用快速计算（仅对标准 IOU 有效）
        **kwargs: 特定 IOU 类型的参数

    Returns:
        距离矩阵 (1 - IoU)
    """
    # 如果扩展模块可用，使用扩展模块
    if IOU_EXTENSIONS_AVAILABLE and iou_type != 'iou':
        return compute_iou_distance(atracks, btracks, iou_type, **kwargs)

    # 否则使用标准 IOU 计算
    if atracks and isinstance(atracks[0], np.ndarray) or btracks and isinstance(btracks[0], np.ndarray):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.xywha if track.angle is not None else track.xyxy for track in atracks]
        btlbrs = [track.xywha if track.angle is not None else track.xyxy for track in btracks]

    if len(atlbrs) == 0 or len(btlbrs) == 0:
        return np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float32)

    # 使用快速或标准计算
    if use_fast and IOU_EXTENSIONS_AVAILABLE:
        try:
            from .iou_extensions import compute_iou_matrix_fast
            ious = compute_iou_matrix_fast(
                np.ascontiguousarray(atlbrs, dtype=np.float32),
                np.ascontiguousarray(btlbrs, dtype=np.float32),
                kwargs.get('iou_thresh', 0.01)
            )
        except:
            # 回退到标准计算
            ious = _compute_standard_iou(atlbrs, btlbrs)
    else:
        ious = _compute_standard_iou(atlbrs, btlbrs)

    return 1 - ious


def _compute_standard_iou(atlbrs, btlbrs):
    """计算标准 IOU"""
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float32)
    if len(atlbrs) and len(btlbrs):
        if len(atlbrs[0]) == 5 and len(btlbrs[0]) == 5:
            ious = batch_probiou(
                np.ascontiguousarray(atlbrs, dtype=np.float32),
                np.ascontiguousarray(btlbrs, dtype=np.float32),
            ).numpy()
        else:
            ious = bbox_ioa(
                np.ascontiguousarray(atlbrs, dtype=np.float32),
                np.ascontiguousarray(btlbrs, dtype=np.float32),
                iou=True,
            )
    return ious


def iou_distance_with_giou(atracks: list, btracks: list, use_giou: bool = True) -> np.ndarray:
    """GIoU 距离（向后兼容）"""
    return iou_distance(atracks, btracks, iou_type='giou')


def iou_distance_with_diou(atracks: list, btracks: list) -> np.ndarray:
    """DIoU 距离（向后兼容）"""
    return iou_distance(atracks, btracks, iou_type='diou')


def iou_distance_with_ciou(atracks: list, btracks: list) -> np.ndarray:
    """CIoU 距离（向后兼容）"""
    return iou_distance(atracks, btracks, iou_type='ciou')


def iou_distance_with_eiou(atracks: list, btracks: list) -> np.ndarray:
    """EIoU 距离（向后兼容）"""
    return iou_distance(atracks, btracks, iou_type='eiou')


def iou_distance_with_siou(atracks: list, btracks: list) -> np.ndarray:
    """SIoU 距离（向后兼容）"""
    return iou_distance(atracks, btracks, iou_type='siou')


def iou_distance_with_focal_iou(atracks: list, btracks: list, gamma: float = 0.5) -> np.ndarray:
    """Focal IoU 距离（向后兼容）"""
    return iou_distance(atracks, btracks, iou_type='focal_iou', gamma=gamma)


def embedding_distance(tracks: list, detections: list, metric: str = "cosine") -> np.ndarray:
    """计算嵌入距离"""
    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
    if cost_matrix.size == 0:
        return cost_matrix
    det_features = np.asarray([track.curr_feat for track in detections], dtype=np.float32)
    track_features = np.asarray([track.smooth_feat for track in tracks], dtype=np.float32)
    cost_matrix = np.maximum(0.0, cdist(track_features, det_features, metric))
    return cost_matrix


def fuse_score(cost_matrix: np.ndarray, detections: list, use_weighted_fusion: bool = True) -> np.ndarray:
    """融合分数"""
    if cost_matrix.size == 0:
        return cost_matrix

    iou_sim = 1 - cost_matrix
    det_scores = np.array([det.score for det in detections])

    if use_weighted_fusion:
        det_scores = np.power(det_scores, 1.5)
        det_scores = np.clip(det_scores, 0, 1)

    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * det_scores

    return 1 - fuse_sim