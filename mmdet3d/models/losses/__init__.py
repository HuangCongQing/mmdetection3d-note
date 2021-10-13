# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.models.losses import FocalLoss, SmoothL1Loss, binary_cross_entropy
from .axis_aligned_iou_loss import AxisAlignedIoULoss, axis_aligned_iou_loss
from .chamfer_distance import ChamferDistance, chamfer_distance
from .paconv_regularization_loss import PAConvRegularizationLoss

__all__ = [
    'FocalLoss', 'SmoothL1Loss', 'binary_cross_entropy', 'ChamferDistance',
    'chamfer_distance', 'axis_aligned_iou_loss', 'AxisAlignedIoULoss',
    'PAConvRegularizationLoss'
]
# mmdet.models.losses
# https://mmdetection3d.readthedocs.io/zh_CN/latest/api.html#module-mmdet3d.models.losses
""" 
mmdet3d.models.losses.AxisAlignedIoULoss(reduction='mean', loss_weight=1.0)
mmdet3d.models.losses.ChamferDistance(mode='l2', reduction='mean', loss_src_weight=1.0, loss_dst_weight=1.0)
mmdet3d.models.losses.FocalLoss(use_sigmoid=True, gamma=2.0, alpha=0.25, reduction='mean', loss_weight=1.0)
mmdet3d.models.losses.PAConvRegularizationLoss(reduction='mean', loss_weight=1.0)
mmdet3d.models.losses.SmoothL1Loss(beta=1.0, reduction='mean', loss_weight=1.0)
mmdet3d.models.losses.axis_aligned_iou_loss(pred, target)
mmdet3d.models.losses.binary_cross_entropy(pred, label, weight=None, reduction='mean', avg_factor=None, class_weight=None, ignore_index=- 100)
mmdet3d.models.losses.chamfer_distance(src, dst, src_weight=1.0, dst_weight=1.0, criterion_mode='l2', reduction='mean')


"""