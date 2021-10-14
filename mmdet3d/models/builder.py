# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from mmcv.cnn import MODELS as MMCV_MODELS
from mmcv.utils import Registry

from mmdet.models.builder import (BACKBONES, DETECTORS, HEADS, LOSSES, NECKS,
                                  ROI_EXTRACTORS, SHARED_HEADS)
from mmseg.models.builder import SEGMENTORS

MODELS = Registry('models', parent=MMCV_MODELS)

VOXEL_ENCODERS = MODELS
MIDDLE_ENCODERS = MODELS
FUSION_LAYERS = MODELS


def build_backbone(cfg):
    """Build backbone."""
    return BACKBONES.build(cfg)


def build_neck(cfg):
    """Build neck."""
    return NECKS.build(cfg)


def build_roi_extractor(cfg):
    """Build RoI feature extractor."""
    return ROI_EXTRACTORS.build(cfg)


def build_shared_head(cfg):
    """Build shared head of detector."""
    return SHARED_HEADS.build(cfg)


def build_head(cfg):
    """Build head."""
    return HEADS.build(cfg)


def build_loss(cfg):
    """Build loss function."""
    return LOSSES.build(cfg)

# {'type': 'VoxelNet', 'voxel_layer': {'max_num_points': 32, 'point_cloud_range': [0, -39.68, -3, 69.12, 39.68, 1], 'voxel_size': [0.16, 0.16, 4], 'max_voxels': (16000, 40000)}, 'voxel_encoder': {'type': 'PillarFeatureNet', 'in_channels': 4, 'feat_channels': [64], 'with_distance': False, 'voxel_size': [0.16, 0.16, 4], 'point_cloud_range': [0, -39.68, -3, 69.12, 39.68, 1]}, 'middle_encoder': {'type': 'PointPillarsScatter', 'in_channels': 64, 'output_shape': [496, 432]}, 'backbone': {'type': 'DCSECOND', 'in_channels': 64, 'layer_nums': [3, 5, 5], 'layer_strides': [2, 2, 2], 'out_channels': [64, 128, 256]}, 'neck': {'type': 'SECONDFPN', 'in_channels': [64, 128, 256], 'upsample_strides': [1, 2, 4], 'out_channels': [128, 128, 128]}, 'bbox_head': {'type': 'Anchor3DHead', 'num_classes': 3, 'in_channels': 384, 'feat_channels': 384, 'use_direction_classifier': True, 'anchor_generator': {'type': 'Anchor3DRangeGenerator', 'ranges': [[0, -39.68, -0.6, 70.4, 39.68, -0.6], [0, -39.68, -0.6, 70.4, 39.68, -0.6], [0, -39.68, -1.78, 70.4, 39.68, -1.78]], 'sizes': [[0.6, 0.8, 1.73], [0.6, 1.76, 1.73], [1.6, 3.9, 1.56]], 'rotations': [0, 1.57], 'reshape_out': False}, 'diff_rad_by_sin': True, 'bbox_coder': {'type': 'DeltaXYZWLHRBBoxCoder'}, 'loss_cls': {'type': 'FocalLoss', 'use_sigmoid': True, 'gamma': 2.0, 'alpha': 0.25, 'loss_weight': 1.0}, 'loss_bbox': {'type': 'SmoothL1Loss', 'beta': 0.1111111111111111, 'loss_weight': 2.0}, 'loss_dir': {'type': 'CrossEntropyLoss', 'use_sigmoid': False, 'loss_weight': 0.2}, 'train_cfg': {'assigner': [{'type': 'MaxIoUAssigner', 'iou_calculator': {'type': 'BboxOverlapsNearest3D'}, 'pos_iou_thr': 0.5, 'neg_iou_thr': 0.35, 'min_pos_iou': 0.35, 'ignore_iof_thr': -1}, {'type': 'MaxIoUAssigner', 'iou_calculator': {'type': 'BboxOverlapsNearest3D'}, 'pos_iou_thr': 0.5, 'neg_iou_thr': 0.35, 'min_pos_iou': 0.35, 'ignore_iof_thr': -1}, {'type': 'MaxIoUAssigner', 'iou_calculator': {'type': 'BboxOverlapsNearest3D'}, 'pos_iou_thr': 0.6, 'neg_iou_thr': 0.45, 'min_pos_iou': 0.45, 'ignore_iof_thr': -1}], 'allowed_border': 0, 'pos_weight': -1, 'debug': False}, 'test_cfg': {'use_rotate_nms': True, 'nms_across_levels': False, 'nms_thr': 0.01, 'score_thr': 0.1, 'min_bbox_size': 0, 'nms_pre': 100, 'max_num': 50}}, 'train_cfg': {'assigner': [{'type': 'MaxIoUAssigner', 'iou_calculator': {'type': 'BboxOverlapsNearest3D'}, 'pos_iou_thr': 0.5, 'neg_iou_thr': 0.35, 'min_pos_iou': 0.35, 'ignore_iof_thr': -1}, {'type': 'MaxIoUAssigner', 'iou_calculator': {'type': 'BboxOverlapsNearest3D'}, 'pos_iou_thr': 0.5, 'neg_iou_thr': 0.35, 'min_pos_iou': 0.35, 'ignore_iof_thr': -1}, {'type': 'MaxIoUAssigner', 'iou_calculator': {'type': 'BboxOverlapsNearest3D'}, 'pos_iou_thr': 0.6, 'neg_iou_thr': 0.45, 'min_pos_iou': 0.45, 'ignore_iof_thr': -1}], 'allowed_border': 0, 'pos_weight': -1, 'debug': False}, 'test_cfg': {'use_rotate_nms': True, 'nms_across_levels': False, 'nms_thr': 0.01, 'score_thr': 0.1, 'min_bbox_size': 0, 'nms_pre': 100, 'max_num': 50}}
def build_detector(cfg, train_cfg=None, test_cfg=None):
    """Build detector."""
    if train_cfg is not None or test_cfg is not None:
        warnings.warn(
            'train_cfg and test_cfg is deprecated, '
            'please specify them in model', UserWarning)
    assert cfg.get('train_cfg') is None or train_cfg is None, \
        'train_cfg specified in both outer field and model field '
    assert cfg.get('test_cfg') is None or test_cfg is None, \
        'test_cfg specified in both outer field and model field '
    return DETECTORS.build(
        cfg, default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg))


def build_segmentor(cfg, train_cfg=None, test_cfg=None):
    """Build segmentor."""
    if train_cfg is not None or test_cfg is not None:
        warnings.warn(
            'train_cfg and test_cfg is deprecated, '
            'please specify them in model', UserWarning)
    assert cfg.get('train_cfg') is None or train_cfg is None, \
        'train_cfg specified in both outer field and model field '
    assert cfg.get('test_cfg') is None or test_cfg is None, \
        'test_cfg specified in both outer field and model field '
    return SEGMENTORS.build(
        cfg, default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg))


def build_model(cfg, train_cfg=None, test_cfg=None):
    """A function warpper for building 3D detector or segmentor according to
    cfg.

    Should be deprecated in the future.
    """
    if cfg.type in ['EncoderDecoder3D']:
        return build_segmentor(cfg, train_cfg=train_cfg, test_cfg=test_cfg)
    else:
        return build_detector(cfg, train_cfg=train_cfg, test_cfg=test_cfg)

# cfg: {'type': 'PillarFeatureNet', 'in_channels': 4, 'feat_channels': [64], 'with_distance': False, 'voxel_size': [0.16, 0.16, 4], 'point_cloud_range': [0, -39.68, -3, 69.12, 39.68, 1]}
def build_voxel_encoder(cfg):
    """Build voxel encoder."""
    return VOXEL_ENCODERS.build(cfg)


def build_middle_encoder(cfg):
    """Build middle level encoder."""
    return MIDDLE_ENCODERS.build(cfg)


def build_fusion_layer(cfg):
    """Build fusion layer."""
    return FUSION_LAYERS.build(cfg)
