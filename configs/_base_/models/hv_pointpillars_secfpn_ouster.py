'''
Description: 
Author: HCQ
Company(School): UCAS
Email: 1756260160@qq.com
Date: 2021-09-12 11:16:43
LastEditTime: 2021-10-23 18:21:45
FilePath: /mmdetection3d/configs/_base_/models/hv_pointpillars_secfpn_ouster.py
'''

# voxel_size = [0.16, 0.16, 4] # 长宽高
voxel_size = [0.16, 0.16, 8] # 长宽高 z高度要和point_cloud_range高度一致z:[-3, 10]
# RuntimeError: cannot perform reduction function max on tensor with no elements because the operation does not have an identity

model = dict(
    type='VoxelNet', # voxelnet.py模型名字  mmdet3d/models/detectors/__init__.py  mmdet3d/models/detectors/voxelnet.py
    voxel_layer=dict(
        max_num_points=32,  # max_points_per_voxel
        point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1],
        voxel_size=voxel_size,
        max_voxels=(16000, 40000)  # (training, testing) max_voxels
    ),
    # 'pillar_features'torch.Size([17842, 64])  # pillar特征（C, P）的Tensor，特征维度C=64，Pillar非空P=17842个
    voxel_encoder=dict(
        type='PillarFeatureNet',  # init--> from .pillar_encoder import PillarFeatureNet --> mmdet3d/models/voxel_encoders/pillar_encoder.py
        in_channels=4,
        feat_channels=[64],   # pillar特征（C, P）的Tensor，特征维度C=64，Pillar非空P=17842个
        with_distance=False,
        voxel_size=voxel_size,
        point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1]),
    middle_encoder=dict(
        type='PointPillarsScatter', in_channels=64, output_shape=[496, 432]), # 生成伪造图像，图像维度为(1,64,496,432)  mmdet3d/models/middle_encoders/pillar_scatter.py
    backbone=dict( # 调用注册器的backbone
        type='SECOND', # backbone名字 
        # conv_cfg=dict(type='DCN', bias=False) # 
        in_channels=64, #
        layer_nums=[3, 5, 5],
        layer_strides=[2, 2, 2],
        out_channels=[64, 128, 256]), # [64, 128, 256]
    neck=dict(
        type='SECONDFPN',
        in_channels=[64, 128, 256],
        upsample_strides=[1, 2, 4],
        out_channels=[128, 128, 128]),
    bbox_head=dict(
        type='Anchor3DHead',
        num_classes=3,
        in_channels=384,
        feat_channels=384,
        use_direction_classifier=True,
        anchor_generator=dict( # 生成anchor
            type='Anchor3DRangeGenerator',
            ranges=[
                [0, -39.68, -0.6, 70.4, 39.68, -0.6],
                [0, -39.68, -0.6, 70.4, 39.68, -0.6],
                [0, -39.68, -1.78, 70.4, 39.68, -1.78],
            ],
            sizes=[[0.6, 0.8, 1.73], [0.6, 1.76, 1.73], [1.6, 3.9, 1.56]],
            rotations=[0, 1.57],
            reshape_out=False),
        diff_rad_by_sin=True,
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
        # 分类loss，回归loss，朝向loss
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=2.0),
        loss_dir=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.2)),
    # model training and testing settings
    train_cfg=dict(
        assigner=[
            dict(  # for Pedestrian
                type='MaxIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                pos_iou_thr=0.5,
                neg_iou_thr=0.35,
                min_pos_iou=0.35,
                ignore_iof_thr=-1),
            dict(  # for Cyclist
                type='MaxIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                pos_iou_thr=0.5,
                neg_iou_thr=0.35,
                min_pos_iou=0.35,
                ignore_iof_thr=-1),
            dict(  # for Car
                type='MaxIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                pos_iou_thr=0.6,
                neg_iou_thr=0.45,
                min_pos_iou=0.45,
                ignore_iof_thr=-1),
        ],
        allowed_border=0,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        use_rotate_nms=True,
        nms_across_levels=False,
        nms_thr=0.01,
        score_thr=0.1,
        min_bbox_size=0,
        nms_pre=100,
        max_num=50))
