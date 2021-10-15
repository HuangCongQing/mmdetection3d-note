'''
Description: 
Author: HCQ
Company(School): UCAS
Email: 1756260160@qq.com
Date: 2021-09-12 11:16:43
LastEditTime: 2021-10-15 19:35:11
FilePath: /mmdetection3d/configs/pointpillars/01dcn_hv_pointpillars_secfpn_6x8_160e_ouster-3d-3class.py
'''
_base_ = [
    '../_base_/models/hv_pointpillars_secfpn_kitti.py', # pointpillars模型配置  configs/_base_/models/hv_pointpillars_secfpn_kitti.py
    '../_base_/datasets/ouster-3d-3class.py', # ouster数据集
    '../_base_/schedules/cyclic_40e.py', '../_base_/default_runtime.py'
]

# 1 数据集 (dataset)  data = dict()================================================================================
point_cloud_range = [0, -39.68, -3, 69.12, 39.68, 1]
# dataset settings 数据集配置
# data_root = 'data/ouster/'
data_root = 'data/kittiTest/'
class_names = ['Pedestrian', 'Cyclist', 'Car']
# CLASSES =  ('Truck','Car','Pedestrian','Excavator','Widebody','Auxiliary')
# PointPillars adopted a different sampling strategies among classes
# db_sampler = dict(
#     data_root=data_root,
#     info_path=data_root + 'kitti_dbinfos_train.pkl',
#     rate=1.0,
#     prepare=dict(
#         filter_by_difficulty=[-1],
#         filter_by_min_points=dict(Car=5, Pedestrian=10, Cyclist=10)),
#     classes=class_names,
#     sample_groups=dict(Car=15, Pedestrian=10, Cyclist=10))

# PointPillars uses different augmentation hyper parameters
train_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    # dict(type='ObjectSample', db_sampler=db_sampler),
    dict(
        type='ObjectNoise',
        num_try=100,
        translation_std=[0.25, 0.25, 0.25],
        global_rot_range=[0.0, 0.0],
        rot_range=[-0.15707963267, 0.15707963267]),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05]),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(
                type='PointsRangeFilter', point_cloud_range=point_cloud_range),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ])
]

data = dict(
    train=dict(dataset=dict(pipeline=train_pipeline, classes=class_names)),
    val=dict(pipeline=test_pipeline, classes=class_names),
    test=dict(pipeline=test_pipeline, classes=class_names))

#  模型 (nodel) 个人添加======================================================================


model = dict(
    backbone=dict(
        # _delete_=True,  # 覆盖基础配置文件里的部分内容
        type='DCSECOND',  # # 优化1：代替原来的SECOND
        in_channels=64,
        layer_nums=[3, 5, 5],
        layer_strides=[2, 2, 2],
        # out_channels=[128, 128, 256], # [128, 128, 256] 覆盖的是_base_ 的 configs/_base_/models/hv_pointpillars_secfpn_kitti.py [64, 128, 256]
        # dcn_config=dict(type='DCN'), # 可加可不加，初始就是DCN
        # stage_with_dcn=(False, False, True, True), #报错  DCSECOND: __init__() got an unexpected keyword argument 'stage_with_dcn'
    ),
    
    # 优化2：多检测融合（无效果）
    # neck=dict(
    # _delete_=True,  # 覆盖基础配置文件里的部分内容
    #     type='SECONDFPNMULTI', # 优化
    #     in_channels=[64, 128, 256],
    #     upsample_strides=[1, 2, 4],
    #     out_channels=[128, 128, 128]),
)

#  训练策略 (schedule) ======================================================================
# In practice PointPillars also uses a different schedule
# optimizer
lr = 0.001
optimizer = dict(lr=lr)
# max_norm=35 is slightly better than 10 for PointPillars in the earlier
# development of the codebase thus we keep the setting. But we does not
# specifically tune this parameter.
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# PointPillars usually need longer schedule than second, we simply double
# the training schedule. Do remind that since we use RepeatDataset and
# repeat factor is 2, so we actually train 160 epochs.
runner = dict(max_epochs=80) # 80个epochs

# Use evaluation interval=2 reduce the number of evaluation timese 每隔2轮评测一次
evaluation = dict(interval=2) # 参数
