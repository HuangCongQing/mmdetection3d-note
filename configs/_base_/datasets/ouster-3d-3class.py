# dataset settings
dataset_type = 'OusterDataset' # # 数据集类型  mmdet3d/datasets/ouster_dataset.py
data_root = 'data/ouster/' # # 数据路径
class_names = ['Pedestrian', 'Cyclist', 'Car'] # 类的名称
point_cloud_range = [0, -40, -3, 70.4, 40, 1]
input_modality = dict(use_lidar=True, use_camera=False)
# db_sampler = dict( # mmdet3d/datasets/pipelines/dbsampler.py
#     data_root=data_root,
#     info_path=data_root + 'ouster_dbinfos_train.pkl', # 
#     rate=1.0,
#     prepare=dict(
#         filter_by_difficulty=[-1],
#         filter_by_min_points=dict(Car=5, Pedestrian=10, Cyclist=10)),
#     classes=class_names,
#     sample_groups=dict(Car=12, Pedestrian=6, Cyclist=6))

file_client_args = dict(backend='disk')
# Uncomment the following if use ceph or other file clients.
# See https://mmcv.readthedocs.io/en/latest/api.html#mmcv.fileio.FileClient
# for more details.
# file_client_args = dict(
#     backend='petrel', path_mapping=dict(data='s3://ouster_data/'))

# train的配置文件  # 训练流水线，更多细节请参考 mmdet3d.datasets.pipelines
train_pipeline = [
    dict(
        type='LoadPointsFromFile',  # mmdet3d/datasets/pipelines/loading.py 第一个流程，用于读取点，更多细节请参考 mmdet3d.datasets.pipelines.indoor_loading
        coord_type='LIDAR',  # 雷达数据
        load_dim=4,  # 读取的点的维度   x,y,z,r
        use_dim=4, # 使用所读取点的哪些维度
        file_client_args=file_client_args),
    dict(
        type='LoadAnnotations3D',  # # mmdet3d/datasets/pipelines/loading.py 第二个流程，用于读取标注GT，更多细节请参考 mmdet3d.datasets.pipelines.indoor_loading
        with_bbox_3d=True,  # 是否读取 3D 框
        with_label_3d=True,# 是否读取 3D 框对应的类别标签
        file_client_args=file_client_args),
    # dict(type='ObjectSample', db_sampler=db_sampler),
    dict(
        type='ObjectNoise',
        num_try=100,
        translation_std=[1.0, 1.0, 0.5],
        global_rot_range=[0.0, 0.0],
        rot_range=[-0.78539816, 0.78539816]),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5), # 数据增广流程，随机翻转点和 3D 框
    dict(
        type='GlobalRotScaleTrans',  # 数据增广流程，旋转并放缩点和 3D 框，更多细节请参考 mmdet3d.datasets.pipelines.indoor_augment
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05]),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])  # 最后一个流程，mmdet3d/datasets/pipelines/formating.py 决定哪些键值对应的数据会被输入给检测器，更多细节请参考 mmdet3d.datasets.pipelines.formating
]
# # 测试流水线，更多细节请参考 mmdet3d.datasets.pipelines
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        file_client_args=file_client_args),
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
# construct a pipeline for data and gt loading in show function
# please keep its loading function consistent with test_pipeline (e.g. client)
# # 模型验证或可视化所使用的流水线，更多细节请参考 mmdet3d.datasets.pipelines
eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        file_client_args=file_client_args),
    dict(
        type='DefaultFormatBundle3D',
        class_names=class_names,
        with_label=False),
    dict(type='Collect3D', keys=['points'])
]

data = dict(
    samples_per_gpu=6, # 单张 GPU 上的样本数
    workers_per_gpu=4, # 每张 GPU 上用于读取数据的进程数
    train=dict(  # 训练数据集配置
        type='RepeatDataset',  # 数据集嵌套，更多细节请参考 https://github.com/open-mmlab/mmdetection/blob/master/mmdet/datasets/dataset_wrappers.py
        times=2,  # 重复次数
        dataset=dict(
            type=dataset_type, # ousterDataset # 数据集类型
            data_root=data_root, # data_root = 'data/ouster/'
            ann_file=data_root + 'ouster_infos_train.pkl',
            split='training',
            pts_prefix='velodyne_reduced',
            pipeline=train_pipeline, # # 流水线，这里传入的就是上面创建的训练流水线变量 train_pipeline = [] 上面有配置文件 
            modality=input_modality,
            classes=class_names, # 类别名称
            test_mode=False,
            # we use box_type_3d='LiDAR' in ouster and nuscenes dataset
            # and box_type_3d='Depth' in sunrgbd and scannet dataset.
            box_type_3d='LiDAR')),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'ouster_infos_val.pkl',
        split='training',
        pts_prefix='velodyne_reduced',
        pipeline=test_pipeline,
        modality=input_modality,
        classes=class_names,
        test_mode=True,
        box_type_3d='LiDAR'),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'ouster_infos_val.pkl',
        split='training',
        pts_prefix='velodyne_reduced',
        pipeline=test_pipeline,
        modality=input_modality,
        classes=class_names,
        test_mode=True,
        box_type_3d='LiDAR'))
# 流水线，这里传入的就是上面创建的验证流水线变量
evaluation = dict(interval=1, pipeline=eval_pipeline)
