_base_ = [
    '../_base_/datasets/ouster-3d-3class.py', #dataset
    '../_base_/models/centerpoint_01voxel_second_secfpn_kitti.py', #model base
    '../_base_/schedules/cyclic_40e.py', #training schedule
    '../_base_/default_runtime.py' #running
]

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
# point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
# point_cloud_range = [0, -40, -3, 70.4, 40, 1]
# # For nuScenes we usually do 10-class detection
# class_names = [
#     'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
#     'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
# ]

# class_names = ['Pedestrian', 'Cyclist', 'Car']
class_names =  ('Truck','Auxiliary','Car','Excavator','Widebody','Pedestrian','Others')
point_cloud_range =  [0, -39.68, -3, 70.4, 39.68, 7.5]

model = dict(
    type='CenterPoint',
    pts_voxel_layer=dict(point_cloud_range=point_cloud_range),
    pts_voxel_encoder=dict(point_cloud_range=point_cloud_range),
    # pts_backbone=dict(type='DCSECOND', dcn_config=dict(type='DCN')),
    pts_bbox_head=dict(
        # type='DCenterHead',
        #                separate_head=dict(separate_head=dict(
        #                                 type='DCNSeparateHead',
        #                                 dcn_config=dict(
        #                                     type='DCN',
        #                                     in_channels=64,
        #                                     out_channels=64,
        #                                     kernel_size=3,
        #                                     padding=1,
        #                                     groups=4),
        #                                 init_bias=-2.19,
        #                                 final_kernel=3)),
                        # second_loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
                        # second_loss_reg=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
                        tasks=[
                            dict(num_class=1, class_names=['Truck']),
                            dict(num_class=1, class_names=['Car']),
                            dict(num_class=1, class_names=['Auxiliary'])
                        ],
                       bbox_coder=dict(pc_range=point_cloud_range[:2],
                                       code_size=7),
                       # loss_bbox=dict(type='BalancedL1Loss_GZ', reduction='none', loss_weight=0.25)
                       ),
    # model training and testing settings
    train_cfg=dict(pts=dict(point_cloud_range=point_cloud_range,
                            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])),
    test_cfg=dict(pts=dict(pc_range=point_cloud_range[:2]))
)

#dataset_type = 'NuScenesDataset'
#data_root = 'data/nuscenes/'
dataset_type = 'OusterDataset'
data_root = "data/ouster/"
file_client_args = dict(backend='disk')

# PointPillars uses different augmentation hyper parameters

# db_sampler = dict(
#     data_root=data_root,
#     info_path=data_root + 'kitti_dbinfos_train.pkl',
#     rate=1.0,
#     prepare=dict(
#     filter_by_difficulty=[-1],
#     filter_by_min_points=dict(Car=5, Pedestrian=10, Cyclist=10)),
#     classes=class_names,
#     sample_groups=dict(Car=15, Pedestrian=10, Cyclist=10))


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

data = dict(samples_per_gpu=8,
    train=dict(times=8, dataset=dict(pipeline=train_pipeline, classes=class_names)),
    val=dict(pipeline=test_pipeline, classes=class_names),
    test=dict(pipeline=test_pipeline, classes=class_names))

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
runner = dict(max_epochs=80)

# Use evaluation interval=2 reduce the number of evaluation timese
evaluation = dict(interval=2)
