voxel_size = [0.16, 0.16, 10.5]
# point_cloud_range = [0, -40, -3, 70.4, 40, 1]

model = dict(
    type='CenterPoint',
    pts_voxel_layer=dict(
            max_num_points=20, voxel_size=voxel_size, max_voxels=(30000, 40000), point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 7.5]
            ),
    pts_voxel_encoder=dict(
            type='PillarFeatureNet',
            in_channels=4,
            feat_channels=[64],
            with_distance=False,
            voxel_size=voxel_size,
            point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 7.5]
            ),
    pts_middle_encoder=dict(
            type='PointPillarsScatter', in_channels=64, output_shape=[496, 432]),
    pts_backbone=dict( # 调用注册器的backbone
        type='SECOND', # backbone名字
        # conv_cfg=dict(type='DCN', bias=False) #
        in_channels=64, #
        layer_nums=[3, 5, 5],
        layer_strides=[2, 2, 2],
        out_channels=[64, 128, 256]),
    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[64, 128, 256],
        upsample_strides=[1, 2, 4],
        out_channels=[128, 128, 128]),
    pts_bbox_head=dict(
        type='CenterHead',
        in_channels=sum([128, 128, 128]),
        tasks=[
            dict(num_class=1, class_names=['Pedestrian']),
            dict(num_class=1, class_names=['Cyclist']),
            dict(num_class=1, class_names=['Car'])
        ],
        common_heads=dict(
            reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2)),#, vel=(2, 2)),
        share_conv_channel=64,
        bbox_coder=dict(
            type='CenterPointBBoxCoder',
            post_center_range=[0, -39.68, -3, 69.12, 39.68, 7.5],
            max_num=500,
            score_threshold=0.1,
            out_size_factor=2,
            voxel_size=voxel_size[:2],
            code_size=7),
        separate_head=dict(
            type='SeparateHead', init_bias=-2.19, final_kernel=3),
        loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
        norm_bbox=True),
    # model training and testing settings
    train_cfg=dict(
        pts=dict(
            grid_size=[432, 496, 1],
            voxel_size=voxel_size,
            out_size_factor=2,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])),
    test_cfg=dict(
        pts=dict(
            post_center_limit_range=[0, -39.68, -3, 69.12, 39.68, 7.5],
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[2, 0.85, 0.175],
            score_threshold=0.1,
            out_size_factor=2,
            voxel_size=voxel_size[:2],
            nms_type='rotate',
            pre_max_size=1000,
            post_max_size=83,
            nms_thr=0.2)))
