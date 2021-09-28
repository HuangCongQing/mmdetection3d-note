# https://mmdetection3d.readthedocs.io/zh_CN/latest/tutorials/config.html#votenet
model = dict(
    type='VoteNet',  # 检测器的类型，更多细节请参考 mmdet3d.models.detectors
    backbone=dict(
        type='PointNet2SASSG', # 主干网络的类型，更多细节请参考 mmdet3d.models.backbones
        in_channels=4,  # 点云输入通道数
        num_points=(2048, 1024, 512, 256),  # 每个 SA 模块采样的中心点的数量
        radius=(0.2, 0.4, 0.8, 1.2), # 每个 SA 层的半径
        num_samples=(64, 32, 16, 16),  # 每个 SA 层聚集的点的数量
        sa_channels=((64, 64, 128), (128, 128, 256), (128, 128, 256),
                     (128, 128, 256)),  # SA 模块中每个多层感知器的输出通道数
        fp_channels=((256, 256), (256, 256)),  # FP 模块中每个多层感知器的输出通道数
        norm_cfg=dict(type='BN2d'),  # 归一化层的配置
        sa_cfg=dict(   # 点集抽象 (SA) 模块的配置
            type='PointSAModule',
            pool_mod='max',
            use_xyz=True,
            normalize_xyz=True)),
    bbox_head=dict(
        type='VoteHead',   # 检测框头的类型，更多细节请参考 mmdet3d.models.dense_heads
        vote_module_cfg=dict(
            in_channels=256,
            vote_per_seed=1,
            gt_per_seed=3,
            conv_channels=(256, 256),
            conv_cfg=dict(type='Conv1d'),
            norm_cfg=dict(type='BN1d'),
            norm_feats=True,
            vote_loss=dict(
                type='ChamferDistance',
                mode='l1',
                reduction='none',
                loss_dst_weight=10.0)),
        vote_aggregation_cfg=dict(
            type='PointSAModule',
            num_point=256,
            radius=0.3,
            num_sample=16,
            mlp_channels=[256, 128, 128, 128],
            use_xyz=True,
            normalize_xyz=True),
        pred_layer_cfg=dict(
            in_channels=128, shared_conv_channels=(128, 128), bias=True),
        conv_cfg=dict(type='Conv1d'),  # 卷积的配置
        norm_cfg=dict(type='BN1d'),   # 归一化层的配置
        # loss
        objectness_loss=dict(  # 物体性 (objectness) 损失函数的配置
            type='CrossEntropyLoss', # 损失函数类型
            class_weight=[0.2, 0.8], # 损失函数对每一类的权重
            reduction='sum',
            loss_weight=5.0),
        center_loss=dict(   # 中心 (center) 损失函数的配置
            type='ChamferDistance',
            mode='l2',
            reduction='sum',
            loss_src_weight=10.0,
            loss_dst_weight=10.0),
        dir_class_loss=dict(   # 方向分类损失函数的配置
            type='CrossEntropyLoss', reduction='sum', loss_weight=1.0),
        dir_res_loss=dict(   # 方向残差 (residual) 损失函数的配置
            type='SmoothL1Loss', reduction='sum', loss_weight=10.0),
        size_class_loss=dict(  # 尺寸分类损失函数的配置
            type='CrossEntropyLoss', reduction='sum', loss_weight=1.0),
        size_res_loss=dict(  # 尺寸残差损失函数的配置
            type='SmoothL1Loss', reduction='sum', loss_weight=10.0 / 3.0),
        semantic_loss=dict(  # 语义损失函数的配置
            type='CrossEntropyLoss', reduction='sum', loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(  # VoteNet 训练的超参数配置
        pos_distance_thr=0.3,    # 距离 >= 0.3 阈值的样本将被视为正样本
        neg_distance_thr=0.6,    # 距离 < 0.6 阈值的样本将被视为负样本
        sample_mod='vote'), # 采样方法的模式
    test_cfg=dict(  # VoteNet 测试的超参数配置
        sample_mod='seed',  # 采样方法的模式
        nms_thr=0.25,      # NMS 中使用的阈值
        score_thr=0.05,  # 剔除框的阈值
        per_class_proposal=True))  # 是否使用逐类提议框 (proposal)
