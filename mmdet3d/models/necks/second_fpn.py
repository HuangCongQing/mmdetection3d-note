# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
from mmcv.cnn import build_conv_layer, build_norm_layer, build_upsample_layer
from mmcv.runner import BaseModule, auto_fp16
from torch import nn as nn

from mmdet.models import NECKS


@NECKS.register_module()
class SECONDFPN(BaseModule):
    """FPN used in SECOND/PointPillars/PartA2/MVXNet.

    Args:
        in_channels (list[int]): Input channels of multi-scale feature maps.
        out_channels (list[int]): Output channels of feature maps.
        upsample_strides (list[int]): Strides used to upsample the
            feature maps.
        norm_cfg (dict): Config dict of normalization layers.
        upsample_cfg (dict): Config dict of upsample layers.
        conv_cfg (dict): Config dict of conv layers.
        use_conv_for_no_stride (bool): Whether to use conv when stride is 1.
    """

    def __init__(self,
                 in_channels=[128, 128, 256],
                 out_channels=[256, 256, 256], # 最终输出的维度相同
                 upsample_strides=[1, 2, 4], # 分别将backbone的3. 分别将三个下采样特征上采样至相同大小
                 norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                 upsample_cfg=dict(type='deconv', bias=False),
                 conv_cfg=dict(type='Conv2d', bias=False),
                 use_conv_for_no_stride=False,
                 init_cfg=None):
        # if for GroupNorm,
        # cfg is dict(type='GN', num_groups=num_groups, eps=1e-3, affine=True)
        super(SECONDFPN, self).__init__(init_cfg=init_cfg)
        assert len(out_channels) == len(upsample_strides) == len(in_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fp16_enabled = False

        deblocks = []
        for i, out_channel in enumerate(out_channels):
            stride = upsample_strides[i] # [1, 2, 4]
            # build_upsample_layer
            if stride > 1 or (stride == 1 and not use_conv_for_no_stride):
                upsample_layer = build_upsample_layer( # build_upsample_layer上采样
                    upsample_cfg,# 上采样
                    in_channels=in_channels[i],
                    out_channels=out_channel,
                    kernel_size=upsample_strides[i], # 
                    stride=upsample_strides[i])
            else: # 正常卷积
                stride = np.round(1 / stride).astype(np.int64) # np.round()函数的作用：对给定的数组进行四舍五入
                upsample_layer = build_conv_layer( # 正常卷积
                    conv_cfg, # 正常卷积
                    in_channels=in_channels[i],
                    out_channels=out_channel,
                    kernel_size=stride,
                    stride=stride)

            deblock = nn.Sequential(upsample_layer,
                                    build_norm_layer(norm_cfg, out_channel)[1],
                                    nn.ReLU(inplace=True))
            deblocks.append(deblock)
        self.deblocks = nn.ModuleList(deblocks) # 保存到ModuleList
        # 初始化参数
        if init_cfg is None:
            self.init_cfg = [
                dict(type='Kaiming', layer='ConvTranspose2d'),
                dict(type='Constant', layer='NaiveSyncBatchNorm2d', val=1.0)
            ]

    @auto_fp16()
    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): 4D Tensor in (N, C, H, W) shape.
            x = {tuple:3}
            0 = {6,C, H/2, W/2 }= (6,64,248,216)
            1 = {6,2C, H/4, W/4 }
            2 = {6,4C, H/8, W/8 }

        Returns:
            list[torch.Tensor]: Multi-level feature maps.
        """
        # print('-----------------------')
        # print(len(x))
        # print(x[0].shape)
        # print(x[1].shape)
        # print(x[2].shape)
        assert len(x) == len(self.in_channels)
        np.save('input_data.npy', x[0].cpu().detach().numpy())
        ups = [deblock(x[i]) for i, deblock in enumerate(self.deblocks)] # self.deblocks = nn.ModuleList(deblocks)
        # print(len(ups))
        # print(ups[0].shape)
        # print(ups[1].shape)
        # print(ups[2].shape)
        # ups = {list:3}
        # 0=(6,128,248,216) = (6, 2C, W/2,H/2)
        # 1=(6,128,248,216) = (6, 2C, W/2,H/2)
        # 1=(6,128,248,216) = (6, 2C, W/2,H/2)

        if len(ups) > 1:
            out = torch.cat(ups, dim=1) #  concat结合 Tensor: (6, 6C, W/2,H/2)
            # print(out.shape)
        else:
            out = ups[0]
        np.save('output_data.npy', out[0].cpu().detach().numpy())
        return [out] # {list: 1}[ Tensor: (6, 6C, W/2,H/2)]. 对三个特征图进行上采样至相同大小，然后进行concatenation
