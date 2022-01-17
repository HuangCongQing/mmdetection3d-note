# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from mmcv.cnn import build_conv_layer, build_norm_layer # 
from mmcv.runner import BaseModule
from torch import nn as nn

from mmdet.models import BACKBONES

# 把下面second类注册成backbones
@BACKBONES.register_module()
class SECOND(BaseModule):
    """Backbone network for SECOND/PointPillars/PartA2/MVXNet.

    Args:
        in_channels (int): Input channels.
        out_channels (list[int]): Output channels for multi-scale feature maps.
        layer_nums (list[int]): Number of layers in each stage.
        layer_strides (list[int]): Strides of each stage.
        norm_cfg (dict): Config dict of normalization layers.
        conv_cfg (dict): Config dict of convolutional layers.
    """

    def __init__(self,
                 in_channels=128,
                 out_channels=[128, 128, 256],
                 layer_nums=[3, 5, 5],
                 layer_strides=[2, 2, 2],
                 norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                 conv_cfg=dict(type='Conv2d', bias=False),  # 对应backbone Conv2d  默认？？  configs/_base_/models/hv_pointpillars_secfpn_kitti.py
                 init_cfg=None,
                 pretrained=None):
        super(SECOND, self).__init__(init_cfg=init_cfg)
        assert len(layer_strides) == len(layer_nums) # 3 
        assert len(out_channels) == len(layer_nums) # 3 

        in_filters = [in_channels, *out_channels[:-1]]
        # note that when stride > 1, conv2d with same padding isn't
        # equal to pad-conv2d. we should use pad-conv2d.
        blocks = [] # append的block
        for i, layer_num in enumerate(layer_nums):
            block = [
                build_conv_layer(
                    conv_cfg,
                    in_filters[i],
                    out_channels[i],
                    3,
                    stride=layer_strides[i],
                    padding=1),
                build_norm_layer(norm_cfg, out_channels[i])[1],
                nn.ReLU(inplace=True),
            ]
            for j in range(layer_num):
                block.append(
                    build_conv_layer(
                        conv_cfg,
                        out_channels[i],
                        out_channels[i],
                        3,
                        padding=1))
                block.append(build_norm_layer(norm_cfg, out_channels[i])[1])
                block.append(nn.ReLU(inplace=True))

            block = nn.Sequential(*block) # 
            blocks.append(block)

        self.blocks = nn.ModuleList(blocks) # 存放在

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be setting at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is a deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        else:
            self.init_cfg = dict(type='Kaiming', layer='Conv2d')

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): Input with shape (N, C, H, W).

        Returns:
            tuple[torch.Tensor]: Multi-scale features.
        """
        outs = []
        for i in range(len(self.blocks)):
            # print(self.blocks[i]) # 
            # print(x.shape)          # torch.Size([1, 64, 496, 432]) torch.Size([1, 64, 248, 216]) torch.Size([1, 128, 124, 108])
            x = self.blocks[i](x)
            # print(x.shape)          # torch.Size([1, 64, 248, 216]) torch.Size([1, 128, 124, 108]) torch.Size([1, 256, 62, 54])
            outs.append(x)
        return tuple(outs)  # 输出元组类型
