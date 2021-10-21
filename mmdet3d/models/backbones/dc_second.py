import warnings
from mmcv.cnn import build_conv_layer, build_norm_layer
from mmcv.runner import BaseModule
from torch import nn as nn

from mmdet.models import BACKBONES

@BACKBONES.register_module()
class DCSECOND(BaseModule):
    def __init__(self,
                 in_channels=128,
                 out_channels=[128, 128, 256], # 不起作用，base其作用，在最终配置的时候修改可以更改base配置
                 layer_nums=[3, 5, 5],
                 layer_strides=[2, 2, 2],
                 norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                 conv_cfg=dict(type='Conv2d', bias=False),
                 dcn_config=dict(type='DCN'),  # 初始化
                 init_cfg=None,
                 pretrained=None
                 ):

        assert init_cfg is None, 'For avoiding abnormal initialization, init_cfg is not allowed to be set'
        super(DCSECOND, self).__init__(init_cfg)

        assert len(layer_strides) == len(layer_nums)
        assert len(out_channels) == len(layer_nums)
        in_filters = [in_channels, *out_channels[:-1]]
        # note that when stride > 1, conv2d with same padding isn't
        # equal to pad-conv2d. we should use pad-conv2d.
        blocks = []
        for i, layer_num in enumerate(layer_nums): # 循环3次
            # 每个block包含(Conv2d BN ReLU)
            block = [
                # 卷积层(conv)
                build_conv_layer( # 来自 mmcv.cnn 
                    dcn_config,  # 调用DCN
                    in_filters[i],
                    out_channels[i],
                    3,
                    stride=layer_strides[i],
                    padding=1),
                # 归一化(BN)
                build_norm_layer(norm_cfg, out_channels[i])[1],
                # 非线性层(ReLU)
                nn.ReLU(inplace=True),
            ]
            for j in range(layer_num): # 分别循环3，5，5次layer_nums=[3, 5, 5],
                block.append(
                    build_conv_layer( # 卷积层(conv)、归一化(BN)、非线性层(ReLU)
                        conv_cfg, # conv_cfg=dict(type='Conv2d', bias=False),
                        out_channels[i],
                        out_channels[i],
                        3,
                        padding=1))
                block.append(build_norm_layer(norm_cfg, out_channels[i])[1])
                block.append(nn.ReLU(inplace=True))

            block = nn.Sequential(*block) # 没循环一次生成一个Sequential
            blocks.append(block) # 

        self.blocks = nn.ModuleList(blocks) # 保存到self.blocks

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
            x = self.blocks[i](x)
            outs.append(x) # backbone有三个输出
        return tuple(outs)