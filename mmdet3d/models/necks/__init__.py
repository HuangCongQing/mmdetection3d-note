'''
Description: 
Author: HCQ
Company(School): UCAS
Email: 1756260160@qq.com
Date: 2021-10-06 23:06:07
LastEditTime: 2021-10-19 15:52:00
FilePath: /mmdetection3d/mmdet3d/models/necks/__init__.py
'''
# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.models.necks.fpn import FPN
from .imvoxel_neck import OutdoorImVoxelNeck
from .second_fpn import SECONDFPN
from .second_fpn_multi import  SECONDFPNMULTI
# from .cbam import CBAM
__all__ = ['FPN', 'SECONDFPN', 'OutdoorImVoxelNeck', 'SECONDFPNMULTI']
