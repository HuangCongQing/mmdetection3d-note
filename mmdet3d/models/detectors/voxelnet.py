# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.runner import force_fp32
from torch.nn import functional as F

from mmdet3d.core import bbox3d2result, merge_aug_bboxes_3d
from mmdet3d.ops import Voxelization
from mmdet.models import DETECTORS
from .. import builder # mmdet3d/models/builder.py
from .single_stage import SingleStage3DDetector


@DETECTORS.register_module()
class VoxelNet(SingleStage3DDetector):
    r"""`VoxelNet <https://arxiv.org/abs/1711.06396>`_ for 3D detection."""

    def __init__(self,
                 voxel_layer,
                 voxel_encoder,
                 middle_encoder,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 pretrained=None):
        super(VoxelNet, self).__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
            pretrained=pretrained)
        self.voxel_layer = Voxelization(**voxel_layer) # 
        self.voxel_encoder = builder.build_voxel_encoder(voxel_encoder) # PillarFeatureNet.py
        self.middle_encoder = builder.build_middle_encoder(middle_encoder) # PointPillarsScatter.py
    # 提取特征（feature, backbone,neck）
    def extract_feat(self, points, img_metas=None):
        """Extract features from points."""
        voxels, num_points, coors = self.voxelize(points)
        voxel_features = self.voxel_encoder(voxels, num_points, coors) # step1
        batch_size = coors[-1, 0].item() + 1
        x = self.middle_encoder(voxel_features, coors, batch_size) # step2 中间encoder  
        x = self.backbone(x) # backbone   step3
        if self.with_neck:
            x = self.neck(x) # neck pointpillars  # step4
        return x

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        """Apply hard voxelization to points."""
        voxels, coors, num_points = [], [], []
        for res in points:
            res_voxels, res_coors, res_num_points = self.voxel_layer(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return voxels, num_points, coors_batch
    # 继承自mmdet3d/models/detectors/base.py中的forward中的forward_train()
    def forward_train(self, 
                      points,
                      img_metas,
                      gt_bboxes_3d,
                      gt_labels_3d,
                      gt_bboxes_ignore=None):
        """Training forward function.

        Args:
            points (list[torch.Tensor]): Point cloud of each sample.
            img_metas (list[dict]): Meta information of each sample
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.

        Returns:
            dict: Losses of each branch.
        """
        x = self.extract_feat(points, img_metas) # 提取特征（feature, backbone,neck）
        
        # print(len(x))
        # print(x[0].shape)
        outs = self.bbox_head(x) # 检测头

        # print(len(outs)) # 3
        # print(len(outs[0])) # 1
        # print(outs[0][0].shape)  # torch.Size([6, 2, 248, 216])     6为batch size
        # print(outs[1][0].shape)  # torch.Size([6, 14, 248, 216])
        # print(outs[2][0].shape)  # torch.Size([6, 4, 248, 216])
        
        # 同anchor3d_head.py里的 cls_score, bbox_pred, dir_cls_preds
        # python tools/train.py configs/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car.py   --gpu-ids 4
        # 输入x len为1
        # torch.Size([6, 384, 248, 216])
        # 输出 len为3
        # torch.Size([6, 2, 248, 216])
        # torch.Size([6, 14, 248, 216])
        # torch.Size([6, 4, 248, 216])
        
        
        # python tools/train.py configs/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class.py   --gpu-ids 4 
        # 输入x len为1
        # torch.Size([6, 384, 248, 216])
        # 输出 len为3
        # torch.Size([6, 18, 248, 216])
        # torch.Size([6, 42, 248, 216])
        # torch.Size([6, 12, 248, 216])
        
        # python tools/train.py configs/pointpillars/01dcn_hv_pointpillars_secfpn_6x8_160e_ouster-3d-3class.py  --gpu-ids 4
        # 输入x len为1 ，torch.Size([4, 384, 248, 216])
        # 输出 len为3
        # torch.Size([4, 42, 248, 216])
        # torch.Size([4, 42, 248, 216])
        # torch.Size([4, 12, 248, 216])
        # ouster输出异常，是因为class_names包含7类，但是anchor只设置了3类
        loss_inputs = outs + (gt_bboxes_3d, gt_labels_3d, img_metas)
        losses = self.bbox_head.loss( # calculate loss
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

    def simple_test(self, points, img_metas, imgs=None, rescale=False):
        """Test function without augmentaiton."""
        x = self.extract_feat(points, img_metas)
        outs = self.bbox_head(x)
        # print(len(outs)) # 3
        # print(outs[0][0].shape)  # torch.Size([1, 2, 248, 216])
        # print(outs[1][0].shape)  # torch.Size([1, 14, 248, 216])
        # print(outs[2][0].shape)  # torch.Size([1, 4, 248, 216])
        bbox_list = self.bbox_head.get_bboxes( # mmdet3d/models/dense_heads/anchor3d_head.py
            *outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels) # mmdet3d/core/bbox/transforms.py
            for bboxes, scores, labels in bbox_list
        ]
        # print(len(bbox_results))
        # print(bbox_results[0])
        return bbox_results

    def aug_test(self, points, img_metas, imgs=None, rescale=False):
        """Test function with augmentaiton."""
        feats = self.extract_feats(points, img_metas)

        # only support aug_test for one sample
        aug_bboxes = []
        for x, img_meta in zip(feats, img_metas):
            outs = self.bbox_head(x)
            bbox_list = self.bbox_head.get_bboxes(
                *outs, img_meta, rescale=rescale)
            bbox_list = [
                dict(boxes_3d=bboxes, scores_3d=scores, labels_3d=labels)
                for bboxes, scores, labels in bbox_list
            ]
            aug_bboxes.append(bbox_list[0])

        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes = merge_aug_bboxes_3d(aug_bboxes, img_metas,
                                            self.bbox_head.test_cfg)

        return [merged_bboxes]
