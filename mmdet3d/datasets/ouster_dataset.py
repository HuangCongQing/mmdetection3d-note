# Copyright (c) OpenMMLab. All rights reserved.
import copy
import mmcv
import numpy as np
import os
import tempfile
import torch
from mmcv.utils import print_log
from os import path as osp

from mmdet.datasets import DATASETS
from ..core import show_multi_modality_result, show_result
from ..core.bbox import (Box3DMode, CameraInstance3DBoxes, Coord3DMode,
                         LiDARInstance3DBoxes, points_cam2img)
from .custom_3d import Custom3DDataset
from .pipelines import Compose # 组装


@DATASETS.register_module()
class OusterDataset(Custom3DDataset):
    r"""KITTI Dataset.

    This class serves as the API for experiments on the `KITTI Dataset
    <http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d>`_.

    Args:
        data_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        split (str): Split of input data.
        pts_prefix (str, optional): Prefix of points files.
            Defaults to 'velodyne'.
        pipeline (list[dict], optional): Pipeline used for data processing.
            Defaults to None.
        classes (tuple[str], optional): Classes used in the dataset.
            Defaults to None.
        modality (dict, optional): Modality to specify the sensor data used
            as input. Defaults to None.
        box_type_3d (str, optional): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'LiDAR' in this dataset. Available options includes

            - 'LiDAR': Box in LiDAR coordinates.
            - 'Depth': Box in depth coordinates, usually for indoor dataset.
            - 'Camera': Box in camera coordinates.
        filter_empty_gt (bool, optional): Whether to filter empty GT.
            Defaults to True.
        test_mode (bool, optional): Whether the dataset is in test mode.
            Defaults to False.
        pcd_limit_range (list): The range of point cloud used to filter
            invalid predicted boxes. Default: [0, -40, -3, 70.4, 40, 0.0].
    """
    # CLASSES = ('car', 'pedestrian', 'cyclist') #
    CLASSES =  ('Truck','Auxiliary','Car','Excavator','Widebody','Pedestrian')
    # 修改=============================================================================
    def __init__(self,
                 data_root,
                 ann_file,
                 split,
                 pts_prefix='velodyne',
                 pipeline=None,
                 classes=None,
                 modality=None,
                 box_type_3d='LiDAR',
                 filter_empty_gt=True,
                 test_mode=False,
                 pcd_limit_range=[0, -40, -3, 70.4, 40, 0.0]):
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            classes=classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode)

        self.split = split
        self.root_split = os.path.join(self.data_root, split)
        assert self.modality is not None
        self.pcd_limit_range = pcd_limit_range
        self.pts_prefix = pts_prefix

    def _get_pts_filename(self, idx):
        """Get point cloud filename according to the given index.

        Args:
            index (int): Index of the point cloud file to get.

        Returns:
            str: Name of the point cloud file.
        """
        pts_filename = osp.join(self.root_split, self.pts_prefix,
                                f'{idx:06d}.bin')
        return pts_filename
    # ===================================================================================
    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - img_prefix (str | None): Prefix of image files.
                - img_info (dict): Image info.
                - lidar2img (list[np.ndarray], optional): Transformations \
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        # {'point_cloud': {'pc_idx': 0, 'num_features': 4, 'velodyne_path': 'training/velodyne/000000.bin'}, 'annos': {'name': array(['Pedestrian'], dtype='<U10'), 'truncated': array([0.]), 'occluded': array([0]), 'alpha': array([-0.2]), 'bbox': array([[712.4 , 143.  , 810.73, 307.92]]), 'dimensions': array([[1.2 , 1.89, 0.48]]), 'location': array([[1.84, 1.47, 8.41]]), 'rotation_y': array([0.01]), 'score': array([0.]), 'index': array([0], dtype=int32), 'group_ids': array([0], dtype=int32), 'difficulty': array([0], dtype=int32)}}
        info = self.data_infos[index]

        # 修改=============================================================================
        # sample_idx = info['image']['image_idx'] # 报错
        sample_idx = info['point_cloud']['pc_idx'] # 
        # img_filename = os.path.join(self.data_root,
        #                             info['image']['image_path'])

        # TODO: consider use torch.Tensor only
        # rect = info['calib']['R0_rect'].astype(np.float32)
        # Trv2c = info['calib']['Tr_velo_to_cam'].astype(np.float32)
        # P2 = info['calib']['P2'].astype(np.float32)
        # lidar2img = P2 @ rect @ Trv2c

        pts_filename = self._get_pts_filename(sample_idx)
        input_dict = dict(
            sample_idx=sample_idx,
            pts_filename=pts_filename,
            img_prefix=None,
            # img_info=dict(filename=img_filename),
            # lidar2img=lidar2img
            )

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos # =======！！！！

        return input_dict
    #   调用 get_anno_info() ，加载 anno 里面的 boxes， 格式为 (x_lidar, y_lidar, z_lidar, dx, dy, dz, yaw)
    # 需要修改gt_bboxes_3d和gt_labels_3d=====================================
    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): \
                    3D ground truth bboxes.
                - gt_labels_3d (np.ndarray): Labels of ground truths.====================================================================
                - gt_bboxes (np.ndarray): 2D ground truth bboxes. # 不用
                - gt_labels (np.ndarray): Labels of ground truths. # 不用
                - gt_names (list[str]): Class names of ground truths.
        """
        # Use index to get the annos, thus the evalhook could also use this api
        info = self.data_infos[index]
        # rect = info['calib']['R0_rect'].astype(np.float32)
        # Trv2c = info['calib']['Tr_velo_to_cam'].astype(np.float32)

        annos = info['annos']
        # we need other objects to avoid collision when sample
        # annos = self.remove_dontcare(annos)
        loc = annos['location']
        dims = annos['dimensions']
        rots = annos['rotation_y']
        gt_names = annos['name']
        # 组合成(x_lidar, y_lidar, z_lidar, dx, dy, dz, yaw)
        gt_bboxes_3d = np.concatenate([loc, dims, rots[..., np.newaxis]],
                                      axis=1).astype(np.float32)
        # 修改 https://www.yuque.com/huangzhongqing/hre6tf/ha7d5f#mqcp4
        # 雷达坐标系：https://mmdetection3d.readthedocs.io/zh_CN/latest/api.html#mmdet3d.core.bbox.LiDARInstance3DBoxes
        # LiDARInstance3DBoxes(tensor, box_dim=7, with_yaw=True, origin=(0.5, 0.5, 0)) 目标框的0.5的长 0.5倍的宽和0倍的高
        gt_bboxes_3d = LiDARInstance3DBoxes(gt_bboxes_3d, origin=(0.5, 0.5, 0.5))# .convert_to(self.box_mode_3d) # box_mode_3d= Box3DMode.LIDAR====================================================================
        # convert gt_bboxes_3d to velodyne coordinates  格式为 (x_lidar, y_lidar, z_lidar, dx, dy, dz, yaw)
        # 参考：https://mmdetection3d.readthedocs.io/zh_CN/latest/api.html#mmdet3d.core.bbox.CameraInstance3DBoxes
        # gt_bboxes_3d = CameraInstance3DBoxes(gt_bboxes_3d).convert_to(
        #     self.box_mode_3d, np.linalg.inv(rect @ Trv2c))
        
        # gt_bboxes = annos['bbox']
        # selected = self.drop_arrays_by_name(gt_names, ['DontCare']) # 不要DonCare
        # gt_bboxes = gt_bboxes[selected].astype('float32')
        # gt_names = gt_names[selected]


        gt_labels_3d = []
        for cat in gt_names:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d).astype(np.int64)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d, # GTbbox
            gt_labels_3d=gt_labels_3d,
            # bboxes=gt_bboxes,
            # labels=gt_labels,
            gt_names=gt_names)
        return anns_results

    def drop_arrays_by_name(self, gt_names, used_classes):
        """Drop irrelevant ground truths by name.

        Args:
            gt_names (list[str]): Names of ground truths.
            used_classes (list[str]): Classes of interest.

        Returns:
            np.ndarray: Indices of ground truths that will be dropped.
        """
        inds = [i for i, x in enumerate(gt_names) if x not in used_classes]
        inds = np.array(inds, dtype=np.int64)
        return inds

    def keep_arrays_by_name(self, gt_names, used_classes):
        """Keep useful ground truths by name.

        Args:
            gt_names (list[str]): Names of ground truths.
            used_classes (list[str]): Classes of interest.

        Returns:
            np.ndarray: Indices of ground truths that will be keeped.
        """
        inds = [i for i, x in enumerate(gt_names) if x in used_classes]
        inds = np.array(inds, dtype=np.int64)
        return inds

    def remove_dontcare(self, ann_info):
        """Remove annotations that do not need to be cared.

        Args:
            ann_info (dict): Dict of annotation infos. The ``'DontCare'``
                annotations will be removed according to ann_file['name'].

        Returns:
            dict: Annotations after filtering.
        """
        img_filtered_annotations = {}
        relevant_annotation_indices = [
            i for i, x in enumerate(ann_info['name']) if x != 'DontCare'
        ]
        for key in ann_info.keys():
            img_filtered_annotations[key] = (
                ann_info[key][relevant_annotation_indices]) # TypeError: list indices must be integers or slices, not list
        return img_filtered_annotations

    def format_results(self,
                       outputs,
                       pklfile_prefix=None,
                       submission_prefix=None):
        """Format the results to pkl file.

        Args:
            outputs (list[dict]): Testing results of the dataset.
            pklfile_prefix (str | None): The prefix of pkl files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            submission_prefix (str | None): The prefix of submitted files. It
                includes the file path and the prefix of filename, e.g.,
                "a/b/prefix". If not specified, a temp file will be created.
                Default: None.

        Returns:
            tuple: (result_files, tmp_dir), result_files is a dict containing \
                the json filepaths, tmp_dir is the temporal directory created \
                for saving json files when jsonfile_prefix is not specified.
        """
        if pklfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            pklfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None

        if not isinstance(outputs[0], dict): #  #outputs[0]:  [boxes_3d,scores_3d,labels_3d]
            result_files = self.bbox2result_kitti2d(outputs, self.CLASSES, #===================================================
                                                    pklfile_prefix,
                                                    submission_prefix)
        elif 'pts_bbox' in outputs[0] or 'img_bbox' in outputs[0]: # 不执行
            result_files = dict()
            for name in outputs[0]: #  # [boxes_3d,scores_3d,labels_3d]
                results_ = [out[name] for out in outputs]
                pklfile_prefix_ = pklfile_prefix + name
                if submission_prefix is not None:
                    submission_prefix_ = submission_prefix + name
                else:
                    submission_prefix_ = None
                if 'img' in name:
                    result_files = self.bbox2result_kitti2d(
                        results_, self.CLASSES, pklfile_prefix_,
                        submission_prefix_)
                else:
                    result_files_ = self.bbox2result_kitti(
                        results_, self.CLASSES, pklfile_prefix_,
                        submission_prefix_)
                result_files[name] = result_files_
        else:#从这执行========
            result_files = self.bbox2result_kitti(outputs, self.CLASSES, #从这执行===================================================
                                                  pklfile_prefix,
                                                  submission_prefix)
        return result_files, tmp_dir
    # 评测（继承mmdet3d/datasets/custom_3d.py）====================评测main
    def evaluate(self,
                 results, # ['boxes_3d', 'scores_3d', 'labels_3d']  5个文件bin文件list就是{list:5}
                 metric=None, # None
                 logger=None,
                 pklfile_prefix=None, # 提交结果是ture
                 submission_prefix=None,
                 show=False,
                 out_dir=None,
                 pipeline=None): # pipeline：【'LoadPointsFromFile'，'DefaultFormatBundle3D'，'Collect3D' 】
        """Evaluation in KITTI protocol.

        Args:
            results (list[dict]): Testing results of the dataset.数据结果
            metric (str | list[str]): Metrics to be evaluated. 被评测的标准
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            pklfile_prefix (str | None): The prefix of pkl files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            submission_prefix (str | None): The prefix of submission datas.
                If not specified, the submission data will not be generated.
            show (bool): Whether to visualize.
                Default: False.
            out_dir (str): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.

        Returns:
            dict[str, float]: Results of each evaluation metric.
        """
        # 预测结果
        result_files, tmp_dir = self.format_results(results, pklfile_prefix)  # ===========================================================================

        from mmdet3d.core.evaluation import kitti_eval, ouster_eval# mmdet3d/core/evaluation/kitti_utils/eval.py 评测=================================================
        gt_annos = [info['annos'] for info in self.data_infos] # 得到GT  def get_label_anno_ouster(label_path):   tools/data_converter/kitti_data_utils.py

        if isinstance(result_files, dict):
            ap_dict = dict()
            for name, result_files_ in result_files.items(): # for循环 遍历
                # eval_types = ['bbox', 'bev', '3d'] # 只要3D就行
                eval_types = [ '3d'] # 只要3D就行====================================
                if 'img' in name:
                    eval_types = ['bbox']
                ap_result_str, ap_dict_ = ouster_eval( # 开始评测的入口 ，调用 mmdet3d/core/evaluation/kitti_utils/eval.py
                    gt_annos, # 输入GT======================================================================
                    result_files_, # result_files_ 输入预测结果=====================================================================
                    self.CLASSES, # CLASSES =  ('Truck','Auxiliary','Car','Excavator','Widebody','Pedestrian')
                    eval_types=eval_types)
                for ap_type, ap in ap_dict_.items():
                    ap_dict[f'{name}/{ap_type}'] = float('{:.4f}'.format(ap))

                print_log(
                    f'Results of {name}:\n' + ap_result_str, logger=logger)
        else:
            if metric == 'img_bbox': # 
                ap_result_str, ap_dict = ouster_eval(
                    gt_annos, result_files, self.CLASSES, eval_types=['bbox'])
            else:
                ap_result_str, ap_dict = ouster_eval(gt_annos, result_files,
                                                    self.CLASSES)
            print_log('\n' + ap_result_str, logger=logger)

        if tmp_dir is not None:
            tmp_dir.cleanup()
        if show:
            self.show(results, out_dir, pipeline=pipeline)
        return ap_dict
    #转换成3D 结果===================================================================
    def bbox2result_kitti(self,
                          net_outputs,
                          class_names,
                          pklfile_prefix=None,
                          submission_prefix=None):
        """Convert 3D detection results to kitti format for evaluation and test
        submission.

        Args:
            net_outputs (list[np.ndarray]): List of array storing the \
                inferenced bounding boxes and scores.
            class_names (list[String]): A list of class names.
            pklfile_prefix (str | None): The prefix of pkl file.
            submission_prefix (str | None): The prefix of submission file.

        Returns:
            list[dict]: A list of dictionaries with the kitti format.
        """
        assert len(net_outputs) == len(self.data_infos), \
            'invalid list length of network outputs'
        if submission_prefix is not None:
            mmcv.mkdir_or_exist(submission_prefix)

        det_annos = []
        print('\nConverting prediction to KITTI format【将预测结果转化为kitti格式ouster_dataset.py】')
        for idx, pred_dicts in enumerate(
                mmcv.track_iter_progress(net_outputs)):
            annos = []
            info = self.data_infos[idx] # 两类数据： ['point_cloud','annos']
            # sample_idx = info['image']['image_idx'] # 报错# KeyError: 'image'
            sample_idx = info['point_cloud']['pc_idx'] #
            # image_shape = info['image']['image_shape'][:2]

            # box_dict = self.convert_valid_bboxes(pred_dicts, info) # 转化为合法的bboxes，和相机对比？？？  预测结果======================================================
            box_dict = self.convert_type(pred_dicts, info) # 转化为合法的bboxes， 预测结果======================================================
            """ 
                # bbox=box_2d_preds[valid_inds, :].numpy(),  不需要
                # box3d_camera=box_preds_camera[valid_inds].tensor.numpy(),  不需要
                 box3d_lidar=box_preds[valid_inds].tensor.numpy(),
                scores=scores[valid_inds].numpy(),
                label_preds=labels[valid_inds].numpy(),
                sample_idx=sample_idx)
             """
            anno = {
                'name': [],
                # 'truncated': [],
                # 'occluded': [],
                # 'alpha': [],
                # 'bbox': [],
                'dimensions': [],
                'location': [],
                'rotation_y': [],
                'score': []
            }
            # if len(box_dict['bbox']) > 0: # 有目标修改
            if len(box_dict['box3d_lidar']) > 0: # 有目标
                # box_2d_preds = box_dict['bbox']
                # box_preds = box_dict['box3d_camera']
                scores = box_dict['scores']
                box_preds_lidar = box_dict['box3d_lidar']
                label_preds = box_dict['label_preds']

                for  box_lidar, score, label in zip(
                        box_preds_lidar, scores,
                        label_preds):
                    # bbox[2:] = np.minimum(bbox[2:], image_shape[::-1])
                    # bbox[:2] = np.maximum(bbox[:2], [0, 0])
                    anno['name'].append(class_names[int(label)])
                    # anno['truncated'].append(0.0)
                    # anno['occluded'].append(0)
                    # anno['alpha'].append(
                    #     -np.arctan2(-box_lidar[1], box_lidar[0]) + box[6]) # 卡车中心与相机中心构成的矢量与在bird view下的夹角
                    # anno['bbox'].append(bbox)
                    anno['dimensions'].append(box_lidar[3:6]) # 长宽高  ，convert_type
                    anno['location'].append(box_lidar[:3]) # 中心点
                    anno['rotation_y'].append(box_lidar[6])
                    anno['score'].append(score)

                anno = {k: np.stack(v) for k, v in anno.items()}
                annos.append(anno)
            else:
                anno = {
                    'name': np.array([]),
                    # 'truncated': np.array([]),
                    # 'occluded': np.array([]),
                    # 'alpha': np.array([]),
                    # 'bbox': np.zeros([0, 4]),
                    'dimensions': np.zeros([0, 3]), # array([], shape=(0, 3), dtype=float64)
                    'location': np.zeros([0, 3]),
                    'rotation_y': np.array([]),
                    'score': np.array([]),
                }
                annos.append(anno)

            if submission_prefix is not None:
                curr_file = f'{submission_prefix}/{sample_idx:06d}.txt' # 
                with open(curr_file, 'w') as f:
                    bbox = anno['bbox']
                    loc = anno['location']
                    dims = anno['dimensions']  # lhw -> hwl

                    for idx in range(len(bbox)):
                        print(
                            '{} -1 -1 {:.4f} {:.4f} {:.4f} {:.4f} '
                            '{:.4f} {:.4f} {:.4f} '
                            '{:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}'.format(
                                anno['name'][idx], anno['alpha'][idx],
                                bbox[idx][0], bbox[idx][1], bbox[idx][2],
                                bbox[idx][3], dims[idx][1], dims[idx][2],
                                dims[idx][0], loc[idx][0], loc[idx][1],
                                loc[idx][2], anno['rotation_y'][idx],
                                anno['score'][idx]),
                            file=f)

            annos[-1]['sample_idx'] = np.array(
                [sample_idx] * len(annos[-1]['score']), dtype=np.int64)

            det_annos += annos # 返回结果

        if pklfile_prefix is not None:
            if not pklfile_prefix.endswith(('.pkl', '.pickle')):
                out = f'{pklfile_prefix}.pkl'
            mmcv.dump(det_annos, out)
            print(f'Result is saved to {out}.')

        return det_annos
    # 转换成2D 检测结果
    def bbox2result_kitti2d(self,
                            net_outputs,
                            class_names,
                            pklfile_prefix=None,
                            submission_prefix=None):
        """Convert 2D detection results to kitti format for evaluation and test
        submission.

        Args:
            net_outputs (list[np.ndarray]): List of array storing the \
                inferenced bounding boxes and scores.
            class_names (list[String]): A list of class names.
            pklfile_prefix (str | None): The prefix of pkl file.
            submission_prefix (str | None): The prefix of submission file.

        Returns:
            list[dict]: A list of dictionaries have the kitti format
        """
        assert len(net_outputs) == len(self.data_infos), \
            'invalid list length of network outputs'
        det_annos = []
        print('\nConverting prediction to KITTI format')
        for i, bboxes_per_sample in enumerate(
                mmcv.track_iter_progress(net_outputs)):
            annos = []
            anno = dict(
                name=[],
                truncated=[],
                occluded=[],
                alpha=[],
                bbox=[],
                dimensions=[],
                location=[],
                rotation_y=[],
                score=[])
            sample_idx = self.data_infos[i]['image']['image_idx']

            num_example = 0
            for label in range(len(bboxes_per_sample)):
                bbox = bboxes_per_sample[label]
                for i in range(bbox.shape[0]):
                    anno['name'].append(class_names[int(label)])
                    anno['truncated'].append(0.0)
                    anno['occluded'].append(0)
                    anno['alpha'].append(0.0)
                    anno['bbox'].append(bbox[i, :4])
                    # set dimensions (height, width, length) to zero
                    anno['dimensions'].append(
                        np.zeros(shape=[3], dtype=np.float32))
                    # set the 3D translation to (-1000, -1000, -1000)
                    anno['location'].append(
                        np.ones(shape=[3], dtype=np.float32) * (-1000.0))
                    anno['rotation_y'].append(0.0)
                    anno['score'].append(bbox[i, 4])
                    num_example += 1

            if num_example == 0:
                annos.append(
                    dict(
                        name=np.array([]),
                        truncated=np.array([]),
                        occluded=np.array([]),
                        alpha=np.array([]),
                        bbox=np.zeros([0, 4]),
                        dimensions=np.zeros([0, 3]),
                        location=np.zeros([0, 3]),
                        rotation_y=np.array([]),
                        score=np.array([]),
                    ))
            else:
                anno = {k: np.stack(v) for k, v in anno.items()}
                annos.append(anno)

            annos[-1]['sample_idx'] = np.array(
                [sample_idx] * num_example, dtype=np.int64)
            det_annos += annos

        if pklfile_prefix is not None:
            # save file in pkl format
            pklfile_path = (
                pklfile_prefix[:-4] if pklfile_prefix.endswith(
                    ('.pkl', '.pickle')) else pklfile_prefix)
            mmcv.dump(det_annos, pklfile_path)

        if submission_prefix is not None:
            # save file in submission format
            mmcv.mkdir_or_exist(submission_prefix)
            print(f'Saving KITTI submission to {submission_prefix}')
            for i, anno in enumerate(det_annos):
                sample_idx = self.data_infos[i]['image']['image_idx']
                cur_det_file = f'{submission_prefix}/{sample_idx:06d}.txt'
                with open(cur_det_file, 'w') as f:
                    bbox = anno['bbox']
                    loc = anno['location']
                    dims = anno['dimensions'][::-1]  # lhw -> hwl
                    for idx in range(len(bbox)):
                        print(
                            '{} -1 -1 {:4f} {:4f} {:4f} {:4f} {:4f} {:4f} '
                            '{:4f} {:4f} {:4f} {:4f} {:4f} {:4f} {:4f}'.format(
                                anno['name'][idx],
                                anno['alpha'][idx],
                                *bbox[idx],  # 4 float
                                *dims[idx],  # 3 float
                                *loc[idx],  # 3 float
                                anno['rotation_y'][idx],
                                anno['score'][idx]),
                            file=f,
                        )
            print(f'Result is saved to {submission_prefix}')

        return det_annos
    # numpy和tensor转换
    def convert_type(self, box_dict, info):
        """Convert the predicted boxes into valid ones.

        Args:
            box_dict (dict): Box dictionaries to be converted.

                - boxes_3d (:obj:`LiDARInstance3DBoxes`): 3D bounding boxes.
                - scores_3d (torch.Tensor): Scores of boxes.
                - labels_3d (torch.Tensor): Class labels of boxes.
            info (dict): Data info.

        Returns:
            dict: Valid predicted boxes.

                - bbox (np.ndarray): 2D bounding boxes. # 不需要
                - box3d_camera (np.ndarray): 3D bounding boxes in \
                    camera coordinate. #不需要
                - box3d_lidar (np.ndarray): 3D bounding boxes in \
                    LiDAR coordinate. # 需要
                - scores (np.ndarray): Scores of boxes.# 需要
                - label_preds (np.ndarray): Class label predictions. # 需要
                - sample_idx (int): Sample index.
        """

        # TODO: refactor this function
        box_preds = box_dict['boxes_3d'] # 7维
        scores = box_dict['scores_3d'] # 1维度
        labels = box_dict['labels_3d'] # label结果
        # sample_idx = info['image']['image_idx']
        sample_idx = info['point_cloud']['pc_idx'] # GT
        # TODO: remove the hack of yaw
        box_preds.tensor[:, -1] = box_preds.tensor[:, -1] - np.pi # 最后一列==============================================================
        box_preds.limit_yaw(offset=0.5, period=np.pi * 2) # self.tensor[:, 6]最后一维

        if len(box_preds) == 0: # 直接返回0
            return dict(
                # bbox=np.zeros([0, 4]),
                # box3d_camera=np.zeros([0, 7]),
                box3d_lidar=np.zeros([0, 7]),
                scores=np.zeros([0]),
                label_preds=np.zeros([0, 4]),
                sample_idx=sample_idx)
        # # 2D 结果转换
        # rect = info['calib']['R0_rect'].astype(np.float32)
        # Trv2c = info['calib']['Tr_velo_to_cam'].astype(np.float32)
        # P2 = info['calib']['P2'].astype(np.float32)
        # img_shape = info['image']['image_shape']
        # P2 = box_preds.tensor.new_tensor(P2)

        # box_preds_camera = box_preds.convert_to(Box3DMode.CAM, rect @ Trv2c)

        # box_corners = box_preds_camera.corners
        # box_corners_in_image = points_cam2img(box_corners, P2)
        # # box_corners_in_image: [N, 8, 2]
        # minxy = torch.min(box_corners_in_image, dim=1)[0]
        # maxxy = torch.max(box_corners_in_image, dim=1)[0]
        # box_2d_preds = torch.cat([minxy, maxxy], dim=1)
        # # Post-processing   后处理
        # # check box_preds_camera    相机处理
        # image_shape = box_preds.tensor.new_tensor(img_shape)
        # valid_cam_inds = ((box_2d_preds[:, 0] < image_shape[1]) &
        #                   (box_2d_preds[:, 1] < image_shape[0]) &
        #                   (box_2d_preds[:, 2] > 0) & (box_2d_preds[:, 3] > 0))
        # # check box_preds
        # limit_range = box_preds.tensor.new_tensor(self.pcd_limit_range)
        # valid_pcd_inds = ((box_preds.center > limit_range[:3]) &
        #                   (box_preds.center < limit_range[3:]))
        # valid_inds = valid_cam_inds & valid_pcd_inds.all(-1) # 得到合法的结果进行判断=========================================
        return dict(
            # bbox=box_2d_preds[valid_inds, :].numpy(),
            # box3d_camera=box_preds_camera[valid_inds].tensor.numpy(),
            box3d_lidar=box_preds.tensor.numpy(),
            scores=scores.numpy(),
            label_preds=labels.numpy(),
            sample_idx=sample_idx)

        # if valid_inds.sum() > 0: # 合法
        #     return dict(
        #         # bbox=box_2d_preds[valid_inds, :].numpy(),
        #         # box3d_camera=box_preds_camera[valid_inds].tensor.numpy(),
        #         box3d_lidar=box_preds[valid_inds].tensor.numpy(),
        #         scores=scores[valid_inds].numpy(),
        #         label_preds=labels[valid_inds].numpy(),
        #         sample_idx=sample_idx)
        # else:
        #     return dict(
        #         # bbox=np.zeros([0, 4]),
        #         # box3d_camera=np.zeros([0, 7]),
        #         box3d_lidar=np.zeros([0, 7]),
        #         scores=np.zeros([0]),
        #         label_preds=np.zeros([0, 4]),
        #         sample_idx=sample_idx)

    # 雷达和相机对比得到合法的预测结果
    def convert_valid_bboxes(self, box_dict, info):
        """Convert the predicted boxes into valid ones.

        Args:
            box_dict (dict): Box dictionaries to be converted.

                - boxes_3d (:obj:`LiDARInstance3DBoxes`): 3D bounding boxes.
                - scores_3d (torch.Tensor): Scores of boxes.
                - labels_3d (torch.Tensor): Class labels of boxes.
            info (dict): Data info.

        Returns:
            dict: Valid predicted boxes.

                - bbox (np.ndarray): 2D bounding boxes. # 不需要
                - box3d_camera (np.ndarray): 3D bounding boxes in \
                    camera coordinate. #不需要
                - box3d_lidar (np.ndarray): 3D bounding boxes in \
                    LiDAR coordinate. # 需要
                - scores (np.ndarray): Scores of boxes.# 需要
                - label_preds (np.ndarray): Class label predictions. # 需要
                - sample_idx (int): Sample index.
        """
        # TODO: refactor this function
        box_preds = box_dict['boxes_3d']
        scores = box_dict['scores_3d']
        labels = box_dict['labels_3d'] # label结果
        # sample_idx = info['image']['image_idx']
        sample_idx = info['point_cloud']['pc_idx'] #
        # TODO: remove the hack of yaw
        box_preds.tensor[:, -1] = box_preds.tensor[:, -1] - np.pi
        box_preds.limit_yaw(offset=0.5, period=np.pi * 2)

        if len(box_preds) == 0: # 直接返回0
            return dict(
                bbox=np.zeros([0, 4]),
                box3d_camera=np.zeros([0, 7]),
                box3d_lidar=np.zeros([0, 7]),
                scores=np.zeros([0]),
                label_preds=np.zeros([0, 4]),
                sample_idx=sample_idx)
        # 2D 结果转换
        rect = info['calib']['R0_rect'].astype(np.float32)
        Trv2c = info['calib']['Tr_velo_to_cam'].astype(np.float32)
        P2 = info['calib']['P2'].astype(np.float32)
        img_shape = info['image']['image_shape']
        P2 = box_preds.tensor.new_tensor(P2)

        box_preds_camera = box_preds.convert_to(Box3DMode.CAM, rect @ Trv2c)

        box_corners = box_preds_camera.corners
        box_corners_in_image = points_cam2img(box_corners, P2)
        # box_corners_in_image: [N, 8, 2]
        minxy = torch.min(box_corners_in_image, dim=1)[0]
        maxxy = torch.max(box_corners_in_image, dim=1)[0]
        box_2d_preds = torch.cat([minxy, maxxy], dim=1)
        # Post-processing   后处理
        # check box_preds_camera    相机处理
        image_shape = box_preds.tensor.new_tensor(img_shape)
        valid_cam_inds = ((box_2d_preds[:, 0] < image_shape[1]) &
                          (box_2d_preds[:, 1] < image_shape[0]) &
                          (box_2d_preds[:, 2] > 0) & (box_2d_preds[:, 3] > 0))
        # check box_preds
        limit_range = box_preds.tensor.new_tensor(self.pcd_limit_range)
        valid_pcd_inds = ((box_preds.center > limit_range[:3]) &
                          (box_preds.center < limit_range[3:]))
        valid_inds = valid_cam_inds & valid_pcd_inds.all(-1) # 得到合法的结果

        if valid_inds.sum() > 0: # 合法
            return dict(
                # bbox=box_2d_preds[valid_inds, :].numpy(),
                # box3d_camera=box_preds_camera[valid_inds].tensor.numpy(),
                box3d_lidar=box_preds[valid_inds].tensor.numpy(),
                scores=scores[valid_inds].numpy(),
                label_preds=labels[valid_inds].numpy(),
                sample_idx=sample_idx)
        else:
            return dict(
                # bbox=np.zeros([0, 4]),
                # box3d_camera=np.zeros([0, 7]),
                box3d_lidar=np.zeros([0, 7]),
                scores=np.zeros([0]),
                label_preds=np.zeros([0, 4]),
                sample_idx=sample_idx)

    def _build_default_pipeline(self):
        """Build the default pipeline for this dataset."""
        pipeline = [
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=4,
                use_dim=4,
                file_client_args=dict(backend='disk')),
            dict(
                type='DefaultFormatBundle3D',
                class_names=self.CLASSES,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ]
        if self.modality['use_camera']:
            pipeline.insert(0, dict(type='LoadImageFromFile'))
        return Compose(pipeline)
    # 可视化
    def show(self, results, out_dir, show=True, pipeline=None):
        """Results visualization.

        Args:
            results (list[dict]): List of bounding boxes results.
            out_dir (str): Output directory of visualization result.
            show (bool): Visualize the results online.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.
        """
        assert out_dir is not None, 'Expect out_dir, got none.'
        pipeline = self._get_pipeline(pipeline)
        for i, result in enumerate(results):
            if 'pts_bbox' in result.keys():
                result = result['pts_bbox']
            data_info = self.data_infos[i]
            pts_path = data_info['point_cloud']['velodyne_path']
            file_name = osp.split(pts_path)[-1].split('.')[0]
            points, img_metas, img = self._extract_data(
                i, pipeline, ['points', 'img_metas', 'img'])
            points = points.numpy()
            # for now we convert points into depth mode
            points = Coord3DMode.convert_point(points, Coord3DMode.LIDAR,
                                               Coord3DMode.DEPTH)
            gt_bboxes = self.get_ann_info(i)['gt_bboxes_3d'].tensor.numpy()
            show_gt_bboxes = Box3DMode.convert(gt_bboxes, Box3DMode.LIDAR,
                                               Box3DMode.DEPTH)
            pred_bboxes = result['boxes_3d'].tensor.numpy()
            show_pred_bboxes = Box3DMode.convert(pred_bboxes, Box3DMode.LIDAR,
                                                 Box3DMode.DEPTH)
            show_result(points, show_gt_bboxes, show_pred_bboxes, out_dir,
                        file_name, show)

            # multi-modality visualization
            if self.modality['use_camera'] and 'lidar2img' in img_metas.keys():
                img = img.numpy()
                # need to transpose channel to first dim
                img = img.transpose(1, 2, 0)
                show_pred_bboxes = LiDARInstance3DBoxes(
                    pred_bboxes, origin=(0.5, 0.5, 0))
                show_gt_bboxes = LiDARInstance3DBoxes(
                    gt_bboxes, origin=(0.5, 0.5, 0))
                show_multi_modality_result(
                    img,
                    show_gt_bboxes,
                    show_pred_bboxes,
                    img_metas['lidar2img'],
                    out_dir,
                    file_name,
                    box_mode='lidar',
                    show=show)
