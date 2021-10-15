# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from collections import OrderedDict
from concurrent import futures as futures
from os import path as osp
from pathlib import Path
from skimage import io


def get_image_index_str(img_idx, use_prefix_id=False):
    if use_prefix_id:
        return '{:07d}'.format(img_idx)
    else:
        return '{:06d}'.format(img_idx)


def get_kitti_info_path(idx,
                        prefix,
                        info_type='image_2',
                        file_tail='.png',
                        training=True,
                        relative_path=True,
                        exist_check=True,
                        use_prefix_id=False):
    img_idx_str = get_image_index_str(idx, use_prefix_id)
    img_idx_str += file_tail
    prefix = Path(prefix)
    if training:
        file_path = Path('training') / info_type / img_idx_str
    else:
        file_path = Path('testing') / info_type / img_idx_str
    if exist_check and not (prefix / file_path).exists():
        raise ValueError('file not exist: {}'.format(file_path))
    if relative_path:
        return str(file_path)
    else:
        return str(prefix / file_path)

# 图像路径image_2
def get_image_path(idx,
                   prefix,
                   training=True,
                   relative_path=True,
                   exist_check=True,
                   info_type='image_2',
                   use_prefix_id=False):
    return get_kitti_info_path(idx, prefix, info_type, '.png', training,
                               relative_path, exist_check, use_prefix_id)

#  label_2
def get_label_path(idx,
                   prefix,
                   training=True,
                   relative_path=True,
                   exist_check=True,
                   info_type='label_2',
                   use_prefix_id=False):
    return get_kitti_info_path(idx, prefix, info_type, '.txt', training,
                               relative_path, exist_check, use_prefix_id)

# velodyne bin文件
def get_velodyne_path(idx,
                      prefix,
                      training=True,
                      relative_path=True,
                      exist_check=True,
                      use_prefix_id=False):
    return get_kitti_info_path(idx, prefix, 'velodyne', '.bin', training,
                               relative_path, exist_check, use_prefix_id)

# calib文件
def get_calib_path(idx,
                   prefix,
                   training=True,
                   relative_path=True,
                   exist_check=True,
                   use_prefix_id=False):
    return get_kitti_info_path(idx, prefix, 'calib', '.txt', training,
                               relative_path, exist_check, use_prefix_id)

# waymo才用到
def get_pose_path(idx,
                  prefix,
                  training=True,
                  relative_path=True,
                  exist_check=True,
                  use_prefix_id=False):
    return get_kitti_info_path(idx, prefix, 'pose', '.txt', training,
                               relative_path, exist_check, use_prefix_id)

# GT数据===========================================================
def get_label_anno(label_path): # 'data/kitti/training/label_2/000010.txt'
    annotations = {}
    annotations.update({
        'name': [],
        'truncated': [],
        'occluded': [],
        'alpha': [],
        'bbox': [],
        'dimensions': [],
        'location': [],
        'rotation_y': []
    })
    with open(label_path, 'r') as f: # 
        lines = f.readlines() # 打开文件
    # if len(lines) == 0 or len(lines[0]) < 15:
    #     content = []
    # else:
    content = [line.strip().split(' ') for line in lines] # list:5x15   GT内容
    num_objects = len([x[0] for x in content if x[0] != 'DontCare']) # 
    annotations['name'] = np.array([x[0] for x in content])
    num_gt = len(annotations['name'])
    annotations['truncated'] = np.array([float(x[1]) for x in content])
    annotations['occluded'] = np.array([int(x[2]) for x in content])
    annotations['alpha'] = np.array([float(x[3]) for x in content])
    annotations['bbox'] = np.array([[float(info) for info in x[4:8]]
                                    for x in content]).reshape(-1, 4)
    # dimensions will convert hwl format to standard lhw(camera) format.
    annotations['dimensions'] = np.array([[float(info) for info in x[8:11]]
                                          for x in content
                                          ]).reshape(-1, 3)[:, [2, 0, 1]]
    annotations['location'] = np.array([[float(info) for info in x[11:14]]
                                        for x in content]).reshape(-1, 3)
    annotations['rotation_y'] = np.array([float(x[14])
                                          for x in content]).reshape(-1)
    if len(content) != 0 and len(content[0]) == 16:  # have score
        annotations['score'] = np.array([float(x[15]) for x in content])
    else:
        annotations['score'] = np.zeros((annotations['bbox'].shape[0], ))
    index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
    annotations['index'] = np.array(index, dtype=np.int32)
    annotations['group_ids'] = np.arange(num_gt, dtype=np.int32)
    return annotations

# Ouster GT数据===========================================================
def get_label_anno_ouster(label_path): # 'data/kitti/training/label_2/000010.txt'
    annotations = {}
    annotations.update({
        'name': [],
        'truncated': [],
        'occluded': [],
        'alpha': [],
        'bbox': [],
        'dimensions': [],
        'location': [],
        'rotation_y': []
    })
    with open(label_path, 'r') as f: # 
        lines = f.readlines() # 打开文件
    # if len(lines) == 0 or len(lines[0]) < 15:
    #     content = []
    # else:
    content = [line.strip().split(' ') for line in lines] # GT内容
    print(content)
    num_objects = len([x[0] for x in content if x[0] != 'DontCare']) # 除了Dontare，有多少类，这里是2类
    annotations['name'] = np.array([x[0] for x in content]) # 第一列：['Car' 'Van' 'DontCare' 'DontCare' 'DontCare']
    num_gt = len(annotations['name'])
    annotations['location'] = np.array([[float(info) for info in x[1:4]] # 中心xyz坐标
                                        for x in content]).reshape(-1, 3)
    annotations['dimensions'] = np.array([[float(info) for info in x[4:7]] #  表示该车的高度，宽度，和长度，单位为米。（H,W,L）
                                          for x in content
                                          ]).reshape(-1, 3)[:, [0, 2, 1]] # 长宽高位置--> 长度 高度，宽度？？？？
    annotations['rotation_y'] = np.array([float(x[7]) # 表示车体朝向，绕相机坐标系y轴的弧度值
                                          for x in content]).reshape(-1)
    # 如果有第9列置信度
    print(len(content[0]))
    if len(content) != 0 and len(content[0]) == 9:  # have score #  (预测有score，但label_2标签文件不包含score)
        annotations['score'] = np.array([float(x[8]) for x in content])
    else:
        annotations['score'] = np.zeros((annotations['rotation_y'].shape[0], ))
    index = list(range(num_objects)) + [-1] * (num_gt - num_objects) # [0, 1, -1, -1, -1] = [0,1 ] +  [-1, -1, -1]
    annotations['index'] = np.array(index, dtype=np.int32)
    annotations['group_ids'] = np.arange(num_gt, dtype=np.int32) # [0 1 2 3 4]
    return annotations


def _extend_matrix(mat):
    mat = np.concatenate([mat, np.array([[0., 0., 0., 1.]])], axis=0)
    return mat

# kitti主函数!!!!!!
def get_kitti_image_info(path,
                         training=True,
                         label_info=True, # label_info 必须
                         velodyne=False, # True 必须
                         calib=False,#  label_info  不必须
                         image_ids=7481, # 循环次数
                         extend_matrix=True,
                         num_worker=8,
                         relative_path=True,
                         with_imageshape=True):
    """
    KITTI annotation format version 2:
    {
        [optional]points: [N, 3+] point cloud
        [optional, for kitti]image: { # 图像
            image_idx: ...
            image_path: ...
            image_shape: ...
        }
        point_cloud: {  点云
            num_features: 4
            velodyne_path: ...
        }
        [optional, for kitti]calib: { # 标定
            R0_rect: ...
            Tr_velo_to_cam: ...
            P2: ...
        }
        annos: {
            location: [num_gt, 3] array
            dimensions: [num_gt, 3] array
            rotation_y: [num_gt] angle array
            name: [num_gt] ground truth name array
            [optional]difficulty: kitti difficulty
            [optional]group_ids: used for multi-part object
        }
    }
    """
    root_path = Path(path)
    if not isinstance(image_ids, list): # image_ids=7481,
        image_ids = list(range(image_ids)) # 转成list格式============================
    #处理单帧数据 [>>>>>>>>>>>>>>>>>>>>>>>>>>>] 3712/3712, 
    def map_func(idx):
        # 三块： 点云信息+标定+图像
        info = {}
        pc_info = {'num_features': 4} # 点云信息
        calib_info = {} # 标定信息
        image_info = {'image_idx': idx} # 图像信息

        annotations = None
        if velodyne:
            pc_info['velodyne_path'] = get_velodyne_path(
                idx, path, training, relative_path) # 路径
        image_info['image_path'] = get_image_path(idx, path, training,
                                                  relative_path)
        if with_imageshape:
            img_path = image_info['image_path']
            if relative_path:
                img_path = str(root_path / img_path)
            image_info['image_shape'] = np.array(
                io.imread(img_path).shape[:2], dtype=np.int32)
        if label_info: # 有label信息
            label_path = get_label_path(idx, path, training, relative_path) # 得到路径
            if relative_path:
                label_path = str(root_path / label_path)
            annotations = get_label_anno(label_path) # GT数据 处理需要修改
        info['image'] = image_info # 图像信息
        info['point_cloud'] = pc_info # 路径信息
        if calib:
            calib_path = get_calib_path(
                idx, path, training, relative_path=False)
            with open(calib_path, 'r') as f:
                lines = f.readlines()
            P0 = np.array([float(info) for info in lines[0].split(' ')[1:13]
                           ]).reshape([3, 4])
            P1 = np.array([float(info) for info in lines[1].split(' ')[1:13]
                           ]).reshape([3, 4])
            P2 = np.array([float(info) for info in lines[2].split(' ')[1:13]
                           ]).reshape([3, 4])
            P3 = np.array([float(info) for info in lines[3].split(' ')[1:13]
                           ]).reshape([3, 4])
            if extend_matrix:
                P0 = _extend_matrix(P0)
                P1 = _extend_matrix(P1)
                P2 = _extend_matrix(P2)
                P3 = _extend_matrix(P3)
            R0_rect = np.array([
                float(info) for info in lines[4].split(' ')[1:10]
            ]).reshape([3, 3])
            if extend_matrix:
                rect_4x4 = np.zeros([4, 4], dtype=R0_rect.dtype)
                rect_4x4[3, 3] = 1.
                rect_4x4[:3, :3] = R0_rect
            else:
                rect_4x4 = R0_rect

            Tr_velo_to_cam = np.array([
                float(info) for info in lines[5].split(' ')[1:13]
            ]).reshape([3, 4])
            Tr_imu_to_velo = np.array([
                float(info) for info in lines[6].split(' ')[1:13]
            ]).reshape([3, 4])
            if extend_matrix:
                Tr_velo_to_cam = _extend_matrix(Tr_velo_to_cam)
                Tr_imu_to_velo = _extend_matrix(Tr_imu_to_velo)
            calib_info['P0'] = P0
            calib_info['P1'] = P1
            calib_info['P2'] = P2
            calib_info['P3'] = P3
            calib_info['R0_rect'] = rect_4x4
            calib_info['Tr_velo_to_cam'] = Tr_velo_to_cam
            calib_info['Tr_imu_to_velo'] = Tr_imu_to_velo
            info['calib'] = calib_info

        if annotations is not None: 
            info['annos'] = annotations
            add_difficulty_to_annos(info)
        return info # 返回info
    # 循环处理    [>>>>>>>>>>>>>>>>>>>>>>>>>>>] 3712/3712
    with futures.ThreadPoolExecutor(num_worker) as executor:
        image_infos = executor.map(map_func, image_ids) # n info # 返回info

    return list(image_infos) # 返回

# ouster主函数!!!!!! include    get_label_anno_ouster(label_path)====================================================================
def get_ouster_image_info(path,
                         training=True,
                         label_info=True, # label_info 必须
                         velodyne=False, # True 必须
                         calib=False,#  label_info  不必须
                         image_ids=7481,
                         extend_matrix=True,
                         num_worker=8,
                         relative_path=True,
                         with_imageshape=True):
    """
    ouster annotation format version 2:
    {
        [optional]points: [N, 3+] point cloud
        [optional, for kitti]image: { # 图像
            image_idx: ...
            image_path: ...
            image_shape: ...
        }
        point_cloud: {  点云
            num_features: 4
            velodyne_path: ...
        }
        [optional, for kitti]calib: { # 标定
            R0_rect: ...
            Tr_velo_to_cam: ...
            P2: ...
        }
        annos: {
            location: [num_gt, 3] array
            dimensions: [num_gt, 3] array
            rotation_y: [num_gt] angle array
            name: [num_gt] ground truth name array
            [optional]difficulty: kitti difficulty
            [optional]group_ids: used for multi-part object
        }
    }
    """
    root_path = Path(path)
    if not isinstance(image_ids, list): # image_ids=7481,
        image_ids = list(range(image_ids)) # 转成list格式
    #处理单帧数据
    def map_func(idx):
        info = {}
        pc_info = {'pc_idx': idx, 'num_features': 4} # 添加下标
        calib_info = {}
        image_info = {'image_idx': idx}

        annotations = None
        if velodyne: # 原始点云数据======================================
            pc_info['velodyne_path'] = get_velodyne_path(
                idx, path, training, relative_path)
        # image_info['image_path'] = get_image_path(idx, path, training,
        #                                           relative_path)
        # if with_imageshape:
        #     img_path = image_info['image_path']
        #     if relative_path:
        #         img_path = str(root_path / img_path)
            # image_info['image_shape'] = np.array(
            #     io.imread(img_path).shape[:2], dtype=np.int32)
        if label_info: # 有label信息
            label_path = get_label_path(idx, path, training, relative_path) # 得到路径
            if relative_path:
                label_path = str(root_path / label_path)
            annotations = get_label_anno_ouster(label_path) # GT数据=====================================================最重要
        # info['image'] = image_info # 图像信息
        info['point_cloud'] = pc_info # 点云数据
        # if calib:
        #     calib_path = get_calib_path(
        #         idx, path, training, relative_path=False)
        #     with open(calib_path, 'r') as f:
        #         lines = f.readlines()
        #     P0 = np.array([float(info) for info in lines[0].split(' ')[1:13]
        #                    ]).reshape([3, 4])
        #     P1 = np.array([float(info) for info in lines[1].split(' ')[1:13]
        #                    ]).reshape([3, 4])
        #     P2 = np.array([float(info) for info in lines[2].split(' ')[1:13]
        #                    ]).reshape([3, 4])
        #     P3 = np.array([float(info) for info in lines[3].split(' ')[1:13]
        #                    ]).reshape([3, 4])
        #     if extend_matrix:
        #         P0 = _extend_matrix(P0)
        #         P1 = _extend_matrix(P1)
        #         P2 = _extend_matrix(P2)
        #         P3 = _extend_matrix(P3)
        #     R0_rect = np.array([
        #         float(info) for info in lines[4].split(' ')[1:10]
        #     ]).reshape([3, 3])
        #     if extend_matrix:
        #         rect_4x4 = np.zeros([4, 4], dtype=R0_rect.dtype)
        #         rect_4x4[3, 3] = 1.
        #         rect_4x4[:3, :3] = R0_rect
        #     else:
        #         rect_4x4 = R0_rect
        #
        #     Tr_velo_to_cam = np.array([
        #         float(info) for info in lines[5].split(' ')[1:13]
        #     ]).reshape([3, 4])
        #     Tr_imu_to_velo = np.array([
        #         float(info) for info in lines[6].split(' ')[1:13]
        #     ]).reshape([3, 4])
        #     if extend_matrix:
        #         Tr_velo_to_cam = _extend_matrix(Tr_velo_to_cam)
        #         Tr_imu_to_velo = _extend_matrix(Tr_imu_to_velo)
        #     calib_info['P0'] = P0
        #     calib_info['P1'] = P1
        #     calib_info['P2'] = P2
        #     calib_info['P3'] = P3
        #     calib_info['R0_rect'] = rect_4x4
        #     calib_info['Tr_velo_to_cam'] = Tr_velo_to_cam
        #     calib_info['Tr_imu_to_velo'] = Tr_imu_to_velo
        #     info['calib'] = calib_info

        if annotations is not None: 
            info['annos'] = annotations
            # add_difficulty_to_annos(info) #
        return info # 返回info

    with futures.ThreadPoolExecutor(num_worker) as executor:
        image_infos = executor.map(map_func, image_ids) # n info # 返回info

    return list(image_infos) # 返回

# waymo主要函数
def get_waymo_image_info(path,
                         training=True,
                         label_info=True,
                         velodyne=False,
                         calib=False,
                         pose=False, # False
                         image_ids=7481,
                         extend_matrix=True,
                         num_worker=8,
                         relative_path=True,
                         with_imageshape=True,
                         max_sweeps=5):
    """
    Waymo annotation format version like KITTI:
    {
        [optional]points: [N, 3+] point cloud
        [optional, for kitti]image: {
            image_idx: ...
            image_path: ...
            image_shape: ...
        }
        point_cloud: {
            num_features: 6
            velodyne_path: ...
        }
        [optional, for kitti]calib: {
            R0_rect: ...
            Tr_velo_to_cam0: ...
            P0: ...
        }
        annos: {
            location: [num_gt, 3] array
            dimensions: [num_gt, 3] array
            rotation_y: [num_gt] angle array
            name: [num_gt] ground truth name array
            [optional]difficulty: kitti difficulty
            [optional]group_ids: used for multi-part object
        }
    }
    """
    root_path = Path(path)
    if not isinstance(image_ids, list):
        image_ids = list(range(image_ids))

    def map_func(idx):
        info = {}
        pc_info = {'num_features': 6}
        calib_info = {}

        image_info = {'image_idx': idx}
        annotations = None
        if velodyne:
            pc_info['velodyne_path'] = get_velodyne_path(
                idx, path, training, relative_path, use_prefix_id=True)
            points = np.fromfile(
                Path(path) / pc_info['velodyne_path'], dtype=np.float32)
            points = np.copy(points).reshape(-1, pc_info['num_features'])
            info['timestamp'] = np.int64(points[0, -1])
            # values of the last dim are all the timestamp
        image_info['image_path'] = get_image_path(
            idx,
            path,
            training,
            relative_path,
            info_type='image_0',
            use_prefix_id=True)
        if with_imageshape:
            img_path = image_info['image_path']
            if relative_path:
                img_path = str(root_path / img_path)
            image_info['image_shape'] = np.array(
                io.imread(img_path).shape[:2], dtype=np.int32)
        if label_info:
            label_path = get_label_path(
                idx,
                path,
                training,
                relative_path,
                info_type='label_all',
                use_prefix_id=True)
            if relative_path:
                label_path = str(root_path / label_path)
            annotations = get_label_anno(label_path)
        info['image'] = image_info
        info['point_cloud'] = pc_info
        if calib:
            calib_path = get_calib_path(
                idx, path, training, relative_path=False, use_prefix_id=True)
            with open(calib_path, 'r') as f:
                lines = f.readlines()
            P0 = np.array([float(info) for info in lines[0].split(' ')[1:13]
                           ]).reshape([3, 4])
            P1 = np.array([float(info) for info in lines[1].split(' ')[1:13]
                           ]).reshape([3, 4])
            P2 = np.array([float(info) for info in lines[2].split(' ')[1:13]
                           ]).reshape([3, 4])
            P3 = np.array([float(info) for info in lines[3].split(' ')[1:13]
                           ]).reshape([3, 4])
            P4 = np.array([float(info) for info in lines[4].split(' ')[1:13]
                           ]).reshape([3, 4])
            if extend_matrix:
                P0 = _extend_matrix(P0)
                P1 = _extend_matrix(P1)
                P2 = _extend_matrix(P2)
                P3 = _extend_matrix(P3)
                P4 = _extend_matrix(P4)
            R0_rect = np.array([
                float(info) for info in lines[5].split(' ')[1:10]
            ]).reshape([3, 3])
            if extend_matrix:
                rect_4x4 = np.zeros([4, 4], dtype=R0_rect.dtype)
                rect_4x4[3, 3] = 1.
                rect_4x4[:3, :3] = R0_rect
            else:
                rect_4x4 = R0_rect

            Tr_velo_to_cam = np.array([
                float(info) for info in lines[6].split(' ')[1:13]
            ]).reshape([3, 4])
            if extend_matrix:
                Tr_velo_to_cam = _extend_matrix(Tr_velo_to_cam)
            calib_info['P0'] = P0
            calib_info['P1'] = P1
            calib_info['P2'] = P2
            calib_info['P3'] = P3
            calib_info['P4'] = P4
            calib_info['R0_rect'] = rect_4x4
            calib_info['Tr_velo_to_cam'] = Tr_velo_to_cam
            info['calib'] = calib_info
        if pose: # False 没有
            pose_path = get_pose_path(
                idx, path, training, relative_path=False, use_prefix_id=True)
            info['pose'] = np.loadtxt(pose_path)

        if annotations is not None:
            info['annos'] = annotations
            info['annos']['camera_id'] = info['annos'].pop('score')
            add_difficulty_to_annos(info) # waymo

        sweeps = []
        prev_idx = idx
        while len(sweeps) < max_sweeps:
            prev_info = {}
            prev_idx -= 1
            prev_info['velodyne_path'] = get_velodyne_path(
                prev_idx,
                path,
                training,
                relative_path,
                exist_check=False,
                use_prefix_id=True)
            if_prev_exists = osp.exists(
                Path(path) / prev_info['velodyne_path'])
            if if_prev_exists:
                prev_points = np.fromfile(
                    Path(path) / prev_info['velodyne_path'], dtype=np.float32)
                prev_points = np.copy(prev_points).reshape(
                    -1, pc_info['num_features'])
                prev_info['timestamp'] = np.int64(prev_points[0, -1])
                prev_pose_path = get_pose_path(
                    prev_idx,
                    path,
                    training,
                    relative_path=False,
                    use_prefix_id=True)
                prev_info['pose'] = np.loadtxt(prev_pose_path)
                sweeps.append(prev_info)
            else:
                break
        info['sweeps'] = sweeps

        return info

    with futures.ThreadPoolExecutor(num_worker) as executor:
        image_infos = executor.map(map_func, image_ids)

    return list(image_infos)

def kitti_anno_to_label_file(annos, folder):
    folder = Path(folder)
    for anno in annos:
        image_idx = anno['metadata']['image_idx']
        label_lines = []
        for j in range(anno['bbox'].shape[0]):
            label_dict = {
                'name': anno['name'][j],
                'alpha': anno['alpha'][j],
                'bbox': anno['bbox'][j],
                'location': anno['location'][j],
                'dimensions': anno['dimensions'][j],
                'rotation_y': anno['rotation_y'][j],
                'score': anno['score'][j],
            }
            label_line = kitti_result_line(label_dict)
            label_lines.append(label_line)
        label_file = folder / f'{get_image_index_str(image_idx)}.txt'
        label_str = '\n'.join(label_lines)
        with open(label_file, 'w') as f:
            f.write(label_str)

# 添加annos['difficulty']   用于gt数据库增强
def add_difficulty_to_annos(info):
    min_height = [40, 25,
                  25]  # minimum height for evaluated groundtruth/detections
    max_occlusion = [
        0, 1, 2
    ]  # maximum occlusion level of the groundtruth used for evaluation
    max_trunc = [
        0.15, 0.3, 0.5
    ]  # maximum truncation level of the groundtruth used for evaluation
    annos = info['annos']
    dims = annos['dimensions']  # lhw format
    bbox = annos['bbox']
    height = bbox[:, 3] - bbox[:, 1] # 使用到了bbox
    occlusion = annos['occluded']
    truncation = annos['truncated']
    diff = []
    easy_mask = np.ones((len(dims), ), dtype=np.bool)
    moderate_mask = np.ones((len(dims), ), dtype=np.bool)
    hard_mask = np.ones((len(dims), ), dtype=np.bool)
    i = 0
    for h, o, t in zip(height, occlusion, truncation):
        if o > max_occlusion[0] or h <= min_height[0] or t > max_trunc[0]:
            easy_mask[i] = False
        if o > max_occlusion[1] or h <= min_height[1] or t > max_trunc[1]:
            moderate_mask[i] = False
        if o > max_occlusion[2] or h <= min_height[2] or t > max_trunc[2]:
            hard_mask[i] = False
        i += 1
    is_easy = easy_mask
    is_moderate = np.logical_xor(easy_mask, moderate_mask)
    is_hard = np.logical_xor(hard_mask, moderate_mask)

    for i in range(len(dims)):
        if is_easy[i]:
            diff.append(0)
        elif is_moderate[i]:
            diff.append(1)
        elif is_hard[i]:
            diff.append(2)
        else:
            diff.append(-1)
    annos['difficulty'] = np.array(diff, np.int32)
    return diff


def kitti_result_line(result_dict, precision=4):
    prec_float = '{' + ':.{}f'.format(precision) + '}'
    res_line = []
    all_field_default = OrderedDict([
        ('name', None),
        ('truncated', -1),
        ('occluded', -1),
        ('alpha', -10),
        ('bbox', None),
        ('dimensions', [-1, -1, -1]),
        ('location', [-1000, -1000, -1000]),
        ('rotation_y', -10),
        ('score', 0.0),
    ])
    res_dict = [(key, None) for key, val in all_field_default.items()]
    res_dict = OrderedDict(res_dict)
    for key, val in result_dict.items():
        if all_field_default[key] is None and val is None:
            raise ValueError('you must specify a value for {}'.format(key))
        res_dict[key] = val

    for key, val in res_dict.items():
        if key == 'name':
            res_line.append(val)
        elif key in ['truncated', 'alpha', 'rotation_y', 'score']:
            if val is None:
                res_line.append(str(all_field_default[key]))
            else:
                res_line.append(prec_float.format(val))
        elif key == 'occluded':
            if val is None:
                res_line.append(str(all_field_default[key]))
            else:
                res_line.append('{}'.format(val))
        elif key in ['bbox', 'dimensions', 'location']:
            if val is None:
                res_line += [str(v) for v in all_field_default[key]]
            else:
                res_line += [prec_float.format(v) for v in val]
        else:
            raise ValueError('unknown key. supported key:{}'.format(
                res_dict.keys()))
    return ' '.join(res_line)
