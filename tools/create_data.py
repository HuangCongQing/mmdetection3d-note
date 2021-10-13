# Copyright (c) OpenMMLab. All rights reserved.
# 生成pkl文件
import argparse
from os import path as osp

from tools.data_converter import indoor_converter as indoor
from tools.data_converter import kitti_converter as kitti # kitti数据转换=====
from tools.data_converter import lyft_converter as lyft_converter
from tools.data_converter import nuscenes_converter as nuscenes_converter
from tools.data_converter.create_gt_database import create_groundtruth_database # gt数据库

# kitti数据预处理生成pkl文件
# python tools/create_data.py kitti --root-path ./data/kitti --out-dir ./data/kitti --extra-tag kitti
def kitti_data_prep(root_path, info_prefix, version, out_dir):
    """Prepare data related to Kitti dataset.

    Related data consists of '.pkl' files recording basic infos,
    2D annotations and groundtruth database.

    Args:
        root_path (str): Path of dataset root.  参数：./data/kitti
        info_prefix (str): The prefix of info filenames.  参数：kitti
        version (str): Dataset version.
        out_dir (str): Output directory of the groundtruth database info.  gtbase参数  ./data/kitti
    """
    kitti.create_kitti_info_file(root_path, info_prefix) # 第一步
    kitti.create_reduced_point_cloud(root_path, info_prefix) #
    # 4个pkl文件路径
    info_train_path = osp.join(root_path, f'{info_prefix}_infos_train.pkl')
    info_val_path = osp.join(root_path, f'{info_prefix}_infos_val.pkl')
    info_trainval_path = osp.join(root_path,
                                  f'{info_prefix}_infos_trainval.pkl')
    info_test_path = osp.join(root_path, f'{info_prefix}_infos_test.pkl')
    # 生成4个pkl文件(得到2D的标注信息)
    kitti.export_2d_annotation(root_path, info_train_path)
    kitti.export_2d_annotation(root_path, info_val_path)
    kitti.export_2d_annotation(root_path, info_trainval_path)
    kitti.export_2d_annotation(root_path, info_test_path)
    # 创建gtbase
    create_groundtruth_database( # tools/data_converter/create_gt_database.py
        'KittiDataset',
        root_path,
        info_prefix,
        f'{out_dir}/{info_prefix}_infos_train.pkl',  # data/kitti/kitti_dbinfos_train.pkl
        relative_path=False,
        mask_anno_path='instances_train.json',
        with_mask=(version == 'mask'))


def nuscenes_data_prep(root_path,
                       info_prefix,
                       version,
                       dataset_name,
                       out_dir,
                       max_sweeps=10):
    """Prepare data related to nuScenes dataset.

    Related data consists of '.pkl' files recording basic infos,
    2D annotations and groundtruth database.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        version (str): Dataset version.
        dataset_name (str): The dataset class name.
        out_dir (str): Output directory of the groundtruth database info.
        max_sweeps (int): Number of input consecutive frames. Default: 10
    """
    nuscenes_converter.create_nuscenes_infos(
        root_path, info_prefix, version=version, max_sweeps=max_sweeps)

    if version == 'v1.0-test':
        info_test_path = osp.join(root_path, f'{info_prefix}_infos_test.pkl')
        nuscenes_converter.export_2d_annotation(
            root_path, info_test_path, version=version)
        return

    info_train_path = osp.join(root_path, f'{info_prefix}_infos_train.pkl')
    info_val_path = osp.join(root_path, f'{info_prefix}_infos_val.pkl')
    nuscenes_converter.export_2d_annotation(
        root_path, info_train_path, version=version)
    nuscenes_converter.export_2d_annotation(
        root_path, info_val_path, version=version)
    create_groundtruth_database(dataset_name, root_path, info_prefix,
                                f'{out_dir}/{info_prefix}_infos_train.pkl')


def lyft_data_prep(root_path, info_prefix, version, max_sweeps=10):
    """Prepare data related to Lyft dataset.

    Related data consists of '.pkl' files recording basic infos.
    Although the ground truth database and 2D annotations are not used in
    Lyft, it can also be generated like nuScenes.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        version (str): Dataset version.
        max_sweeps (int, optional): Number of input consecutive frames.
            Defaults to 10.
    """
    lyft_converter.create_lyft_infos(
        root_path, info_prefix, version=version, max_sweeps=max_sweeps)


def scannet_data_prep(root_path, info_prefix, out_dir, workers):
    """Prepare the info file for scannet dataset.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        out_dir (str): Output directory of the generated info file.
        workers (int): Number of threads to be used.
    """
    indoor.create_indoor_info_file(
        root_path, info_prefix, out_dir, workers=workers)


def s3dis_data_prep(root_path, info_prefix, out_dir, workers):
    """Prepare the info file for s3dis dataset.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        out_dir (str): Output directory of the generated info file.
        workers (int): Number of threads to be used.
    """
    indoor.create_indoor_info_file(
        root_path, info_prefix, out_dir, workers=workers)


def sunrgbd_data_prep(root_path, info_prefix, out_dir, workers):
    """Prepare the info file for sunrgbd dataset.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        out_dir (str): Output directory of the generated info file.
        workers (int): Number of threads to be used.
    """
    indoor.create_indoor_info_file(
        root_path, info_prefix, out_dir, workers=workers)

# waymo
def waymo_data_prep(root_path,
                    info_prefix,
                    version,
                    out_dir,
                    workers,
                    max_sweeps=5):
    """Prepare the info file for waymo dataset.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        out_dir (str): Output directory of the generated info file.
        workers (int): Number of threads to be used.
        max_sweeps (int): Number of input consecutive frames. Default: 5 \
            Here we store pose information of these frames for later use.
    """
    from tools.data_converter import waymo_converter as waymo # waymo转化为kitti格式

    splits = ['training', 'validation', 'testing']
    for i, split in enumerate(splits):
        load_dir = osp.join(root_path, 'waymo_format', split)
        if split == 'validation':
            save_dir = osp.join(out_dir, 'kitti_format', 'training')
        else:
            save_dir = osp.join(out_dir, 'kitti_format', split)
        converter = waymo.Waymo2KITTI(
            load_dir,
            save_dir,
            prefix=str(i),
            workers=workers,
            test_mode=(split == 'test'))
        converter.convert() # 转换
    # Generate waymo infos
    out_dir = osp.join(out_dir, 'kitti_format')
    kitti.create_waymo_info_file(out_dir, info_prefix, max_sweeps=max_sweeps) # 创建waymo_info文件 pkl
    create_groundtruth_database(
        'WaymoDataset',
        out_dir,
        info_prefix,
        f'{out_dir}/{info_prefix}_infos_train.pkl',
        relative_path=False,
        with_mask=False)

# 新建ouster
# python tools/create_data.py ouster --root-path ./data/ouster --out-dir ./data/ouster --extra-tag ouster
def ouster_data_prep(root_path, info_prefix, version, out_dir):
    """Prepare data related to ouster dataset.

    Related data consists of '.pkl' files recording basic infos,
    2D annotations and groundtruth database.

    Args:
        root_path (str): Path of dataset root.  参数：./data/ouster
        info_prefix (str): The prefix of info filenames.  参数：ouster
        version (str): Dataset version.
        out_dir (str): Output directory of the groundtruth database info.  gtbase参数  ./data/ouster
    """
    kitti.create_ouster_info_file(root_path, info_prefix) # 第一步  ouster
    # kitti.create_reduced_point_cloud(root_path, info_prefix) #
    # 4个pkl文件路径
    info_train_path = osp.join(root_path, f'{info_prefix}_infos_train.pkl')
    info_val_path = osp.join(root_path, f'{info_prefix}_infos_val.pkl')
    info_trainval_path = osp.join(root_path,
                                  f'{info_prefix}_infos_trainval.pkl')
    info_test_path = osp.join(root_path, f'{info_prefix}_infos_test.pkl')
    # 生成4个pkl文件(得到2D的标注信息)
    # kitti.export_2d_annotation(root_path, info_train_path)
    # kitti.export_2d_annotation(root_path, info_val_path)
    # kitti.export_2d_annotation(root_path, info_trainval_path)
    # kitti.export_2d_annotation(root_path, info_test_path)
    # 创建gtbase
    create_groundtruth_database( # tools/data_converter/create_gt_database.py
        'OusterDataset',
        root_path,
        info_prefix,
        f'{out_dir}/{info_prefix}_infos_train.pkl',  # data/kitti/kitti_dbinfos_train.pkl
        relative_path=False,
        mask_anno_path='instances_train.json',
        with_mask=(version == 'mask'))


parser = argparse.ArgumentParser(description='Data converter arg parser')
parser.add_argument('dataset', metavar='kitti', help='name of the dataset')
parser.add_argument(
    '--root-path', # --root-path ./data/kitti --out-dir ./data/kitti --extra-tag kitti
    type=str,
    default='./data/kitti',
    help='specify the root path of dataset')
parser.add_argument(
    '--version',
    type=str,
    default='v1.0',
    required=False,
    help='specify the dataset version, no need for kitti')
parser.add_argument(
    '--max-sweeps',
    type=int,
    default=10,
    required=False,
    help='specify sweeps of lidar per example')
parser.add_argument(
    '--out-dir', # --root-path ./data/kitti --out-dir ./data/kitti --extra-tag kitti
    type=str,
    default='./data/kitti',
    required='False',
    help='name of info pkl')
parser.add_argument('--extra-tag', type=str, default='kitti')
parser.add_argument(
    '--workers', type=int, default=4, help='number of threads to be used')
args = parser.parse_args()

# python tools/create_data.py kitti --root-path ./data/kitti --out-dir ./data/kitti --extra-tag kitti
if __name__ == '__main__':
    if args.dataset == 'kitti': # 判断kitti参数 
        kitti_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=args.version,
            out_dir=args.out_dir)
    elif args.dataset == 'nuscenes' and args.version != 'v1.0-mini':
        train_version = f'{args.version}-trainval'
        nuscenes_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=train_version,
            dataset_name='NuScenesDataset',
            out_dir=args.out_dir,
            max_sweeps=args.max_sweeps)
        test_version = f'{args.version}-test'
        nuscenes_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=test_version,
            dataset_name='NuScenesDataset',
            out_dir=args.out_dir,
            max_sweeps=args.max_sweeps)
    elif args.dataset == 'nuscenes' and args.version == 'v1.0-mini':
        train_version = f'{args.version}'
        nuscenes_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=train_version,
            dataset_name='NuScenesDataset',
            out_dir=args.out_dir,
            max_sweeps=args.max_sweeps)
    elif args.dataset == 'lyft':
        train_version = f'{args.version}-train'
        lyft_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=train_version,
            max_sweeps=args.max_sweeps)
        test_version = f'{args.version}-test'
        lyft_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=test_version,
            max_sweeps=args.max_sweeps)
    elif args.dataset == 'waymo': # waymo
        waymo_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=args.version,
            out_dir=args.out_dir,
            workers=args.workers,
            max_sweeps=args.max_sweeps)
    elif args.dataset == 'scannet':
        scannet_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            out_dir=args.out_dir,
            workers=args.workers)
    elif args.dataset == 's3dis':
        s3dis_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            out_dir=args.out_dir,
            workers=args.workers)
    elif args.dataset == 'sunrgbd':
        sunrgbd_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            out_dir=args.out_dir,
            workers=args.workers)
    elif args.dataset == 'ouster': # 判断ouster参数 ============================================
        ouster_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=args.version,
            out_dir=args.out_dir)
