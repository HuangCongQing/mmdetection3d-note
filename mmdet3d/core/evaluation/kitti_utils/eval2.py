'''
Description: https://blog.csdn.net/weixin_44128857/article/details/108532437
Author: HCQ
Company(School): UCAS
Email: 1756260160@qq.com
Date: 2021-10-16 21:35:10
LastEditTime: 2021-10-16 21:35:10
FilePath: /mmdetection3d/mmdet3d/core/evaluation/kitti_utils/eval3.py
'''
import io as sysio

import numba
import numpy as np

from .rotate_iou import rotate_iou_gpu_eval


@numba.jit
def get_thresholds(scores: np.ndarray, num_gt, num_sample_pts=41):
    scores.sort()             # 将得分的一维数组 升序排列，如[1,2,3,4]
    scores = scores[::-1]        # 再将得分数组降序排列
    current_recall = 0  
    thresholds = []
    for i, score in enumerate(scores):
        l_recall = (i + 1) / num_gt
        if i < (len(scores) - 1):
            r_recall = (i + 2) / num_gt
        else:
            r_recall = l_recall
        if (((r_recall - current_recall) < (current_recall - l_recall)) and (i < (len(scores) - 1))):
            continue
    
        thresholds.append(score)
        current_recall += 1 / (num_sample_pts - 1.0)
    return thresholds


#这个函数是处理一帧的数据, current_class是5个类别中的其中一类
def clean_data(gt_anno, dt_anno, current_class, difficulty):
    
    '''
        print("____________clean_data() args:________________")
        print('current_class  :  ',current_class)
        print('difficulty : ',difficulty)
            ____________clean_data() args:________________
            current_class  :   0
            difficulty :  0
    '''

    CLASS_NAMES = ['vehicle', 'big_vehicle', 'pedestrian','bicycle','cone','huge_vehicle','motorcycle','tricycle','unknown']

    ignored_gt, ignored_dt =  [], []

    # 这一句的作用是：将current_class中对应的类别的名字找出来，
    # 如0 对应 vehicle。方法.lower()的意思是将字符串中的大写全部变成小写
    current_cls_name = CLASS_NAMES[current_class].lower()

    '''
        print("________________current_cls_name________________")
        print(current_cls_name)
        #得到的是：vehicle
    '''

    # 获取当前帧中物体object的个数
    num_gt = len(gt_anno["name"])
    num_dt = len(dt_anno["name"])
    num_valid_gt = 0

    #对num_gt中每一个物体object：
    for i in range(num_gt):

        #获取这个物体的name，并小写
        gt_name = gt_anno["name"][i].lower()

        valid_class = -1

        # 如果该物体正好是 需要处理的当前的object，将valid_class值为 1
        if (gt_name == current_cls_name):
            valid_class = 1
        else:
            valid_class = -1
        
        ignore = False
        if valid_class == 1 and not ignore:
            # 如果 为有效的物体， 且该物体object不忽略，
            # 则ignored_gt上该值为0，有效的物体数num_valid_gt+1
            ignored_gt.append(0)
            num_valid_gt += 1
        else:
            ignored_gt.append(-1)

    #对num_dt中每一个物体object：
    for i in range(num_dt):
        if (dt_anno["name"][i].lower() == current_cls_name):
            valid_class = 1
        else:
            valid_class = -1

        if valid_class == 1:
            ignored_dt.append(0)
        else:
            ignored_dt.append(-1)
    
    '''
        print("__________num_valid_gt____________")
        print(num_valid_gt)
        print("__________ignored_gt____________")
        print(ignored_gt)
        print("__________ignored_dt____________")
        print(ignored_dt)
    
        该函数的输出结果是
            __________num_valid_gt____________
            76
            __________ignored_gt____________
            [0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, -1, 
            -1, -1, -1, 0, -1, 0, 0, -1, 0, 0, 0, 0, 0, -1, 0, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, 
            -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, -1, 0, 0, -1, -1, -1, -1, -1, -1, 
            -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0, -1, 0, -1, -1, -1, -1, -1, 0, 0,
            0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 
            -1, -1, -1, 0, -1, 0, 0, -1, -1, -1, -1, 0, -1, -1, -1, 0, 0, -1]
            __________ignored_dt____________
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, -1, -1, 0, -1, 0, 0, -1, -1,
            0, 0, 0, -1, -1, -1, -1, -1, 0, 0, -1, -1, 0, 0, 0, -1, 0, 0, 0, -1, 
            -1, -1, -1, 0, -1, 0, -1, -1, 0, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1, 
            0, -1, -1, 0, 0, -1, -1, -1, 0, -1, 0, -1, 0, -1, 0, -1, -1, -1, -1, -1, 
            -1, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, 0, -1, -1, -1, 0, 
            0, 0, -1, -1, -1, -1, -1, 0, -1, 0, -1, 0, 0, -1, -1, -1, -1, -1, -1]
    '''
    return num_valid_gt, ignored_gt, ignored_dt


@numba.jit(nopython=True)
def image_box_overlap(boxes, query_boxes, criterion=-1):
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = np.zeros((N, K), dtype=boxes.dtype)
    for k in range(K):
        qbox_area = ((query_boxes[k, 2] - query_boxes[k, 0]) *
                     (query_boxes[k, 3] - query_boxes[k, 1]))
        for n in range(N):
            iw = (min(boxes[n, 2], query_boxes[k, 2]) -
                  max(boxes[n, 0], query_boxes[k, 0]))
            if iw > 0:
                ih = (min(boxes[n, 3], query_boxes[k, 3]) -
                      max(boxes[n, 1], query_boxes[k, 1]))
                if ih > 0:
                    if criterion == -1:
                        ua = (
                            (boxes[n, 2] - boxes[n, 0]) *
                            (boxes[n, 3] - boxes[n, 1]) + qbox_area - iw * ih)
                    elif criterion == 0:
                        ua = ((boxes[n, 2] - boxes[n, 0]) *
                              (boxes[n, 3] - boxes[n, 1]))
                    elif criterion == 1:
                        ua = qbox_area
                    else:
                        ua = 1.0
                    overlaps[n, k] = iw * ih / ua
    return overlaps


def bev_box_overlap(boxes, qboxes, criterion=-1):
    riou = rotate_iou_gpu_eval(boxes, qboxes, criterion)
    return riou


#@numba.jit(nopython=True, parallel=True)
def d3_box_overlap_kernel(boxes, qboxes, rinc, criterion=-1):
    # ONLY support overlap in CAMERA, not lider.
    N, K = boxes.shape[0], qboxes.shape[0]
    for i in range(N):
        for j in range(K):
            if rinc[i, j] > 0:
                # iw = (min(boxes[i, 1] + boxes[i, 4], qboxes[j, 1] +
                #         qboxes[j, 4]) - max(boxes[i, 1], qboxes[j, 1]))
                iw = (min(boxes[i, 1], qboxes[j, 1]) - max(
                    boxes[i, 1] - boxes[i, 4], qboxes[j, 1] - qboxes[j, 4]))

                if iw > 0:
                    area1 = boxes[i, 3] * boxes[i, 4] * boxes[i, 5]
                    area2 = qboxes[j, 3] * qboxes[j, 4] * qboxes[j, 5]
                    inc = iw * rinc[i, j]
                    if criterion == -1:
                        ua = (area1 + area2 - inc)
                    elif criterion == 0:
                        ua = area1
                    elif criterion == 1:
                        ua = area2
                    else:
                        ua = inc
                    rinc[i, j] = inc / ua
                else:
                    rinc[i, j] = 0.0


def d3_box_overlap(boxes, qboxes, criterion=-1):
    rinc = rotate_iou_gpu_eval(boxes[:, [0, 2, 3, 5, 6]],
                               qboxes[:, [0, 2, 3, 5, 6]], 2)
    d3_box_overlap_kernel(boxes, qboxes, rinc, criterion)
    return rinc


@numba.jit(nopython=True)

def compute_statistics_jit(overlaps,
                           gt_datas,
                           dt_datas,
                           ignored_gt,
                           ignored_det,
                           metric,
                           min_overlap,
                           thresh=0,
                           compute_fp=False,
                           compute_aos=False):

    det_size = dt_datas.shape[0]
    gt_size = gt_datas

    dt_scores = dt_datas[:, -1]   #获取预测的得分情况
    #dt_scores = dt_datas

    assigned_detection = [False] * det_size # 存储是否每个检测都分配给了一个gt。
    ignored_threshold = [False] * det_size    # 如果检测分数低于阈值，则存储数组
    if compute_fp:
        for i in range(det_size):
            if (dt_scores[i] < thresh):
                ignored_threshold[i] = True
    
    NO_DETECTION = -10000000
    tp, fp, fn, similarity = 0, 0, 0, 0

    thresholds = np.zeros((gt_size,))
    thresh_idx = 0
    delta = np.zeros((gt_size, ))
    delta_idx = 0

    for i in range(gt_size):
        if ignored_gt[i] == -1:
            #如果不是当前class，如vehicle类别，
            # 则跳过当前循环，继续判断下一个类别
            continue

        det_idx = -1            #! 储存对此gt存储的最佳检测的idx
        valid_detection = NO_DETECTION      
        max_overlap = 0
        assigned_ignored_det = False

        # 遍历det中的所有数据，找到一个与真实值最高得分的框
        for j in range(det_size):
            # 如果该数据 无效，则跳过继续判断
            if (ignored_det[j] == -1):
                continue
            if (assigned_detection[j]):
                continue
            if (ignored_threshold[j]):
                continue

            # 获取 overlaps 中相应的数值
            overlap = overlaps[j, i]
            # 获取这个预测框的得分 
            dt_score = dt_scores[j]
            if (not compute_fp and (overlap > min_overlap) and dt_score > valid_detection):
                det_idx = j
                valid_detection = dt_score
            elif (compute_fp and (overlap > min_overlap)
                  and (overlap > max_overlap or assigned_ignored_det)
                  and ignored_det[j] == 0):
                max_overlap = overlap
                det_idx = j
                valid_detection = 1
                assigned_ignored_det = False
            elif (compute_fp and (overlap > min_overlap)
                  and (valid_detection == NO_DETECTION)
                  and ignored_det[j] == 1):
                # 不存在该类别，： ignored_det[j] == 1
                det_idx = j
                valid_detection = 1
                assigned_ignored_det = True

        if (valid_detection == NO_DETECTION) and ignored_gt[i] == 0:
            # 如果没有找到，valid_detection还等于 NO_DETECTION，
            # 且真实框确实属于vehicle类别，则fn+1
            fn += 1
        elif ((valid_detection != NO_DETECTION) and (ignored_gt[i] == 1 or ignored_det[det_idx] == 1)):
            # 这种情况不存在：ignored_gt[i] == 1
            assigned_detection[det_idx] = True
        elif valid_detection != NO_DETECTION:
            # 这种情况是检测出来了，且是正确的
            tp += 1
            # thresholds.append(dt_scores[det_idx])
            thresholds[thresh_idx] = dt_scores[det_idx]
            thresh_idx += 1
            
            assigned_detection[det_idx] = True
    
    
    if compute_fp:
        #遍历验证det中的每一个：
        for i in range(det_size):
            if (not (assigned_detection[i] or ignored_det[i] == -1
                     or ignored_det[i] == 1 or ignored_threshold[i])):
                fp += 1
        nstuff = 0
        fp -= nstuff

        if compute_aos:
            tmp = np.zeros((fp + delta_idx, ))
            # tmp = [0] * fp
            for i in range(delta_idx):
                tmp[i + fp] = (1.0 + np.cos(delta[i])) / 2.0
                # tmp.append((1.0 + np.cos(delta[i])) / 2.0)
            # assert len(tmp) == fp + tp
            # assert len(delta) == tp
            if tp > 0 or fp > 0:
                similarity = np.sum(tmp)
            else:
                similarity = -1
    return tp, fp, fn, similarity, thresholds[:thresh_idx]


#@numba.jit(nopython=True)
def compute_statistics_jit1(
                           overlaps,
                           gt_datas,
                           dt_datas,
                           ignored_gt,
                           ignored_det,
                           metric,
                           min_overlap,
                           thresh=0,
                           compute_fp=False,
                           compute_aos=False):

    #print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
    #print(ignored_gt)
    #print(ignored_det)
    det_size = dt_datas.shape[0]
    gt_size = gt_datas

    dt_scores = dt_datas  #获取预测的得分情况
    #dt_scores = dt_datas

    assigned_detection = [False] * det_size # 存储是否每个检测都分配给了一个gt。
    ignored_threshold = [False] * det_size    # 如果检测分数低于阈值，则存储数组
    if compute_fp:
        for i in range(det_size):
            if (dt_scores[i] < thresh):
                ignored_threshold[i] = True
    
    NO_DETECTION = -10000000
    tp, fp, fn, similarity = 0, 0, 0, 0

    thresholds = np.zeros((gt_size,))
    thresh_idx = 0
    delta = np.zeros((gt_size, ))
    delta_idx = 0

    for i in range(gt_size):
        if ignored_gt[i] == -1:
            #如果不是当前class，如vehicle类别，
            # 则跳过当前循环，继续判断下一个类别
            continue

        det_idx = -1            #! 储存对此gt存储的最佳检测的idx
        valid_detection = NO_DETECTION      
        max_overlap = 0
        assigned_ignored_det = False

        # 遍历det中的所有数据，找到一个与真实值最高得分的框
        for j in range(det_size):
            # 如果该数据 无效，则跳过继续判断
            if (ignored_det[j] == -1):
                continue
            if (assigned_detection[j]):
                continue
            if (ignored_threshold[j]):
                continue

            # 获取 overlaps 中相应的数值
            overlap = overlaps[j, i]
            # 获取这个预测框的得分 
            dt_score = dt_scores[j]
            if (not compute_fp and (overlap > min_overlap) and dt_score > valid_detection):
                det_idx = j
                valid_detection = dt_score
            elif (compute_fp and (overlap > min_overlap)
                  and (overlap > max_overlap or assigned_ignored_det)
                  and ignored_det[j] == 0):
                max_overlap = overlap
                det_idx = j
                valid_detection = 1
                assigned_ignored_det = False
            elif (compute_fp and (overlap > min_overlap)
                  and (valid_detection == NO_DETECTION)
                  and ignored_det[j] == 1):
                # 不存在该类别，： ignored_det[j] == 1
                det_idx = j
                valid_detection = 1
                assigned_ignored_det = True

        if (valid_detection == NO_DETECTION) and ignored_gt[i] == 0:
            # 如果没有找到，valid_detection还等于 NO_DETECTION，
            # 且真实框确实属于vehicle类别，则fn+1
            fn += 1
        elif ((valid_detection != NO_DETECTION) and (ignored_gt[i] == 1 or ignored_det[det_idx] == 1)):
            # 这种情况不存在：ignored_gt[i] == 1
            assigned_detection[det_idx] = True
        elif valid_detection != NO_DETECTION:
            # 这种情况是检测出来了，且是正确的
            tp += 1
            # thresholds.append(dt_scores[det_idx])
            thresholds[thresh_idx] = dt_scores[det_idx]
            thresh_idx += 1
            
            assigned_detection[det_idx] = True
    
    
    if compute_fp:
        #遍历验证det中的每一个：
        for i in range(det_size):
            if (not (assigned_detection[i] or ignored_det[i] == -1
                     or ignored_det[i] == 1 or ignored_threshold[i])):
                fp += 1
        nstuff = 0
        fp -= nstuff

        if compute_aos:
            tmp = np.zeros((fp + delta_idx, ))
            # tmp = [0] * fp
            for i in range(delta_idx):
                tmp[i + fp] = (1.0 + np.cos(delta[i])) / 2.0
                # tmp.append((1.0 + np.cos(delta[i])) / 2.0)
            # assert len(tmp) == fp + tp
            # assert len(delta) == tp
            if tp > 0 or fp > 0:
                similarity = np.sum(tmp)
            else:
                similarity = -1
    return tp, fp, fn, similarity, thresholds[:thresh_idx]


def get_split_parts(num, num_part):
    same_part = num // num_part
    remain_num = num % num_part
    if same_part == 0:
        return [num]

    if remain_num == 0:
        return [same_part] * num_part
    else:
        return [same_part] * num_part + [remain_num]


#@numba.jit(nopython=True)
def fused_compute_statistics(
                             overlaps,
                             pr,
                             gt_nums,
                             dt_nums,
                             gt_datas,
                             dt_datas,
                             ignored_gts,
                             ignored_dets,
                             metric,
                             min_overlap,
                             thresholds,
                             compute_aos=False):
    '''
                    (
                        parted_overlaps[0],
                        pr,
                        total_gt_num,
                        total_dt_num,
                        gt_datas_part,
                        dt_datas_part,
                        ignored_gts_part,
                        ignored_dets_part,
                        metric,
                        min_overlap=min_overlap,
                        thresholds=thresholds,
                        compute_aos=compute_aos)
                ______________gt_datas__________________
                [164, 121, 152, 42, 31, 36, 154, 151, 78, 49, 78, 76, 120, 
                43, 81, 13, 33, 21, 20, 16, 38, 19, 204, 206, 66, 66, 38, 114,
                 103, 83, 24, 91, 64, 119, 87, 103, 133, 53, 62, 94, 34, 130, 121, 
                 114, 103, 104, 10, 48, 40, 18, 41]

        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        print("______________overlaps__________________")
        print(overlaps)
        print("______________pr__________________")
        print(pr)                        #[[0. 0. 0. 0.],[0. 0. 0. 0.]]
    
        print("______________gt_nums__________________")
        print(gt_nums.shape)   #输出是（51，）
        print(gt_nums)         # #表示 当前帧中真实物体的个数 ，等于gt_datas

        print("______________dt_nums__________________")
        print(dt_nums)   #预测得到的每一帧的物体的个数，组成一个列表成一个列表
        
        print("______________gt_datas__________________")
        print(gt_datas)                #表示 当前帧中真实物体的个数
        print("______________dt_datas__________________")
        print(len(dt_datas))                #得到的是51组数组（N,1）的数组组成的一个列表
        print(dt_datas[0].shape)         #(N,1).表示当前帧中，预测得到了N个物体
        print(dt_datas[0])                    #打印当前帧中，N个物体的得分

        print("______________ignored_gts__________________")
        print(len(ignored_gts))
        print(ignored_gts[0])
        print("______________ignored_dets__________________")
        print(len(ignored_dets))
        print(ignored_dets[0])
        print("______________metric__________________")
        print(metric)
        print("______________min_overlap__________________")
        print(min_overlap)

        print("______________thresholds__________________")
        print(thresholds)
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
    '''
    
    gt_num = 0
    dt_num = 0
    # 传入的数据是10帧数据，分10次进行运行
    for i in range(gt_nums.shape[0]):            
        for t,thresh in enumerate(thresholds):
            overlap = overlaps[dt_num:dt_num+dt_nums[i],gt_num:gt_num+gt_nums[i]]
            gt_data = gt_datas[i]
            dt_data = dt_datas[i]
            ignored_gt = ignored_gts[i]
            ignored_det = ignored_dets[i]

            tp,fp,fn,similarity, _ = compute_statistics_jit1(
                overlap,
                gt_data,
                dt_data,
                ignored_gt,
                ignored_det,
                metric,
                min_overlap=min_overlap,
                thresh=thresh,
                compute_fp=True,
                compute_aos=compute_aos)

            pr[t,0]+=tp
            pr[t,1]+=fp
            pr[t,2]+=fn
            if similarity !=-1:
                pr[t,3]+=similarity
            
        gt_num += gt_nums[i]
        dt_num += dt_nums[i]


def calculate_iou_partly(gt_annos, dt_annos, metric, num_parts=5):
    """fast iou algorithm. this function can be used independently to
    do result analysis. Must be used in CAMERA coordinate system.
    Args:
        gt_annos: dict, must from get_label_annos() in kitti_common.py
        dt_annos: dict, must from get_label_annos() in kitti_common.py
        metric: eval type. 0: bbox, 1: bev, 2: 3d
        num_parts: int. a parameter for fast calculate algorithm
    """
    #如果长度不相等，直接报错
    assert len(gt_annos) == len(dt_annos)

    #计算每一帧中包含物体的个数，组成一个列表[164,121,152...]
    #即： 每个文件中批注数量的list
    total_dt_num = np.stack([len(a["name"]) for a in dt_annos], 0)
    total_gt_num = np.stack([len(a["name"]) for a in gt_annos], 0)

    '''  
        print("_____________total_dt_num__________________")
        print(total_dt_num)
        print("_____________:total_gt_num__________________")
        print(total_gt_num)
         输出的结果是:每一帧框的个数
        _____________:total_dt_num__________________
        [164 121 152  42  31  36 154 151  78  49  78  76 120  43  81  13  33  21
        20  16  38  19 204 206  66  66  38 114 103  83  24  91  64 119  87 103
        133  53  62  94  34 130 121 114 103 104  10  48  40  18  41]
        _____________:total_gt_num__________________
        [142 125 137  77  79  80 134 135 134  98 159 167 150 112 148  32  58 112
        102 111  72  86 143 165 137 130  71 146 130  79  36 152  82 137 101 137
        178  65 119 178  60 182 117 132 117 122  85 147 104  97 128]
    '''

    num_examples = len(gt_annos)
    split_parts = get_split_parts(num_examples, num_parts)
    
    '''
        print("___________num_examples______________")
        print(num_examples)
        print("___________split_parts______________")
        print(split_parts)
        print(type(split_parts))
        ___________num_examples______________
        51
        ___________split_parts______________
        [10,10,10,10,10,1]
        <class 'list'>
    '''
    
    parted_overlaps = []
    example_idx = 0

    for num_part in split_parts:
        # # 基本上将数据集分成多个部分并进行迭代
        gt_annos_part = gt_annos[example_idx:example_idx + num_part]
        dt_annos_part = dt_annos[example_idx:example_idx + num_part]
        
        '''
            print("_______________gt_annos_part__________")
            print(len(gt_annos_part))
            print(type(gt_annos_part))
            print("_______________dt_annos_part__________")
            print(len(dt_annos_part))
            print(type(dt_annos_part))
            还是相当于是原数据
            _______________gt_annos_part__________
            51 <class 'list'>
            _______________dt_annos_part__________
            51  <class 'list'>
        '''
        
        if metric == 0:
            #这个是针对bbox的，robosense中不涉及
            gt_boxes = np.concatenate([a["bbox"] for a in gt_annos_part], 0)
            dt_boxes = np.concatenate([a["bbox"] for a in dt_annos_part], 0)
            overlap_part = image_box_overlap(gt_boxes, dt_boxes)
        

        elif metric == 1:
            
            loc = np.concatenate([a["box_center"][:, [0, 2]] for a in gt_annos_part], 0)
            dims = np.concatenate([a["box_size"][:, [0, 2]] for a in gt_annos_part], 0)
            rots = np.concatenate([a["box_rotation"][:,-1] for a in gt_annos_part], 0)
            gt_boxes = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1)
            
            loc = np.concatenate([a["box_center"][:, [0, 2]] for a in dt_annos_part], 0)
            dims = np.concatenate([a["box_size"][:, [0, 2]] for a in dt_annos_part], 0)
            rots = np.concatenate([a["box_rotation"][:,-1] for a in dt_annos_part], 0)
            dt_boxes = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1)
            
            #计算iou   (N1+N2+...+N_10,N1+N2+...+N_10)，10帧数据里：
            # 前面的是测试集中的物体总个数，后面是真实值中的物体的总数
            overlap_part = bev_box_overlap(gt_boxes, dt_boxes).astype(np.float64)
            ''' 得到的结果是：
                print("______________overlap_part_________________")
                print(overlap_part)
                print(type(overlap_part))
                print(overlap_part.shape)
                ______________overlap_part_________________
                [[0. 0. 0. ... 0. 0. 0.]
                [0. 0. 0. ... 0. 0. 0.]
                [0. 0. 0. ... 0. 0. 0.]
                ...
                [0. 0. 0. ... 0. 0. 0.]
                [0. 0. 0. ... 0. 0. 0.]
                [0. 0. 0. ... 0. 0. 0.]]
                <class 'numpy.ndarray'>
                (5927, 4009)     
            '''
        

        elif metric == 2:
            loc = np.concatenate([a["box_center"] for a in gt_annos_part], 0)
            dims = np.concatenate([a["box_size"] for a in gt_annos_part], 0)
            rots = np.concatenate([a["box_rotation"][:,-1] for a in gt_annos_part], 0)
            gt_boxes = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1)
            
            loc = np.concatenate([a["box_center"] for a in dt_annos_part], 0)
            dims = np.concatenate([a["box_size"] for a in dt_annos_part], 0)
            rots = np.concatenate([a["box_rotation"][:,-1] for a in dt_annos_part], 0)
            dt_boxes = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1)
            
            #计算iou
            overlap_part = d3_box_overlap(gt_boxes, dt_boxes).astype(np.float64)
            '''
                print("______________overlap_part_________________")
                print(overlap_part)
                print(type(overlap_part))
                print(overlap_part.shape)
            '''
        
        else:
            raise ValueError("unknown metric")
        
        # 最终是数据集的b/n个部分的iou矩阵的列表
        parted_overlaps.append(overlap_part)
        example_idx += num_part
        '''
            print("______________parted_overlaps_________________")
            print(parted_overlaps)
            print(type(parted_overlaps))
            print(len(parted_overlaps))
        '''
    
    overlaps = []
    example_idx = 0

    for j, num_part in enumerate(split_parts):         # 遍历每一个部分
        gt_annos_part = gt_annos[example_idx:example_idx + num_part]
        dt_annos_part = dt_annos[example_idx:example_idx + num_part]
        gt_num_idx, dt_num_idx = 0, 0
        for i in range(num_part):
            gt_box_num = total_gt_num[example_idx + i]
            dt_box_num = total_dt_num[example_idx + i]
            overlaps.append(
                parted_overlaps[j][gt_num_idx:gt_num_idx + gt_box_num,
                                   dt_num_idx:dt_num_idx + dt_box_num])
            gt_num_idx += gt_box_num
            dt_num_idx += dt_box_num
        example_idx += num_part
    
    '''
        overlaps是矩阵的list。 列表的长度为图像帧的数量，len（dt_annos）= len（gt_annos）
        每个索引中都有一个iou矩阵，形状:（gt boxes的数量，dt boxes的数量）
        parted_overlaps是数据集各部分之间的重叠矩阵
        # total_gt_num是每个图像中的boxes数量的list。
        输出：
            overlaps：<class 'list'>  51
            ____________parted_overlaps_______________
                    <class 'list'>
                    长度：1
                    [array([[0., 0., 0., ..., 0., 0., 0.],
                    [0., 0., 0., ..., 0., 0., 0.],
                    [0., 0., 0., ..., 0., 0., 0.],
                    ...,
                    [0., 0., 0., ..., 0., 0., 0.],
                    [0., 0., 0., ..., 0., 0., 0.],
                    [0., 0., 0., ..., 0., 0., 0.]])]
            ____________total_gt_num_______________
                <class 'numpy.ndarray'>
                51
                [142 125 137  77  79  80 134 135 134  98 159 167 150 112 148  32  58 112
                102 111  72  86 143 165 137 130  71 146 130  79  36 152  82 137 101 137
                178  65 119 178  60 182 117 132 117 122  85 147 104  97 128]
            ____________total_dt_num_______________
                <class 'numpy.ndarray'>
                51
                [164 121 152  42  31  36 154 151  78  49  78  76 120  43  81  13  33  21
                20  16  38  19 204 206  66  66  38 114 103  83  24  91  64 119  87 103
                133  53  62  94  34 130 121 114 103 104  10  48  40  18  41]
    '''
    # print的一些语句
    '''
        print("____________overlaps_______________")
        print(type(overlaps))            
        print(len(overlaps))
        print(overlaps[5].shape,overlaps[7].shape,overlaps[25].shape)        #(80, 36) (135, 151) (130, 66)
        
        print("____________parted_overlaps_______________")
        print(type(parted_overlaps))
        print(len(parted_overlaps))
        print(parted_overlaps)

        print("____________total_gt_num_______________")
        print(type(total_gt_num))
        print(len(total_gt_num))
        print(total_gt_num)

        print("____________total_dt_num_______________")
        print(type(total_dt_num))
        print(len(total_dt_num))
        print(total_dt_num)    
    '''
    
    return overlaps, parted_overlaps, total_gt_num, total_dt_num


#参数difficulty是int类型，为0,1,2
def _prepare_data(gt_annos, dt_annos, current_class, difficulty):
    
    #数据初始化
    gt_datas_list = []
    dt_datas_list = []
    ignored_gts, ignored_dets = [], []
    total_num_valid_gt = 0

    # 对于每一帧的数据进行操作
    for i in range(len(gt_annos)):
        
        #得到的是参数，当前帧的这个类别的 有效物体数，和有效物体的索引列表
        rets = clean_data(gt_annos[i], dt_annos[i], current_class, difficulty)
        
        num_valid_gt, ignored_gt, ignored_det  = rets

        # 将每一帧的ignored_gt数据类型进行转换为numpy格式，再添加到ignored_gts
        ignored_gts.append(np.array(ignored_gt, dtype=np.int64))
        ignored_dets.append(np.array(ignored_det, dtype=np.int64))
        total_num_valid_gt += num_valid_gt

        gt_datas_num = len(gt_annos[i]["name"])
        gt_datas_list.append(gt_datas_num)

        #dt_datas_score = dt_annos[i]["score"]
        dt_datas_score = dt_annos[i]["scores"][..., np.newaxis]
        dt_datas_list.append(dt_datas_score)

    return (
                    gt_datas_list,  #存放的是 每一帧物体的个数
                    dt_datas_list,  #存放的是每一帧 不同物体的得分的情况，是（N,1）
                    ignored_gts, ignored_dets,   #存在
                    total_num_valid_gt                 #存在
                    )               


def eval_class(gt_annos,
               dt_annos,
               current_classes,
               difficultys,
               metric,
               min_overlaps,
               compute_aos=False,
               num_parts=5):
    """Kitti eval. support 2d/bev/3d/aos eval. support 0.5:0.05:0.95 coco AP.
        Args:
            gt_annos: dict, must from get_label_annos() in kitti_common.py
            dt_annos: dict, must from get_label_annos() in kitti_common.py
            current_classes: list of int, 0: car, 1: pedestrian, 2: cyclist
            difficultys: list of int. eval difficulty, 0: easy, 1: normal, 2: hard
            metric: eval type. 0: bbox, 1: bev, 2: 3d
            min_overlaps: float, min overlap. format: [num_overlap, metric, class].
            num_parts: int. a parameter for fast calculate algorithm

        Returns:
            dict of recall, precision and aos
        
            kitti中的数据字典里的键是：{
                'name': [],
                'truncated': [],
                'occluded': [],
                'alpha': [],
                'bbox': [],
                'dimensions': [],
                'location': [],
                'rotation_y': []
                'score'    # 对于验证集的数据，kitti中有score这一列的数字
            }
            min_overlaps:
                                    # (2, 3, num_classes) 其中:
                                    # 2 表示阈值为中等或者容易
                                    # 3 表示表示不同的指标 (bbox, bev, 3d), 
                                    # num_classes用于每个类的阈值

            参数difficultys:[0, 1, 2],<class 'list'>
    """

    #如果验证集gt_annos中的帧数 和 从model中验证出来dt_annos帧的长度不一致，直接报错！
    assert len(gt_annos) == len(dt_annos)
    # 验证集中帧的总数是 num_examples:51
    num_examples = len(gt_annos)
    #得到的split_parts是一个list的类型，num_parts=5,
    # 意思是将51分为5部分，经过一下函数得到的是：split_parts：[10,10,10,10,10,1]
    split_parts = get_split_parts(num_examples, num_parts)
    
    '''
        print('________________num_examples________________')
        print (num_examples)
        print('________________split_parts________________')
        print (split_parts)
        print(type(split_parts))
        print('________________difficultys________________')
        print (difficultys)
        print(type(difficultys))
        print("________________________min_overlaps:__________________________")
        print(min_overlaps)
        print(type(min_overlaps))
        print(min_overlaps.shape)
    '''

    #计算iou
    #rets = calculate_iou_partly(gt_annos,dt_annos, metric, num_parts)
    rets = calculate_iou_partly(dt_annos, gt_annos, metric, num_parts)
    overlaps, parted_overlaps, total_dt_num, total_gt_num = rets

    N_SAMPLE_PTS = 41

    #获取min_overlaps的各个的维度，得到的是(2, 3, 5)
    # 获取当前类别的个数num_class：5，难度的个数为3
    num_minoverlap = len(min_overlaps)            #得到长度为2
    num_class = len(current_classes)
    num_difficulty = len(difficultys)

    #初始化precision，recall，aos
    precision = np.zeros([num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS])
    recall = np.zeros([num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS])
    aos = np.zeros([num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS])

    #每个类别：
    for m, current_class in enumerate(current_classes):
        # 每个难度：
        for l, difficulty in enumerate(difficultys):
            
            #参数difficulty是int类型，为0,1,2
            rets = _prepare_data(gt_annos, dt_annos, current_class, difficulty)
            (gt_datas_list, dt_datas_list, ignored_gts, ignored_dets, total_num_valid_gt) = rets
            
            # 运行两次，首先进行中等难度的总体设置，然后进行简单设置。
            for k, min_overlap in enumerate(min_overlaps[:, metric, m]):
                
                thresholdss = []
                
                # 循环浏览数据集中的图像。因此一次只显示一张图片。
                for i in range(len(gt_annos)):
                    
                    # 调用  函数
                    rets = compute_statistics_jit(
                        overlaps[i],     # 单个图像的iou值b/n gt和dt
                        gt_datas_list[i],       # 是一个数，表示当前帧中的物体个数
                        dt_datas_list[i],       # N x 1阵列，表示的是预测得到的N个物体的得分情况
                        ignored_gts[i],         # 长度N数组，-1、0
                        ignored_dets[i],        # 长度N数组，-1、0
                        metric,                             # 0, 1, 或 2 (bbox, bev, 3d)
                        min_overlap=min_overlap,         # 浮动最小IOU阈值为正
                        thresh=0.0,                 # 忽略得分低于此值的dt。
                        compute_fp=False)
                    
                    #print("___________rets_____________")
                    #print(rets)
                    #返回的结果：___________rets_____________
                    #(10, 0, 1, 0.0, array([0.52749819, 0.76324338, 0.60215807, 0.29757985, 0.72033411,
                    # 0.11587256, 0.31741855, 0.32567033, 0.36515915, 0.29665849]))

                    tp, fp, fn, similarity, thresholds = rets
                    thresholdss += thresholds.tolist()
            
                #一维数组，记录匹配的dts分数，将list转为np格式
                thresholdss = np.array(thresholdss)
                '''
                print("=======================================")
                print(thresholdss)
                print(thresholdss.shape)
                '''
                
                # total_num_valid_gt是51帧数据里，vehicle出现的总个数
                thresholds = get_thresholds(thresholdss, total_num_valid_gt)
                thresholds = np.array(thresholds)
                '''
                print(thresholds)
                print(thresholds.shape)
                print(total_num_valid_gt)
                print("class:",current_class)
                print("difficulty:",difficulty)
                print("min_overlap",min_overlap)
                print("")
                '''

                # thresholds是 N_SAMPLE_PTS长度的一维数组，记录分数，递减，表示阈值
                # 储存有关gt/dt框的信息（是否忽略，fn，tn，fp）
                pr = np.zeros([len(thresholds), 4])

                idx = 0
                for j,num_part in enumerate(split_parts):

                    gt_datas_part = np.array(gt_datas_list[idx:idx+num_part])
                    dt_datas_part = np.array(dt_datas_list[idx:idx+num_part])
                    ignored_dets_part = np.array(ignored_dets[idx:idx+num_part])
                    ignored_gts_part = np.array(ignored_gts[idx:idx+num_part])

                    # 再将各部分数据融合
                    fused_compute_statistics(
                        parted_overlaps[j],
                        pr,
                        total_gt_num[idx:idx+num_part],
                        total_dt_num[idx:idx+num_part],
                        gt_datas_part,
                        dt_datas_part,
                        ignored_gts_part,
                        ignored_dets_part,
                        metric,
                        min_overlap=min_overlap,
                        thresholds=thresholds
                    )
                    idx += num_part

                #计算recall和precision
                for i in range(len(thresholds)):
                    recall[m, l, k, i] = pr[i, 0] / (pr[i, 0] + pr[i, 2])
                    precision[m, l, k, i] = pr[i, 0] / (pr[i, 0] + pr[i, 1])
                    if compute_aos:
                        aos[m, l, k, i] = pr[i, 3] / (pr[i, 0] + pr[i, 1])

                # 返回各自序列的最值
                for i in range(len(thresholds)):
                    precision[m, l, k, i] = np.max(precision[m, l, k, i:], axis=-1)
                    recall[m, l, k, i] = np.max(recall[m, l, k, i:], axis=-1)
                    if compute_aos:
                        aos[m, l, k, i] = np.max(aos[m, l, k, i:], axis=-1)
    ret_dict = {
        "recall": recall,             # [num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS]  
        "precision": precision,          # RECALLING RECALL的顺序，因此精度降低
        "orientation": aos,
    }
    return ret_dict


def get_mAP(prec):
    sums = 0
    for i in range(0, prec.shape[-1], 4):
        sums = sums + prec[..., i]
    return sums / 11 * 100


def get_mAP_R40(prec):
    sums = 0
    for i in range(1, prec.shape[-1]):
        sums = sums + prec[..., i]
    return sums / 40 * 100


#打印结果的函数
def print_str(value, *arg, sstream=None):
    if sstream is None:
        sstream = sysio.StringIO()
    sstream.truncate(0)
    sstream.seek(0)
    print(value, *arg, file=sstream)
    return sstream.getvalue()


# 该函数是实现计算和评估的具体的函数
def do_eval(gt_annos,
            dt_annos,
            current_classes,
            min_overlaps,
            compute_aos=False,
            PR_detail_dict=None):

    # min_overlaps: [num_minoverlap, metric, num_class]
    #     #由上面得到的min_overlaps的形状是（2,3,5），
    # 这个是每个类别的IOU达到这个阈值时判断是否预测正确

    difficultys = [0, 1, 2]

    #metric: eval type. 0: bbox, 1: bev, 2: 3d
    # 重点是eval_class这个函数！！！！！！！！！！
    ret = eval_class(gt_annos, dt_annos, current_classes, difficultys, 1, min_overlaps)

    mAP_bev = get_mAP(ret["precision"])
    mAP_bev_R40 = get_mAP_R40(ret["precision"])

    ret = eval_class(gt_annos, dt_annos, current_classes, difficultys, 2,min_overlaps)
    mAP_3d = get_mAP(ret["precision"])
    mAP_3d_R40 = get_mAP_R40(ret["precision"])

    #return mAP_bbox, mAP_bev, mAP_3d, mAP_aos, mAP_bbox_R40, mAP_bev_R40, mAP_3d_R40, mAP_aos_R40
    return  mAP_bev, mAP_3d, mAP_bev_R40, mAP_3d_R40


#定义函数，实现
def do_coco_style_eval(gt_annos, dt_annos, current_classes, overlap_ranges,
                       compute_aos):
    
    # overlap_ranges: [range, metric, num_class]
    #overlap_ranges的形状是：（3,3,5）
    min_overlaps = np.zeros([10, *overlap_ranges.shape[1:]])
    for i in range(overlap_ranges.shape[1]):
        for j in range(overlap_ranges.shape[2]):
            min_overlaps[:, i, j] = np.linspace(*overlap_ranges[:, i, j])
    #由上面得到的min_overlaps的形状是（10,3,5），
    # 这个是每个类别的IOU达到这个阈值时判断是否预测正确

    print("________________________overlap_ranges:__________________________")
    print(overlap_ranges)
    print(type(overlap_ranges))
    print(overlap_ranges.shape)

    print("________________________min_overlaps:__________________________")
    print(min_overlaps)
    print(type(min_overlaps))
    print(min_overlaps.shape)
    
    
    mAP_bev, mAP_3d, mAP_bev_R40, mAP_3d_R40 = do_eval(
        gt_annos, dt_annos, current_classes, min_overlaps, compute_aos)
    
    # ret: [num_class, num_diff, num_minoverlap]
    #mAP_bbox = mAP_bbox.mean(-1)
    mAP_bev = mAP_bev.mean(-1)
    mAP_3d = mAP_3d.mean(-1)
    
    return  mAP_bev, mAP_3d, mAP_bev_R40, mAP_3d_R40


#传入的参数current_classes 是五个类别：
#['vehicle', 'big_vehicle', 'pedestrian', 'bicycle', 'cone']
def get_coco_eval_result1(gt_annos, dt_annos, current_classes):
    class_to_name = {
        0: 'vehicle',
        1: 'big_vehicle',
        2: 'pedestrian',
        3: 'bicycle',
        4: 'cone',
        5:'huge_vehicle',
        6:'motorcycle',
        7:'tricycle',
        8:'unknown'
    }
    class_to_range = {
        0: [0.5, 0.95, 10],
        1: [0.25, 0.7, 10],
        2: [0.25, 0.7, 10],
        3: [0.5, 0.95, 10],
        4: [0.25, 0.7, 10],
    }
    #定义由名字变成类别，如0,1,2之类的字典
    name_to_class = {v: n for n, v in class_to_name.items()}

    #如果传入的current_classes 不是list或者tuple，则转为列表list格式
    if not isinstance(current_classes, (list, tuple)):
        current_classes = [current_classes]
    #定义一个空列表，如果current_classes中每一类为str类型，则存入相应的类别号
    #如当前判断的vehicle，big_vehicle，pedestrian，bicycle,cone,则current_classes_int=[ 0,1,2,3,4]
    current_classes_int = []
    for curcls in current_classes:
        if isinstance(curcls, str):
            current_classes_int.append(name_to_class[curcls])
        else:
            current_classes_int.append(curcls)
    current_classes = current_classes_int
    #现在current_classes = [0,1,2,3,4],#当前的类别变成了含有数字的列表

    #定义一个三维的全零数组，形状是(3,3,5),有5个类别
    overlap_ranges = np.zeros([3, 3, len(current_classes)])
    for i, curcls in enumerate(current_classes):
        overlap_ranges[:, :, i] = np.array(
            class_to_range[curcls])[:, np.newaxis]
    # [:, np.newaxis]的作用是将class_to_range中每一值的行向量转化为列向量，并赋值为overlap_ranges中
    # 由此得到了，overlap_ranges = [[[0.5,0.25,0.25,0.5,0.25],[],[]],[[]]...]
    
    print("________________________overlap_ranges:__________________________")
    print(overlap_ranges)
    print(type(overlap_ranges))
    print(overlap_ranges.shape)

    result = ''
    
    # check whether alpha is valid
    compute_aos = False
    for anno in dt_annos:
        ##判断dt_annos中每一帧中有几个物体objects，如果不为零
        #接着判断，第一个值是否为-10，如果不是，将计算aos的标志compute_aos设为true
        if anno['name'].shape[0] != 0:
            if anno['box_rotation'][0][2] != -10:
                compute_aos = True
            break
    
    #调用函数，计算各个指标的结果
    # 函数的返回值是mAP_bev, mAP_3d, mAP_bev_R40, mAP_3d_R40
    mAPbev, mAP3d,mAP_bev_R40, mAP_3d_R40= do_coco_style_eval(gt_annos, dt_annos, current_classes, overlap_ranges, compute_aos)
    mAP_bev_R40, mAP_3d_R40=mAP_bev_R40, mAP_3d_R40
    
    print('_____________mAPbev________________')
    print(mAPbev)
    print('_____________mAP3d________________')
    print(mAP3d)

    #打印输出，，，，current_classes = [0,1,2,3,4],#当前的类别变成了含有数字的列表
    #循环当前的每个类别
    for j, curcls in enumerate(current_classes):
        # mAP threshold array: [num_minoverlap, metric, class]
        # mAP result: [num_class, num_diff, num_minoverlap]

        #由class_to_range[curcls]得到的是一个类别的[0.5,0.95,10]的数组，
        # 再[[0, 2, 1]]，得到的是[0.5,10,0.95]，即顺序反了一下
        o_range = np.array(class_to_range[curcls])[[0, 2, 1]]
        #下面计算得到，= （0.95-0.5）/（10-1）= 0.9/9 =0.01,则o_range=[0.5,0.01,0.95]
        o_range[1] = (o_range[2] - o_range[0]) / (o_range[1] - 1)

        result += print_str((f"{class_to_name[curcls]} "
                             "coco AP@{:.2f}:{:.2f}:{:.2f}:".format(*o_range)))

        result += print_str((f"bev  AP:{mAPbev[j, 0]:.2f}, "
                             f"{mAPbev[j, 1]:.2f}, "
                             f"{mAPbev[j, 2]:.2f}"))
        result += print_str((f"3d   AP:{mAP3d[j, 0]:.2f}, "
                             f"{mAP3d[j, 1]:.2f}, "
                             f"{mAP3d[j, 2]:.2f}"))


    print("_________________result:___________________")
    print(result)
    return result



def get_official_eval_result(gt_annos, dt_annos, current_classes, PR_detail_dict=None):
    overlap_0_7 = np.array([[0.7, 0.5, 0.5, 0.7,0.5, 0.7, 0.5, 0.5, 0.7], 
                                                        [0.7, 0.5, 0.5, 0.7, 0.5, 0.7, 0.5, 0.5, 0.7],
                                                        [0.7, 0.5, 0.5, 0.7, 0.5, 0.7, 0.5, 0.5, 0.7]])
    overlap_0_5 = np.array([[0.7, 0.5, 0.5, 0.7, 0.5, 0.5, 0.7, 0.5, 0.5],
                                                     [0.5, 0.25, 0.25, 0.5, 0.25, 0.5, 0.25, 0.5, 0.25],
                                                     [0.5, 0.25, 0.25, 0.5, 0.25, 0.5, 0.25, 0.5, 0.25]])
    #目的是给不同的类别设置不同的阈值
    #每个类别的输出 2×2个结果，AP 或 AR_R40，overlap为0.7或0.5
    #通过下面一行，min_overlaps的形状是(2, 3, 9)的三维数组，9是因为有9个类别
    min_overlaps = np.stack([overlap_0_7, overlap_0_5], axis=0)  # [2, 3, 9]
    
    #每一类对应上面的每一列的内容
    class_to_name = {
        0: 'vehicle',                        ##[[0.7,0.7,0.7],[0.5, 0.5, 0.5]]等于上面数组的：min_overlaps[:,:,0]
        1: 'big_vehicle',           ##[[0.5,0.5,0.5],[0.5,0.25,0.25]] ,等于上面数组的：min_overlaps[:,:,1]
        2: 'pedestrian',
        3: 'bicycle',
        4: 'cone',
        5: 'huge_vehicle',
        6:'motorcycle',
        7:'tricycle',
        8:'unknown'
    }
    #将名字和对应的类别号反一下，便于索引
    name_to_class = {v: n for n, v in class_to_name.items()}

    if not isinstance(current_classes, (list, tuple)):
        current_classes = [current_classes]
    
    #定义一个空列表，如果current_classes中每一类为str类型，则存入相应的类别号
    #如当前判断的Car，Pedestrian，Cyclist，则current_classes_int=[ 0,1,2]
    current_classes_int = []
    for curcls in current_classes:
        if isinstance(curcls, str):
            current_classes_int.append(name_to_class[curcls])
        else:
            current_classes_int.append(curcls)
    #当前的类别变成了含有数字的列表   current_classes=[ 0,1,2]
    current_classes = current_classes_int

    #下面一行的作用：min_overlaps[:,:,[0,1,2,3,4]]，
    # 取min_overlaps的前5列，因为有5个类别是需要分类和计算的
    #得到的min_overlaps的形状：（2,3,5）
    min_overlaps = min_overlaps[:, :, current_classes]
    
    result = ''
    # check whether name is valid
    compute_aos = False
    '''
    for anno in dt_annos:
        #判断dt_annos中每一帧中有几个物体objects，如果不为零
        #接着判断，第一个值是否为-10，如果不是，将计算aos的标志compute_aos设为true
        if anno['name'].shape[0] != 0:
            if anno['box_rotation'][0][2] != -10:
                compute_aos = True
            break
    '''
    #调用函数，计算各个值，4个指标
    mAP_bev, mAP_3d, mAP_bev_R40, mAP_3d_R40 = do_eval(
        gt_annos, dt_annos, current_classes, min_overlaps, compute_aos, PR_detail_dict=PR_detail_dict)

    #将结果打印并返回
    ret_dict = {}
    for j, curcls in enumerate(current_classes):
        # mAP threshold array: [num_minoverlap, metric, class]
        # mAP result: [num_class, num_diff, num_minoverlap]
        for i in range(min_overlaps.shape[0]):
            result += print_str(
                (f"{class_to_name[curcls]} "
                 "AP@{:.2f}, {:.2f}, {:.2f}:".format(*min_overlaps[i, :, j])))
            
            result += print_str((f"bev  AP:{mAP_bev[j, 0, i]:.4f}, "
                                 f"{mAP_bev[j, 1, i]:.4f}, "
                                 f"{mAP_bev[j, 2, i]:.4f}"))
            result += print_str((f"3d   AP:{mAP_3d[j, 0, i]:.4f}, "
                                 f"{mAP_3d[j, 1, i]:.4f}, "
                                 f"{mAP_3d[j, 2, i]:.4f}"))

            result += print_str(
                (f"{class_to_name[curcls]} "
                 "AP_R40@{:.2f}, {:.2f}, {:.2f}:".format(*min_overlaps[i, :, j])))

            result += print_str((f"bev  AP:{mAP_bev_R40[j, 0, i]:.4f}, "
                                 f"{mAP_bev_R40[j, 1, i]:.4f}, "
                                 f"{mAP_bev_R40[j, 2, i]:.4f}"))
            result += print_str((f"3d   AP:{mAP_3d_R40[j, 0, i]:.4f}, "
                                 f"{mAP_3d_R40[j, 1, i]:.4f}, "
                                 f"{mAP_3d_R40[j, 2, i]:.4f}"))

            if i == 0:
                ret_dict['%s_3d/easy_R40' % class_to_name[curcls]] = mAP_3d_R40[j, 0, 0]
                ret_dict['%s_3d/moderate_R40' % class_to_name[curcls]] = mAP_3d_R40[j, 1, 0]
                ret_dict['%s_3d/hard_R40' % class_to_name[curcls]] = mAP_3d_R40[j, 2, 0]
                ret_dict['%s_bev/easy_R40' % class_to_name[curcls]] = mAP_bev_R40[j, 0, 0]
                ret_dict['%s_bev/moderate_R40' % class_to_name[curcls]] = mAP_bev_R40[j, 1, 0]
                ret_dict['%s_bev/hard_R40' % class_to_name[curcls]] = mAP_bev_R40[j, 2, 0]

    return result, ret_dict



