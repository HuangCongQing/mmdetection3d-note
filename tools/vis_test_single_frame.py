import open3d as o3d
import numpy as np
import math

# def bbox_projection(T, values):
#     ## this function is to project a bbox to a xoy-plane-based one
#     vertices = from_7value_to_8pts(values)

# def eulerAnglesToRotationMatrix(theta):  #theta -> [roll, pitch, yaw]
    
#     R_x = np.array([[1,         0,                  0                   ],
#                     [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
#                     [0,         math.sin(theta[0]), math.cos(theta[0])  ]
#                     ])       
                    
#     R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
#                     [0,                     1,      0                   ],
#                     [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
#                     ])
                
#     R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
#                     [math.sin(theta[2]),    math.cos(theta[2]),     0],
#                     [0,                     0,                      1]
#                     ])         
                    
#     R = np.dot(R_z, np.dot( R_y, R_x ))

#     return R

    ## suppose 7 values are l,w,h,cx,cy,cz
def from_7value_to_8pts(values):
    ## bottom center point -> (cx,cy,cz)
    #h, w, l, cx, cy, cz, yaw = values
    cx, cy, cz, l, w, h, yaw = values
    #yaw *= -1
    #convert from cm to m
    #cx /= 100
    #cy /= 100
    #cz /= 100


    ### rad or degree? 
    #yaw = yaw / 180 * math.pi

    R = np.array([
        [np.cos(yaw), -np.sin(yaw), 0], 
        [np.sin(yaw), np.cos(yaw), 0], 
        [0, 0, 1]
        ])

    x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2]
    y_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2]
    
    ### center or bottom center?
    #z_corners = [0,0,0,0,h,h,h,h]
    z_corners = [-h/2,-h/2,-h/2,-h/2,h/2,h/2,h/2,h/2]

    #rotation
    vertices = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    #position
    vertices += np.vstack([cx, cy, cz])
    return vertices


def transform_from_world_to_lidar(gt_info):  ###for single frame!!!
    #删除还未启动的车 cx==0
    mask_array = gt_info[:, 3] == 0
    gt_info = gt_info[~mask_array]

    #世界坐标系为W 载雷达车为C 雷达为L，已知点在W下的坐标值，求在L下的坐标值
    #已知C相对W为T1，C相对L为T2，求W相对C的T3
    #T3 = T2 * (T1^-1)

    #数据最后一行是雷达转载车的info
    ##这个地方最好能读到雷达相对车体的位置 这样可以适应不同车种类   
    l, w, h, C2W_x, C2W_y, C2W_z, C2W_yaw = gt_info[-1]
    C2W_yaw = C2W_yaw / 180 * math.pi
    #convert from cm to m
    C2W_x /= 100
    C2W_y /= 100
    C2W_z /= 100

    T1 = np.array([
    [np.cos(C2W_yaw), -np.sin(C2W_yaw), 0, C2W_x], 
    [np.sin(C2W_yaw), np.cos(C2W_yaw), 0, C2W_y], 
    [0, 0, 1, C2W_z],
    [0, 0, 0, 1]
    ])

    #7.5 4.9 根据雷达相对车体坐标系确定
    T2 = np.array([
    [1, 0, 0, -7.5], 
    [0, 1, 0, 0], 
    [0, 0, 1, -4.9],
    [0, 0, 0, 1]
    ])

    T3 = np.dot(T2, np.linalg.inv(T1))
    
    bbox_list = []
    for idx in range(gt_info.shape[0]):
        truck_info_world = gt_info[idx]
        truck_bbox_world = from_7value_to_8pts(truck_info_world)  #3x8
        truck_bbox_lidar = np.dot(T3, np.vstack([truck_bbox_world, np.ones(8)])) #4x8

        #TODO 删除lidar检测范围外的真值框
        ##TODO 将真值框贴合lidar坐标系的xoy
        bbox_list.append(truck_bbox_lidar[:3]) 

    #返回单个场景里所有的真值框
    return bbox_list


if __name__ == "__main__":
    #410 边框上浮
    #350 比较理想
    #filename_bbox = '/home/keenthan/Documents/AI_training/groundtrue/GroundTrueData_Frame410.txt'
    #filename_bbox = '/home/keenthan/Desktop/14/20211213103031002/10.25-2/_2021-10-25-14-59-51/livox/000013.txt'
    filename_bbox = '/home/hcq/data/2022anno/finaul_result/txt_result/000315.txt'
    #l,w,h,cx,cy,cz,rotz
    #l,w,h in meters   cx,cy,cz in centimeters  rotz in degree   
    #points_box_7value_array = np.loadtxt(filename_bbox, delimiter=',', usecols=(2,3,4,5,6,7,8))
    #points_box_7value_array = np.loadtxt(filename_bbox, delimiter=' ', usecols=(8,9,10,11,12,13,14)).reshape(-1,7)
    points_box_7value_array = np.loadtxt(filename_bbox, delimiter=' ', usecols=(1,2,3,4,5,6,7)).reshape(-1,7)# delimiter=' ' 几个空格
    #print(points_box_7value_array.shape)
    

    ##########################
    #### if from fangzhen ####
    ##########################
    #bbox_list = transform_from_world_to_lidar(points_box_7value_array)

    ##########################
    #### if only for vis  ####
    ##########################
    bbox_list = []
    for idx in range(points_box_7value_array.shape[0]):
        truck_info_world_ = points_box_7value_array[idx]
        truck_bbox_world_ = from_7value_to_8pts(truck_info_world_)  #3x8
        #np.savetxt('bbox_%d.txt' % idx, truck_bbox_world_.transpose(), fmt="%.18f,%.18f,%.18f", delimiter="\n")
        bbox_list.append(truck_bbox_world_)

    





    # 3维点云
    #filename_pc = '/home/keenthan/Documents/AI_training/pointcloud/frame409.txt'
    #pc = np.loadtxt(filename_pc, delimiter=',', usecols=(0,1,2))
    filename_pc = '/home/hcq/data/2022anno/finaul_result/pcd_result/000315.pcd'
    pcd_load = o3d.io.read_point_cloud(filename_pc)
    pc = np.asarray(pcd_load.points)
  
    lines_box = np.array([[0, 1], [1, 2], [0, 3], [2, 3], [4, 5], [4, 7], [5, 6], [6, 7],
                            [0, 4], [1, 5], [2, 6], [3, 7]])
    colors = np.array([[255, 255, 255] for j in range(len(lines_box))])
    
    #code below for visualization
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(pc[:,:3])

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(point_cloud)
    

    #add lines for each bbox
    for points_box in bbox_list:
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points_box.transpose())  #for drawing points_box should be 8x3
        line_set.lines = o3d.utility.Vector2iVector(lines_box)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        vis.add_geometry(line_set)
   
    render_option = vis.get_render_option()
    render_option.point_size = 4
    render_option.background_color = np.asarray([0, 0, 0])
    vis.run()
    vis.destroy_window()

