'''
Description: 
Author: HCQ
Company(School): UCAS
Email: 1756260160@qq.com
Date: 2021-10-24 18:35:21
LastEditTime: 2021-10-24 20:09:50
FilePath: /mmdetection3d/demo/pcl_viewer_obj.py
'''
__Author__ = "Shliang"
__Email__ = "shliang0603@gmail.com"

import sys
import os


obj_name = sys.argv[1]
obj_file_abspath = os.path.abspath(obj_name)
obj_file_dir = os.path.dirname(obj_file_abspath)
output_pcd_name = obj_file_abspath.split("/")[-1].split(".")[0] + ".pcd"
output_pcd_file_abspath = os.path.join(obj_file_dir, output_pcd_name)

def pcl_obj2pcd(rm_pcd_file=True):
    # 转换obj文件为pcd类型文件
    cmd1 = "pcl_obj2pcd " + obj_name + " " + output_pcd_file_abspath
    print(f"cmd1: {cmd1}")
    os.system(cmd1) # 运行shell脚本命令
    # 使用pcl_viewer可视化pcd文件
    cmd2 = "pcl_viewer " + output_pcd_file_abspath
    print(f"cmd2: {cmd2}")
    os.system(cmd2)


    # 程序退出时，删除
    if rm_pcd_file:
        os.remove(output_pcd_file_abspath)
        sys.exit()

if __name__ == '__main__':
    pcl_obj2pcd(False)
