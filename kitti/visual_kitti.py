import numpy as np
import os
import open3d as o3d


def visual_pcl(pcl1, pcl2):
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(pcl1[:,:3])
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(pcl2[:,:3])
    pcd1.paint_uniform_color([1, 0.706, 0])
    pcd2.paint_uniform_color([0, 0.651, 0.929])
    o3d.visualization.draw_geometries([pcd1, pcd2], window_name='Open3D Origin', width=1920, height=1080, left=50, top=50,
                                      point_show_normal=False, mesh_show_wireframe=False, mesh_show_back_face=False)


if __name__ == '__main__':

    path = '/home/qzj/code/catkin/catkin_lego/src/LeGO-LOAM/LeGO-LOAM/generatedMap/kitti_format'
    num1 = 0
    num2 = 5

    fname1 = os.path.join(path,'lidar', '{}.bin'.format(str(num1).zfill(6)))
    fname2 = os.path.join(path,'lidar', '{}.bin'.format(str(num2).zfill(6)))

    poses = np.loadtxt(os.path.join(path,'odom_key_frames.txt'))
    pose_w_1=np.eye(4)
    pose_w_2=np.eye(4)
    # 第num1帧到世界坐标系w的变换矩阵
    pose_w_1[:3,:]=poses[num1].reshape(3,4)
    pose_w_2[:3,:]=poses[num2].reshape(3,4)

    # 第num1帧在机体坐标系下的点云
    pcl1_transformed = np.fromfile(fname1, dtype=np.float32).reshape(-1, 4)
    pcl1 = np.fromfile(fname1, dtype=np.float32).reshape(-1, 4)
    pcl2 = np.fromfile(fname2, dtype=np.float32).reshape(-1, 4)
    # 求两帧之间的相对位姿
    T_2_1 = np.linalg.inv(pose_w_2)@pose_w_1

    pcl1_transformed[:,:3] = (T_2_1[:3,:3]@pcl1[:,:3].T+T_2_1[:3,3].reshape(3,1)).T
    visual_pcl(pcl1, pcl2)
    visual_pcl(pcl1_transformed, pcl2)
