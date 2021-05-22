import numpy as np
import open3d as o3d
from lgsvl.normalize_pcl import normalize


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

    file1 = open('../benchmark_datasets/lgsvl_raw/00/velodyne/000003.bin')
    pcl1 = np.fromfile(file1, dtype=np.float32).reshape(-1, 4)
    pcl1 = normalize(pcl1)

    file1 = open('../benchmark_datasets/GTA5/round1/pcl/10.bin')
    pcl1 = np.fromfile(file1, dtype=np.float64).reshape(-1, 3)

    file2 = open('../benchmark_datasets/oxford/2014-05-19-13-20-57/pointcloud_20m_10overlap/1400505898024354.bin')
    pcl2 = np.fromfile(file2, dtype=np.float64).reshape(-1, 3)

    print(np.max(np.abs(pcl1), axis=0))
    print(np.max(np.abs(pcl2), axis=0))
    visual_pcl(pcl1, pcl2)