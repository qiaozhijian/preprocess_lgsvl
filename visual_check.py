import numpy as np
from glob import glob
from tqdm import tqdm
import os
import open3d as o3d
import pandas as pd
import shutil
from os.path import join
from sklearn.neighbors import KDTree
from sklearn.decomposition import PCA
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
import re
from pcl_normalize import normalize
import argparse


def visual_pcl(pcl1, pcl2):
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(pcl1[:,:])
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(pcl2[:,:])
    pcd1.paint_uniform_color([1, 0.706, 0])
    pcd2.paint_uniform_color([0, 0.651, 0.929])
    o3d.visualization.draw_geometries([pcd1, pcd2], window_name='Open3D Origin', width=1920, height=1080, left=50, top=50,
                                      point_show_normal=False, mesh_show_wireframe=False, mesh_show_back_face=False)


if __name__ == '__main__':
    path = '/home/jimmy/PycharmProjects/benchmark_datasets/data_save/00/pointcloud_4m_2overlap'
    fname1 = os.path.join(path, '10005.bin')
    fname2 = os.path.join(path, '10223.bin')

    pcl1 = np.fromfile(fname1, dtype=np.float64).reshape(-1, 3)
    pcl2 = np.fromfile(fname2, dtype=np.float64).reshape(-1, 3)

    visual_pcl(pcl1, pcl2)
