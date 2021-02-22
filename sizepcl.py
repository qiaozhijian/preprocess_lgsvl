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
from normalize_pcl import normalize
import argparse


def visual_pcl(pcl):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcl)
    o3d.visualization.draw_geometries([pcd], window_name='Open3D Origin', width=1920, height=1080, left=50, top=50,
                                      point_show_normal=False, mesh_show_wireframe=False, mesh_show_back_face=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', default='../newdata', type=str)
    parser.add_argument('--num', default=23, type=int)
    args = parser.parse_args()

    dataset_dir = args.dir
    for i in tqdm(np.arange(1, args.num + 1)):
        seq = str(i).zfill(2)
        dataset_dir_seq = os.path.join(dataset_dir, seq)
        gps_dir = os.path.join(dataset_dir, seq, 'gps')
        dataset_dir_seq = os.path.join(dataset_dir_seq, "pointcloud_20m_10overlap/*")
        pcl_all = []
        for file in glob(dataset_dir_seq):
            pcl = np.fromfile(file).reshape(-1, 3)
            pcl_all.append(pcl)
            print(np.max(pcl,axis=0)," ",np.min(pcl,axis=0))
            # visual_pcl(pcl)
        # pcl_all = np.concatenate(pcl_all, axis=0)
        # pcl_all = pcl_all
