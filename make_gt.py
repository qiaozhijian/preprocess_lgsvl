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
from math import pi
from scipy.spatial.transform import Rotation


class gt_maker:
    def __init__(self):
        dirs = '../lgsvl_raw'
        seqs = [str(3).zfill(2)]
        self.dirs_seq = []
        self.dirs_gps = []
        self.dirs_pcl = []
        for seq in seqs:
            self.dirs_seq.append(os.path.join(dirs, seq))
            self.dirs_gps.append(os.path.join(dirs, seq, 'gps'))
            self.dirs_pcl.append(os.path.join(dirs, seq, 'velodyne'))
        self.calib_txt = join(dirs, seq, "calib","000001.txt")
        self.Tr_imu_to_velo, self.Tr_velo_to_imu = self.get_vel2cam()
        self.degree2rad = 1.0*pi/180.0

    def get_vel2cam(self):
        file = open(self.calib_txt)
        lines = file.readlines()
#changed
        line = lines[6]
        line = line.replace('Tr_imu_to_velo: ', '')
        eles = [float(e) for e in line.split(' ') if e != '']
        Tr_imu_to_velo = np.eye(4)
        for i in range(3):
            for j in range(4):
                Tr_imu_to_velo[i, j] = eles[i * 4 + j]
        return np.linalg.inv(Tr_imu_to_velo),Tr_imu_to_velo

    def get_WGS_84(self, file):
        f = open(file)
        line = f.readlines()[1]
        line = line.replace('Transform(position=Vector(', '')
        line = line.replace('), rotation=Vector(', ' ')
        line = line.replace('))\n', '')
        line = line.replace(',', ' ')
        gps = [float(e) for e in line.split(' ') if e!='']
        T = np.eye(4)

#changed
        rotvec = np.asarray([gps[5]*self.degree2rad, gps[3]*self.degree2rad, gps[4]*self.degree2rad])
        T[:3,:3] = Rotation.from_rotvec(rotvec).as_matrix()
        T[0, 3] = gps[2]
        T[1, 3] = gps[0]
        T[2, 3] = gps[1]
        return T

    def write_odo_file(self, dir_gps, odometry):
        dir = join(dir_gps, "..", "odometry_lidar")
        if not os.path.exists(dir):
            os.mkdir(dir)
        file = open(join(dir, 'odometry_lidar.txt'), 'w')
        for pose in odometry:
            for i in range(3):
                for j in range(4):
                    file.write(("%6e" % pose[i, j]))
                    if i != 2 or j != 3:
                        file.write(" ")
            file.write("\n")

    def make(self):

        for dir_gps in self.dirs_gps:
            gpsfiles = sorted(glob(join(dir_gps, '*')))
            T_init = np.eye(4)
            init = True
            odometry = []
            for gpsfile in gpsfiles:
                T = self.get_WGS_84(gpsfile)
                if init:
                    T_init = T
                    init = False
                # 
                T = np.linalg.inv(T_init) @ T
                # T = np.linalg.inv(T) @ T_init
                # T = T @ np.linalg.inv(T_init)
                # T = T_init @ np.linalg.inv(T)
                T = self.Tr_velo_to_imu @ T @ self.Tr_imu_to_velo
                odometry.append(T)
            self.write_odo_file(dir_gps, odometry)

    def get_odometry(self, file):

        file = open(file)
        gts = file.readlines()
        T_odo = []
        for gt in gts:
            pose = np.eye(4)
            eles = [float(ele) for ele in gt.split(' ')]
            for i in range(3):
                for j in range(4):
                    pose[i, j] = eles[i*4+j]
            T_odo.append(pose)
        return T_odo

    def draw_traj(self,gts):
        traj = []
        for gt in gts:
            traj.append(gt[:3,3].reshape(1,3))
        traj = np.concatenate(traj, axis=0)
        self.visual_one_pcl(traj)

    def refine_registration(self, pcl1, pcl2):

        source = o3d.geometry.PointCloud()
        source.points = o3d.utility.Vector3dVector(pcl1)
        target = o3d.geometry.PointCloud()
        target.points = o3d.utility.Vector3dVector(pcl2)
        source.paint_uniform_color([1, 0.706, 0])
        target.paint_uniform_color([0, 0.651, 0.929])

        result = o3d.pipelines.registration.registration_icp(
            source, target, 100, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint())
        source.paint_uniform_color([1, 0.706, 0])
        target.paint_uniform_color([0, 0.651, 0.929])
        # a=np.asarray(source_temp.points)
        source.transform(result.transformation)
        print(result.transformation)
        # b=np.asarray(source_temp.points)
        o3d.visualization.draw_geometries([source, target])
        return result


    def check(self):
        for dir_pcl in self.dirs_pcl:
            pclfiles = sorted(glob(join(dir_pcl, '*')))

            one = 1
            othor = one + 20
            pcl1 = np.fromfile(pclfiles[one], dtype=np.float32).reshape(-1, 4)[:,:3]
            pcl2 = np.fromfile(pclfiles[othor], dtype=np.float32).reshape(-1, 4)[:,:3]

            gts = self.get_odometry(join(join(dir_pcl, "..", "odometry_lidar"), 'odometry_lidar.txt'))
            # self.draw_traj(gts)
            pose1 = gts[one]
            pose2 = gts[othor]
            #self.refine_registration(pcl1, pcl2)
            # self.visual_pcl(pcl1, pcl2, np.linalg.inv(pose1) @ pose2)

    def visual_pcl(self, pcl1, pcl2, T=np.eye(4)):

        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(pcl1)
        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(pcl2)
        pcd1.paint_uniform_color([1, 0.706, 0])
        pcd2.paint_uniform_color([0, 0.651, 0.929])
        pcd2 = pcd2.transform(T)
        o3d.visualization.draw_geometries([pcd1, pcd2], window_name='Open3D Origin', width=1920, height=1080, left=50,
                                           top=50,
                                           point_show_normal=False, mesh_show_wireframe=False, mesh_show_back_face=False)

    def visual_one_pcl(self, pcl, color = [1, 0.706, 0]):

        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(pcl)
        pcd1.paint_uniform_color(color)

        o3d.visualization.draw_geometries([pcd1], window_name='Open3D Origin')



if __name__ == '__main__':
    maker = gt_maker()
    maker.make()
    # maker.check()
