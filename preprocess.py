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

Degree2Rad = 1.0 * pi / 180.0

class Prepross:
    def __init__(self, source_dir, target_dir):
        self.source_dir = source_dir
        self.target_dir = target_dir
        self.mkdir_dirs()
        self.make_gt()
        self.generate_samples()
        self.move_pcl()
        self.normalize_pcl()

    def mkdir_dirs(self):

        self.source_odometry_lidar = join(self.source_dir, "odometry_lidar")
        self.mkdir_safe(self.source_odometry_lidar)
        self.source_calib = join(self.source_dir, "calib")
        self.source_velodyne = join(self.source_dir, "velodyne")
        self.source_gps = join(self.source_dir, "gps")

        self.mkdir_safe(self.target_dir)
        self.tar_train_loc = join(self.target_dir, "./pointcloud_locations_20m_10overlap.csv")
        self.tar_test_loc = join(self.target_dir, "./pointcloud_locations_20m.csv")
        self.tar_train_dir = os.path.join(self.target_dir, 'pointcloud_20m_10overlap')
        self.mkdir_safe(self.tar_train_dir)
        self.tar_test_dir = os.path.join(self.target_dir, 'pointcloud_20m')
        self.mkdir_safe(self.tar_test_dir)
        
    def make_gt(self):
        self.Tr_imu_to_velo, self.Tr_velo_to_imu = self.get_vel2cam()
        gps_files = glob(join(self.source_gps,'*'))
        T_init = np.eye(4)
        init = True
        odometry = []
        self.odometry_dict = {}
        for gps_file in gps_files:
            T = self.get_pose(gps_file)
            if init:
                T_init = T
                init = False
            T = np.linalg.inv(T_init) @ T
            T = self.Tr_velo_to_imu @ T @ self.Tr_imu_to_velo
            odometry.append(T)
            self.odometry_dict[gps_file] = T
        self.write_odo_file(odometry)

    def generate_samples(self):
        self.positions_train = [[0, 0, 0]]
        self.positions_test = [[0, 0, 0]]
        self.timestamp_list_train = []
        self.timestamp_list_test = []
        gps_files = sorted(glob(os.path.join(self.source_gps, '*')))
        for gps_file in gps_files:
            timestamp = os.path.splitext(gps_file)[0].split('/')[-1]
            T = self.odometry_dict[gps_file]
            position = T[:3,3].tolist()
            if self.findNearest(position, self.positions_train, 20):
                self.positions_train.append(position)
                self.timestamp_list_train.append(timestamp)
            if self.findNearest(position, self.positions_test, 40):
                self.positions_test.append(position)
                self.timestamp_list_test.append(timestamp)

        # 去除第一个[0,0,0]
        positions_train = np.asarray(self.positions_train[1:]).reshape(-1, 3)
        positions_test = np.asarray(self.positions_test[1:]).reshape(-1, 3)

        # 字典中的key值即为csv中列名
        dataframe = pd.DataFrame({'timestamp': self.timestamp_list_train, 'northing': positions_train[:, 0].reshape(-1),
                                  'easting': positions_train[:, 1].reshape(-1),
                                  'altitude': positions_train[:, 2].reshape(-1)})
        # 将DataFrame存储为csv,index表示是否显示行名，default=True
        dataframe.to_csv(self.tar_train_loc, index=False, sep=',')

        dataframe = pd.DataFrame({'timestamp': self.timestamp_list_test, 'northing': positions_test[:, 0].reshape(-1),
                                  'easting': positions_test[:, 1].reshape(-1),
                                  'altitude': positions_test[:, 2].reshape(-1)})
        dataframe.to_csv(self.tar_test_loc, index=False, sep=',')

    def move_pcl(self):
        for timestamp in self.timestamp_list_train:
            pcl_file = os.path.join(self.source_velodyne, timestamp + '.bin')
            pcl_file_new = os.path.join(self.tar_train_dir, timestamp + '.bin')
            shutil.copy(pcl_file, pcl_file_new)
        for timestamp in self.timestamp_list_test:
            pcl_file = os.path.join(self.source_velodyne, timestamp + '.bin')
            pcl_file_new = os.path.join(self.tar_test_dir, timestamp + '.bin')
            shutil.copy(pcl_file, pcl_file_new)

    
    def findNearest(self, pos, database, thresold=0.125):
        database = np.asarray(database).reshape(-1, 3)
        pos = np.asarray(pos).reshape(-1, 3)
        tree = KDTree(database)
        dist, ind = tree.query(pos, k=1)
        if dist[0] >thresold:
            return True
        return False

    def normalize_pcl(self):
        ranges = []
        for pcl_file in glob(os.path.join(self.tar_train_dir, '*.bin')):
            pcl = np.fromfile(pcl_file, dtype=np.float32).reshape(-1, 4)
            # pcl = normalize(pcl)
            ranges.append(np.max(np.abs(pcl), axis=0).reshape(1,-1))
            pcl.tofile(pcl_file)
        ranges = np.concatenate(ranges, axis=0)
        ranges = np.mean(ranges, axis=0)
        print("ranges x: {}, y: {}, z: {}".format(ranges[0], ranges[1], ranges[2]))

        for pcl_file in glob(os.path.join(self.tar_test_dir, '*.bin')):
            pcl = np.fromfile(pcl_file, dtype=np.float32).reshape(-1, 4)
            # pcl = normalize(pcl)
            pcl.tofile(pcl_file)


    def get_WGS_84(self, file):
        f = open(file)
        line = f.readlines()[0]
        line = line.replace('GpsData(latitude=', '')
        line = line.replace(', longitude=', ' ')
        line = line.replace(', northing=', ' ')
        line = line.replace(', easting=', ' ')
        line = line.replace(', altitude=', ' ')
        line = line.replace(', orientation=', ' ')
        line = line.replace(')', '')
        gps = [float(e) for e in line.split(' ')]
        northing = gps[2]
        easting = gps[3]
        altitude = gps[-2]
        orientation = gps[-1]
        return northing, easting, altitude, orientation

    def get_pose(self, file):
        f = open(file)
        line = f.readlines()[1]
        line = line.replace('Transform(position=Vector(', '')
        line = line.replace('), rotation=Vector(', ' ')
        line = line.replace('))\n', '')
        line = line.replace(',', ' ')
        gps = [float(e) for e in line.split(' ') if e != '']
        T = np.eye(4)

        # changed
        rotvec = np.asarray([gps[5] * Degree2Rad, gps[3] * Degree2Rad, gps[4] * Degree2Rad])
        T[:3, :3] = Rotation.from_rotvec(rotvec).as_matrix()
        T[0, 3] = gps[2]
        T[1, 3] = gps[0]
        T[2, 3] = gps[1]
        return T

    def write_odo_file(self, odometry):
        file = open(join(self.source_odometry_lidar, 'odometry_lidar.txt'), 'w')
        for pose in odometry:
            for i in range(3):
                for j in range(4):
                    file.write(("%6e" % pose[i, j]))
                    if i != 2 or j != 3:
                        file.write(" ")
            file.write("\n")

    def get_vel2cam(self):
        calib_txt = join(self.source_dir, "calib", "000001.txt")
        file = open(calib_txt)
        lines = file.readlines()
        # changed
        line = lines[6]
        line = line.replace('Tr_imu_to_velo: ', '')
        eles = [float(e) for e in line.split(' ') if e != '']
        Tr_imu_to_velo = np.eye(4)
        for i in range(3):
            for j in range(4):
                Tr_imu_to_velo[i, j] = eles[i * 4 + j]
        return np.linalg.inv(Tr_imu_to_velo), Tr_imu_to_velo
    
    def mkdir_safe(self, path):
        os.makedirs(path, exist_ok=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='../lgsvl_raw', type=str)
    parser.add_argument('--target', default='../LGSVL_new', type=str)
    args = parser.parse_args()

    for seq_source_dir in tqdm(glob(join(args.source, "[0-9][0-9]"))):
        seq_target_dir = seq_source_dir.replace(args.source, args.target)
        preprocess = Prepross(seq_source_dir, seq_target_dir)

