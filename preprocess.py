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

class Prepross:
    def __init__(self):
        self.positions_train = [[0, 0, 0]]
        self.positions_test = [[0, 0, 0]]
        self.num = 0
        self.timestamp_list_train = []
        self.timestamp_list_test = []
        self.train_interval = 0.125
        self.test_interval = 0.25
        # 除以尺度
        self.scale = 40.0

    def findNearest(self, pos, database, criterion='train'):
        database = np.asarray(database).reshape(-1, 3)
        pos = np.asarray(pos).reshape(-1, 3)
        tree = KDTree(database)
        dist, ind = tree.query(pos, k=1)
        if criterion == 'train' and dist[0] > 0.125:
            return True
        if criterion == 'test' and dist[0] > 0.25:
            return True
        return False

    def generate_samples(self, gps_dir, dest_dir_seq):
        print('start analyse gps from {}'.format(gps_dir))
        gps_files = sorted(glob(os.path.join(gps_dir, '*')))
        for gps_file in gps_files:
            timestamp = os.path.splitext(gps_file)[0].split('/')[-1]
            northing, easting, altitude, orientation = self.get_WGS_84(gps_file)
            position = [northing, easting, altitude]
            position = [i / self.scale for i in position]
            bTrain = self.findNearest(position, self.positions_train, criterion='train')
            bTest = self.findNearest(position, self.positions_test, criterion='test')
            if bTrain or bTest:
                if bTrain:
                    self.positions_train.append(position)
                    self.timestamp_list_train.append(timestamp)
                if bTest:
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
        dataframe.to_csv(join(dest_dir_seq, "./pointcloud_locations_20m_10overlap.csv"), index=False, sep=',')

        dataframe = pd.DataFrame({'timestamp': self.timestamp_list_test, 'northing': positions_test[:, 0].reshape(-1),
                                  'easting': positions_test[:, 1].reshape(-1),
                                  'altitude': positions_test[:, 2].reshape(-1)})
        dataframe.to_csv(join(dest_dir_seq, "./pointcloud_locations_20m.csv"), index=False, sep=',')

    def generate_pcl(self, pcl_dir,  dest_dir_seq):
        print('start copy {} from {}'.format(dest_dir_seq, pcl_dir))
        train_dir = os.path.join(dest_dir_seq,'pointcloud_20m_10overlap')
        test_dir = os.path.join(dest_dir_seq,'pointcloud_20m')
        if not os.path.exists(test_dir):
            os.mkdir(test_dir)
        if not os.path.exists(train_dir):
            os.mkdir(train_dir)
        for timestamp in self.timestamp_list_train:
            pcl_file = os.path.join(pcl_dir,timestamp+'.bin')
            pcl_file_new = os.path.join(train_dir,timestamp+'.bin')
            shutil.copy(pcl_file, pcl_file_new)
        for timestamp in self.timestamp_list_test:
            pcl_file = os.path.join(pcl_dir,timestamp+'.bin')
            pcl_file_new = os.path.join(test_dir,timestamp+'.bin')
            shutil.copy(pcl_file, pcl_file_new)

    def normalize_pcl(self, dest_dir_seq):
        print('start copy {} from {}'.format(dest_dir_seq, pcl_dir))
        train_dir = os.path.join(dest_dir_seq,'pointcloud_20m_10overlap')
        test_dir = os.path.join(dest_dir_seq,'pointcloud_20m')
        for pcl_file in glob(os.path.join(train_dir, '*.bin')):
            try:
                pcl = np.fromfile(pcl_file, dtype=np.float32).reshape(-1,4)
                pcl = normalize(pcl, self.scale)
                pcl.tofile(pcl_file)
            except:
                print('wrong train_dir {}'.format(pcl_file))
        for pcl_file in glob(os.path.join(test_dir, '*.bin')):
            try:
                pcl = np.fromfile(pcl_file, dtype=np.float32).reshape(-1,4)
                pcl = normalize(pcl, self.scale)
                pcl.tofile(pcl_file)
            except:
                print('wrong train_dir {}'.format(pcl_file))
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', default='../', type=str)
    parser.add_argument('--dest', default='../../benchmark_datasets/LGSVL', type=str)
    parser.add_argument('--num', default=23, type=int)
    args = parser.parse_args()

    dataset_dir = args.dir
    dest_dir = args.dest
    img = 'image_jpg'
    for i in tqdm(np.arange(1,args.num+1)):
        seq = str(i).zfill(2)
        dest_dir_seq = os.path.join(dest_dir, seq)
        if not os.path.exists(dest_dir_seq):
            os.makedirs(dest_dir_seq, 0o777)
        gps_dir = os.path.join(dataset_dir, seq, 'gps')
        pcl_dir = os.path.join(dataset_dir, seq, 'velodyne')

        preprocess = Prepross()
        preprocess.generate_samples(gps_dir=gps_dir, dest_dir_seq=dest_dir_seq)
        preprocess.generate_pcl(pcl_dir, dest_dir_seq=dest_dir_seq)
        preprocess.normalize_pcl(dest_dir_seq=dest_dir_seq)

        print('finish')

