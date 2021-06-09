import numpy as np
from glob import glob
from tqdm import tqdm
import os
import open3d as o3d
import pandas as pd
import shutil
from augmentation import TrainTransform
from os.path import join
from sklearn.neighbors import KDTree
from normalize_pcl import normalize
import argparse
from math import pi
from scipy.spatial.transform import Rotation
import concurrent.futures as futures
Degree2Rad = 1.0 * pi / 180.0



class Preprocess:
    ####### 点云原文件名时000000,这里为训练方便起见保存时转成100000  #######
    def __init__(self, source_dir, target_dir, train_dist, test_dist, scope, vox_size):
        self.train_dist = train_dist
        self.test_dist = test_dist
        self.target_dir = target_dir
        seq = os.path.split(source_dir)[-1]
        self.source_dir = join(source_dir, 'velodyne')
        self.traj_files = os.path.join(self.source_dir, '../../../poses/{}.txt'.format(seq))
        self.aug_times = 5

        self.aug_fun = TrainTransform(aug_mode=1)

        self.mkdir_dirs()
        self.generate_samples()
        self.normalize_pcl(scope, vox_size)


    def mkdir_dirs(self):
        self.mkdir_safe(self.source_dir)
        self.mkdir_safe(self.target_dir)

        # later used to store the csv and binary file
        self.tar_train_loc = join(self.target_dir, "./pointcloud_locations_20m_10overlap.csv")
        self.tar_test_loc = join(self.target_dir, "./pointcloud_locations_20m.csv")
        self.tar_train_dir = os.path.join(self.target_dir, 'pointcloud_20m_10overlap')
        self.mkdir_safe(self.tar_train_dir)
        self.tar_test_dir = os.path.join(self.target_dir, 'pointcloud_20m')
        self.mkdir_safe(self.tar_test_dir)


    def generate_samples(self):
        self.positions_train = [[0, 0, 0]]
        self.positions_test = [[0, 0, 0]]
        self.timestamp_list_train = []
        self.timestamp_list_test = []

        positions = self.get_traj_from_odom(self.traj_files)   #timestamp and [northing, easting, altitude]

        # get the timestamp of the cloud point file
        cloud_files = sorted(glob(os.path.join(self.source_dir, '*')))
        timestamps = []
        for cloud_file in cloud_files:
            timestamp = os.path.splitext(cloud_file)[0].split('/')[-1]
            timestamps.append(timestamp)

        # generate the train and test set
        for index in range(positions.shape[0]):
            position = positions[index]
            if self.findNearest(position, self.positions_train, self.train_dist):
                self.positions_train.append(position)
                self.timestamp_list_train.append(100000 + int(timestamps[index]))
            if self.findNearest(position, self.positions_test, self.test_dist):
                for i in range(self.aug_times):
                    self.positions_test.append(position + np.full_like(position, 0.00001)*i)
                    self.timestamp_list_test.append(100000*(i+1) + int(timestamps[index]))

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


    def findNearest(self, pos, database, thresold=0.125):
        database = np.asarray(database).reshape(-1, 3)
        pos = np.asarray(pos).reshape(-1, 3)
        tree = KDTree(database)
        dist, ind = tree.query(pos, k=1)
        if dist[0] > thresold:
            return True
        return False

    def normalize(self, timestamp, tar_dir):
        # 因为是机体坐标系，取样时中心是0
        center_points = np.zeros([1, 3])
        cloud_idx = timestamp % 100000;
        is_origin = (timestamp // 100000) == 1
        # 把编号的首位1去掉
        cloud_fname = str(cloud_idx).zfill(6) + '.bin'
        save_fname = str(timestamp) + '.bin'
        cloud_file = os.path.join(self.source_dir, cloud_fname)
        pclCloud = np.fromfile(cloud_file, dtype=np.float32).reshape(-1, 4)
        # return a set with 4096 points, it requires the input to be 3D rather than 4D
        pclCloud_normed = normalize(center_points, pclCloud, scope, vox_size)

        if not is_origin:
            pclCloud_normed = pclCloud_normed
            pclCloud_normed[:,:3] = self.aug_fun(pclCloud_normed[:,:3])
            pclCloud_normed = pclCloud_normed[np.sum(pclCloud_normed[:,:3], axis=1) <= 0.001, :]

        fname = os.path.join(tar_dir, save_fname)
        pclCloud_normed = pclCloud_normed[:,:3].astype(np.float64)
        pclCloud_normed.tofile(fname)

    def normalize_train(self, timestamp):
        self.normalize(timestamp=timestamp, tar_dir = self.tar_train_dir)

    def normalize_test(self, timestamp):
        self.normalize(timestamp=timestamp, tar_dir = self.tar_test_dir)

    def normalize_pcl(self, scope, vox_size):
        train_traj = pd.read_csv(self.tar_train_loc)
        train_index = train_traj['timestamp']
        # train_traj = train_traj[:, 1:]

        test_traj = pd.read_csv(self.tar_test_loc)
        test_index = test_traj['timestamp']
        # test_traj = test_traj[:, 1:]

        # print("normalize_test start")
        # with futures.ThreadPoolExecutor(8) as executor:
        #     image_infos = executor.map(self.normalize_test, test_index)
        # print("normalize_test finished")
        #
        # print("normalize_train start")
        # with futures.ThreadPoolExecutor(8) as executor:
        #     image_infos = executor.map(self.normalize_train, train_index)
        # print("normalize_train finished")

        for timestamp in tqdm(train_index):
            self.normalize(timestamp, tar_dir=self.tar_train_dir)
        for timestamp in tqdm(test_index):
            self.normalize(timestamp, tar_dir=self.tar_test_dir)


    def get_traj_from_odom(self, odom_fname):
        # the file is a pcd file
        poses = np.loadtxt(odom_fname)
        traj = []
        for pose in poses:
            traj_temp = pose.reshape(3, 4)[:, 3]
            traj.append(traj_temp)

        return np.asarray(traj).reshape(-1, 3)


    def mkdir_safe(self, path):
        os.makedirs(path, exist_ok=True)


def map_reconstruct(cloud_path, odom_fname, traj_fname, target_path, scope, vox_size):
    # 统一到世界坐标系并重构地图
    cloud_files = sorted(glob(os.path.join(cloud_path, '*')))
    poses = np.loadtxt(odom_fname)
    trajs = get_traj(traj_fname)

    center_points = np.zeros([1, 3])
    for cloud_file, pose, pclTraj in zip(tqdm(cloud_files), poses, trajs):
        timestamp = os.path.splitext(cloud_file)[0].split('/')[-1]

        pose_new = np.eye(4)
        pose_new[:3, :] = pose.reshape(3, 4)
        T_w_new = pose_new

        # get the coordinates of the clouds with respect to the
        pclCloud = np.fromfile(cloud_file, dtype=np.float32).reshape(-1, 4)
        pclCloud_trans = np.fromfile(cloud_file, dtype=np.float32).reshape(-1, 4)

        pclCloud_trans[:, :3] = pcl_transform(pclCloud, T_w_new)
        # get the coordinate of the origin in the world coordinate
        center_points[0, :] = pcl_transform(np.zeros([1, 3]), T_w_new)

        # return a set with 4096 points, it requires the input to be 3D rather than 4D
        pclCloud_normed = normalize(center_points, pclCloud_trans, scope, vox_size)

        target_fname = os.path.join(target_path, timestamp + '.bin')
        pclCloud_normed.tofile(target_fname)



def pcl_transform(pcl, trans_matrix):
    pcl_trans = (trans_matrix[:3, :3] @ pcl[:, :3].T + trans_matrix[:3, 3].reshape(3, 1)).T

    return pcl_trans


def centralization(center, pcl_set):
    pcl_set[:, :3] -= center[:, :3]

    return pcl_set


def get_traj(file):
    # the file is a pcd file
    traj_pcd = o3d.io.read_point_cloud(file)
    traj_points = np.asarray(traj_pcd.points)

    return traj_points


def generate_map(target_cloud_path):
    # only data of 3D are saved
    map = []
    cloud_files = sorted(glob(os.path.join(target_cloud_path, '*')))

    for cloud_file in tqdm(cloud_files):
        pcl = np.fromfile(cloud_file, dtype=np.float64).reshape(-1, 3)
        map.append(pcl)
    map = np.concatenate(map[:], axis=0)

    map_pcd = o3d.geometry.PointCloud()
    map_pcd.points = o3d.utility.Vector3dVector(map)
    # map_pcd = map_pcd.voxel_down_sample(voxel_size=0.2)
    map_pcd.paint_uniform_color([1, 0.706, 0])
    o3d.visualization.draw_geometries([map_pcd], window_name='Open3D Origin', width=1920, height=1080, left=50, top=50,
                                      point_show_normal=False, mesh_show_wireframe=False, mesh_show_back_face=False)


def visual_check(target_path):
    num1 = 0
    num2 = 100
    fname1 = os.path.join(target_path, '{}.bin'.format(str(num1).zfill(6)))
    fname2 = os.path.join(target_path, '{}.bin'.format(str(num2).zfill(6)))
    pcl1 = np.fromfile(fname1, dtype=np.float64).reshape(-1, 3)
    pcl2 = np.fromfile(fname2, dtype=np.float64).reshape(-1, 3)
    visual_pcl(pcl1, pcl2)


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
    source_path = '../../dataset/SJTU/odometry/lidar/dataset'
    target_path = '../benchmark_datasets/sjtu'
    # source_path = '/media/qzj/Dataset/slamDataSet/kitti/odometry/data_odometry_velodyne/dataset'
    # target_path = '../benchmark_datasets/kitti'
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default=source_path, type=str)
    parser.add_argument('--target', default=target_path, type=str)
    args = parser.parse_args()

    # the units of the distance and scope are metre
    train_dist = 1
    test_dist = 1.5
    scope = 30  # variable 'scope' is the range of the cube
    vox_size = 0.03  # 1cm

    for seq_source_dir in tqdm(glob(join(args.source, "sequences", "[0-0][0-1]"))):
        seq = os.path.split(seq_source_dir)[-1]
        preprocess = Preprocess(seq_source_dir, join(args.target, seq), train_dist, test_dist, scope, vox_size)

    # #### 还原地图部分
    # cloud_path = os.path.join(args.source, 'lidar')
    # odom_fname = os.path.join(args.source, 'odom_key_frames.txt')
    # # map_reconstruct(cloud_path, odom_fname, traj_fname, args.target, scope, vox_size)
    # generate_map(target_path)

