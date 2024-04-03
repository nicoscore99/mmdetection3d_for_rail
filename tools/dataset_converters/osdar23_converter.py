# Import libraries

import os
import os.path as osp
import sys
import copy
import glob
import random
import argparse
import raillabel
from io import BytesIO


import mmengine
import numpy as np
import tensorflow as tf
from mmengine import print_log
from nuscenes.utils.geometry_utils import view_points
from PIL import Image
from waymo_open_dataset.utils import range_image_utils, transform_utils
from waymo_open_dataset.utils.frame_utils import \
    parse_range_image_and_camera_projection

from mmdet3d.datasets.convert_utils import post_process_coords
from mmdet3d.structures import Box3DMode, LiDARInstance3DBoxes, points_cam2img


class OSDaR2_KITTY_Converter(object):
    def __init__(self,
                 load_dir,
                 save_dir,
                 prefix,
                 workers=64,
                 test_mode=True,
                 save_sensor_data=True,
                 save_cam_syn_instances=False,
                 save_cam_instances=True,
                 info_prefix='osdar23',
                 max_sweeps=10,
                 split='train'):
        

        self.load_dir = load_dir
        self.save_dir = save_dir
        self.prefix = prefix
        self.workers = int(workers)
        self.test_mode = test_mode
        self.save_sensor_data = save_sensor_data
        self.save_cam_syn_instances = save_cam_syn_instances
        self.save_cam_instances = save_cam_instances
        self.info_prefix = info_prefix
        self.max_sweeps = max_sweeps
        self.split = split
        
        # No idea what this does
        if int(tf.__version__.split('.')[0]) < 2:
            tf.enable_eager_execution()

        self.camera_sensor_list = TODO
        self.lidar_sensor_list = TODO
        
        # Classes of the underlying dataset to be considered
        self.class_names = ['CAR', 'PEDESTRIAN', 'CYCLIST']

        self.info_map = {
            'training': '_infos_train.pkl',
            'validation': '_infos_val.pkl',
            'testing': '_infos_test.pkl',
            'testing_3d_camera_only_detection': '_infos_test_cam_only.pkl'
        }

        self.filter_empty_3dboxes = True
        self.filter_no_label_zone_points = True
        self.save_track_id = False

        self.image_dir = osp.join(self.save_dir, 'image')
        self.lidar_dir = osp.join(self.save_dir, 'velodyne')
        self.sets_dir = osp.join(self.save_dir, 'ImageSets')

        # Check if there are directories or if they need to be created
        mmengine.mkdir_or_exist(self.image_dir)
        mmengine.mkdir_or_exist(self.lidar_dir)

    def convert(self):
        raise NotImplementedError
        
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('load_dir', help='Directory to load Waymo Open Dataset tfrecords')
    parser.add_argument('save_dir', help='Directory to save converted KITTI-format data')
    parser.add_argument('--prefix', default='', help='Prefix to be added to converted file names')
    parser.add_argument('--num_proc', default=1, help='Number of processes to spawn')
    args = parser.parse_args()

    converter = OSDaR2_KITTY_Converter(args.load_dir, args.save_dir, args.prefix, args.num_proc)
    converter.convert()