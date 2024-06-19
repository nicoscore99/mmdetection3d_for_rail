# Import libraries

import os
import os.path as osp
import sys
import copy
import math
import json
from pypcd import pypcd
import shutil
import glob
import random
import argparse
# import raillabel
from io import BytesIO
from scipy.spatial.transform import Rotation as R
import numpy as np


import mmengine
import numpy as np
from mmengine import print_log
from mmdet3d.datasets.convert_utils import post_process_coords
from mmdet3d.structures import Box3DMode, LiDARInstance3DBoxes, points_cam2img


class OSDaR23_2_KITTI_Converter(object):
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
                 sensor_idcs=[0, 1, 2, 3, 4, 5, 6],
                #  sensor_idcs=[0],
                 split='train'):
        
        """
        The purpose of this script is to implement the functionality to cast the OSDaR23 dataset into the KITTI format (for LIDAR-based 3D detection only).

        Input format of the OSDaR23 dataset:

        mmdetection3d
        ├── mmdet3d
        ├── tools
        ├── configs
        ├── data
        |   ├── osdar23
        |   |   ├── orig
        |   |   |   ├── scene_1
        |   |   |   |   // lots of content
        |   |   |   ├── scene_2
        |   |   |   |   // lots of content
        |   |   |   ├── scene_3
        |   |   |   |   ...

        Output format of the KITTI dataset:
        
        mmdetection3d
        ├── mmdet3d
        ├── tools
        ├── configs
        ├── data
        |   ├── osdar23
        |   |   ├── orig
        |   |   ├── ImageSets
                    // Contains the data split into train, val, test, and test_cam_only
        |   |   |   ├── train.txt
        |   |   |   ├── val.txt
        |   |   |   ├── test.txt
        |   |   |   ├── test_cam_only.txt
        |   |   ├── points 
                    // Contains the converted point clouds in binary format
        |   |   |   ├── scene_1_000000.bin
        |   |   |   ├── scene_1_000001.bin
        |   |   |   ├── ...
        |   |   ├── labels
                    // Not even required for just training in lidar points

        Note: It's unnecessary to already here split between train, val, and test. This script implements a functionality to generate data split files 
        that can be defined in the config file. Training, testing and validation data will be loaded accordingly.

        """
    
        self.load_dir = load_dir # Should point to the 'osdar23' folder in the 'data' directory
        self.save_dir = save_dir # Should point to the 'osdar23' folder in the 'data' directory
        # Assert identical save and load directories
        assert self.load_dir == self.save_dir, 'The load and save directories must be identical for this operation'

        self.prefix = prefix
        self.workers = int(workers)
        self.test_mode = test_mode
        self.save_sensor_data = save_sensor_data
        self.save_cam_syn_instances = save_cam_syn_instances
        self.save_cam_instances = save_cam_instances
        self.info_prefix = info_prefix
        self.max_sweeps = max_sweeps
        self.sensor_idcs = sensor_idcs

        self.train_partition = 0.80
        self.val_partition = 0.20

        # self.camera_sensor_list = ['rgb_center', 'rgb_left', 'rgb_right']
        self.lidar_sensor_list = ['lidar']

        self.OSDAR23_CLASSES = ['person',
                                'crowd',
                                'train',
                                'wagons',
                                'bicycle',
                                'group_of_bicycles',
                                'motorcycle',
                                'road_vehicle',
                                'animal',
                                'group_of_animals',
                                'wheelchair',
                                'track',
                                'transition',
                                'switch',
                                'catenary_pole',
                                'signal_pole',
                                'signal',
                                'signal_bridge',
                                'buffer_stop',
                                'flame',
                                'smoke']
        
        # Mapping of the OSDAR classes into the classes we want to consider
        # Hint: Transition and switch are not considered since they are not 3D bounding box labeled in the LiDAR data
        self.class_names = {
            'pedestrian': ['person', 'crowd'],
            'cyclist': ['bicycle', 'group_of_bicycles', 'motorcycle'],
            'car': ['road_vehicle'],
            'train': ['train', 'wagons'],
            'unknown': ['animal', 'group_of_animals'],
            # 'dontcare': ['track', 'catenary_pole', 'signal_pole', 'signal', 'signal_bridge', 'buffer_stop', 'flame', 'smoke', 'switch', 'wheelchair'],
        }

        # self.class_names = {
        #     'pedestrian': ['person', 'crowd'],
        #     'car': ['road_vehicle'],
        #     # 'train': ['train', 'wagons'],
        #     'cyclist': ['bicycle', 'group_of_bicycles', 'motorcycle'],
        #     # 'unknown': ['animal', 'group_of_animals'],
        #     # 'dontcare': ['track', 'catenary_pole', 'signal_pole', 'signal', 'signal_bridge', 'buffer_stop', 'flame', 'smoke', 'switch', 'wheelchair'],
        # }
        
        # self.class_names = {
        #     'pedestrian': ['person', 'crowd'],
        #     'cyclist': ['bicycle', 'group_of_bicycles', 'motorcycle'],
        #     'car': ['road_vehicle'],
        #     # 'train': ['train', 'wagons'],
        #     # 'cyclist': ['bicycle', 'group_of_bicycles', 'motorcycle'],
        #     # 'unknown': ['animal', 'group_of_animals'],
        #     # 'dontcare': ['track', 'catenary_pole', 'signal_pole', 'signal', 'signal_bridge', 'buffer_stop', 'flame', 'smoke', 'switch', 'wheelchair'],
        # }

        self.ensure_mapping_consistency(self.OSDAR23_CLASSES, self.class_names)

        # Mapping of the different sets to the corresponding info files
        # self.info_map = {
        #     'training': '_infos_train.pkl',
        #     'validation': '_infos_val.pkl',
        #     'testing': '_infos_test.pkl',
        #     'testing_3d_camera_only_detection': '_infos_test_cam_only.pkl'
        # }

        self.filter_empty_3dboxes = True
        self.filter_no_label_zone_points = True
        self.save_track_id = False

        self.image_dir = osp.join(self.save_dir, 'image')
        self.lidar_dir = osp.join(self.save_dir, 'points')
        self.sets_dir = osp.join(self.save_dir, 'ImageSets')
        self.labels_dir = osp.join(self.save_dir, 'labels')
        self.calib_dir = osp.join(self.save_dir, 'calib')

        # Check if there are directories or if they need to be created
        mmengine.mkdir_or_exist(self.image_dir)
        mmengine.mkdir_or_exist(self.lidar_dir)
        mmengine.mkdir_or_exist(self.sets_dir)
        mmengine.mkdir_or_exist(self.labels_dir)
        mmengine.mkdir_or_exist(self.calib_dir)

        
    ########################################################
    # Helper functions
    ########################################################

    def x1y1x2y2_to_xywh(self, x1y1x2y2):
        """
            Convert x1y1x2y2 format to xywh format.
        """
        x1, y1, x2, y2 = x1y1x2y2
        return [x1, y1, x2 - x1, y2 - y1]
    
    def xywh_to_x1y1x2y2(self, xywh):
        """
            Convert xywh format to x1y1x2y2 format.
        """
        x, y, w, h = xywh
        return [x, y, x + w, y + h]
    
    def osdarbbox3d_to_kittibbox3d(self, osdar_bbox3d):
        '''
            Convert an OSDaR23 bounding box to a KITTI bounding box.

            OSDaR23 format: [cp_x, cp_y, cp_z, q_x, q_y, q_z, omega, d_x, d_y, d_z]

            where cp_x, cp_y, cp_z are the coordinates of the center point,
            q_x, q_y, q_z, omega are the quaternion values,
            d_x, d_y, d_z are the dimensions of the bounding box.

            KITTI format: [x, y, z, h, w, l, yaw]
        '''

        cp_x, cp_y, cp_z, q_x, q_y, q_z, omega, d_x, d_y, d_z = osdar_bbox3d

        # Convert quaternion to rotation matrix
        transform = R.from_quat([q_x, q_y, q_z, omega])
        yaw = transform.as_euler('zxy')[0]

        # Convert the OSDaR23 bounding box to the KITTI bounding box
        # x, y, z, h, w, l, yaw
        kitti_bbox3d = [cp_x, cp_y, cp_z-d_z/2, d_y, d_z, d_x, yaw]
        return kitti_bbox3d


    def ensure_mapping_consistency(self, osdar_classes, class_names):

        # First, check that all classes in the class_names dict are in the OSDAR23_CLASSES list
        for key, value in class_names.items():
            for val in value:
                if val not in osdar_classes:
                    raise ValueError(f'Class {val} in class_names dict is not in OSDAR23_CLASSES list')
                
        # Second, check which classes are not mapped
        classes_not_found = []
        for osdar_class in osdar_classes:
            found = False
            for key, value in class_names.items():
                if osdar_class in value:
                    found = True
                    break
            if not found:
                classes_not_found.append(osdar_class)

        # Log which classes have not been mapped
        if len(classes_not_found) > 0:
            print_log(f'The following classes have not been mapped: {classes_not_found}', logger='current')
            
    def map_osdar23_to_training_classes(self, osdar23_class):
        """
            Map the OSDaR23 classes to the classes we want to consider in the training.
        """
        for key, value in self.class_names.items():
            if osdar23_class in value:
                return key
        
        return None
    
    def quat2eulers(self, q0:float, q1:float, q2:float, q3:float) -> tuple:
        """
        Compute yaw-pitch-roll Euler angles from a quaternion.
        
        Args
        ----
            q0: Scalar component of quaternion.
            q1, q2, q3: Vector components of quaternion.
        
        Returns
        -------
            (roll, pitch, yaw) (tuple): 321 Euler angles in radians
        """
        roll = math.atan2(
            2 * ((q2 * q3) + (q0 * q1)),
            q0**2 - q1**2 - q2**2 + q3**2
        )  # radians
        pitch = math.asin(2 * ((q1 * q3) - (q0 * q2)))
        yaw = math.atan2(
            2 * ((q1 * q2) + (q0 * q3)),
            q0**2 + q1**2 - q2**2 - q3**2
        )
        return (roll, pitch, yaw)
    
    def create_homogenous_matrix(self, rotation, translation):
        """
            Create a homogenous matrix from a quaternion and a translation.
        """
        homogenous_matrix = np.eye(4)
        homogenous_matrix[:3, :3] = rotation
        homogenous_matrix[:3, 3] = translation
        return homogenous_matrix
    
    def cart_to_homo(self, mat):
        """Convert transformation matrix in Cartesian coordinates to
        homogeneous format.

        Args:
            mat (np.ndarray): Transformation matrix in Cartesian.
                The input matrix shape is 3x3 or 3x4.

        Returns:
            np.ndarray: Transformation matrix in homogeneous format.
                The matrix shape is 4x4.
        """
        ret = np.eye(4)
        if mat.shape == (3, 3):
            ret[:3, :3] = mat
        elif mat.shape == (3, 4):
            ret[:3, :] = mat
        else:
            raise ValueError(mat.shape)
        return ret

    def reformat_pc(self, pcd_uri):
        """
            Change pc format.
        """

        pc = pypcd.PointCloud.from_path(pcd_uri)

        np_x = (np.array(pc.pc_data['x'], dtype=np.float32)).astype(np.float32)
        np_y = (np.array(pc.pc_data['y'], dtype=np.float32)).astype(np.float32)
        np_z = (np.array(pc.pc_data['z'], dtype=np.float32)).astype(np.float32)
        np_i = (np.array(pc.pc_data['intensity'], dtype=np.float32)).astype(np.float32)
        np_sensor_index = (np.array(pc.pc_data['sensor_index'], dtype=np.float32)).astype(np.float32)

        points_32 = np.transpose(np.vstack((np_x, np_y, np_z, np_i, np_sensor_index)))

        # Filter for sensor_index
        if self.sensor_idcs is not None:
            points_32 = points_32[np.isin(points_32[:, -1], self.sensor_idcs)]

        # Omit the sensor index
        points_32 = points_32[:, :-1]

        return points_32
    
    ########################################################
    # Main functions
    ########################################################

    def convert(self):
        """Convert action."""

        print_log('Start converting OSDaR23 dataset to KITTI format', logger='current')

        # All folders in the load directory are considered scenes, add orig_folder to the load directory
        self.orig_folder = osp.join(self.load_dir, 'orig')
        scene_folders = os.listdir(self.orig_folder)

        if self.workers == 0:
            mmengine.track_progress(self.convert_scene, scene_folders)
        else:
            mmengine.track_parallel_progress(self.convert_scene, scene_folders, self.workers)

    def convert_scene(self, scene_folder):
        """
            OSDaR23 dataset comes in folders that contain all the information on a single scene. 
        """
        
        print_log(f'Converting scene {scene_folder}...', logger='current')

        # Access the scene label file (only json file in the scene folder)
        label_file_path = osp.join(self.orig_folder, scene_folder, f'{scene_folder}_labels.json')
        label_file = json.load(open(label_file_path, 'r'))
        frames_dict_keys = list(label_file["openlabel"]['frames'].keys())

        print(f'Frames in label file: {frames_dict_keys}')
        
        # Available lidar frames
        lidar_path = osp.join(self.orig_folder, scene_folder, 'lidar')
        lidar_files_frame_keys = [str(int(file[:3])) for file in os.listdir(lidar_path)]


        print(f'Frames in lidar folder: {lidar_files_frame_keys}')

        # Find the intersection of the lidar files and the frames in the label file
        common_frame_keys = list(set(frames_dict_keys) & set(lidar_files_frame_keys))

        # Copy the lidar files to the output directory
        self.copy_lidar_files(scene_folder, common_frame_keys, lidar_path)

        # Create labels
        self.create_labels(scene_folder, common_frame_keys, label_file)

        # Create calibration file
        self.create_calib(scene_folder, common_frame_keys, label_file)

    def create_calib(self, scene_folder, _common_frame_keys, label_file):
        """
            Create the calibration file for the scene.
        """

        out_dir = osp.join(self.save_dir, 'calib')

        # Check if the output directory exits
        if not os.path.exists(out_dir):
            raise ValueError(f'Output directory {out_dir} does not exist')
        
        
        rgb_left_intrinsic = label_file['openlabel']['streams']['rgb_left']['stream_properties']['intrinsics_pinhole']['camera_matrix']
        rgb_right_intrinsic = label_file['openlabel']['streams']['rgb_right']['stream_properties']['intrinsics_pinhole']['camera_matrix']
        rgb_center_intrinsic = label_file['openlabel']['streams']['rgb_center']['stream_properties']['intrinsics_pinhole']['camera_matrix']

        rgb_left_translation = label_file['openlabel']['coordinate_systems']['rgb_left']['pose_wrt_parent']['translation']
        rgb_left_quaternion = label_file['openlabel']['coordinate_systems']['rgb_left']['pose_wrt_parent']['quaternion']
        rgb_right_translation = label_file['openlabel']['coordinate_systems']['rgb_right']['pose_wrt_parent']['translation']
        rgb_right_quaternion = label_file['openlabel']['coordinate_systems']['rgb_right']['pose_wrt_parent']['quaternion']
        rgb_center_translation = label_file['openlabel']['coordinate_systems']['rgb_center']['pose_wrt_parent']['translation']
        rgb_center_quaternion = label_file['openlabel']['coordinate_systems']['rgb_center']['pose_wrt_parent']['quaternion']

        for frame_key in _common_frame_keys:
                
                frame_key_name = frame_key
                if len(frame_key) == 1:
                    frame_key_name = '00' + frame_key
                elif len(frame_key) == 2:
                    frame_key_name = '0' + frame_key

                # Write the bounding box to the label file
                out_path = osp.join(out_dir, f'{scene_folder}_{frame_key_name}.txt')

                if osp.exists(out_path):
                    print_log(f'File {out_path} already exists', logger='current')
                    continue
                
                # If the file does not exist, create it in any case
                with open(out_path, 'w') as file:
                    print_log(f'Created new file {out_path}.')
                    pass

                with open(out_path, 'a') as f:
                    f.write(f'rgb_left_intrinsic {rgb_left_intrinsic[0]} {rgb_left_intrinsic[1]} {rgb_left_intrinsic[2]} {rgb_left_intrinsic[3]} {rgb_left_intrinsic[4]} {rgb_left_intrinsic[5]} {rgb_left_intrinsic[6]} {rgb_left_intrinsic[7]} {rgb_left_intrinsic[8]} {rgb_left_intrinsic[9]} {rgb_left_intrinsic[10]} {rgb_left_intrinsic[11]}\n')
                    f.write(f'rgb_right_intrinsic {rgb_right_intrinsic[0]} {rgb_right_intrinsic[1]} {rgb_right_intrinsic[2]} {rgb_right_intrinsic[3]} {rgb_right_intrinsic[4]} {rgb_right_intrinsic[5]} {rgb_right_intrinsic[6]} {rgb_right_intrinsic[7]} {rgb_right_intrinsic[8]} {rgb_right_intrinsic[9]} {rgb_right_intrinsic[10]} {rgb_right_intrinsic[11]}\n')
                    f.write(f'rgb_center_intrinsic {rgb_center_intrinsic[0]} {rgb_center_intrinsic[1]} {rgb_center_intrinsic[2]} {rgb_center_intrinsic[3]} {rgb_center_intrinsic[4]} {rgb_center_intrinsic[5]} {rgb_center_intrinsic[6]} {rgb_center_intrinsic[7]} {rgb_center_intrinsic[8]} {rgb_center_intrinsic[9]} {rgb_center_intrinsic[10]} {rgb_center_intrinsic[11]}\n')
                    f.write(f'rgb_left_translation {rgb_left_translation[0]} {rgb_left_translation[1]} {rgb_left_translation[2]}\n')
                    f.write(f'rgb_center_translation {rgb_center_translation[0]} {rgb_center_translation[1]} {rgb_center_translation[2]}\n')
                    f.write(f'rgb_right_translation {rgb_right_translation[0]} {rgb_right_translation[1]} {rgb_right_translation[2]}\n')
                    f.write(f'rgb_left_quaternion {rgb_left_quaternion[0]} {rgb_left_quaternion[1]} {rgb_left_quaternion[2]} {rgb_left_quaternion[3]}\n')
                    f.write(f'rgb_right_quaternion {rgb_right_quaternion[0]} {rgb_right_quaternion[1]} {rgb_right_quaternion[2]} {rgb_right_quaternion[3]}\n')
                    f.write(f'rgb_center_quaternion {rgb_center_quaternion[0]} {rgb_center_quaternion[1]} {rgb_center_quaternion[2]} {rgb_center_quaternion[3]}\n')


    def create_labels(self, scene_folder, _common_frame_keys, label_file):

        # Label format taken by the MMDetection3D framework does not match exactly with the KITTI format
        
        # The format of the 3D bounding box in OSDaR23 is as follows:
        # [cp_x, cp_y, cp_z, q_x, q_y, q_z, omega, d_x, d_y, d_z]

        # ['name', 'truncated', 'occluded', 'alpha', 'bbox', 'dimensions', 'location', 'rotation_y', 'score', 'index', 'group_ids']
        # with dimensions being [h, w, l] and location being [x, y, z]

        out_dir = osp.join(self.save_dir, 'labels')

        # Check if the output directory exits
        if not os.path.exists(out_dir):
            raise ValueError(f'Output directory {out_dir} does not exist')
        
        for frame_key in _common_frame_keys:
            
            objects_in_frame = label_file["openlabel"]['frames'][frame_key]['objects']

            frame_key_name = frame_key
            if len(frame_key) == 1:
                frame_key_name = '00' + frame_key
            elif len(frame_key) == 2:
                frame_key_name = '0' + frame_key

            # Write the bounding box to the label file
            out_path = osp.join(out_dir, f'{scene_folder}_{frame_key_name}.txt')

            if osp.exists(out_path):
                print_log(f'File {out_path} already exists', logger='current')
                continue
            
            # If the file does not exist, create it in any case
            with open(out_path, 'w') as file:
                print_log(f'Created new file {out_path}.')
                pass

            for obj_key, obj_dict in objects_in_frame.items():
                object_data_dict = obj_dict['object_data']

                if 'cuboid' in object_data_dict:

                    class_name = label_file['openlabel']['objects'][obj_key]['type']
                    projected_class_name = self.map_osdar23_to_training_classes(class_name)

                    if projected_class_name is None:

                        # MMDection3D omits the class if it is called "DontCare", thus don't care about this case

                        print_log(f'Class {class_name} not found in the mapping', logger='current')
                    else:

                        cuboid_data = object_data_dict['cuboid']

                        for cuboid in cuboid_data:
                            bbox3d = cuboid['val']
                            
                            # roll, pitch, yaw = self.quat2eulers(q0=bbox3d[6], q1=bbox3d[3], q2=bbox3d[4], q3=bbox3d[5])
                            # Convert the OSDaR23 bounding box to the KITTI bounding box
                            kitti_bbox3d = self.osdarbbox3d_to_kittibbox3d(bbox3d)

                            truncated = 0.0 # Not imblemented
                            occlusion = 0 # Not implemented
                            left = -1   # Not implemented
                            top = -1    # Not implemented
                            right = -1  # Not implemented
                            bottom = -1 # Not implemented
                            x, y, z, height, width, length, rotation_y = kitti_bbox3d
                            alpha = rotation_y # Temporary, probably wrong since its not with respect to the camera
                            # height = bbox3d[9] # d_z
                            # width = bbox3d[7] # d_x
                            # length = bbox3d[8] # d_y
                            # x = bbox3d[0]
                            # y = bbox3d[1]
                            # z = bbox3d[2]
                            # rotation_y = yaw # Temporary, probably wrong since its not with respect to the camera

                            with open(out_path, 'a') as f:
                                f.write(f'{projected_class_name} {truncated} {occlusion} {alpha} {left} {top} {right} {bottom} {height} {width} {length} {x} {y} {z} {rotation_y}\n')

    def map_occlusion(self, occlusion):
        raise NotImplementedError

    def copy_lidar_files(self, scene_folder, _common_frame_keys, lidar_path):
        
        out_dir = osp.join(self.save_dir, 'points')

        # Check if the output directory exits
        if not os.path.exists(out_dir):
            raise ValueError(f'Output directory {out_dir} does not exist')
        
        available_files = os.listdir(lidar_path)
    
        for frame_key in _common_frame_keys:
            # if the frame key is only 2 characters, fill up with zeros
            if len(frame_key) == 1:
                frame_key = '00' + frame_key
            elif len(frame_key) == 2:
                frame_key = '0' + frame_key    
            
            # Find the corresponding lidar file
            lidar_file = [file for file in available_files if frame_key in file[:3]]

            print(lidar_file)

            if len(lidar_file) == 0:
                raise ValueError(f'No lidar file found for frame {frame_key} in scene {scene_folder}')
            elif len(lidar_file) > 1:
                print(_common_frame_keys)
                print(lidar_file)
                raise ValueError(f'Multiple lidar files found for frame {frame_key} in scene {scene_folder}')
            
            copy_path = osp.join(self.orig_folder, scene_folder, 'lidar', lidar_file[0])
            out_path = osp.join(out_dir, f'{scene_folder}_{frame_key}.bin')

            if not osp.exists(out_path):
                pc_xyzi = self.reformat_pc(copy_path)
                pc_xyzi.tofile(out_path)
            else:
                print_log(f'File {out_path} already exists', logger='current')

    def generate_datasplit(self):
        """Generate data split."""

        train_file = osp.join(self.sets_dir, 'train.txt')
        val_file = osp.join(self.sets_dir, 'val.txt')
        test_file = osp.join(self.sets_dir, 'test.txt')
        trainval_file = osp.join(self.sets_dir, 'trainval.txt')

        if any([osp.exists(train_file), osp.exists(val_file), osp.exists(test_file), osp.exists(trainval_file)]):
            print_log('Data split files already exist. Did not save again. Please delete them if you want to regenerate.', logger='current')
            return
        
        # labels and points folder path
        labels_dir = osp.join(self.save_dir, 'labels')
        points_dir = osp.join(self.save_dir, 'points')

        # get all the files in the labels directory
        label_files = os.listdir(labels_dir)
        label_files = [file[:-4] for file in label_files if file.endswith('.txt')]
        
        # get all the files in the points directory
        point_files = os.listdir(points_dir)
        point_files = [file[:-4] for file in point_files if file.endswith('.bin')]

        # Assert that all label files have corresponding point files
        for label_file in label_files:
            if label_file not in point_files:
                raise ValueError(f'No point file found for label file {label_file}')
            
        # Shuffle the files
        random.shuffle(point_files)
        num_point_files = len(point_files)

        # Split the files into train, val, and test
        train_files = point_files[:int(self.train_partition * num_point_files)]
        val_files = point_files[int(self.train_partition * num_point_files):]

        # Write the files to the corresponding text files
        with open(train_file, 'w') as f:
            for file in train_files:
                f.write(f'{file}\n')

        with open(val_file, 'w') as f:
            for file in val_files:
                f.write(f'{file}\n')

        with open(trainval_file, 'w') as f:
            for file in point_files:
                f.write(f'{file}\n')

        with open(test_file, 'w') as f:
            for file in point_files:
                f.write(f'{file}\n')

        print(f'Training set length: {len(train_files)}')
        print(f'Validation set length: {len(val_files)}')
        print('Data split files have been generated successfully.')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('load_dir', help='Directory to load OSDaR Open Dataset tfrecords')
    parser.add_argument('save_dir', help='Directory to save converted KITTI-format data')
    parser.add_argument('--prefix', default='', help='Prefix to be added to converted file names')
    parser.add_argument('--num_proc', default=1, help='Number of processes to spawn')
    args = parser.parse_args()

    converter = OSDaR23_2_KITTI_Converter(args.load_dir, args.save_dir, args.prefix, args.num_proc)
    converter.convert()
    converter.generate_datasplit()