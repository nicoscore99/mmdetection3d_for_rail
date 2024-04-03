# Import libraries

import os
import os.path as osp
import sys
import copy
import utils
import json
import pypcd
import glob
import random
import argparse
import raillabel
from io import BytesIO
from scipy.spatial.transform import Rotation as R
import numpy as np


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

        self.camera_sensor_list = ['rgb_center', 'rgb_left', 'rgb_right']
        self.lidar_sensor_list = ['lidar']
        
        # Classes of the underlying dataset to be considered
        self.class_names = ['CAR', 'PEDESTRIAN', 'CYCLIST']

        self.OSDAR23_CLASSES = ['person',
                                'crowd',
                                'train',
                                'wagons',
                                'bicycle',
                                'group_of_bicycles',
                                'motorcycle',
                                'street_vehicle',
                                'animal',
                                'group_of_animals',
                                'wheelchair',
                                'track',
                                'transition',
                                'switch',
                                'cantenary_pole',
                                'signal_pole',
                                'signal',
                                'signal_bridge',
                                'buffer_stop',
                                'flame',
                                'smoke']
        
        # Mapping of the OSDAR classes into the classes we want to consider
        # Hint: Transition and switch are not considered since they are not 3D bounding box labeled in the LiDAR data
        self.class_names = {
            'PEDESTRIAN': ['person', 'crowd', 'wheelchair', 'animal', 'group_of_animals'],
            'VEHICLE': ['train', 'wagons', 'street_vehicle'],
            'CYCLIST': ['bicycle', 'group_of_bicycles', 'motorcycle'],
            'OBSTACLE': ['track', 'cantenary_pole', 'signal_pole', 'signal', 'signal_bridge', 'buffer_stop', 'flame', 'smoke']
        }

        self.ensure_mapping_consistency(self.OSDAR23_CLASSES, self.class_names)

        # Mapping of the different sets to the corresponding info files
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
        # R = R.from_quat([q_x, q_y, q_z, omega]).as_matrix()

        # Convert the OSDaR23 bounding box to the KITTI bounding box
        kitti_bbox3d = [cp_x, cp_y, cp_z, d_z, d_x, d_y, omega]
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

    def convert(self):
        """Convert action."""

        print_log(f'Start converting {self.split} set...', logger='current')

        # All folders in the load directory are considered scenes
        scene_folders = os.listdir(self.load_dir)

        if self.workers == 0:
            data_infos = mmengine.track_progress(self.convert_scene,
                                                 scene_folders)
        else:
            data_infos = mmengine.track_parallel_progress(
                self.convert_scene, scene_folders, self.workers)
            
        data_list = []
        for data_info in data_infos:
            data_list.extend(data_info)
        metainfo = dict()
        metainfo['dataset'] = 'osdar23'
        metainfo['version'] = 'osdar23_v1'
        metainfo['info_version'] = 'mmdet3d_v1.4'

        osdar_infos = dict(data_list=data_list, metainfo=metainfo)
        filenames = osp.join(
            osp.dirname(self.save_dir),
            f'{self.info_prefix + self.info_map[self.split]}')
        print_log(f'Saving {self.split} dataset infos into {filenames}')
        mmengine.dump(osdar_infos, filenames)

    def convert_scene(self, scene_folder):
        """
            OSDaR23 dataset comes in folders folders that contain all the information on a single scene. 

            Adapted form the 'convert_one' function in the 'Waymo2KITTIConverter' class in the 'waymo2kitti_converter.py' file.
        """
        
        print_log(f'Converting scene {scene_folder}...', logger='current')

        # Access the scene label file (only json file in the scene folder)
        label_file_path = osp.join(self.load_dir, scene_folder, f'{scene_folder}_labels.json')
        label_file = json.load(open(label_file_path, 'r'))
        _frames = label_file["openlabel"]['frames']

        # Copy the image files to the correct location
        for frame_key, frame_dict in _frames.items():
            image_uri = frame_dict['frame_properties']['streams'][self.camera_sensor_list[0]]['uri']

            # Copy the image file to the correct location
            image_path = osp.join(self.load_dir, scene_folder, image_uri)
            image_save_path = osp.join(self.image_dir, f'{scene_folder.split("/")[0]}_{image_uri.split("/")[-1]}')
            os.system(f'cp {image_path} {image_save_path}')

        file_infos = [] # List of all infos (format: dict) for each frame contained in the scene

        # Copy the lidar files to the correct location
        for frame_key, frame_dict in _frames.items():
            lidar_pcd_uri = frame_dict['frame_properties']['streams'][self.lidar_sensor_list[0]]['uri']

            _pc_xyz = self.reformat_pc(lidar_pcd_uri)

            lidar_filename = lidar_pcd_uri.split("/")[-1]

            lidar_bin_outpath = osp.join(self.lidar_dir, f'{scene_folder.split("/")[0]}_{lidar_pcd_uri.split("/")[-1].replace(".pcd", ".bin")}')

            # Save the lidar file as bin
            pypcd.save_point_cloud_bin(_pc_xyz, lidar_bin_outpath)

            frame_info = self.create_frame_info_file(scene_folder, label_file, frame_key, frame_dict, lidar_filename)
            file_infos.append(frame_info)
                                        
        return file_infos
    
    def create_frame_info_file(self, scene_folder, label_file, frame_key, frame_dict, lidar_filename):
        """
            Training is based on .pkl files that contain the training information of each frame.

            See detailed documentation: https://mmdetection3d.readthedocs.io/en/latest/advanced_guides/datasets/waymo.html 
        """
        
        frame_info = dict()
        cam_info = dict()

        frame_info['sample_idx'] = f'{frame_key}_{frame_dict["frame_properties"]["timestamp"]}'
        frame_info['timestamp'] = frame_dict['frame_properties']['timestamp']
        frame_info['ego2global'] = np.eye(4).tolist() # TODO: Check if this is correct
        frame_info['context_name'] = scene_folder   
        frame_info['lidar_points'] = {
            'lidar_path': lidar_filename,
            'num_pts_feats': 4
        }
        
        # ---- Lidar sweeps ----
        frame_info['lidar_sweeps'] = []

        # ---- Images ----
        frame_info['images'] = dict()

        camera_calibs = []
        Tr_velo_to_cams = []
        T_osdar_ref_to_kitti_cam = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])

        # ---- Camera information ----
        for sensor in self.camera_sensor_list:
            # Camera extrinsics
            Q_cam_to_vehicle_base = label_file["openlabel"]['coordinate_systems'][sensor]['pose_wrt_parent']['quaternion']
            T_cam_to_vehicle_base = label_file["openlabel"]['coordinate_systems'][sensor]['pose_wrt_parent']['translation']
            R_cam_to_vehicle_base = R.from_quat(Q_cam_to_vehicle_base).as_matrix()
            H_cam_to_vehicle_base = self.create_homogenous_matrix(R_cam_to_vehicle_base, T_cam_to_vehicle_base)
            H_vehicle_base_to_cam = np.linalg.inv(H_cam_to_vehicle_base)
            Tr_velo_to_cam = self.cart_to_homo(T_osdar_ref_to_kitti_cam) @ H_vehicle_base_to_cam
            Tr_velo_to_cams.append(Tr_velo_to_cam)

            # Camera intrinsics
            camera_matrix_lst = label_file["openlabel"]['streams'][sensor]['stream_properties']['intrinsics_pinhole']['camera_matrix']
            disto_coeff = label_file["openlabel"]['streams'][sensor]['stream_properties']['intrinsics_pinhole']['distortion_coeffs']

            # Cast len 12 list into 3x4 np matrix
            camera_calib = np.array(camera_matrix_lst).reshape(3, 4)
            camera_calibs.append(camera_calib)

            cam_infos = {
                'height': label_file["openlabel"]['streams'][sensor]['stream_properties']['intrinsics_pinhole']['height_px'],
                'width': label_file["openlabel"]['streams'][sensor]['stream_properties']['intrinsics_pinhole']['width_px'],
                'lidar2cam': Tr_velo_to_cam.astype(np.float32).tolist(), # TODO adapt something wrong
                'cam2img': camera_calib.astype(np.float32).tolist(),
                'lidar2img': (camera_calib @ Tr_velo_to_cam).astype(np.float32).tolist()
            }

            # Add camera information to frame_info dict
            frame_info['images'][sensor] = cam_infos

        # ---- Imapges sweeps ----
        frame_info['image_sweeps'] = []


        # ---- Annotations ----
        instances = []
        objects = frame_dict['objects']
        for obj_key, obj_dict in objects.items():
            object_label = obj_dict['object_data'][0]['name'].split('_')[-1]

            # TODO: There is a better way to get the object key checking the label
            mapped_object_key = self.map_osdar23_to_training_classes(object_label)

            # Don't consider any objects that are not in the classes we want to consider
            if mapped_object_key is None:
                continue
        
            instance_dict = dict()

            if 'bbox' in obj_dict['object_data'].keys():
                avaialble_views = []
                # TODO: Consider whcih boundingbox we are taking.
                for _bbox in obj_dict['object_data']['bbox']:
                    avaialble_views.append(_bbox['coordinate_system'])
            
                if list(set(avaialble_views) & set(self.camera_sensor_list)) == 0:
                    
                    # But now we first have to check which camera view we rely on

                    optimal_sensor_key = self.find_optimal_sensor_key()

                    bbox_xywh = obj_dict['object_data']['bbox']['val']
                    instance_dict['bbox'] = self.x1y1x2y2_to_xywh(bbox_xywh)
                    instance_dict['bbox_label'] = mapped_object_key
            else:
                # Ignore if no bbox is present
                instance_dict['label'] = -1

            if 'cuboid' in obj_dict['object_data'].keys():
                osdar_bbox3d = obj_dict['object_data']['cuboid']['val']
                instance_dict['bbox_3d'] = self.osdarbbox3d_to_kittibbox3d(osdar_bbox3d)
                instance_dict['bbox_label_3d'] = mapped_object_key
            else:
                # Ignore if no cuboid is present
                instance_dict['bbox_label_3d'] = -1

            # TODO: Here we will probabily need to count the number of lidar points inside the bounding box...
            if 'vec' in obj_dict['object_data'].keys():
                instance_dict['num_lidar_pts'] = len(obj_dict['object_data']['vec']['val'])
        
            instance_dict['camera_id'] = None
            instance_dict['group_id'] = None

            instances.append(instance_dict)

        frame_info['instances'] = instances

        # ---- Camera sync instances ----
        frame_info['cam_sync_instances'] = []


        # ---- Camera instances ----
        frame_info['cam_instances']

    def find_optimal_sensor_key():
        """
            Find the optimal sensor key for a given object.
        """
        raise NotImplementedError

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
            Convert a pcd file to a binary file.

            Adapted from the 'convert_pcd_to_bin' function in the 'Waymo2KITTIConverter' class in the 'waymo2kitti_converter.py' file.
        """

        pc = pypcd.PointCloud.from_path(pcd_uri)

        np_x = (np.array(pc.pc_data['x'], dtype=np.float32)).astype(np.float32)
        np_y = (np.array(pc.pc_data['y'], dtype=np.float32)).astype(np.float32)
        np_z = (np.array(pc.pc_data['z'], dtype=np.float32)).astype(np.float32)

        points_32 = np.transpose(np.vstack((np_x, np_y, np_z)))

        pc_xyz = pypcd.make_xyz_point_cloud(points_32)

        return pc_xyz
    
    def generate_datasplit(self, data_infos):
        """Generate data split."""
        raise NotImplementedError

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('load_dir', help='Directory to load OSDaR Open Dataset tfrecords')
    parser.add_argument('save_dir', help='Directory to save converted KITTI-format data')
    parser.add_argument('--prefix', default='', help='Prefix to be added to converted file names')
    parser.add_argument('--num_proc', default=1, help='Number of processes to spawn')
    args = parser.parse_args()

    converter = OSDaR2_KITTY_Converter(args.load_dir, args.save_dir, args.prefix, args.num_proc)
    converter.convert()