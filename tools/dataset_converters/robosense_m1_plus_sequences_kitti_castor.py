# Import libraries

import os
import os.path as osp
import json
import shutil
import random
import argparse
import mmengine
import time
from mmengine import print_log


class ROBOSENSE_M1_PLUS_SEQUENCES_KITTI_CASTOR(object):
    def __init__(self,
                 load_dir,
                 save_dir,
                 prefix,
                 workers=64,
                 release_file_name='robosense_m1_plus_sequences-v0.1.json',
                 test_mode=True,
                 info_prefix='robosense_m1_plus_sequences'):
        
        """
        
        The purpose of this class is to convert the RoboSense M1+ sequences to KITTI format.
        
        The folder structure of the M1+ sequences dataset is as follows:
        
        mmdetection3d
        ├── mmdet3d
        ├── tools
        ├── configs
        ├── data
        |   ├── robosense_m1_plus_sequences
        |   |   ├── scenes
        |   |   |   ├── robosense_m1_plus_sequences-releasefile.json
        |   |   |   ├── Alte_Winterthurerstrasse_1
        |   |   |   |   ├── points
        |   |   |   |   |   ├── Alte_Winterthurerstrasse_1_000000.bin
        |   |   |   |   |   ├── Alte_Winterthurerstrasse_1_000001.bin
        |   |   |   |   |   ├── ...
        |   |   |   |   ├── images
        |   |   |   |   |   ├── Alte_Winterthurerstrasse_1_000000.png
        |   |   |   |   |   ├── Alte_Winterthurerstrasse_1_000001.png
        |   |   |   |   |   ├── ...
        |   |   |   ├── ...
        
        
        Folder structure after processing the dataset:
        
        mmdetection3d
        ├── mmdet3d
        ├── tools
        ├── configs
        ├── data
        |   ├── robosense_m1_plus_sequences
        |   |   |   ├── robosense_m1_plus_sequences-releasefile.json
        |   |   ├── scenes
        |   |   |   ├── Alte_Winterthurerstrasse_1
        |   |   |   |   ├── points
        |   |   |   |   |   ├── Alte_Winterthurerstrasse_1_000000.bin
        |   |   |   |   |   ├── Alte_Winterthurerstrasse_1_000001.bin
        |   |   |   |   |   ├── ...
        |   |   |   |   ├── images
        |   |   |   |   |   ├── Alte_Winterthurerstrasse_1_000000.png
        |   |   |   |   |   ├── Alte_Winterthurerstrasse_1_000001.png
        |   |   |   |   |   ├── ...
        |   |   ├── ImageSets
        |   |   ├── points
        |   |   |   ├── Alte_Winterthurerstrasse_1_000000.bin
        |   |   |   ├── Alte_Winterthurerstrasse_1_000001.bin
        |   |   |   ├── ...
        |   |   ├── labels
        |   |   |   ├── Alte_Winterthurerstrasse_1_000000.txt
        |   |   |   ├── Alte_Winterthurerstrasse_1_000001.txt
        |   |   |   ├── ...
        
        The conversion process is as follows:
        
        1. Transform the robosense_m1_plus_sequences-releasefile.json to the label files
        2. Copy the point cloud files to the points folder
        3. Create the data split
        
        Release data structure:
        
        "dataset": {
            "samples": [
                {
                    "name": "Alte_Winterthurerstrasse_1_segment_1",
                    "attributes": {
                        "frames": [
                            {
                                "name": "Alte_Winterthurerstrasse_1_000000",
                            },
                            {
                                "name": "Alte_Winterthurerstrasse_1_000001",
                            },
                            ...
                        ]
                    }
                    "metadata": {},
                    "labels": {
                        "ground-truth": {
                            "label_status": "LABELED",
                            "attributes": {
                                "frames": [
                                    {
                                        "format_version": "1.0",
                                        "annotations": [
                                            {
                                                "attributes": {
                                                    "position": null,
                                                    "dimensions": null,
                                                    "rotation": null
                                                },
                                                "position": {
                                                    "x": 105.086331,
                                                    "y": -23.255824,
                                                    "z": 1.519104
                                                },
                                                "id": 1,
                                                "category_id": 4,
                                                "track_id": 1,
                                                "index": 0,
                                                "is_keyframe": true,
                                                "type": "cuboid",
                                                "dimensions": {
                                                    "x": 9.629924,
                                                    "y": 3.234876,
                                                    "z": 5.109832
                                                },
                                                "rotation": {
                                                    "qx": 0,
                                                    "qy": 0,
                                                    "qz": 0.998526,
                                                    "qw": 0.054284
                                                },
                                                "yaw": 3.032971
                                            },
                                            {
                                                "attributes": {
                                                    "position": null,
                                                    "dimensions": null,
                                                    "rotation": null
                                                },
                                                "position": {
                                                    "x": 14.892328,
                                                    "y": 2.770917,
                                                    "z": -0.026322
                                                },
                                                "id": 2,
                                                "category_id": 1,
                                                "track_id": 2,
                                                "index": 0,
                                                "is_keyframe": true,
                                                "type": "cuboid",
                                                "dimensions": {
                                                    "x": 0.861574,
                                                    "y": 0.98558,
                                                    "z": 2.052645
                                                },
                                                "rotation": {
                                                    "qx": 0,
                                                    "qy": 0,
                                                    "qz": 0.997203,
                                                    "qw": 0.074739
                                                },
                                                "yaw": 2.991976
                                            }
                                        ], 
                                        "image_attributes": {},
                                        "timestamp": null
                                    }
                                ]
                        }        
                    }
        
        """

        self.load_dir = load_dir
        self.save_dir = save_dir
        self.prefix = prefix
        self.workers = workers
        self.release_file_name = release_file_name
        self.test_mode = test_mode
        self.info_prefix = info_prefix
        self.release_data = None
        self.labeled_frame_names = None
        
        self.train_partitions = 0.0
        self.val_partitions = 1.0
        
        self.lidar_sensors_list = ['lidar']
        
        self.ROBOSENSE_M1_PLUS_SEQUENCES_KITTI_CASTOR_CLASSES = [
            'Pedestrian',
            'Car',
            'Train',
            'Truck',
            'Bike',
            'Motorcycle',
            'Van',
            'Tram',
            'Person (sitting)',
            'Unknown',
            'Wheelchair',
            'Animal',
            'Bus'
        ]
        
        # self.class_name_mapping = {
        #     'Pedestrian': ['Pedestrian', 'Person (sitting)', 'Wheelchair'],
        #     'Car': ['Car', 'Van'],
        #     'Truck': ['Truck', 'Bus'],
        #     'Train': ['Train', 'Tram'],
        #     'Cyclist': ['Bike', 'Motorcycle'],
        #     'Unknown': ['Unknown', 'Animal']
        # }
        
        self.class_name_mapping = {
            'Pedestrian': ['Pedestrian', 'Person (sitting)', 'Wheelchair'],
            'Cyclist': ['Bike', 'Motorcycle'],
            'RoadVehicle': ['Car', 'Van', 'Truck', 'Bus'],
            'Train': ['Train', 'Tram'],
        }
        
        self.lidar_dir = osp.join(self.save_dir, 'points')
        self.labels_dir = osp.join(self.save_dir, 'labels')
        self.image_dir = osp.join(self.save_dir, 'images')
        self.set_dir = osp.join(self.save_dir, 'ImageSets')
        self.scene_folder = osp.join(self.load_dir, 'scenes')
        
        mmengine.mkdir_or_exist(self.lidar_dir)
        mmengine.mkdir_or_exist(self.labels_dir)
        mmengine.mkdir_or_exist(self.image_dir)
        mmengine.mkdir_or_exist(self.set_dir)
        
        self.ensure_mapping_consistency(self.ROBOSENSE_M1_PLUS_SEQUENCES_KITTI_CASTOR_CLASSES, self.class_name_mapping) 
            
        
    ####################################################
    # Helper functions
    ####################################################
    
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
            
    def map_dataset_to_training_class(self, dataset_class):
        """
            Map the dataset_class classes to the classes we want to consider in the training.
        """
        for key, value in self.class_name_mapping.items():
            if dataset_class in value:
                return key
            
        raise ValueError(f'Class {dataset_class} not found in class_names dict')
    
    def is_mapped(self, class_asked):
        """
            Check if the class_asked is in the class_name_mapping dict.
        """
        for key, value in self.class_name_mapping.items():
            if class_asked in value:
                return True
        return False
    
    
    ####################################################
    # Functions to process the release file
    ####################################################
    
    def check_frame_consistency(self, frame_names, frame_annotations):
        
        if len(frame_names) != len(frame_annotations):
            raise ValueError('Length of frame names and frame annotations is not the same')
        
        for frame_name, annotations in zip(frame_names, frame_annotations):
            if len(annotations) == 0:
                # raise ValueError(f'No annotations found for frame {frame_name}')
                print(f'No annotations found for frame {frame_name}')
    
    def get_labeled_samples(self, data):
            
            labeled_samples = []
            
            for sample in data['dataset']['samples']:
                
                if 'labels' in sample:
                    if sample['labels']['ground-truth']:
                        if sample['labels']['ground-truth']['label_status'] == 'REVIEWED':
                            labeled_samples.append(sample)
                    
            
            # Log percentage of labeled samples
            print_log('Percentage of labeled samples: {:.2f}%'.format(len(labeled_samples) / len(data['dataset']['samples']) * 100))         
                    
            return labeled_samples
        
    def get_frame_names(self, samples):
        
        frame_names = []
        
        for sample in samples:
            for frame in sample['attributes']['frames']:
                frame_names.append(frame['name'])
                
        return frame_names
    
    def get_frame_annotations(self, samples):
        
        frame_annotations = []
        
        for sample in samples:
            for frame in sample['labels']['ground-truth']['attributes']['frames']:
                frame_annotations.append(frame['annotations'])
                
        return frame_annotations
    
    def get_category_names(self):
        
        category_names = []
        
        for category in self.release_data['dataset']['task_attributes']['categories']:
            category_names.append(category['name'])
            
        assert category_names == self.ROBOSENSE_M1_PLUS_SEQUENCES_KITTI_CASTOR_CLASSES, f'Category names do not match. Expected {self.ROBOSENSE_M1_PLUS_SEQUENCES_KITTI_CASTOR_CLASSES}, got {category_names}'
        
        print_log('Category names match', logger='current')       
    
    def annotation_to_kitti(self, annotation) -> dict:
        
        """
        
        KITTI format annotation dict:
        
        {
            "object-type": "Car",
            "truncated": 0,
            "occluded": 0,
            "alpha": 0.0, # in radians
            "bbox_2d": {
                "x1": 0,
                "y1": 0,
                "x2": 0,
                "y2": 0
            }
            "bbox_3d_dim": {
                "h": 0,
                "w": 0,
                "l": 0
            }
            "bbox_3d_loc": {
                "x": 0,
                "y": 0,
                "z": 0
            }
            "rotation_y": 0.0, # in radians (yaw in LiDAR coordinates, different from original KITTI)
        }
        
        """
        
        _dict = {}
        
        # Check that the type is "cuboid"
        if not annotation['type'] == 'cuboid':
            raise ValueError('Found annotation type that is not "cuboid"')
        
        # Get the category_id
        category_id = annotation['category_id']
        
        # Get the category name
        category_name = self.ROBOSENSE_M1_PLUS_SEQUENCES_KITTI_CASTOR_CLASSES[category_id-1]
        
        # Check if the category name is in the class name mapping
        if not self.is_mapped(category_name):
            print(f"Category name {category_name} not found in class name mapping")
            return None
        
        # Get the category name from the class name mapping
        category_name = self.map_dataset_to_training_class(category_name)
        
        # Get the position
        position = annotation['position']
        
        # Get the dimensions
        dimensions = annotation['dimensions']
        
        # Get the rotation
        yaw = annotation['yaw']
        
        # Get the rotation quaternion
        rotation = annotation['rotation']
        
        _dict['object-type'] = category_name
        _dict['truncated'] = 0
        _dict['occluded'] = 0
        _dict['alpha'] = 0.0
        _dict['bbox_2d'] = {
            'x1': 0,
            'y1': 0,
            'x2': 0,
            'y2': 0
        }
        _dict['bbox_3d_dim'] = {
            'h': dimensions['y'],
            'w': dimensions['z'],
            'l': dimensions['x']
        }
        
        
        # From segments, we get the cuboid center
        _dict['bbox_3d_loc'] = {
            'x': position['x'],
            'y': position['y'],
            'z': position['z'] - dimensions['z'] / 2
        }
        
        _dict['rotation_y'] = yaw
        
        return _dict
    
    ####################################################
    # Conversion functions
    ####################################################
    
    def process_release_data(self):
        
        self.labeled_samples = self.get_labeled_samples(self.release_data)
        
        self.labeled_frame_names = self.get_frame_names(self.labeled_samples)
        
        self.frame_annotations = self.get_frame_annotations(self.labeled_samples)
        
        self.check_frame_consistency(self.labeled_frame_names, self.frame_annotations)
        
        for frame_name, annotations in zip(self.labeled_frame_names, self.frame_annotations):
            
            label_file = osp.join(self.labels_dir, frame_name[:-4] + '.txt')
            
            # Check that the label file does not exist
            if osp.exists(label_file):
                print_log(f'Label file {label_file} already exists. Will not overwrite')
                continue
            
            annotation_dicts = [self.annotation_to_kitti(annotation) for annotation in annotations]
            
            with open(label_file, 'w') as f:
                
                for annotation_dict in annotation_dicts:
                    if annotation_dict is not None:
                        f.write(f"{annotation_dict['object-type']} {annotation_dict['truncated']} {annotation_dict['occluded']} {annotation_dict['alpha']} {annotation_dict['bbox_2d']['x1']} {annotation_dict['bbox_2d']['y1']} {annotation_dict['bbox_2d']['x2']} {annotation_dict['bbox_2d']['y2']} {annotation_dict['bbox_3d_dim']['h']} {annotation_dict['bbox_3d_dim']['w']} {annotation_dict['bbox_3d_dim']['l']} {annotation_dict['bbox_3d_loc']['x']} {annotation_dict['bbox_3d_loc']['y']} {annotation_dict['bbox_3d_loc']['z']} {annotation_dict['rotation_y']}\n")
                        
            print_log(f'Processed frame {frame_name}')
            
    def copy_lidar_files(self):
        
        for scene in os.listdir(self.scene_folder):
            
            scene_dir = osp.join(self.scene_folder, scene)
            
            if not osp.isdir(scene_dir):
                continue
            
            point_cloud_dir = osp.join(scene_dir, 'points')
            
            if not osp.exists(point_cloud_dir):
                print_log(f'Point cloud directory {point_cloud_dir} does not exist. Skipping...', logger='current')
                continue
            
            for point_cloud_file in os.listdir(point_cloud_dir):
                
                if not point_cloud_file.endswith('.bin'):
                    continue
                
                src = osp.join(point_cloud_dir, point_cloud_file)
                dst = osp.join(self.lidar_dir, point_cloud_file)
                
                if osp.exists(dst):
                    print_log(f'Point cloud file {dst} already exists. Will not overwrite', logger='current')
                    continue
                
                shutil.copyfile(src, dst)
                
                print_log(f'Copied point cloud file {src} to {dst}', logger='current')
                
        print_log('Copying process completed', logger='current')
    
    def generate_datasplit(self):
        
        train_file = osp.join(self.set_dir, 'train.txt')
        val_file = osp.join(self.set_dir, 'val.txt')
        test_file = osp.join(self.set_dir, 'test.txt')
        trainval_file = osp.join(self.set_dir, 'trainval.txt')
        
        if any([osp.exists(train_file), osp.exists(val_file), osp.exists(test_file), osp.exists(trainval_file)]):
            print_log('Data split files already exist. Did not save again. Please delete them if you want to regenerate.', logger='current')
            return
    
        random.shuffle(self.labeled_frame_names)
        
        num_train = int(len(self.labeled_frame_names) * self.train_partitions)
        num_val = int(len(self.labeled_frame_names) * self.val_partitions)
        
        train_frames = self.labeled_frame_names[:num_train]
        val_frames = self.labeled_frame_names[num_train:num_train + num_val]
        
        with open(train_file, 'w') as f:
            for frame in train_frames:
                f.write(frame[:-4] + '\n')
                
        with open(val_file, 'w') as f:
            for frame in val_frames:
                f.write(frame[:-4] + '\n')
                
        with open(trainval_file, 'w') as f:
            for frame in train_frames + val_frames:
                f.write(frame[:-4]+ '\n')
                
        with open(test_file, 'w') as f:
            for frame in val_frames:
                f.write(frame[:-4] + '\n')
                
        print_log('Generated data split files', logger='current')
        
        print_log(f'Number of training frames: {len(train_frames)}', logger='current')
        print_log(f'Number of validation frames: {len(val_frames)}', logger='current')
        
    def validity_checks(self):
        
        # Check that for every label file, there is a corresponding point cloud file
        for label_file in os.listdir(self.labels_dir):
                
                if not label_file.endswith('.txt'):
                    continue
                
                point_cloud_file = osp.join(self.lidar_dir, label_file.replace('.txt', '.bin'))
                
                if not osp.exists(point_cloud_file):
                    raise FileNotFoundError(f'Point cloud file {point_cloud_file} does not exist for label file {label_file}')
                
        print_log('All label files have corresponding point cloud files', logger='current')
        
        # Check that for every entry in the data split files, there is a corresponding label file
        for split_file in os.listdir(self.set_dir):
                    
                    if not split_file.endswith('.txt'):
                        continue
                    
                    with open(osp.join(self.set_dir, split_file), 'r') as f:
                        frame_names = f.readlines()
                        
                        for frame_name in frame_names:
                            label_file = osp.join(self.labels_dir, frame_name.strip() + '.txt')
                            
                            if not osp.exists(label_file):
                                raise FileNotFoundError(f'Label file {label_file} does not exist for frame {frame_name.strip()}')
                            
        print_log('All data split files have corresponding label files', logger='current')
        
        print_log('All validity checks passed', logger='current')
    
    def convert(self):
        
        print_log('Converting RoboSense M1+ sequences to KITTI format...')
        
        # Ensure that the scenes folder exists
        if not osp.exists(self.scene_folder):
            raise FileNotFoundError('The scenes folder does not exist')

        # Ensure that the release file exists
        release_file = osp.join(self.load_dir, self.release_file_name)
        
        # Load the .json release file
        with open(release_file, 'r') as f:
            self.release_data = json.load(f)
        
        time.sleep(3)
        
        self.get_category_names()
        
        # Sleep for 3 seconds
        time.sleep(3)
            
        self.copy_lidar_files()
        
        time.sleep(3)
            
        self.process_release_data()
        
        time.sleep(3)
        
        self.generate_datasplit()
        
        time.sleep(3)
        
        self.validity_checks()
        
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Convert RoboSense M1+ sequences to KITTI format')
    
    parser.add_argument('load_dir', help='Directory to load OSDaR Open Dataset')
    parser.add_argument('save_dir', help='Directory to save converted KITTI-format data')
    parser.add_argument('release_file', help='Release file name')
    parser.add_argument('--prefix', default='', help='Prefix to be added to converted file names')
    parser.add_argument('--num_proc', default=1, help='Number of processes to spawn')
    args = parser.parse_args()

    converter = ROBOSENSE_M1_PLUS_SEQUENCES_KITTI_CASTOR(args.load_dir, args.save_dir, args.prefix, args.num_proc, args.release_file)
    converter.convert()
    converter.generate_datasplit()