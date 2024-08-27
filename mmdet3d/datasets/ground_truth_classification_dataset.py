from typing import Callable, List, Optional, Set, Union

import numpy as np

from mmdet3d.registry import DATASETS
from mmdet3d.structures import CameraInstance3DBoxes
from .det3d_dataset import Det3DDataset
from mmengine.dataset import BaseDataset
from terminaltables import AsciiTable
from mmengine.logging import print_log


import os.path as osp
import torch


from mmdet3d.structures import get_box_type

# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
from os import path as osp
from typing import Callable, List, Optional, Set, Union

import numpy as np
import torch
from mmengine.dataset import BaseDataset
from mmengine.logging import print_log
from terminaltables import AsciiTable

from mmdet3d.registry import DATASETS
from mmdet3d.structures import get_box_type



@DATASETS.register_module()
class GroundTruthClassificationDataset(BaseDataset):
    r"""Ground Truth Classification Dataset.
    
    This clases serves as the API for experiments on the Ground Truth Labels that get extracted by MMDetection3D from any KITTI-like dataset.
    
    Args:
        data_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        pipeline (List[dict]): Pipeline used for data processing.
            Defaults to []..
        load_type (str): Type of loading mode. Defaults to 'frame_based'.
            - 'frame_based': Load all of the instances in the frame.
            - 'mv_image_based': Load all of the instances in the frame and need
              to convert to the FOV-based data type to support image-based
              detector.
            - 'fov_image_based': Only load the instances inside the default
              cam, and need to convert to the FOV-based data type to support
              image-based detector.
        box_type_3d (str): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'LiDAR' in this dataset. Available options includes:

            - 'LiDAR': Box in LiDAR coordinates.
            - 'Depth': Box in depth coordinates, usually for indoor dataset.
            - 'Camera': Box in camera coordinates.
        min_num_pts (int): Minimum number of points in data sample.
    """
    
    METAINFO = {
        'classes': ('Pedestrian', 'Cyclist', 'Car', 'Van', 'Truck',
                    'Person_sitting', 'Tram', 'Misc'),
        'palette': [(106, 0, 228), (119, 11, 32), (165, 42, 42), (0, 0, 192),
                    (197, 226, 255), (0, 60, 100), (0, 0, 142), (255, 77, 255)]
    }
    
    def __init__(self,
                    data_root: str,
                    ann_file: str,
                    pipeline: List[Union[dict, Callable]] = [],
                    metainfo: Optional[dict] = None,
                    modality: dict = dict(use_lidar=True),
                    data_prefix: dict = dict(pts='velodyne', img=''),
                    min_num_pts: int = 1024,
                    box_type_3d: dict = 'LiDAR',
                    test_mode = False,
                    max_refetch: int = 1000,
                    **kwargs):
        
            self.min_num_pts = min_num_pts
            _default_modality_keys = ('use_lidar', 'use_camera')
            if modality is None:
                modality = dict()
            
                    # Defaults to False if not specify
            for key in _default_modality_keys:
                if key not in modality:
                    modality[key] = False
            self.modality = modality
            assert self.modality['use_lidar'] or self.modality['use_camera'], (
                'Please specify the `modality` (`use_lidar` '
                f', `use_camera`) for {self.__class__.__name__}')

            self.box_type_3d, self.box_mode_3d = get_box_type(box_type_3d)
            
            
            if metainfo is not None and 'classes' in metainfo:
                # we allow to train on subset of self.METAINFO['classes']
                # map unselected labels to -1
                self.label_mapping = {
                    i: -1
                    for i in range(len(self.METAINFO['classes']))
                }
                self.label_mapping[-1] = -1
                for label_idx, name in enumerate(metainfo['classes']):
                    try:
                        ori_label = self.METAINFO['classes'].index(name)
                        self.label_mapping[ori_label] = label_idx
                    except Exception:
                        raise ValueError(
                            f'class {name} is not in {self.METAINFO["classes"]}')

                self.num_ins_per_cat = [0] * len(metainfo['classes'])
            else:
                self.label_mapping = {
                    i: i
                    for i in range(len(self.METAINFO['classes']))
                }
                self.label_mapping[-1] = -1

                self.num_ins_per_cat = [0] * len(self.METAINFO['classes'])

            
            super().__init__(
                ann_file=ann_file,
                metainfo=metainfo,
                data_root=data_root,
                data_prefix=data_prefix,
                pipeline=pipeline,
                test_mode=test_mode,
                max_refetch=max_refetch,
                **kwargs)
            
    def get_ann_info(self, index: int) -> dict:
        """Get annotation info according to the given index.

        Use index to get the corresponding annotations, thus the
        evalhook could use this api.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: Annotation information.
        """
        data_info = self.get_data_info(index)
        # test model
        if 'ann_info' not in data_info:
            ann_info = self.parse_ann_info(data_info)
        else:
            ann_info = data_info['ann_info']

        return ann_info
            
    def parse_ann_info(self, info: dict) -> dict:
        """ Process the 'instance' in data info to 'ann_info'.
        
        Args:
            info (dict): Data information of single data sample.
            
        Returns:
            dict or None: Processed `ann_info`.
            
        info will look like this:
        
        {'name': 'Pedestrian',
        'path': 'kitti_gt_database/0_Pedestrian_0.bin',
        'image_idx': 0,
        'gt_idx': 0,
        'box3d_lidar': array([ 8.73138  , -1.8559175, -1.5996994,  1.2      ,  0.48     ,
                1.89     , -1.5807964], dtype=float32),
        'num_points_in_gt': 377,
        'difficulty': 0,
        'group_id': 0,
        'score': 0.0}
        """
        
        # check if there are any keys in the info dict
        
        if not info:
            return None
        
        # check that the 'name' is one of the classes in the metainfo
        if info['name'] not in self.METAINFO['classes']:
            raise ValueError(f"Class {info['name']} is not in {self.METAINFO['classes']}")
        
        # Turn name from string to index and add to the ann_info dict
        info['class_idx'] = self.METAINFO['classes'].index(info['name'])

        return info     
            
    def parse_data_info(self, info: dict) -> dict:
        """
        
        Process the raw data.
        
        Convert all relative path of needed modality data file to
        the absolute path. And process the `instances` field to
        `ann_info` in training stage.
        
        Args:
            info (dict): Raw info dict.
            
        Returns:
            dict: Has `ann_info` in training stage. And
            all path has been converted to absolute path.
        
        """
        
        info['path'] = osp.join(
                    self.data_prefix.get('pts', ''),
                    info['path'])
        
        info_clone = copy.deepcopy(info)
        info['ann_info'] = self.parse_ann_info(info_clone)
        
        return info
        
    def prepare_data(self, index: int) -> Union[dict, None]:
        """Data preparation for both training and testing stage.

        Called by `__getitem__`  of dataset.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict or None: Data dict of the corresponding index.
        """
        ori_input_dict = self.get_data_info(index)

        # deepcopy here to avoid inplace modification in pipeline.
        input_dict = copy.deepcopy(ori_input_dict)

        # box_type_3d (str): 3D box type.
        input_dict['box_type_3d'] = self.box_type_3d
        # box_mode_3d (str): 3D box mode.
        input_dict['box_mode_3d'] = self.box_mode_3d

        example = self.pipeline(input_dict)

        if not self.test_mode and self.min_num_pts:
            # after pipeline drop the example with empty annotations
            # return None to random another in `__getitem__`
            num_pts = example['data_samples'].num_points_in_gt
            if example is None or example['data_samples'].num_points_in_gt < self.min_num_pts:
                return None

        return example
    
    def get_cat_ids(self, idx: int) -> Set[int]:
        """Get category ids by index. Dataset wrapped by ClassBalancedDataset
        must implement this method.

        The ``CBGSDataset`` or ``ClassBalancedDataset``requires a subclass
        which implements this method.

        Args:
            idx (int): The index of data.

        Returns:
            set[int]: All categories in the sample of specified index.
        """
        info = self.get_data_info(idx)
        gt_label = info['ann_info']['class_idx']
        return [gt_label]
