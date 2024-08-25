# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Sequence, Union

import mmengine
import numpy as np
import torch
from mmcv import BaseTransform
from mmengine.structures import InstanceData
from numpy import dtype
import inspect

from mmdet3d.registry import TRANSFORMS
from mmdet3d.structures import Cls3DDataSample, BaseInstance3DBoxes
from mmdet3d.structures.points import BasePoints

def to_tensor(
    data: Union[torch.Tensor, np.ndarray, Sequence, int,
                float]) -> torch.Tensor:
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.

    Args:
        data (torch.Tensor | numpy.ndarray | Sequence | int | float): Data to
            be converted.

    Returns:
        torch.Tensor: the converted data.
    """

    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        if data.dtype is dtype('float64'):
            data = data.astype(np.float32)
        return torch.from_numpy(data)
    elif isinstance(data, Sequence) and not mmengine.is_str(data):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(f'type {type(data)} cannot be converted to tensor.')


@TRANSFORMS.register_module()
class PackClsInputs(BaseTransform):
    """ Pack the data for a classification task."""
    INPUT_KEYS = ('points')
    INSTANCEDATA_3D_KEYS = [
        'gt_bboxes_3d'
    ]
    def __init__(self, 
                 keys=('path', 'gt_idx', 'name', 'num_points_in_gt', 'box3d_lidar', 'difficulty', 'group_id', 'score', 'ann_info'),
                 meta_keys=('img_path', 'ori_shape', 'img_shape', 'lidar2img',
                            'depth2img', 'cam2img', 'pad_shape',
                            'scale_factor', 'flip', 'pcd_horizontal_flip',
                            'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d',
                            'img_norm_cfg', 'num_pts_feats', 'pcd_trans',
                            'sample_idx', 'pcd_scale_factor', 'pcd_rotation',
                            'pcd_rotation_angle', 'lidar_path',
                            'transformation_3d_flow', 'trans_mat',
                            'affine_aug', 'sweep_img_metas', 'ori_cam2img',
                            'cam2global', 'crop_offset', 'img_crop_offset',
                            'resize_img_shape', 'lidar2cam', 'ori_lidar2img',
                            'num_ref_frames', 'num_views', 'ego2global',
                            'axis_align_matrix')):
        
        self.keys = keys
        self.meta_keys = meta_keys
        
    def transform(self, results: dict) -> dict:
        """Transform the data for a classification task.
        
        Args:
            results (dict): The input data.
            
        Returns:
            dict: The packed data.
        """
        
    # augtest
        if isinstance(results, list):
            if len(results) == 1:
                # simple test
                return self.pack_single_results(results[0])
            pack_results = []
            for single_result in results:
                pack_results.append(self.pack_single_results(single_result))
            return pack_results
        # norm training and simple testing
        elif isinstance(results, dict):
            pack_results = self.pack_single_results(results)
            return pack_results
        else:
            raise NotImplementedError("The input data should be a dict or a list of dict.")
        
        
    def pack_single_results(self, results: dict) -> dict:
        """Pack the data for a single sample.
        
        Args:
            results (dict): The input data.
            
        Returns:
            dict: The packed data.
        """
        
        if 'points' in results:
            if isinstance(results['points'], BasePoints):
                results['points'] = results['points'].tensor
        else:
            raise ValueError("The input data should contain 'points'.")
        
        data_sample = Cls3DDataSample()
        
        data_metas = {}
        for key in self.meta_keys:
            if key in results:
                data_metas[key] = results[key]
            elif 'lidar_points' in results:
                if key in results['lidar_points']:
                    data_metas[key] = results['lidar_points'][key]
        data_sample.set_metainfo(data_metas)
        
        for key in self.keys:
            if key in results:
                setattr(data_sample, key, results[key])
        
        inputs = {'points': to_tensor(results['points'])}          
                    
        packed_results = dict()
        packed_results['data_samples'] = data_sample
        packed_results['inputs'] = inputs
        
        return packed_results
                
        
    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(keys={self.keys})'
        repr_str += f'(meta_keys={self.meta_keys})'
        return repr_str  
    