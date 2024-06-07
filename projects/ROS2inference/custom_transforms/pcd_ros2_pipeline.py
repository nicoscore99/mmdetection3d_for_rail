import copy
from typing import List, Optional, Union

import mmcv
import mmengine
import numpy as np
from sensor_msgs_py import point_cloud2 as pc2
from mmcv.transforms import LoadImageFromFile
from mmcv.transforms.base import BaseTransform
from mmdet.datasets.transforms import LoadAnnotations
from mmengine.fileio import get

from mmdet3d.registry import TRANSFORMS
from mmdet3d.structures.bbox_3d import get_box_type
from mmdet3d.structures.points import BasePoints, get_points_type

# 

@TRANSFORMS.register_module()
class LoadPointsFromPointcloud2(BaseTransform):
    "Load points from the ROS2 message fromat Pointcloud2."

    def __init__(self,
                 coord_type: str,
                 load_dim: int = 4,
                 use_dim: Union[int, List[int]] = [0, 1, 2],
                 shift_height: bool = False,
                 use_color: bool = False,
                 norm_intensity: bool = False,
                 norm_elongation: bool = False,
                 backend_args: Optional[dict] = None) -> None:
        super().__init__()
        
        # TODO: Perhaps here we should specify the skip_nans argument
        
    def _load_points(self, pcd2_msg) -> np.ndarray:
        # Load the pointcloud2 message from the file
        return pc2.read_points(pcd2_msg, field_names=("x", "y", "z", "intensity"), skip_nans=True)
    
    def transform(self, results: dict) -> dict:

        pc2_msg = results['lidar_points']
        points = self._load_points(pc2_msg)
        # This step is reduncant since the point dimension is already specified in the _load_points function
        points = points.reshape(-1, self.load_dim)
        points = points[:, self.use_dim]
        if self.norm_intensity:
            assert len(self.use_dim) >= 4, \
                f'When using intensity norm, expect used dimensions >= 4, got {len(self.use_dim)}'
            
    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__ + '('
        repr_str += f'shift_height={self.shift_height}, '
        repr_str += f'use_color={self.use_color}, '
        repr_str += f'backend_args={self.backend_args}, '
        repr_str += f'load_dim={self.load_dim}, '
        repr_str += f'use_dim={self.use_dim})'
        repr_str += f'norm_intensity={self.norm_intensity})'
        repr_str += f'norm_elongation={self.norm_elongation})'
        return repr_str