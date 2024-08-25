from typing import Dict, List, Optional, Tuple, Union

import torch
from mmdet.structures import DetDataSample
from mmengine.structures import InstanceData
from mmengine.structures import BaseDataElement, InstanceData, PixelData



from .point_data import PointData

class Cls3DDataSample(BaseDataElement):
    """Data sample for 3D classification task.

    For classification task, the data sample contains the following attributes:

    - path (str): Path of the data sample.
    - gt_idx (int): The ground truth index of the sample.
    - name (str): The ground truth label of the sample.
    - num_points_in_gt (int): The number of points in the ground truth.
    - box3d_lidar (BaseInstance3DBoxes): The 3D bounding box in lidar
        coordinate.
    - difficulty (int): The difficulty level of the sample.
    - group_id: (int): Unknown
    - score (float): The score of the sample.
    - ann_info (dict): The annotation information of the sample.
    
    """
    

    
    
    
    
    
    
    
    
    