import copy
from typing import List, Union
import torch
from mmcv.transforms.base import BaseTransform

from mmdet3d.registry import TRANSFORMS

import sys
sys.path.append('/home/cws-ml-lab/ros2_ws/build/depth_clustering_for_rail/pybind/install/depth_clustering_for_rail/lib/')
import depth_cluster


@TRANSFORMS.register_module()
class ClusterPointCloud(BaseTransform):

    def __init__(
        self,
        keys: tuple = (),
        meta_keys: tuple = ()
        ) -> None:
        self.keys = keys
        self.meta_keys = meta_keys
        
        self.depth_cluster_module = depth_cluster.DepthCluster()
    
    def transform(self, results: Union[dict,
                                       List[dict]]) -> Union[dict, List[dict]]:
        """ Method to cluster the complete point cloud scenes into clusters.
        
        Args:
            results (Union[dict, List[dict]]): The input data to be clustered.
        """
        
        # deep copy
        _results = copy.deepcopy(results)
        
        inputs = _results['inputs']
        points = inputs['points']
        
        # if points is tensor, to cpu numpy
        if isinstance(points, torch.Tensor):
            points = points.cpu().numpy()
            
        points_xyz = points[:, 0:3]       
        point_input = copy.deepcopy(points_xyz)
        clusters, bboxes = self.depth_cluster_module.process_cloud(point_input)
        
        # to tensor
        clusters = [torch.tensor(cluster, dtype=torch.float32) for cluster in clusters]
        bboxes = [torch.tensor(bbox, dtype=torch.float32) for bbox in bboxes]
        
        # move to device cuda if necessary
        if torch.cuda.is_available():
            clusters = [cluster.cuda() for cluster in clusters]
            bboxes = [bbox.cuda() for bbox in bboxes]
                
        inputs['points'] = clusters
        inputs['bboxes'] = bboxes
        
        _results['inputs'] = inputs
        
        return _results
        

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(keys={self.keys})'
        repr_str += f'(meta_keys={self.meta_keys})'
        return repr_str         # noqa: E501