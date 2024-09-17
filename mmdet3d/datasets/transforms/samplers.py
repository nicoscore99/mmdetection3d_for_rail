# from scipy.spatial import Voronoi, voronoi_plot_2d, Delaunay, cKDTree
from mmcv.transforms import BaseTransform
import pypcd.pypcd as pypcd
import numpy as np
import open3d as o3d
import torch
import copy

# baseclass abc
from abc import ABCMeta, abstractmethod
from mmengine.registry import TRANSFORMS

@TRANSFORMS.register_module()
class Open3DAlphaShape(BaseTransform):
    """Base sampler for point cloud data."""

    def __init__(self,
                 alpha: float = 0.2,
                 min_points: int = 50,
                 num_pts_sample: int = 256,
                    **kwargs):
        
        self.alpha = alpha
        self.min_points = min_points
        self.num_pts_sample = num_pts_sample
        
    def transform(self, data):       
        _data = copy.deepcopy(data)
        # _points = data['points'].tensor.numpy()

        # if data is tensor, convert to numpy
        if torch.is_tensor(data['points'].tensor):
            _points = data['points'].tensor.numpy()
        else:
            _points = data['points'].tensor
        
        try:
            if _points.shape[0] >= self.min_points and _points.shape[0] < self.num_pts_sample:
                num_pts_to_generate = self.num_pts_sample - _points.shape[0]
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(_points)
                mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, self.alpha)
                new_points = mesh.sample_points_poisson_disk(num_pts_to_generate)
                new_points_numpy = np.asarray(new_points)
                _new_cloud = torch.from_numpy(new_points_numpy).float()
                _points = torch.from_numpy(_points).float()
                _data['points'].tensor = torch.cat((_points, _new_cloud), 0)
                assert _data['points'].tensor.shape[0] == self.num_pts_sample
                _data['num_points_in_gt'] = self.num_pts_sample
        except Exception as e:
            print(f"Error: {e}")
            print(f"Upsampling inpossible for {data['path']}")
            
        return _data
    
    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__ + '('
        repr_str += f'alpha={self.alpha}, '
        repr_str += f'min_points={self.min_points}, '
        repr_str += f'num_pts_sample={self.num_pts_sample})'
        return repr_str
    
    def __call__(self, data: np.ndarray):
        return self.upsample(data)
    
@TRANSFORMS.register_module()
class Open3DBallPivoting(BaseTransform):
    
    def __init__(self,
                 min_points: int = 50,
                 num_pts_sample: int = 256,
                    **kwargs):
        
        self.min_points = min_points
        self.num_pts_sample = num_pts_sample
        
    def transform(self, data):
        
        _data = copy.deepcopy(data)
        _points = data['points'].tensor.numpy()
        
        try:
            if _points.shape[0] >= self.min_points and _points.shape[0] < self.num_pts_sample:
                num_pts_to_generate = self.num_pts_sample - _points.shape[0]
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(_points)
                pcd.estimate_normals()
                distances = pcd.compute_nearest_neighbor_distance()
                avg_dist = np.mean(distances)
                radius = 2 * avg_dist
                bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector([radius, radius * 2]))
                new_points = bpa_mesh.sample_points_poisson_disk(num_pts_to_generate).points
                # turn points to float32
                new_points_numpy = np.asarray(new_points)
                _new_cloud = torch.from_numpy(new_points_numpy).float()
                _points = torch.from_numpy(_points).float()
                _data['points'].tensor = torch.cat((_points, _new_cloud), 0)
        except Exception as e:
            print(f"Error: {e}")
            print(f"Upsampling inpossible for {data['path']}")
            
        return _data
    
    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__ + '('
        repr_str += f'radius={self.radius}, '
        repr_str += f'min_points={self.min_points}, '
        repr_str += f'num_pts_sample={self.num_pts_sample})'
        return repr_str

@TRANSFORMS.register_module()
class Open3DBallPivotingSequence(BaseTransform):
    
    def __init__(self,
                 min_points: int = 50,
                 num_pts_sample: int = 256,
                    **kwargs):
        
        self.min_points = min_points
        self.num_pts_sample = num_pts_sample
        
    def transform(self, data):
        
        _data = copy.deepcopy(data)
        
        inputs = _data['inputs']
        points = inputs['points']
        
        # If the points are not a sequence, make it a list
        if not isinstance(points, list):
            points = [points]
            
        new_points_list = []
        new_boxes_list = []
            
        for i, cloud in enumerate(points):
            
            # if cloud is tensor, to cpu numpy
            _cloud = cloud
            if isinstance(cloud, torch.Tensor):
                _cloud = cloud.cpu().numpy()

            try:
                if _cloud.shape[0] >= self.min_points and _cloud.shape[0] < self.num_pts_sample:
                    num_pts_to_generate = self.num_pts_sample - _cloud.shape[0]
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(_cloud)
                    pcd.estimate_normals()
                    distances = pcd.compute_nearest_neighbor_distance()
                    avg_dist = np.mean(distances)
                    radius = 3 * avg_dist
                    bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector([radius, radius * 2]))
                    new_points = bpa_mesh.sample_points_poisson_disk(num_pts_to_generate).points
                    # turn points to float32
                    new_points_numpy = np.asarray(new_points)
                    _cloud = torch.from_numpy(_cloud).float()
                    _new_cloud = torch.from_numpy(new_points_numpy).float()
                    
                    # merge the new points with the old points
                    _cloud = torch.cat((_cloud, _new_cloud), 0)
                    
                    # assert that the new cloud has the correct number of points
                    assert _cloud.shape[0] == self.num_pts_sample
                    
            except Exception as e:
                print(f"Error: {e}")
                print(f"Upsampling impossible for cloud {i} with {cloud.shape[0]} points")

            new_points_list.append(_cloud)
            new_boxes_list.append(inputs['bboxes'][i])
            
        inputs['points'] = new_points_list
        inputs['bboxes'] = new_boxes_list
        _data['inputs'] = inputs
            
        return _data
    
    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__ + '('
        repr_str += f'radius={self.radius}, '
        repr_str += f'min_points={self.min_points}, '
        repr_str += f'num_pts_sample={self.num_pts_sample})'
        return repr_str
        