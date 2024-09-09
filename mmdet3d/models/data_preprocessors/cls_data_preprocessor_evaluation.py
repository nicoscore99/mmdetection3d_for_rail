from typing import Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import torch
import copy
from torch import Tensor
from mmdet3d.registry import MODELS
from mmdet3d.structures.det3d_data_sample import SampleList
from mmdet3d.utils import OptConfigType
from mmengine.model import BaseDataPreprocessor

@MODELS.register_module()
class Cls3DDataPreprocessorEvaluation(BaseDataPreprocessor):
    """ Point cloud data preprocessor for 3D object classification.

    Args:
        non_blocking (bool): Whether to perform non-blocking data transfer.
            Defaults to False.
        normalize (bool): Whether to normalize the point cloud. Defaults to
            True.
        downsample (bool): Whether to downsample the point cloud. Defaults to
            False.
        num_pts_downsample (int): Number of points to downsample. Defaults to
            1024.        
    """

    def __init__(self,
                 non_blocking: bool = False,
                 normalize_size: bool = False,
                 normalize_mean: bool = True,
                 downsample: bool = False,
                 num_pts_downsample: int = 1024,) -> None:
        super(Cls3DDataPreprocessorEvaluation, self).__init__(
            non_blocking=non_blocking)

        self.normalize_size = normalize_size
        self.normalize_mean = normalize_mean
        self.downsample = downsample
        self.num_pts_downsample = num_pts_downsample

    def forward(self,
                data: Union[dict, List[dict]],
                training: bool = False) -> Union[dict, List[dict]]:

        if isinstance(data, list):
            num_augs = len(data)
            aug_batch_data = []
            for aug_id in range(num_augs):
                single_aug_batch_data = self.simple_process(
                    data[aug_id], training)
                aug_batch_data.append(single_aug_batch_data)
            return aug_batch_data

        else:
            return self.simple_process(data, training)
        
    def simple_process(self, data: dict, training: bool = False) -> dict:

        if 'inputs' in data and 'data_samples' in data:
            inputs, data_samples = data['inputs'], data['data_samples']
        else:
            inputs = data
            data_samples = []

        batch_inputs = inputs

        if 'points' in batch_inputs:
            if self.normalize_mean or self.normalize_size:
                for i, res in enumerate(batch_inputs['points']):
                    # if res is a tuple, get the first element
                    if isinstance(res, Tuple):
                        res = res[0]                                 
                    if res.shape[0] > 0:
                        if self.normalize_mean:
                            res[:, :3] = res[:, :3] - res[:, :3].mean(axis=0)
                        if self.normalize_size:
                            res[:, :3] = res[:, :3] / res[:, :3].std(axis=0)

            if self.downsample:
                # assert that there is more than the number of points to downsample
                for i, res in enumerate(batch_inputs['points']):
                    if isinstance(res, Tuple):
                        res = res[0]            
                    
                    if res.shape[0] > self.num_pts_downsample:
                        idx = np.random.choice(
                            res.shape[0], self.num_pts_downsample, replace=False)
                        batch_inputs['points'][i] = res[idx]

        return {'inputs': batch_inputs, 'data_samples': data_samples}

    def collate_data(self, data: dict) -> dict:
        data = self.cast_data(data)  # type: ignore
        data.setdefault('data_samples', None)

        return data

    @torch.no_grad()
    def voxelize(self, points: List[Tensor],
                 data_samples: SampleList) -> Dict[str, Tensor]:
        raise NotImplementedError
    
    def ravel_hash(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def sparse_quantize(self,
                        coords: np.ndarray,
                        return_index: bool = False,
                        return_inverse: bool = False) -> List[np.ndarray]:       
        raise NotImplementedError
