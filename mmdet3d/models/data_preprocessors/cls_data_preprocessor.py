# Copyright (c) OpenMMLab. All rights reserved.
import math
from numbers import Number
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from mmdet.models import DetDataPreprocessor
from mmdet.models.utils.misc import samplelist_boxtype2tensor
from mmengine.model import stack_batch
from mmengine.utils import is_seq_of
from torch import Tensor
from torch.nn import functional as F

from mmdet3d.registry import MODELS, INFERENCERS
from mmdet3d.structures.det3d_data_sample import SampleList
from mmdet3d.utils import OptConfigType
from .utils import multiview_img_stack_batch
from .voxelize import VoxelizationByGridShape, dynamic_scatter_3d
from mmengine.model import BaseDataPreprocessor



@MODELS.register_module()
class Cls3DDataPreprocessor(BaseDataPreprocessor):
    """ Point cloud data preprocessor for 3D object classification.

    Args:
        voxel (bool): Whether to apply voxelization to point cloud.
            Defaults to False.
        voxel_type (str): Voxelization type. Two voxelization types are
            provided: 'hard' and 'dynamic', respectively for hard voxelization
            and dynamic voxelization. Defaults to 'hard'.
        voxel_layer (dict or :obj:`ConfigDict`, optional): Voxelization layer
            config. Defaults to None.
        batch_first (bool): Whether to put the batch dimension to the first
            dimension when getting voxel coordinates. Defaults to True.
        max_voxels (int, optional): Maximum number of voxels in each voxel
            grid. Defaults to None.
        non_blocking (bool): Whether to perform non-blocking data transfer.
            Defaults to False.
        normalize (bool): Whether to normalize the point cloud. Defaults to
            True.
        downsample (bool): Whether to downsample the point cloud. Defaults to
            False.
        num_pts_downsample (int): Number of points to downsample. Defaults to
            1024.
        batch_augments (List[dict], optional): Batch augmentation configs.
            Defaults to None.            

    """

    def __init__(self,
                 voxel: bool = False,
                 voxel_type: str = 'hard',
                 voxel_layer: OptConfigType = None,
                 batch_first: bool = True,
                 max_voxels: Optional[int] = None,
                 non_blocking: bool = False,
                 normalize: bool = True,
                 downsample: bool = False,
                 num_pts_downsample: int = 1024,
                 batch_augments: Optional[List[dict]] = None) -> None:
        super(Cls3DDataPreprocessor, self).__init__(
            non_blocking=non_blocking)
        self.voxel = voxel
        self.voxel_type = voxel_type
        self.batch_first = batch_first
        self.max_voxels = max_voxels
        self.normalize = normalize
        self.downsample = downsample
        self.num_pts_downsample = num_pts_downsample
        if voxel:
            self.voxel_layer = VoxelizationByGridShape(**voxel_layer)

    def forward(self,
                data: Union[dict, List[dict]],
                training: bool = False) -> Union[dict, List[dict]]:
        """Perform normalization, padding and bgr2rgb conversion based on
        ``BaseDataPreprocessor``.

        Args:
            data (dict or List[dict]): Data from dataloader. The dict contains
                the whole batch data, when it is a list[dict], the list
                indicates test time augmentation.
            training (bool): Whether to enable training time augmentation.
                Defaults to False.

        Returns:
            dict or List[dict]: Data in the same format as the model input.
        """
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
        """Perform normalization, padding and bgr2rgb conversion for img data
        based on ``BaseDataPreprocessor``, and voxelize point cloud if `voxel`
        is set to be True.

        Args:
            data (dict): Data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.
                Defaults to False.

        Returns:
            dict: Data in the same format as the model input.
        """

        data = self.collate_data(data)

        if 'inputs' in data and 'data_samples' in data:
            inputs, data_samples = data['inputs'], data['data_samples']
        else:
            inputs = data
            data_samples = []


        batch_inputs = dict()

        if 'points' in inputs:
            batch_inputs['points'] = inputs['points']

            if self.normalize:
                for i, res in enumerate(batch_inputs['points']):
                    if res.shape[0] > 0:
                        res[:, :3] = res[:, :3] - res[:, :3].mean(dim=0)
                        # res[:, :3] = res[:, :3] / res[:, :3].std(dim=0)

            if self.downsample:
                # assert that there is more than the number of points to downsample
                assert self.num_pts_downsample <= res.shape[0], \
                    f'Number of points to downsample is greater than the number of points in the point cloud'

                for i, res in enumerate(batch_inputs['points']):
                    if res.shape[0] > self.num_pts_downsample:
                        idx = np.random.choice(
                            res.shape[0], self.num_pts_downsample, replace=False)
                        batch_inputs['points'][i] = res[idx]


        return {'inputs': batch_inputs, 'data_samples': data_samples}

    def collate_data(self, data: dict) -> dict:
        """Copy data to the target device and perform normalization, padding
        and bgr2rgb conversion and stack based on ``BaseDataPreprocessor``.

        Collates the data sampled from dataloader into a list of dict and list
        of labels, and then copies tensor to the target device.

        Args:
            data (dict): Data sampled from dataloader.

        Returns:
            dict: Data in the same format as the model input.
        """
        data = self.cast_data(data)  # type: ignore
        data.setdefault('data_samples', None)

        return data

    @torch.no_grad()
    def voxelize(self, points: List[Tensor],
                 data_samples: SampleList) -> Dict[str, Tensor]:
        """Apply voxelization to point cloud.

        Args:
            points (List[Tensor]): Point cloud in one data batch.
            data_samples: (list[:obj:`Det3DDataSample`]): The annotation data
                of every samples. Add voxel-wise annotation for segmentation.

        Returns:
            Dict[str, Tensor]: Voxelization information.

            - voxels (Tensor): Features of voxels, shape is MxNxC for hard
              voxelization, NxC for dynamic voxelization.
            - coors (Tensor): Coordinates of voxels, shape is Nx(1+NDim),
              where 1 represents the batch index.
            - num_points (Tensor, optional): Number of points in each voxel.
            - voxel_centers (Tensor, optional): Centers of voxels.
        """

        raise NotImplementedError
    
    def ravel_hash(self, x: np.ndarray) -> np.ndarray:
        """Get voxel coordinates hash for np.unique.

        Args:
            x (np.ndarray): The voxel coordinates of points, Nx3.

        Returns:
            np.ndarray: Voxels coordinates hash.
        """
        raise NotImplementedError

    def sparse_quantize(self,
                        coords: np.ndarray,
                        return_index: bool = False,
                        return_inverse: bool = False) -> List[np.ndarray]:
        """Sparse Quantization for voxel coordinates used in Minkunet.

        Args:
            coords (np.ndarray): The voxel coordinates of points, Nx3.
            return_index (bool): Whether to return the indices of the unique
                coords, shape (M,).
            return_inverse (bool): Whether to return the indices of the
                original coords, shape (N,).

        Returns:
            List[np.ndarray]: Return index and inverse map if return_index and
            return_inverse is True.
        """
        
        raise NotImplementedError
