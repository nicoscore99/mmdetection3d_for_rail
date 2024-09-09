# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Tuple
from abc import ABCMeta, abstractmethod
from typing import Dict, List, Union

import numpy as np
import torch
from torch import Tensor
from torch import nn as nn

from mmdet3d.registry import MODELS
from mmdet3d.utils import ConfigType, OptConfigType, OptMultiConfig
from ...structures.det3d_data_sample import OptSampleList, SampleList
from ..utils import add_prefix
from mmengine.model import BaseModel
from abc import ABCMeta, abstractmethod
from mmdet3d.models.segmentors.classifier_base import BaseClassifier

from mmdet3d.structures.det3d_data_sample import (ForwardResults,
                                                  OptSampleList, SampleList)


@MODELS.register_module()
class EncoderCls3D(BaseModel):
    """3D Encoder-Classifier.
    
    Args:

        backbone (ConfigType): Config of backbone.
        cls_head (ConfigType): Config of classification head.
        neck (OptConfigType): Config of neck. Defaults to None.
        loss_regularization (OptMultiConfig): Config of regularization loss. Defaults to None.
        train_cfg (OptConfigType): Config of training. Defaults to None.
        test_cfg (OptConfigType): Config of testing. Defaults to None.
        data_preprocessor (OptConfigType): Config of data preprocessing. Defaults to None.
        init_cfg (OptMultiConfig): Config of initialization. Defaults to None.
        min_points (int): Number of points required to run the inference. Ignored if None. Defaults to 256.
    """

    def __init__(self,
                 backbone: ConfigType,
                 cls_head: ConfigType,
                 neck: OptConfigType = None,
                 loss_regularization: OptMultiConfig = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 num_pts_sample: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super(EncoderCls3D, self).__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)
        self._init_cls_head(cls_head)
        self._init_loss_regularization(loss_regularization)
        self.num_pts_sample = num_pts_sample

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def _init_cls_head(self, head: ConfigType) -> None:
        """Initialize ``cls_head``."""
        self.cls_head = MODELS.build(head)
        self.num_classes = self.cls_head.num_classes

    def _init_loss_regularization(self,
                                  loss_regularization: OptMultiConfig = None
                                  ) -> None:
        """Initialize ``loss_regularization``."""
        if loss_regularization is not None:
            if isinstance(loss_regularization, list):
                self.loss_regularization = nn.ModuleList()
                for loss_cfg in loss_regularization:
                    self.loss_regularization.append(MODELS.build(loss_cfg))
            else:
                self.loss_regularization = MODELS.build(loss_regularization)
                
    @property
    def with_neck(self) -> bool:
        """bool: Whether the segmentor has neck."""
        return hasattr(self, 'neck') and self.neck is not None

    @property
    def with_cls_head(self) -> bool:
        """bool: Whether the segmentor has decode head."""
        return hasattr(self, 'cls_head') and self.cls_head is not None

    @property
    def with_regularization_loss(self) -> bool:
        """bool: Whether the segmentor has regularization loss for weight."""
        return hasattr(self, 'loss_regularization') and \
            self.loss_regularization is not None

                
    def forward(self,
                inputs: Union[dict, List[dict]],
                data_samples: OptSampleList = None,
                mode: str = 'tensor') -> ForwardResults:
        """The unified entry for a forward process in both training and test.

        The method should accept three modes: "tensor", "predict" and "loss":

        - "tensor": Forward the whole network and return tensor or tuple of
          tensor without any post-processing, same as a common nn.Module.
        - "predict": Forward and return the predictions, which are fully
          processed to a list of :obj:`SegDataSample`.
        - "loss": Forward and return a dict of losses according to the given
          inputs and data samples.

        Note that this method doesn't handle neither back propagation nor
        optimizer updating, which are done in the :meth:`train_step`.

        Args:
            inputs (dict or List[dict]): Input sample dict which includes
                'points' and 'imgs' keys.

                - points (List[Tensor]): Point cloud of each sample.
                - imgs (Tensor): Image tensor has shape (B, C, H, W).
            data_samples (List[:obj:`Det3DDataSample`], optional):
                The annotation data of every samples. Defaults to None.
            mode (str): Return what kind of value. Defaults to 'tensor'.

        Returns:
            The return type depends on ``mode``.

            - If ``mode="tensor"``, return a tensor or a tuple of tensor.
            - If ``mode="predict"``, return a list of :obj:`Det3DDataSample`.
            - If ``mode="loss"``, return a dict of tensor.
        """
        if mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            return self.predict(inputs, data_samples)
        elif mode == 'tensor':
            return self._forward(inputs, data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode')

    def extract_feat(self, batch_inputs: Tensor) -> dict:
        """Extract features from points."""
        x = self.backbone(batch_inputs)
        if self.with_neck:
            x = self.neck(x)
        return x

    def encode_classify(self, batch_inputs: Tensor,
                      batch_input_metas: List[dict]) -> Tensor:
        """Encode points with backbone and classify with cls_head.

        Args:
            batch_input (Tensor): Input point cloud sample
            batch_input_metas (List[dict]): Meta information of a batch of
                samples.

        Returns:
            Tensor: Classification logits of shape [B, num_classes].
        """
        
        x = self.extract_feat(batch_inputs)
        classification = self.cls_head.predict(x, batch_input_metas,
                                              self.test_cfg)
        return classification

    def _cls_head_forward_train(
            self, batch_inputs_dict: dict,
            batch_data_samples: SampleList) -> Dict[str, Tensor]:
        """Run forward function and calculate loss for classification head training.
        
        Args:
            batch_input (Tensor): Input point cloud sample
            batch_data_samples (List[:obj:`Det3DDataSample`]): The det3d data
                samples. It usually includes information such as `metainfo` and
                `gt_pts_seg`.

        Returns:
            Dict[str, Tensor]: A dictionary of loss components for cls head.
        """
        losses = dict()
        # loss_decode = self.decode_head.loss(batch_inputs_dict,
        #                                     batch_data_samples, self.train_cfg)
        
        loss_cls = self.cls_head.loss(batch_inputs_dict, batch_data_samples,
                                      self.train_cfg)

        losses.update(add_prefix(loss_cls, 'loss'))
        return losses

    def _loss_regularization_forward_train(self) -> Dict[str, Tensor]:
        """Calculate regularization loss for model weight in training."""
        losses = dict()
        if isinstance(self.loss_regularization, nn.ModuleList):
            for idx, regularize_loss in enumerate(self.loss_regularization):
                loss_regularize = dict(
                    loss_regularize=regularize_loss(self.modules()))
                losses.update(add_prefix(loss_regularize, f'regularize_{idx}'))
        else:
            loss_regularize = dict(
                loss_regularize=self.loss_regularization(self.modules()))
            losses.update(add_prefix(loss_regularize, 'regularize'))

        return losses

    def loss(self, batch_inputs_dict: dict,
             batch_data_samples: SampleList) -> Dict[str, Tensor]:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs_dict (dict): Input sample dict which
                includes 'points' and 'imgs' keys.

                - points (List[Tensor]): Point cloud of each sample.
                - imgs (Tensor, optional): Image tensor has shape (B, C, H, W).
            batch_data_samples (List[:obj:`Det3DDataSample`]): The det3d data
                samples. It usually includes information such as `metainfo` and
                `gt_pts_seg`.

        Returns:
            Dict[str, Tensor]: A dictionary of loss components.
        """

        # extract features using backbone
        points = torch.stack(batch_inputs_dict['points'])

        x = self.extract_feat(points)

        losses = dict()

        loss_cls = self._cls_head_forward_train(x, batch_data_samples)
        losses.update(loss_cls)
        
        if self.with_regularization_loss:
            loss_regularize = self._loss_regularization_forward_train()
            losses.update(loss_regularize)

        return losses

    @staticmethod
    def _input_generation(coords,
                          patch_center: Tensor,
                          coord_max: Tensor,
                          feats: Tensor,
                          use_normalized_coord: bool = False) -> Tensor:
        """Generating model input.

        Generate input by subtracting patch center and adding additional
        features. Currently support colors and normalized xyz as features.

        Args:
            coords (Tensor): Sampled 3D point coordinate of shape [S, 3].
            patch_center (Tensor): Center coordinate of the patch.
            coord_max (Tensor): Max coordinate of all 3D points.
            feats (Tensor): Features of sampled points of shape [S, C].
            use_normalized_coord (bool): Whether to use normalized xyz as
                additional features. Defaults to False.

        Returns:
            Tensor: The generated input data of shape [S, 3+C'].
        """
        # subtract patch center, the z dimension is not centered
        centered_coords = coords.clone()
        centered_coords[:, 0] -= patch_center[0]
        centered_coords[:, 1] -= patch_center[1]

        # normalized coordinates as extra features
        if use_normalized_coord:
            normalized_coord = coords / coord_max
            feats = torch.cat([feats, normalized_coord], dim=1)

        points = torch.cat([centered_coords, feats], dim=1)

        return points

    def whole_inference(self, points: Tensor, batch_input_metas: List[dict],
                        rescale: bool) -> Tensor:
        """Inference with full scene (one forward pass without sliding)."""
        cls_logic = self.encode_classify(points, batch_input_metas)
        return cls_logic

    def inference(self, points: Tensor, batch_input_metas: List[dict],
                  rescale: bool) -> Tensor:
        """Inference with slide/whole style.

        Args:
            points (Tensor): Input points of shape [B, N, 3+C].
            batch_input_metas (List[dict]): Meta information of a batch of
                samples.
            rescale (bool): Whether transform to original number of points.
                Will be used for voxelization based segmentors.

        Returns:
            Tensor: The output segmentation map.
        """
        assert self.test_cfg.mode in ['whole']

        cls_logic = self.whole_inference(points, batch_input_metas,
                                             rescale)
        return cls_logic

    def predict(self,
                batch_inputs_dict: dict,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Simple test with single scene.

        Args:
            batch_inputs_dict (dict): Input sample dict which includes 'points'
            batch_data_samples (List[:obj:`Cls3DDataSample`]): The det3d data and metadata
            rescale (bool): Not needed

        """
        
        cls_list = []
        batch_input_metas = []
        for data_sample in batch_data_samples:
            batch_input_metas.append(data_sample.metainfo)

        points = batch_inputs_dict['points']
        # for point, input_meta in zip(points, batch_input_metas):
        #     inf = self.inference(
        #         point.unsqueeze(0), [input_meta], rescale)[0]
        #     cls_list.append(inf)
        
        for _points in points:
            if isinstance(_points, tuple):
                _points = _points[0]

            # check if torch tensor, if not, convert to torch tensor
            if not isinstance(_points, torch.Tensor):
                _points = torch.tensor(_points)

            # if cuda is available, move to cuda
            if torch.cuda.is_available() and not _points.is_cuda:
                _points = _points.cuda()
                
            if self.num_pts_sample is not None:
                if _points.shape[0] < self.num_pts_sample:
                    # Less than the minimally required number of points
                    # Skip this sample and append a tensor with high negative numbers
                    cls_list.append(torch.tensor([-1e6] * self.num_classes))
                    continue

            inf = self.inference(
                _points.unsqueeze(0), [], rescale)[0]
            cls_list.append(inf)

        return cls_list

    def _forward(self,
                 batch_inputs_dict: dict,
                 batch_data_samples: OptSampleList = None) -> Tensor:
        """Network forward process.

        Args:
            batch_inputs_dict (dict): Input sample dict which includes 'points'
                and 'imgs' keys.

                - points (List[Tensor]): Point cloud of each sample.
                - imgs (Tensor, optional): Image tensor has shape (B, C, H, W).
            batch_data_samples (List[:obj:`Det3DDataSample`]): The det3d data
                samples. It usually includes information such as `metainfo` and
                `gt_pts_seg`.

        Returns:
            Tensor: Forward output of model without any post-processes.
        """
        points = torch.stack(batch_inputs_dict['points'])
        x = self.extract_feat(points)
        return self.cls_head.forward(x)
