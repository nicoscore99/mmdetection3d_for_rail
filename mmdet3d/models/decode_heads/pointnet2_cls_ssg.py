import torch.nn as nn
import torch.nn.functional as F
from .pointnet2_utils import PointNetSetAbstraction
from typing import List, Sequence, Tuple

# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
from typing import Dict, List

from mmcv.cnn.bricks import ConvModule
from torch import Tensor
from torch import nn as nn
import torch

from mmdet3d.models.layers import PointFPModule
from mmdet3d.registry import MODELS
from mmdet3d.utils.typing_utils import ConfigType
from mmengine.model import BaseModule, normal_init

from mmdet3d.registry import MODELS
from mmdet3d.structures.det3d_data_sample import SampleList
from mmdet3d.utils.typing_utils import ConfigType, OptMultiConfig


from mmdet3d.utils.typing_utils import ConfigType, OptMultiConfig
    
@MODELS.register_module()
class PointNet2ClsHead(BaseModule, metaclass=ABCMeta):
    r"""PointNet2 Classification head.
    
    Args:
        num_classes (int): Number of classes.
        in_channels (int): Number of input channels.
        dropout_ratio (float): Ratio of dropout layer. Defaults to 0.5.
        init_cfg (dict or :obj:`ConfigDict` or list[dict or :obj:`ConfigDict`],
            optional): Initialization config dict. Defaults to None.
    """
    
    def __init__(self,
                 num_classes: int,
                 lin_layers = ((1024, 512), (512, 256)), 
                 dropout_ratio: float = 0.5,
                 init_cfg: OptMultiConfig = None):
        super(PointNet2ClsHead, self).__init__(init_cfg=init_cfg)
        
        self.num_classes = num_classes
        self.lin_layers = lin_layers
        
        self.cls_head = nn.Sequential()
        
        for i, (in_channels, out_channels) in enumerate(self.lin_layers):
            self.cls_head.add_module(
                f'fc{i + 1}',
                nn.Linear(in_channels, out_channels)
            )
            self.cls_head.add_module(
                f'bn{i + 1}',
                nn.BatchNorm1d(out_channels)
            )
            self.cls_head.add_module(
                f'dropout{i + 1}',
                nn.Dropout(p=dropout_ratio)
            )
            
        self.cls_head.add_module(
            'fc_final',
            nn.Linear(out_channels, self.num_classes)
        )
        
        self.cls_head.add_module(
            'log_softmax',
            nn.LogSoftmax(dim=-1)
        )
        
    def forward(self, feat_dict: dict) -> Tensor:
        
        feature_vector = feat_dict['sa_features']
        
        # from [B, 1024, 1] to [B, 1024]
        if feature_vector.shape[-1] == 1:
            x = feature_vector.squeeze(-1)
        else:
            x = feature_vector

        for layer in self.cls_head:
            x = layer(x)

        return x
    
    def predict(self, inputs: dict, batch_input_metas: List[dict],
                test_cfg: ConfigType) -> Tensor:
        """Forward function for testing.

        Args:
            inputs (dict): Feature dict from backbone.
            batch_input_metas (List[dict]): Meta information of a batch of
                samples.
            test_cfg (dict or :obj:`ConfigDict`): The testing config.
            
        Returns:
            Tensor: Prediction results.
        """
        
        return self.forward(inputs)
    
    
    def loss(self, inputs: dict, batch_data_samples: SampleList,
             train_cfg: ConfigType) -> Dict[str, Tensor]:
        """Calculate the loss.
        
        Args:
            inputs (dict): Output of the model.
            batch_data_samples (SampleList): Data samples.
            train_cfg (dict): Training config.
            
        Returns:
            dict: Loss dict.
        """
        
        _prediction = self.forward(inputs)
        
        _targets = [sample.ann_info['class_idx'] for sample in batch_data_samples]
        
        # turn list into vertical tensor
        _targets = torch.tensor(_targets).to(_prediction.device)
        
        total_loss = F.nll_loss(_prediction, _targets)
        
        return dict(loss_cls=total_loss)