import os.path as osp
from typing import Dict, List, Optional, Sequence, Union

import mmengine
import numpy as np
import torch
from mmengine.dataset import Compose
from mmengine.fileio import (get_file_backend, isdir, join_path,
                             list_dir_or_file)
from mmengine.infer.infer import ModelType
from mmengine.structures import InstanceData

from mmdet3d.registry import INFERENCERS
from mmdet3d.structures import (CameraInstance3DBoxes, DepthInstance3DBoxes,
                                Det3DDataSample, LiDARInstance3DBoxes)
from mmdet3d.utils import ConfigType
from .base_3d_inferencer import Base3DInferencer

InstanceList = List[InstanceData]
InputType = Union[str, np.ndarray]
InputsType = Union[InputType, Sequence[InputType]]
PredType = Union[InstanceData, InstanceList]
ImgType = Union[np.ndarray, Sequence[np.ndarray]]
ResType = Union[Dict, List[Dict], InstanceData, List[InstanceData]]


@INFERENCERS.register_module(name='cls-lidar')
@INFERENCERS.register_module()
class LidarDet3DInferencer(Base3DInferencer):
    """The inferencer of pointcloud classification
    
        Args:
        model (str, optional): Path to the config file or the model name
            defined in metafile. For example, it could be
            "pointpillars_kitti-3class" or
            "configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py". # noqa: E501
            If model is not specified, user must provide the
            `weights` saved by MMEngine which contains the config string.
            Defaults to None.
        weights (str, optional): Path to the checkpoint. If it is not specified
            and model is a model name of metafile, the weights will be loaded
            from metafile. Defaults to None.
        device (str, optional): Device to run inference. If None, the available
            device will be automatically used. Defaults to None.
        scope (str): The scope of the model. Defaults to 'mmdet3d'.
        palette (str): Color palette used for visualization. The order of
            priority is palette -> config -> checkpoint. Defaults to 'none'.
    """
    
    def __init__(self,
                model: Union[ModelType, str, None] = None,
                weights: Optional[str] = None,
                device: Optional[str] = None,
                which_pipeline: Optional[str] = None,
                scope: str = 'mmdet3d',
                palette: str = 'none') -> None:
        # A global counter tracking the number of frames processed, for
        # naming of the output results
        self.num_visualized_frames = 0
        self.which_pipeline = which_pipeline
        super(LidarDet3DInferencer, self).__init__(
            model=model,
            weights=weights,
            device=device,
            scope=scope,
            palette=palette)
        
    def _init_pipeline(self, cfg: ConfigType) -> Compose:
        """Initialize the test pipeline."""
        pipeline_cfg = cfg.test_dataloader.dataset.pipeline

        if self.which_pipeline == 'LoadPointsFromFile' or self.which_pipeline is None:
            load_point_idx = self._get_transform_idx(pipeline_cfg,
                                                    'LoadPointsFromFile')
        else:
            load_point_idx = self._get_transform_idx(pipeline_cfg,
                                                    self.which_pipeline)

        if load_point_idx == -1:
            raise ValueError(
                'Loadingfile {} is not found in the test pipeline'.format(self.which_pipeline)
            )

        load_cfg = pipeline_cfg[load_point_idx]
        self.coord_type, self.load_dim = load_cfg['coord_type'], load_cfg[
            'load_dim']
        self.use_dim = list(range(load_cfg['use_dim'])) if isinstance(
            load_cfg['use_dim'], int) else load_cfg['use_dim']

        pipeline_cfg[load_point_idx]['type'] = 'LidarClsInferencer'
        return Compose(pipeline_cfg)
    
    def visualize(self,
                inputs: InputsType,
                preds: PredType,
                return_vis: bool = False,
                show: bool = False,
                wait_time: int = -1,
                draw_pred: bool = True,
                pred_score_thr: float = 0.3,
                no_save_vis: bool = False,
                img_out_dir: str = '') -> Union[List[np.ndarray], None]:
        
        raise NotImplementedError('Visualize is not implemented in LidarDet3DInferencer')
    
    
    def visualize_preds_fromfile(self, inputs: InputsType, preds: PredType,
                                 **kwargs) -> Union[List[np.ndarray], None]:
        
        raise NotImplementedError('Visualize_preds_fromfile is not implemented in LidarDet3DInferencer')