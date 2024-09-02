import os.path as osp
from typing import Dict, List, Optional, Sequence, Union, Tuple

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
from mmengine.infer.infer import BaseInferencer, ModelType

InstanceList = List[InstanceData]
InputType = Union[str, np.ndarray]
InputsType = Union[InputType, Sequence[InputType]]
PredType = Union[InstanceData, InstanceList]
ImgType = Union[np.ndarray, Sequence[np.ndarray]]
ResType = Union[Dict, List[Dict], InstanceData, List[InstanceData]]

import numpy as np
import torch.nn as nn
from mmengine import dump, print_log
from mmengine.infer.infer import BaseInferencer, ModelType
from mmengine.model.utils import revert_sync_batchnorm
from mmengine.registry import init_default_scope
from mmengine.runner import load_checkpoint
from mmengine.structures import InstanceData
from mmengine.visualization import Visualizer
from rich.progress import track

from mmdet3d.registry import DATASETS, MODELS
from mmdet3d.structures import Box3DMode, Det3DDataSample
from mmdet3d.utils import ConfigType

@INFERENCERS.register_module(name='cls-lidar')
@INFERENCERS.register_module()
class LidarClsInferencer(BaseInferencer):
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
                scope: str = 'mmdet3d',
                palette: str = 'none') -> None:
        # A global counter tracking the number of frames processed, for
        # naming of the output results
        self.num_visualized_frames = 0
        super(LidarClsInferencer, self).__init__(
            model=model,
            weights=weights,
            device=device,
            scope=scope)
    
    def _convert_syncbn(self, cfg: ConfigType):
        """Convert config's naiveSyncBN to BN.

        Args:
            config (str or :obj:`mmengine.Config`): Config file path
                or the config object.
        """
        if isinstance(cfg, dict):
            for item in cfg:
                if item == 'norm_cfg':
                    cfg[item]['type'] = cfg[item]['type']. \
                                        replace('naiveSyncBN', 'BN')
                else:
                    self._convert_syncbn(cfg[item])

    def _inputs_to_list(self, inputs: Union[dict, list], **kwargs) -> list:
        """Preprocess the inputs to a list.

        Preprocess inputs to a list according to its type:

        - list or tuple: return inputs
        - dict: the value with key 'points' is
            - Directory path: return all files in the directory
            - other cases: return a list containing the string. The string
              could be a path to file, a url or other types of string according
              to the task.

        Args:
            inputs (Union[dict, list]): Inputs for the inferencer.

        Returns:
            list: List of input for the :meth:`preprocess`.
        """

        if isinstance(inputs, dict) and isinstance(inputs['points'], str):
            pcd = inputs['points']
            backend = get_file_backend(pcd)
            if hasattr(backend, 'isdir') and isdir(pcd):
                # Backends like HttpsBackend do not implement `isdir`, so
                # only those backends that implement `isdir` could accept
                # the inputs as a directory
                filename_list = list_dir_or_file(pcd, list_dir=False)
                inputs = [{
                    'points': join_path(pcd, filename)
                } for filename in filename_list]

        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]

        return list(inputs)

    def _init_pipeline(self, cfg: ConfigType) -> Compose:
        return None
    
    def _init_collate(self, cfg: ConfigType) -> Compose:
        return None
    
    def _init_visualizer(self, cfg: ConfigType) -> Optional[Visualizer]:
        return None
    
    def _init_model(
        self,
        cfg: ConfigType,
        weights: str,
        device: str = 'cpu',) -> nn.Module:
        
        self._convert_syncbn(cfg.model)
        cfg.model.train_cfg = None
        model = MODELS.build(cfg.model)

        checkpoint = load_checkpoint(model, weights, map_location='cpu')
        if 'dataset_meta' in checkpoint.get('meta', {}):
            # mmdet3d 1.x
            model.dataset_meta = checkpoint['meta']['dataset_meta']
        elif 'CLASSES' in checkpoint.get('meta', {}):
            # < mmdet3d 1.x
            classes = checkpoint['meta']['CLASSES']
            model.dataset_meta = {'classes': classes}

            if 'PALETTE' in checkpoint.get('meta', {}):  # 3D Segmentor
                model.dataset_meta['palette'] = checkpoint['meta']['PALETTE']
        else:
            # < mmdet3d 1.x
            model.dataset_meta = {'classes': cfg.class_names}

            if 'PALETTE' in checkpoint.get('meta', {}):  # 3D Segmentor
                model.dataset_meta['palette'] = checkpoint['meta']['PALETTE']

        model.cfg = cfg  # save the config in the model for convenience
        model.to(device)
        model.eval()
        return model
    
    # def _get_transform_idx(self, pipeline_cfg: ConfigType, name: str) -> int:
    #     """Returns the index of the transform in a pipeline.

    #     If the transform is not found, returns -1.
    #     """
    #     for i, transform in enumerate(pipeline_cfg):
    #         if transform['type'] == name:
    #             return i
    #     return -1
    
    def __call__(self,
                 inputs: InputsType,
                 batch_size: int = 1,
                 return_datasamples: bool = False,
                 **kwargs) -> Optional[dict]:
        """Call the inferencer.

        Args:
            inputs (InputsTyori_inputspe): Inputs for the inferencer.
            batch_size (int): Batch size. Defaults to 1.
            return_datasamples (bool): Whether to return results as
                :obj:`BaseDataElement`. Defaults to False.
            **kwargs: Key words arguments passed to :meth:`preprocess`,
                :meth:`forward`, :meth:`visualize` and :meth:`postprocess`.
                Each key in kwargs should be in the corresponding set of
                ``preprocess_kwargs``, ``forward_kwargs``, ``visualize_kwargs``
                and ``postprocess_kwargs``.


        Returns:
            dict: Inference and visualization results.
        """

        (
            preprocess_kwargs,
            forward_kwargs,
            visualize_kwargs,
            postprocess_kwargs,
        ) = self._dispatch_kwargs(**kwargs)

        ori_inputs = self._inputs_to_list(inputs)
        inputs = self.preprocess(
            ori_inputs, batch_size=batch_size, **preprocess_kwargs)
        preds = []

        results_dict = {'predictions': []}
        for data in (track(inputs, description='Inference')
                     if self.show_progress else inputs):
            preds.extend(self.forward(data, **forward_kwargs))
            results = self.postprocess(preds, return_datasamples,
                                       **postprocess_kwargs)
            results_dict['predictions'].extend(results['predictions'])
        return results_dict
    
    def preprocess(self, inputs: InputsType, batch_size: int = 1, **kwargs):
        return inputs

    def postprocess(
        self,
        preds: PredType,
        visualization: Optional[List[np.ndarray]] = None,
        return_datasample: bool = False,
        print_result: bool = False,
        no_save_pred: bool = False,
        pred_out_dir: str = '',
    ) -> Union[ResType, Tuple[ResType, np.ndarray]]:
        """Process the predictions and visualization results from ``forward``
        and ``visualize``.

        This method should be responsible for the following tasks:

        1. Convert datasamples into a json-serializable dict if needed.
        2. Pack the predictions and visualization results and return them.
        3. Dump or log the predictions.

        Args:
            preds (List[Dict]): Predictions of the model.
            visualization (np.ndarray, optional): Visualized predictions.
                Defaults to None.
            return_datasample (bool): Whether to use Datasample to store
                inference results. If False, dict will be used.
                Defaults to False.
            print_result (bool): Whether to print the inference result w/o
                visualization to the console. Defaults to False.
            pred_out_dir (str): Directory to save the inference results w/o
                visualization. If left as empty, no file will be saved.
                Defaults to ''.

        Returns:
            dict: Inference and visualization results with key ``predictions``
            and ``visualization``.

            - ``visualization`` (Any): Returned by :meth:`visualize`.
            - ``predictions`` (dict or DataSample): Returned by
              :meth:`forward` and processed in :meth:`postprocess`.
              If ``return_datasample=False``, it usually should be a
              json-serializable dict containing only basic data elements such
              as strings and numbers.
        """
        if no_save_pred is True:
            pred_out_dir = ''

        result_dict = {}
        results = preds
        result_dict['predictions'] = results
        return result_dict
    
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