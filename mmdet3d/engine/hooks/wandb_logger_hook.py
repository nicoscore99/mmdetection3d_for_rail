import os
import os.path as osp
import warnings
import yaml
from pathlib import Path
import wandb
from typing import Dict, Optional, Union
from collections import OrderedDict
import numpy as np
import torch
from collections import OrderedDict
# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.dataset import BaseDataset
from mmengine.hooks import Hook, LoggerHook
from mmengine.model import is_model_wrapper
from mmengine.runner import Runner
from mmengine.visualization import WandbVisBackend

from mmdet3d.datasets.transforms import ObjectSample
from mmdet3d.registry import HOOKS

DATA_BATCH = Optional[Union[dict, tuple, list]]

@HOOKS.register_module()
class WandbLoggerHook(LoggerHook):
    """ Hook to log metrics with Weights & Biases.

    This hook was inspired to a major part by the MMCV WandbLoggerHook.
    See here: https://mmcv.readthedocs.io/en/master/_modules/mmcv/runner/hooks/logger/wandb.html

    """

    def __init__(self,
                 save_dir: str = None,
                 init_kwargs: dict = None,
                 yaml_config_path: str = 'wandb_auth.yaml',
                 metric_cfg: Optional[Dict[str, Union[str, Dict]]] = None,
                 commit: Optional[str] = None,
                 watch_kwargs: Optional[Dict] = None,
                 log_artifact: bool = False):
        super().__init__()

        self.curve_visualization_wandb_logging_map = {
            'roc': self.log_roc_curve,
            'prec': self.log_prec_curve,
            'cm': self.log_cm_curve
        }

        self._save_dir = save_dir
        self._init_kwargs = init_kwargs
        self._yaml_config_path = yaml_config_path
        self._define_metric_cfg = metric_cfg
        self._commit = commit
        self._watch_kwargs = watch_kwargs
        self._log_artifact = log_artifact

        # Code from WandVisBackend
        if not osp.exists(self._save_dir):
            os.makedirs(self._save_dir, exist_ok=True)  # type: ignore
        if self._init_kwargs is None:
            self._init_kwargs = {'dir': self._save_dir}
        else:
            self._init_kwargs.setdefault('dir', self._save_dir)
        try:
            import wandb
        except ImportError:
            raise ImportError(
                'Please run "pip install wandb" to install wandb')

        # resolve the path to the yaml config file
        if not osp.isabs(self._yaml_config_path ):
            self._yaml_config_path  = osp.join(osp.dirname(__file__), self._yaml_config_path )
        with open(self._yaml_config_path , 'r') as file:
            self.wandb_auth_config = yaml.load(file, Loader=yaml.FullLoader)

        self._wandb = wandb

        # Wandb login
        key_configured = self._wandb.login(key=self.wandb_auth_config['api_key'])        
        assert key_configured, '[WandbLoggerHook] wandb api key not configured'

        # Check if wandb is already initialized

        # Initialize wandb with kwards and config
        self.run = self._wandb.init(**self._init_kwargs)
        if self._define_metric_cfg is not None:
            if isinstance(self._define_metric_cfg, dict):
                for metric, summary in self._define_metric_cfg.items():
                    wandb.define_metric(metric, summary=summary)
            elif isinstance(self._define_metric_cfg, list):
                for metric_cfg in self._define_metric_cfg:
                    wandb.define_metric(**metric_cfg)
            else:
                raise ValueError('define_metric_cfg should be dict or list')

    def wandb_login(self):
        key_configure = self.wandb.login(key=self.wandb_auth_config['api_key'], relogin=True, force=True)
        assert key_configure, '[WandbLoggerHook] wandb api key not configured'

    def before_run(self, runner: Runner) -> None:

        # Check that the runner exists
        assert runner is not None, '[WandbLoggerHook] runner must be provided'
        assert isinstance(runner, Runner), f'[WandbLoggerHook] runner must be an instance of Runner, got {type(runner)}'

        self._wandb.watch(runner.model, self._watch_kwargs)
    
    def after_train_iter(self,
                        runner,
                        batch_idx: int,
                        data_batch: DATA_BATCH = None,
                        outputs: Optional[dict] = None) -> None:

        log_output = {'train': outputs}
        if outputs is not None:
            self._wandb.log(log_output, step=runner.iter, commit=self._commit)

    def after_val_iter(self,
                       runner,
                       batch_idx: int,
                       data_batch: DATA_BATCH = None,
                       outputs: Optional[dict] = None) -> None:
        
        log_output = {'val': outputs}
        if outputs is not None:
            self._wandb.log(log_output, step=runner.iter, commit=self._commit)

    def after_val_epoch(self, 
                        runner: Runner, 
                        metrics: Optional[Dict[str, float]] = None) -> None:

        if metrics['General 3D Det metric mmlab/evaluations'].keys():
            self._wandb.log(metrics[''], step=runner.iter, commit=self._commit)

        curves = metrics['General 3D Det metric mmlab/curves']
        if curves.keys():
            for curve_key in curves.keys():
                for level in curves[curve_key].keys():

                    self.curve_visualization_wandb_logging_map[curve_key](runner = runner,
                                                                      curve = curves[curve_key][level],
                                                                      level = level)
                    
    def after_test_epoch(self,
                         runner: Runner,
                         metrics: Optional[Dict[str, float]] = None) -> None:

        self.after_val_epoch(runner, metrics)

    def get_wandb_file(self, path: str) -> str:

        # ensure that the path exists
        assert osp.exists(path), f'Path {path} does not exist'

        all_files = os.listdir(path)
        wandb_file = [f for f in all_files if f.endswith('.wandb')]
        assert len(wandb_file) == 1, f'Expected 1 wandb file, got {len(wandb_file)}'
        wandb_file_abspath = osp.join(path, wandb_file[0])
        return wandb_file_abspath

    def after_run(self, runner: Runner) -> None:
        if self._log_artifact:
            wandb_artifact = self._wandb.Artifact(name='artifacts', type='model')
            latest_run_directory = osp.join(self._save_dir, 'wandb', 'latest-run')
            wandb_artifact.add_file(self.get_wandb_file(latest_run_directory))
            self.run.log_artifact(wandb_artifact)

        # Finish the run and close the wandb logger
        self._wandb.finish()

    def log_roc_curve(self, runner: Runner, curve: dict, level: str) -> None:
        
        try:
            # Check if there is a key which is not called 'all_classes' in curve.keys()
            if any([key for key in curve.keys() if key != 'all_classes']):
                raise ValueError('The case to plot different class dependent curves was not yet implemented here.')
            
            curve = curve['all_classes']
            data = np.column_stack((np.array(curve['fpr']), np.array(curve['tpr'])))
            table = wandb.Table(data=data, columns=["FPR", "TPR"])
            self._wandb.log({"ROC Curve": wandb.plot.line(table, "FPR", "TPR")})

        except Exception as e:
            print(f"Error while logging ROC curve: {e}")


    def log_prec_curve(self, runner: Runner, curve: dict, level: str) -> None:
        
        try:
            # Check if there is a key which is not called 'all_classes' in curve.keys()
            if any([key for key in curve.keys() if key != 'all_classes']):
                raise ValueError('The case to plot different class dependent curves was not yet implemented here.')
            
            curve = curve['all_classes']
            data = np.column_stack((np.array(curve['recall']), np.array(curve['precision'])))
            table = wandb.Table(data=data, columns=["Recall", "Precision"])
            self._wandb.log({"Precision-Recall Curve": wandb.plot.line(table, "Recall", "Precision")})

        except Exception as e:
            print(f"Error while logging precision-recall curve: {e}")

    def idx_to_lables(self, incices: np.array, labels: np.array) -> np.array:
        return np.array([labels[idx] for idx in incices])

    def log_cm_curve(self, runner: Runner, curve: dict, level: str) -> None:

        try:     
            _true = self.idx_to_lables(curve['y_true'], curve['labels'])
            _pred = self.idx_to_lables(curve['y_pred'], curve['labels'])

            self._wandb.log({'cm': wandb.sklearn.plot_confusion_matrix(y_true=_true, 
                                                                    y_pred=_pred,
                                                                    labels=curve['classes'])},
                            step=runner.iter)
            
        except Exception as e:
            print(f"Error while logging confusion matrix: {e}")

