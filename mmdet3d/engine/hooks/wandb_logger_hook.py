import os
import os.path as osp
import warnings
import yaml
from pathlib import Path
import wandb
from typing import Dict, Optional, Union
from collections import OrderedDict
import numpy as np
# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.dataset import BaseDataset
from mmengine.hooks import Hook, LoggerHook
from mmengine.model import is_model_wrapper
from mmengine.runner import Runner

from mmdet3d.datasets.transforms import ObjectSample
from mmdet3d.registry import HOOKS

DATA_BATCH = Optional[Union[dict, tuple, list]]

def flatten_dict(d, parent_key='', sep='/'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

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
                 log_artifact: bool = False,
                 log_model_every_n_epochs: int = 1):
        super().__init__()

        self.curve_visualization_wandb_logging_map = {
            'roc': self.log_roc_curve,
            'prec': self.log_prec_curve
        }

        self.train_outputs_total = None
        self.val_outputs_total = None

        self._save_dir = save_dir
        self._init_kwargs = init_kwargs
        self._yaml_config_path = yaml_config_path
        self._commit = commit
        self._watch_kwargs = watch_kwargs
        self._log_artifact = log_artifact
        self.log_model_every_n_epochs = log_model_every_n_epochs

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
        if not self._wandb.login():
            key_configured = self._wandb.login(key=self.wandb_auth_config['api_key'])        
            assert key_configured, '[WandbLoggerHook] wandb api key not configured'

        # # Initialize wandb with kwards and config
        # self.run = self._wandb.init(**self._init_kwargs)

        # Define the metrics
        self._define_metric_cfg = {
            'train_iter': 'train_iter',     # iter
            'train_epoch': 'train_epoch',   # epoch
            'val_iter': 'val_iter',         # iter
            'val_epoch': 'val_epoch',       # epoch
            'val_metrics': 'val_epoch'      # epoch
        }

        # Define the metrics
        if metric_cfg is not None:
            self._define_metric_cfg = metric_cfg


    def wandb_login(self):
        key_configure = self.wandb.login(key=self.wandb_auth_config['api_key'], relogin=True, force=True)
        assert key_configure, '[WandbLoggerHook] wandb api key not configured'

    def before_run(self, runner: Runner) -> None:
        
        # Initialize wandb with kwards and config
        
        runner_cfg = runner.cfg._to_lazy_dict()  
        self._init_kwargs['config'] = runner_cfg
                
        self.run = self._wandb.init(**self._init_kwargs)

        for values in set(self._define_metric_cfg.values()):
            self._wandb.define_metric(values)

        for key, value in self._define_metric_cfg.items():

            key_all_subsubkeys = key + '/*'
            self._wandb.define_metric(key_all_subsubkeys, step_metric=value)

        # Check that the runner exists
        assert runner is not None, '[WandbLoggerHook] runner must be provided'
        assert isinstance(runner, Runner), f'[WandbLoggerHook] runner must be an instance of Runner, got {type(runner)}'

        self._wandb.watch(runner.model, self._watch_kwargs)
        
    
    def after_train_iter(self,
                        runner,
                        batch_idx: int,
                        data_batch: DATA_BATCH = None,
                        outputs: Optional[dict] = None) -> None:
        
        outputs = {'train_iter': outputs}
        outputs = flatten_dict(outputs)       
        if outputs is not None:
            # self._wandb.log(outputs, step=runner.iter, commit=self._commit)

            for key, value in outputs.items():
                self._wandb.log({key: value, 'train_iter': runner.iter}, commit=self._commit)
                
        lr_dict = runner.optim_wrapper.get_lr()
        if lr_dict is not None:
            self._wandb.log({'lr': lr_dict['lr'][0], 'train_iter': runner.iter}, commit=self._commit)
            
    def after_train_epoch(self,
                          runner: Runner,
                          metrics: Optional[Dict[str, float]] = None) -> None:

        outputs = {'train_epoch': metrics}
        outputs = flatten_dict(outputs)
        if metrics is not None:
            # self._wandb.log(outputs, step=runner.epoch, commit=self._commit)

            for key, value in outputs.items():
                self._wandb.log({key: value, 'train_epoch': runner.epoch}, commit=self._commit)

    def after_val_iter(self,
                       runner,
                       batch_idx: int,
                       data_batch: DATA_BATCH = None,
                       outputs: Optional[dict] = None) -> None:
        
        outputs = {'val_iter': outputs}
        outputs = flatten_dict(outputs)
        if outputs is not None:
            # self._wandb.log(outputs, step=runner.iter, commit=self._commit)

            for key, value in outputs.items():
                self._wandb.log({key: value, 'val_iter': runner.iter}, commit=self._commit)

    def after_val_epoch(self, 
                        runner: Runner, 
                        metrics: Optional[Dict[str, float]] = None) -> None:
        
        print(f'Logging metrics: {metrics}')
        
        if 'log_vars' in metrics:
            if metrics['log_vars'].keys():
                outputs = {'val_epoch': metrics['log_vars']}
                outputs = flatten_dict(outputs)
                # self._wandb.log(outputs, step=runner.epoch, commit=self._commit)

                for key, value in outputs.items():
                    self._wandb.log({key: value, 'val_epoch': runner.epoch}, commit=self._commit)

        if 'General 3D Det metric mmlab/evaluations' in metrics:
            if metrics['General 3D Det metric mmlab/evaluations'].keys():
                outputs = {'val_metrics': metrics['General 3D Det metric mmlab/evaluations']}
                # self._wandb.log(outputs, step=runner.epoch, commit=self._commit)

                for key, value in outputs.items():
                    self._wandb.log({key: value, 'val_epoch': runner.epoch}, commit=self._commit)

        if 'General 3D Det metric mmlab/curves' in metrics:
            curves = metrics['General 3D Det metric mmlab/curves']
            if curves.keys():
                for curve_key in curves.keys():
                    for level in curves[curve_key].keys():
                        self.curve_visualization_wandb_logging_map[curve_key](runner = runner,
                                                                        curve = curves[curve_key])
                    
    def after_test_epoch(self,
                         runner: Runner,
                         metrics: Optional[Dict[str, float]] = None) -> None:

        self.after_val_epoch(runner, metrics)

    def get_wandb_file(self, path: str) -> str:

        # ensure that the path exists
        assert osp.exists(path), f'Path {path} does not exist'

        all_files = os.listdir(path)
        
        print(f'Path: {path}')
        
        checkpoint_files = [f for f in all_files if f.endswith('.pth') or f.endswith('.pt')]
        
        print(f'Found {len(checkpoint_files)} checkpoint files in the directory')
        
        # If there are multiple, take the one with the largest epoch number
        if len(checkpoint_files) > 1:
            checkpoint_files = sorted(checkpoint_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
            wandb_file = checkpoint_files[-1]
        elif len(checkpoint_files) == 0:
            raise ValueError('No checkpoint files found in the directory')            
        else:
            wandb_file = checkpoint_files[0]
        
        wandb_file_abspath = osp.join(path, wandb_file)
        return wandb_file_abspath

    def after_run(self, runner: Runner) -> None:
        try:
            if self._log_artifact:
                wandb_artifact = self._wandb.Artifact(name='artifacts', type='model')
                wandb_artifact.add_file(self.get_wandb_file(self._save_dir))
                self.run.log_artifact(wandb_artifact)
        except Exception as e:
            print(f"Error while logging artifact: {e}")
        # Finish the run and close the wandb logger
        self._wandb.finish()
        
        print("Done. WandB Logging is finished.")

    def log_roc_curve(self, runner: Runner, curve: dict) -> None:
        
        try:
            data = np.column_stack((np.array(curve['fpr']), np.array(curve['tpr'])))
            table = wandb.Table(data=data, columns=["FPR", "TPR"])
            self._wandb.log({"ROC Curve": wandb.plot.line(table, "FPR", "TPR")})

        except Exception as e:
            print(f"Error while logging ROC curve: {e}")


    def log_prec_curve(self, runner: Runner, curve: dict) -> None:
        
        try:
            data = np.column_stack((np.array(curve['recall']), np.array(curve['precision'])))
            table = wandb.Table(data=data, columns=["Recall", "Precision"])
            self._wandb.log({"Precision-Recall Curve": wandb.plot.line(table, "Recall", "Precision")})

        except Exception as e:
            print(f"Error while logging precision-recall curve: {e}")

    def idx_to_lables(self, incices: np.array, labels: np.array) -> np.array:
        return np.array([labels[idx] for idx in incices])

