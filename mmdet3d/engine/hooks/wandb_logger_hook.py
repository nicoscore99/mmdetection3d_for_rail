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
        
        tag, log_str = runner.log_processor.get_log_after_iter(runner, batch_idx, 'train')

        if tag:
            self._wandb.log(tag, step=runner.iter, commit=self._commit)

    def after_train_epoch(self, runner: Runner):

        # Log the tags
        tags = runner.log_processor.get_log_after_epoch(runner, len(runner.train_dataloader), 'val')

        print("print tags after epoch: ", tags)
        print("tags type: ", type(tags))

    def after_val_epoch(self, 
                        runner: Runner, 
                        metrics: Optional[Dict[str, float]] = None) -> None:

        if metrics:
            self._wandb.log(metrics, step=runner.iter, commit=self._commit)

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

    


