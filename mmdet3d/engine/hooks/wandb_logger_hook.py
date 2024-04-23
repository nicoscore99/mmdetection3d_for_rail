# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import warnings
import yaml
from pathlib import Path
import wandb
from typing import Dict, Optional, Union
from collections import OrderedDict
import numpy as np
import torch
# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.dataset import BaseDataset
from mmengine.hooks import Hook, LoggerHook
from mmengine.model import is_model_wrapper
from mmengine.runner import Runner
from mmengine.visualization import WandbVisBackend

from mmdet3d.datasets.transforms import ObjectSample
from mmdet3d.registry import HOOKS

@HOOKS.register_module()
class WandbLoggerHook(Hook):
    """ Hook to log metrics with Weights & Biases.

    This hook was inspired to a major part by the MMCV WandbLoggerHook.
    See here: https://mmcv.readthedocs.io/en/master/_modules/mmcv/runner/hooks/logger/wandb.html

    """

    def __init__(self,
                 yaml_config_path: str = 'wandb_auth.yaml'):

        # resolve the path to the yaml config file
        if not osp.isabs(yaml_config_path):
            yaml_config_path = osp.join(osp.dirname(__file__), yaml_config_path)
        with open(yaml_config_path, 'r') as file:
            self.wandb_auth_config = yaml.load(file, Loader=yaml.FullLoader)

        # Initialize wandb
        self.import_wandb()
        # Login to wandb
        self.wandb_login()


    def import_wandb(self):
        try:
            import wandb
        except ImportError:
            wandb = None
            warnings.warn('wandb is not installed')
        self.wandb = wandb

    def wandb_login(self):
        key_configured = self.wandb.login(key=self.wandb_auth_config['api_key'])        
        assert key_configured, '[WandbLoggerHook] wandb api key not configured'

    def retry_wandb_login(self):
        key_configure = self.wandb.login(key=self.wandb_auth_config['api_key'], relogin=True, force=True)
        assert key_configure, '[WandbLoggerHook] wandb api key not configured'

    def before_run(self, runner: Runner):
        super().before_run(runner)

        # Check that the runner exists
        assert runner is not None, '[WandbLoggerHook] runner must be provided'
        assert isinstance(runner, Runner), f'[WandbLoggerHook] runner must be an instance of Runner, got {type(runner)}'

        wandb.init(
            project='pc_obj_det_learning',
            name= runner.logger.log_file.split('/')[-1],
            config=runner.cfg,
        )
    
    def after_train_epoch(self, runner: Runner):
        tag, log_str = runner.log_processor.get_log_after_epoch(
            runner, len(runner.val_dataloader), 'val')
        json_friendly_tags = self._process_tags(log_str)
        wandb.log(json_friendly_tags, step=runner.epoch)
    
    def after_val_epoch(self, runner: Runner):
        tag, log_str = runner.log_processor.get_log_after_epoch(
            runner, len(runner.val_dataloader), 'val')
        json_friendly_tags = self._process_tags(log_str)
        wandb.log(json_friendly_tags, step=runner.epoch)
    
    def is_last_train_epoch(self, runner) -> bool:
        return NotImplementedError
    
    def _process_tags(tags: dict):
        """Convert tag values to json-friendly type."""

        def process_val(value):
            if isinstance(value, (list, tuple)):
                # Array type of json
                return [process_val(item) for item in value]
            elif isinstance(value, dict):
                # Object type of json
                return {k: process_val(v) for k, v in value.items()}
            elif isinstance(value, (str, int, float, bool)) or value is None:
                # Other supported type of json
                return value
            elif isinstance(value, (torch.Tensor, np.ndarray)):
                return value.tolist()
            # Drop unsupported values.

        processed_tags = OrderedDict(process_val(tags))

        return processed_tags

    


