# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import warnings
from typing import Dict, Optional, Union
# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.dataset import BaseDataset
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmengine.runner import Runner

from mmdet3d.datasets.transforms import ObjectSample
from mmdet3d.registry import HOOKS

@HOOKS.register_module()
class WandbLoggerHook(Hook):
    """ Hook to log metrics with Weights & Biases.

    This hook was inspired to a major part by the MMCV WandbLoggerHook.
    See here: https://mmcv.readthedocs.io/en/master/_modules/mmcv/runner/hooks/logger/wandb.html

    
    """

    def __init__(self) -> None:
        self.import_wandb()

    def import_wandb(self):
        try:
            import wandb
        except ImportError:
            wandb = None
            warnings.warn('wandb is not installed')
        self.wandb = wandb

    def before_run(self, runner: Runner):
        super().before_run(runner)
        if self.wandb is None:
            self.import_wandb()
        if self.init_kwargs:
            self.wandb.init(**self.init_kwargs)  # type: ignore
        else:
            self.wandb.init()  # type: ignore
        


        raise NotImplementedError
    
    def before_train_epoch(self, runner: Runner):
        raise NotImplementedError
    
    def after_train_epoch(self, runner: Runner):
        raise NotImplementedError
    
    def after_val_epoch(self, runner: Runner):
        raise NotImplementedError
    
    def is_last_train_epoch(self, runner) -> bool:
        return NotImplementedError
    

    


