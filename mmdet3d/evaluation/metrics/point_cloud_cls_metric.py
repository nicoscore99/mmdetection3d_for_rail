# Copyright (c) OpenMMLab. All rights reserved.
import tempfile
from os import path as osp
from typing import Dict, List, Optional, Sequence, Tuple, Union

import mmengine
import numpy as np
import torch
from mmengine import load
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger, print_log

from mmdet3d.evaluation import kitti_eval
from mmdet3d.registry import METRICS
from mmdet3d.structures import (Box3DMode, CameraInstance3DBoxes,
                                LiDARInstance3DBoxes, points_cam2img)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


@METRICS.register_module()
class PointCloudClsMetric(BaseMetric):
    
    def __init__(self, 
                 class_names: List[str],
                 save_graphics: bool = False,
                 save_evaluation_resutls: bool = False):
        
        super().__init__()
        
        self.class_names = class_names
        self.save_graphics = save_graphics
        self.save_evaluation_resutls = save_evaluation_resutls
        
        self.ground_truth = []
        self.results = []
    
    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> dict:
        """Process the results to get the evaluation results.
        
        Args:
            results (list): The results to be evaluated.
        
        Returns:
            dict: The evaluation results.
        """
        
        self.ground_truth += data_batch['data_samples']
        self.results += data_samples
        
    def ypred(self, results: List) -> np.ndarray:
        ypred = []
        for result in results:
            # single results is tensor like [-0.0711, -4.3492, -2.8881] form log softmax
            ypred.append(np.argmax(result))
        return np.array(ypred)
    
    def ytrue(self, ground_truth_list: List) -> np.ndarray:
        ytrue = []
        for sample in ground_truth_list:
            ytrue.append(sample.ann_info['class_idx'])
        return np.array(ytrue)
    
    def compute_metrics(self, results: List) -> Dict:
        
        y_pred = self.ypred(results)
        y_true = self.ytrue(self.ground_truth)
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        f1 = f1_score(y_true, y_pred, average='macro')
        cm = confusion_matrix(y_true, y_pred)
        
        # Reset the ground truth and results
        self.ground_truth = []
        self.results = []
        
        return {'evaluations': {'accuracy': accuracy,
                                 'precision': precision,
                                 'recall': recall,
                                 'f1': f1},
                'curves': {'confusion_matrix': cm}}
    

    
    
    