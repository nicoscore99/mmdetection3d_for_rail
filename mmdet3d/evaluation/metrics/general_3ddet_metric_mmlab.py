# Copyright (c) OpenMMLab. All rights reserved.
import tempfile
import os
from os import path as osp
from typing import Dict, List, Optional, Sequence, Tuple, Union
from sklearn import metrics
from abc import ABC, abstractmethod
import json

import sklearn.metrics as sk_metrics
import matplotlib.pyplot as plt
import pprint
import bbox
import torch
import pandas as pd

import mmengine
import numpy as np
import copy
from mmengine import load
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger, print_log
from mmdet3d.registry import METRICS
from mmdet3d.structures.bbox_3d import LiDARInstance3DBoxes
from mmdet3d.models.task_modules.assigners.max_3d_iou_assigner import Max3DIoUAssigner
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mmengine.structures import InstanceData

from mmdet3d.structures import (Box3DMode, CameraInstance3DBoxes,
                                LiDARInstance3DBoxes, points_cam2img)

@METRICS.register_module()
class General_3dDet_Metric_MMLab(BaseMetric):
    """Kitti evaluation metric.

    Args:
        ann_file (str): Annotation file path.
        metric (str or List[str]): Metrics to be evaluated. Defaults to 'bbox'.
        pcd_limit_range (List[float]): The range of point cloud used to filter
            invalid predicted boxes. Defaults to [0, -40, -3, 70.4, 40, 0.0].
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix will
            be used instead. Defaults to None.
        pklfile_prefix (str, optional): The prefix of pkl files, including the
            file path and the prefix of filename, e.g., "a/b/prefix". If not
            specified, a temp file will be created. Defaults to None.
        default_cam_key (str): The default camera for lidar to camera
            conversion. By default, KITTI: 'CAM2', Waymo: 'CAM_FRONT'.
            Defaults to 'CAM2'.
        format_only (bool): Format the output results without perform
            evaluation. It is useful when you want to format the result to a
            specific format and submit it to the test server.
            Defaults to False.
        submission_prefix (str, optional): The prefix of submission data. If
            not specified, the submission data will not be generated.
            Defaults to None.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
    """

    def __init__(self,
                 ann_file: str,
                 metric: str = 'det3d',
                 pcd_limit_range: List[float] = [0, -39.68, -20, 69.12, 39.68, 20],
                #  pcd_limit_range: List[float] = [0, -40, -3, 70.4, 40, 1],
                 force_single_assignement: bool = False,
                 prefix: Optional[str] = None,
                 pklfile_prefix: Optional[str] = None,
                 default_cam_key: str = 'CAM2',
                 format_only: bool = False,
                 submission_prefix: Optional[str] = None,
                 collect_device: str = 'cpu',
                 save_graphics: Optional[bool] = False,
                 save_evaluation_results: Optional[bool] = False,
                 difficulty_levels: Optional[List[float]] = [0.01],
                 classes: Optional[List[str]] = None,
                 output_dir: Optional[str] = None,
                 evaluation_file_name: Optional[str] = 'evaluation_results.json',
                 backend_args: Optional[dict] = None) -> None:
    
        self.default_prefix = 'General 3D Det metric mmlab'
        super(General_3dDet_Metric_MMLab, self).__init__(collect_device=collect_device, prefix=prefix)
        self.pcd_limit_range = pcd_limit_range
        self.ann_file = ann_file
        self.pklfile_prefix = pklfile_prefix
        self.format_only = format_only
        self.submission_prefix = submission_prefix
        self.default_cam_key = default_cam_key
        self.backend_args = backend_args
        self.classes = classes
        self.output_dir = output_dir
        self.save_graphics = save_graphics
        self.force_single_assignement = force_single_assignement
        self.save_evaluation_results = save_evaluation_results
        self.evaluation_file_name = evaluation_file_name
        self.difficulty_levels = difficulty_levels
        self.metric = metric

        if self.format_only:
            assert submission_prefix is not None, 'submission_prefix must be '
            'not None when format_only is True, otherwise the result files '
            'will be saved to a temp directory which will be cleaned up at '
            'the end.'

        # '3d' and 'bev' stand for '3-dimensional object detction metrics' and 'bird's eye view object detection metrics'
        allowed_metrics = ['det3d', 'bev']
       
        # Check that difficulty levels are not empty
        if not self.difficulty_levels:
            raise ValueError('difficulty_levels should not be empty.')

        if self.metric not in allowed_metrics:
            raise KeyError("metric should be one of {allowed_metrics} but got {metric}.")
            
        # Ensure that the classes are not empty
        if not self.classes:
            raise ValueError('classes should not be empty.')
        
        # Ensure that the output directory exists
        if not self.output_dir:
            raise ValueError('output_dir should not be empty.')
        
        # Ensure that the output directory exists or create it
        if not osp.exists(self.output_dir):
            print("Output directory does not exist. Creating it now.")
            os.makedirs(self.output_dir)

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> dict:

        # Append the datasamples Sequence to the results list
        self.results += data_samples

    def compute_metrics(self, results: List[dict]) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (List[dict]): The processed results of the whole dataset.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()

        """
        Required format for the conversion:
        {'_sample_idx_': {
            'bbox_3d': tensor([[ 43.3839, -19.6041,  -2.1520,   0.9658,   0.6022,   2.0313,   2.8344],
                                [ 31.3530, -21.2475,  -2.3134,   0.8766,   0.8507,   1.9997,  -0.3569],
                                [ 95.6967,  -9.9219,  -1.9124,   1.7009,   0.6550,   1.7863,  -0.2158],
                                [ 11.2689,   7.4125,  -1.1116,   4.2148,   1.6975,   1.6820,   2.0275],
                                [ 35.8330, -21.2327,  -1.1420,   3.9669,   1.6993,   1.5309,   1.5040],
                                [ 33.7343, -15.4057,  -1.3091,   4.0293,   1.6570,   1.6490,   1.6808]]),
            'labels_3d': tensor([0, 0, 1, 2, 2, 2]),
            'scores_3d': tensor([0.1432, 0.1336, 0.2081, 0.1142, 0.1091, 0.1007])
        }
        """
             
        gt_annos = self.convert_gt_from_results(results)
        dt_annos = self.convert_dt_from_results(results)
        
        # print("Debug: gt_annos: ", gt_annos)
        # print("Debug: dt_annos: ", dt_annos)
        # Assert that the keys of the gt_annos and dt_annos are the same
        assert gt_annos.keys() == dt_annos.keys(), "The keys of the gt_annos and dt_annos are not the same."

        gt_annos_valid, perc = self.filter_valid_gt_annos(gt_annos)
        print("Percentage of valid bounding boxes for ground truth: ", perc)

        dt_annos_valid, perc = self.filter_valid_dt_annos(dt_annos)
        print("Percentage of valid bounding boxes for detections: ", perc)
        
        # Assert that the keys of the gt_annos_valid and dt_annos_valid are the same
        if not gt_annos_valid.keys() == dt_annos_valid.keys():
            print("Keys that are not in both lists: ", set(gt_annos_valid.keys()) ^ set(dt_annos_valid.keys()))
            raise ValueError("The keys of the gt_annos_valid and dt_annos_valid are not the same.")

        evaluation_results_dict = dict() # Contains all evaluation results
        curves_dict = dict() # Contains all curves
        
        self.evaluator = EvaluatorMetrics(gt_annos_valid=gt_annos_valid, 
                                          dt_annos_valid=dt_annos_valid, 
                                          classes=self.classes, 
                                          metric=self.metric, 
                                          force_single_assignement=self.force_single_assignement)

        # Only perform the evaluation when there is at least 10 gt instances and 10 dt instances
        if not self.evaluator.total_gt_instances >= 10:
            print_log("The number of gt instances is less than 10, skip evaluation.")
            return {'evaluations': evaluation_results_dict, 'curves': curves_dict}
        
        if not self.evaluator.total_dt_instances >= 10:
            print_log("The number of dt instances is less than 10, skip evaluation.")
            return {'evaluations': evaluation_results_dict, 'curves': curves_dict}
        
        evaluation_results_dict['mAP'] = self.evaluator.sklearn_mean_average_precision_score(iou_level=self.difficulty_levels,
                                                                                             class_accuracy_requirements=['easy', 'hard'])
        
        evaluation_results_dict['mAP40'] = self.evaluator.mean_average_precision_score(iou_level=self.difficulty_levels,
                                                                                       class_accuracy_requirements=['easy', 'hard'])

        evaluation_results_dict['F1'] = self.evaluator.sklearn_f1_score(iou_level=self.difficulty_levels,
                                                                        class_accuracy_requirements=['easy', 'hard'])
        

        evaluation_results_dict['precision'] = self.evaluator.compute_precision(iou_level=self.difficulty_levels,
                                                                                class_accuracy_requirements=['easy', 'hard'])

        evaluation_results_dict['AP'] = self.evaluator.sklearn_average_precision_score(iou_level=self.difficulty_levels,
                                                                                       class_accuracy_requirements=['easy', 'hard'],
                                                                                       class_idx=range(len(self.classes)))

        curves_dict['prec'] = self.evaluator.precision_recall_curve(iou_level=0.5)
        curves_dict['roc'] = self.evaluator.roc_curve(iou_level=0.5)
        curves_dict['cm'] = self.evaluator.confusion_matrix(iou_level=0.5)

        if self.save_graphics:
            self.save_plot(plot=self.evaluator.precision_recall_plot(iou_level=0.5), filename = 'precision_recall_plot_pointpillars_kitti.png')
            self.save_plot(plot=self.evaluator.roc_plot(iou_level=0.5), filename = 'roc_plot_pointpillars_kitti.png')
            self.save_plot(plot=self.evaluator.confusion_matrix_plot(iou_level=0.5), filename = 'confusion_matrix_plot_pointpillars_kitti.png')

        # Save the evaluation results to a .json file
        if self.save_evaluation_results:
            save_path = os.join(self.output_dir, self.evaluation_file_name)
            with open(save_path, 'w') as f:
                json.dump(evaluation_results_dict, f, indent=4)
            
        return {'evaluations': evaluation_results_dict, 'curves': curves_dict}
    
    def convert_gt_from_results(self, results: List[dict]) -> dict:

        gt_dict = dict()
        for data_sample in results:

            sample_idx = data_sample['lidar_path']
            gt_bboxes = InstanceData()
            # If 'gt_labels_3d' is not a tensor, convert it to a tensor
            if not isinstance(data_sample['eval_ann_info']['gt_labels_3d'], torch.Tensor):
                data_sample['eval_ann_info']['gt_labels_3d'] = torch.from_numpy(data_sample['eval_ann_info']['gt_labels_3d'])
            
            gt_bboxes.bboxes_3d = data_sample['eval_ann_info']['gt_bboxes_3d'].tensor.to('cpu')
            gt_bboxes.labels_3d = data_sample['eval_ann_info']['gt_labels_3d'].to('cpu')
            gt_dict[sample_idx] = gt_bboxes

        return gt_dict  

    def filter_valid_gt_annos(self, annos: dict) -> dict:
        """
        
        Filter the annotations to only include the valid ones.

        Args:
            annos (Dict[InstanceData]): The annotations to be filtered.

        Returns:
            Dict[InstanceData]: Identical dict that contains only bounding boxes that have their center point within the point cloud range.

        """

        annos_valid = dict()

        for key in annos.keys():

            assert len(annos[key].bboxes_3d) == len(annos[key].labels_3d), "The number of bounding boxes and labels are not the same."

            _bbox_3d = annos[key].bboxes_3d
            
            # Check if any dimension is empty
            if _bbox_3d.size(0) == 0:
                annos_valid[key] = annos[key]
                continue
            elif _bbox_3d.size(1) == 0:
                annos_valid[key] = annos[key]
                continue
            
            bbox_3d_center = _bbox_3d[:, :3]
            valid_inds = ((bbox_3d_center[:, 0] >= self.pcd_limit_range[0]) &
                        (bbox_3d_center[:, 0] <= self.pcd_limit_range[3]) &
                        (bbox_3d_center[:, 1] >= self.pcd_limit_range[1]) &
                        (bbox_3d_center[:, 1] <= self.pcd_limit_range[4]) &
                        (bbox_3d_center[:, 2] >= self.pcd_limit_range[2]) &
                        (bbox_3d_center[:, 2] <= self.pcd_limit_range[5]))
            
            instance_data_valid = InstanceData()
            instance_data_valid.bboxes_3d = annos[key].bboxes_3d[valid_inds]

            # Check if any dimension is empty
            instance_data_valid.labels_3d = annos[key].labels_3d[valid_inds]

            annos_valid[key] = instance_data_valid

        num_bbox_before = sum([len(annos[key].bboxes_3d) for key in annos.keys()])
        num_bbox_after = sum([len(annos_valid[key].bboxes_3d) for key in annos_valid.keys()])
        print("Number of ground thruth bounding boxes after filtering: ", num_bbox_after)
        
        # Devision by zero check
        if num_bbox_before == 0:
            percentage = 0.0
        else:
            percentage = round((num_bbox_after / num_bbox_before) * 100, 2)            
        return annos_valid, percentage
        
    def convert_dt_from_results(self, results: List[dict]) -> dict:
        
        dt_dict = dict()
        for data_sample in results:
                sample_idx = data_sample['lidar_path']
                dt_bboxes = InstanceData()

                if not isinstance(data_sample['pred_instances_3d']['labels_3d'], torch.Tensor):
                    data_sample['pred_instances_3d']['labels_3d'] = torch.from_numpy(data_sample['pred_instances_3d']['labels_3d'])

                dt_bboxes.bboxes_3d = data_sample['pred_instances_3d']['bboxes_3d'].tensor.to('cpu')
                dt_bboxes.labels_3d = data_sample['pred_instances_3d']['labels_3d'].to('cpu')
                dt_bboxes.scores = data_sample['pred_instances_3d']['scores_3d'].to('cpu')
                dt_dict[sample_idx] = dt_bboxes

        return dt_dict
    
    def filter_valid_dt_annos(self, annos: dict) -> dict:
        """
        
        Filter the annotations to only include the valid ones.

        Args:
            annos (Dict[InstanceData]): The annotations to be filtered.

        Returns:
            Dict[InstanceData]: Identical dict that contains only bounding boxes that have their center point within the point cloud range.

        """

        annos_valid = dict()

        for key in annos.keys():

            assert len(annos[key].bboxes_3d) == len(annos[key].labels_3d), "The number of bounding boxes and labels are not the same."

            _bbox_3d = annos[key].bboxes_3d

            # Check if any dimension is empty
            if _bbox_3d.size(0) == 0:
                annos_valid[key] = annos[key]
                continue
            elif _bbox_3d.size(1) == 0:
                annos_valid[key] = annos[key]
                continue

            bbox_3d_center = _bbox_3d[:, :3]

            valid_inds = ((bbox_3d_center[:, 0] >= self.pcd_limit_range[0]) &
                          (bbox_3d_center[:, 0] <= self.pcd_limit_range[3]) &
                          (bbox_3d_center[:, 1] >= self.pcd_limit_range[1]) &
                          (bbox_3d_center[:, 1] <= self.pcd_limit_range[4]) &
                          (bbox_3d_center[:, 2] >= self.pcd_limit_range[2]) &
                          (bbox_3d_center[:, 2] <= self.pcd_limit_range[5]))
            
            instance_data_valid = InstanceData()
            instance_data_valid.bboxes_3d = annos[key].bboxes_3d[valid_inds]
            instance_data_valid.labels_3d = annos[key].labels_3d[valid_inds]
            instance_data_valid.scores = annos[key].scores[valid_inds]

            annos_valid[key] = instance_data_valid

        num_bbox_before = sum([len(annos[key].bboxes_3d) for key in annos.keys()])
        num_bbox_after = sum([len(annos_valid[key].bboxes_3d) for key in annos_valid.keys()])

        if num_bbox_before == 0:
            percentage = 0.0
        else:
            percentage = round((num_bbox_after / num_bbox_before) * 100, 2)

        return annos_valid, percentage
    
    def save_plot(self,
                plot: Union[sk_metrics.ConfusionMatrixDisplay, sk_metrics.PrecisionRecallDisplay, sk_metrics.RocCurveDisplay],
                filename: str) -> None:

        save_dir = self.output_dir + filename

        plt.savefig(save_dir)

class EvaluatorMetrics():

    def __init__(self, 
                 gt_annos_valid: dict,
                 dt_annos_valid: dict,
                 classes: List[str],
                 force_single_assignement: bool = True,
                 metric = str):
        
        self.iou_calculator_map = {
            'det3d': 'BboxOverlaps3D',
            'bev': 'BboxOverlapsNearest3D',
        }
        
        # Explaination for the class_accuarcy_requirement_possible: 'easy' means that for a TP the class does not have to be exact,
        # 'hard' means that for a TP the class has to be exact.
        self.class_accuracy_requirement_possible = ['easy', 'hard']

        self.class_accuracy_requirement_map = {
            'easy': self.tp_detection,
            'hard': self.tp_detection_and_label
        }

        self._gt_annos_valid = gt_annos_valid
        self._dt_annos_valid = dt_annos_valid
        self._classes = classes
        self._iou_calculator = self.iou_calculator_map[metric]
        self.force_single_assignement = force_single_assignement
        
        self.threshold_specific_results_dict = dict()

    @property
    def total_gt_instances(self) -> int:
        return sum([len(self._gt_annos_valid[key].bboxes_3d) for key in self._gt_annos_valid.keys()])
    
    @property
    def total_dt_instances(self) -> int:
        return sum([len(self._dt_annos_valid[key].bboxes_3d) for key in self._dt_annos_valid.keys()])
    
    def val_batch_evaluation(self, 
                             iou_threshold: float) -> dict:
        
        iou_assigner = Max3DIoUAssigner(pos_iou_thr=iou_threshold,
                                        neg_iou_thr=iou_threshold,
                                        min_pos_iou=iou_threshold,
                                        iou_calculator=dict(type=self._iou_calculator, coordinate='lidar'))
        
        dt_labels = []
        dt_scores = []
        assigned_gt_inds = []
        assigned_max_overlaps = []
        assigned_labels = []     
        
        # # select 10 random keys
        # keys = list(self._gt_annos_valid.keys())
        # random_choice = np.random.choice(keys, 50) 
        
        # # set random choice to last 10 keys
        # random_choice = keys[-10:]
        
        for i, key in enumerate(self._gt_annos_valid.keys()):
            
            gt_instance = self._gt_annos_valid[key]
            dt_instance = self._dt_annos_valid[key]
            assign_result = iou_assigner.assign(pred_instances=dt_instance, gt_instances=gt_instance, force_single_assignement=self.force_single_assignement)
            # Maybe something is not fully right here: Does it automatically mean that the labels correspond to the gt_labels?

            dt_labels.append(dt_instance.labels_3d)
            dt_scores.append(dt_instance.scores)
            assigned_gt_inds.append(assign_result.gt_inds)
            assigned_max_overlaps.append(assign_result.max_overlaps)
            assigned_labels.append(assign_result.labels)
            
            # # # Print the bev boxes
            # if key in random_choice:
            #     draw_bev_projection(key=key,
            #             gt_3d_bboxes=self._gt_annos_valid[key].bboxes_3d,
            #             dt_3d_bboxes=self._dt_annos_valid[key].bboxes_3d)
        

        # assert that all out the lists have the same length
        assert len(dt_labels) == len(dt_scores) == len(assigned_gt_inds) == len(assigned_max_overlaps) == len(assigned_labels)

        threshold_specific_results = {
            'dt_labels': torch.cat(dt_labels),
            'confidence': torch.cat(dt_scores),
            'assigned_gt_inds': torch.cat(assigned_gt_inds),
            'assigned_max_overlaps': torch.cat(assigned_max_overlaps),
            'assigned_labels': torch.cat(assigned_labels)
        }
                            
        self.threshold_specific_results_dict[iou_threshold] = threshold_specific_results


    def filter_for_prediction_class(self,
                                    filter_dict: dict,
                                    class_idx: int) -> dict:
        
        citerion = filter_dict['dt_labels'] == class_idx

        filtered_dict = {
            'dt_labels': filter_dict['dt_labels'][citerion],
            'confidence': filter_dict['confidence'][citerion],
            'assigned_gt_inds': filter_dict['assigned_gt_inds'][citerion],
            'assigned_max_overlaps': filter_dict['assigned_max_overlaps'][citerion],
            'assigned_labels': filter_dict['assigned_labels'][citerion]
        }

        return filtered_dict

    def tp_detection_and_label(self,
                               filtered_dict: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        
        Args:
            filtered_dict (dict): A dict that contains the results that were, if this is wanted, already filtered for detection labels.

        Returns:
            tp_binary (torch.Tensor): A tensor that indicates if a detection is a true positive or not.
            score (torch.Tensor): A tensor that contains the confidence scores of the detections.
            y_true (torch.Tensor): A tensor that contains the true labels of the detections.
            y_pred (torch.Tensor): A tensor that contains the predicted labels of the detections.

        ***Attention***: True positives are defined here as detections, that have a valid, corresponding bounding box AND the correct label.
        
        """
        
        tp_criterion = filtered_dict['dt_labels'] == filtered_dict['assigned_labels']
        tp_binary = torch.where(tp_criterion, torch.tensor(1), torch.tensor(0))
        score = filtered_dict['confidence']
        return tp_binary, score, filtered_dict['assigned_labels'], filtered_dict['dt_labels']

    def tp_detection(self,
                     filtered_dict: dict) -> torch.Tensor:
        """
        
        Args:
            filtered_dict (dict): A dict that contains the results that were, if this is wanted, already filtered for detection labels.

        Returns:
            tp (torch.Tensor): A tensor that indicates if a detection is a true positive or not.
            score (torch.Tensor): A tensor that contains the confidence scores of the detections.
            y_true (torch.Tensor): A tensor that contains the true labels of the detections.
            y_pred (torch.Tensor): A tensor that contains the predicted labels of the detections.    
        

        ***Attention***: True positives are defined here as detections, that have a valid, corresponding bounding box (but not 
        necessarily the correct label).
            
        """
        
        tp_criterion = filtered_dict['assigned_gt_inds'] > 0
        tp_binary = torch.where(tp_criterion, torch.tensor(1), torch.tensor(0))
        score = filtered_dict['confidence']
        return tp_binary, score, filtered_dict['assigned_labels'], filtered_dict['dt_labels']  

    def sklearn_average_precision_score(self,
                                        iou_level: Union[float, List[float]], 
                                        class_accuracy_requirements: Union[str, List[str]] = 'easy',
                                        class_idx: Union[int, List[int]] = []):

        if isinstance(iou_level, float):
            iou_level = [iou_level]

        if isinstance(class_idx, int):
            class_idx = [class_idx]

        if isinstance(class_accuracy_requirements, str):
            class_accuracy_requirements = [class_accuracy_requirements]

        # Assure that the class_idx list is not empty
        if not class_idx:
            raise ValueError('class_idx should not be empty. Call property sklearn_mean_average_precision_score instead.')
        
        # Assure that the class_accuracy_requirement list is not empty
        if not class_accuracy_requirements:
            raise ValueError('class_accuracy_requirement should not be empty.')
        
        # Assure that the class_accuracy_requirement list contains one of the possible values
        if not all([class_accuracy_requirement in self.class_accuracy_requirement_possible for class_accuracy_requirement in class_accuracy_requirements]):
            raise ValueError('class_accuracy_requirement should only contain the possible values: ', self.class_accuracy_requirement_possible)

        # Assure that the class_idx list contains only valid indices
        if not all([cls_idx in range(len(self._classes)) for cls_idx in class_idx]):
            raise ValueError('class_idx should only contain valid indices for the classes.')

        level_dict = dict()
        for level in iou_level:
            if not level in self.threshold_specific_results_dict.keys():
                self.val_batch_evaluation(level)
            _threshold_specific_results_dict = self.threshold_specific_results_dict[level]

            class_dict = dict()
            for class_accuracy_requirement in class_accuracy_requirements:
                
                for cls_idx in class_idx:
                    _class_filtered_dict = self.filter_for_prediction_class(filter_dict=_threshold_specific_results_dict, class_idx=cls_idx)
                    tp_binary, score, y_true, y_pred  = self.class_accuracy_requirement_map[class_accuracy_requirement](filtered_dict=_class_filtered_dict)
                    if tp_binary.nelement() > 0 and score.nelement() > 0:
                        class_dict[self._classes[cls_idx]] = round(sk_metrics.average_precision_score(y_true=tp_binary, y_score=score), 3)
                    else:
                        class_dict[self._classes[cls_idx]] = 0.0
            
            level_dict[level] = class_dict

        return level_dict                    

    def sklearn_mean_average_precision_score(self,
                                             iou_level: Union[float, List[float]],
                                             class_accuracy_requirements: Union[str, List[str]] = 'easy'):
        
        if isinstance(iou_level, float):
            iou_level = [iou_level]

        if isinstance(class_accuracy_requirements, str):
            class_accuracy_requirements = [class_accuracy_requirements]

        level_dict = dict()
        for level in iou_level:
            if not level in self.threshold_specific_results_dict.keys():
                self.val_batch_evaluation(level)

            _threshold_specific_results_dict = self.threshold_specific_results_dict[level]

            class_dict = dict()
            for class_accuracy_requirement in class_accuracy_requirements:
                tp_binary, score, y_true, y_pred = self.class_accuracy_requirement_map[class_accuracy_requirement](filtered_dict=_threshold_specific_results_dict)

                if tp_binary.nelement() > 0 and score.nelement() > 0:
                    class_dict[class_accuracy_requirement] = round(sk_metrics.average_precision_score(y_true=tp_binary, y_score=score), 3)
                else:
                    class_dict[class_accuracy_requirement] = 0.0                
            level_dict[level] = class_dict

        return level_dict
    
    def sklearn_f1_score(self,
                        iou_level: Union[float, List[float]],
                        class_accuracy_requirements: Union[str, List[str]] = 'easy'):
        
        if isinstance(iou_level, float):
            iou_level = [iou_level]

        if isinstance(class_accuracy_requirements, str):
            class_accuracy_requirements = [class_accuracy_requirements]

        level_dict = dict()
        for level in iou_level:
            if not level in self.threshold_specific_results_dict.keys():
                self.val_batch_evaluation(level)

            _threshold_specific_results_dict = self.threshold_specific_results_dict[level]

            class_dict = dict()
            for class_accuracy_requirement in class_accuracy_requirements:
                tp_binary, score, y_true, y_pred = self.class_accuracy_requirement_map[class_accuracy_requirement](filtered_dict=_threshold_specific_results_dict)
                
                if tp_binary.nelement() > 0 and score.nelement() > 0:
                    class_dict[class_accuracy_requirement] = round(sk_metrics.f1_score(y_true=y_true, y_pred=y_pred, average='micro'), 3)
                else:
                    class_dict[class_accuracy_requirement] = 0.0

            level_dict[level] = class_dict

        return level_dict
    
    def mean_average_precision_score(self,
                                    iou_level: Union[float, List[float]],
                                    class_accuracy_requirements: Union[str, List[str]] = 'easy'):
        """ 
            This function computes the mean average precision score with the traditional method without using sklearn.

            Args:
                iou_level (Union[float, List[float]]): The IoU levels to be evaluated.
                class_accuracy_requirements (Union[str, List[str]]): The class accuracy requirements to be evaluated. Defaults to 'easy'.

            Returns:
                Dict: A dictionary that contains the mean average precision scores for the different IoU levels and class accuracy requirements.

        """
        
        if isinstance(iou_level, float):
            iou_level = [iou_level]

        if isinstance(class_accuracy_requirements, str):
            class_accuracy_requirements = [class_accuracy_requirements]

        map_level_dict = dict()
        for level in iou_level:

            class_dict = dict()
            for class_accuracy_requirement in class_accuracy_requirements:
                if not level in self.threshold_specific_results_dict.keys():
                    self.val_batch_evaluation(level)
                _threshold_specific_results_dict = self.threshold_specific_results_dict[level]
                
                tp_binary, score, y_true, y_pred = self.class_accuracy_requirement_map[class_accuracy_requirement](filtered_dict=_threshold_specific_results_dict)
                _precision, _recall, _ = sk_metrics.precision_recall_curve(y_true=tp_binary, probas_pred=score)
                
                class_dict[class_accuracy_requirement] = round(self.compute_ap(precision=_precision, recall=_recall, n=40), 3)

            map_level_dict[level] = class_dict

        return map_level_dict
    
    def precision_recall_curve(self,
                               iou_level: Union[float, List[float]],
                               class_idx: Union[int, List[int]] = []):
        
        if isinstance(iou_level, float):
            iou_level = [iou_level]

        if isinstance(class_idx, int):
            class_idx = [class_idx]
        
        level_dict = dict()
        for level in iou_level:
            if not level in self.threshold_specific_results_dict.keys():
                self.val_batch_evaluation(level)
            _threshold_specific_results_dict = self.threshold_specific_results_dict[level]

            class_dict = dict()
            tp_binary, score, y_true, y_pred  = self.tp_detection_and_label(filtered_dict=_threshold_specific_results_dict)
            precision, recall, _ = sk_metrics.precision_recall_curve(y_true=tp_binary, probas_pred=score)
            class_dict['all_classes'] = {
                'precision': precision,
                'recall': recall,
                'y_true': tp_binary,
                'y_probas': score,
                'labels': self._classes
            }

            level_dict[level] = class_dict

        return level_dict
        
    def precision_recall_plot(self,
                              iou_level: float):
        
        level_dict = self.precision_recall_curve(iou_level=iou_level)
        precision, recall = level_dict[iou_level]['all_classes']['precision'], level_dict[iou_level]['all_classes']['recall']
        disp = sk_metrics.PrecisionRecallDisplay(precision=precision, recall=recall, estimator_name='Precision-Recall Curve')
        return disp.plot()

    def roc_curve(self,
                  iou_level: Union[float, List[float]],
                  class_idx: Union[int, List[int]] = []):
        
        if isinstance(iou_level, float):
            iou_level = [iou_level]

        if isinstance(class_idx, int):
            class_idx = [class_idx]

        level_dict = dict()
        for level in iou_level:
            if not level in self.threshold_specific_results_dict.keys():
                self.val_batch_evaluation(level)
            _threshold_specific_results_dict = self.threshold_specific_results_dict[level]

            class_dict = dict()

            tp_binary, score, y_true, y_pred  = self.tp_detection(filtered_dict=_threshold_specific_results_dict)
            fpr, tpr, _ = sk_metrics.roc_curve(y_true=tp_binary, y_score=score)
            class_dict['all_classes'] = {
                'fpr': fpr,
                'tpr': tpr,
                'y_true': tp_binary,
                'y_probas': score,
                'labels': self._classes
            }

            level_dict[level] = class_dict

        return level_dict

    def roc_plot(self,
                 iou_level: float):
        
        level_dict = self.roc_curve(iou_level=iou_level)
        fpr, tpr = level_dict[iou_level]['all_classes']['fpr'], level_dict[iou_level]['all_classes']['tpr']
        roc_auc = sk_metrics.auc(fpr, tpr)
        disp = sk_metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, estimator_name='ROC Curve', roc_auc=roc_auc)
        return disp.plot()

    def confusion_matrix(self,
                         iou_level: float):
        
        if not iou_level in self.threshold_specific_results_dict.keys():
            self.val_batch_evaluation(iou_level)
    
        relevant_inds = self.threshold_specific_results_dict[iou_level]['assigned_labels'] != -1

        # print("Debug: assigned_gt_inds: ", self.threshold_specific_results_dict[iou_level]['assigned_gt_inds'])
        # print("Debug: relevant_inds: ", relevant_inds)

        _y_true = self.threshold_specific_results_dict[iou_level]['assigned_labels'][relevant_inds]

        # print("Debug: assigned_labels: ", self.threshold_specific_results_dict[iou_level]['assigned_labels'])
        # print("Debug: y_true: ", _y_true)

        _y_pred = self.threshold_specific_results_dict[iou_level]['dt_labels'][relevant_inds]

        # print("Debug: dt_labels: ", self.threshold_specific_results_dict[iou_level]['dt_labels'])
        # print("Debug: y_pred: ", _y_pred)

        _labels = range(len(self._classes))

        confusion_matrix = sk_metrics.confusion_matrix(y_true=_y_true, y_pred=_y_pred, labels=_labels)

        confusion_matrix_dict ={
            'y_true': _y_true,
            'y_pred': _y_pred,
            'labels': _labels,
            'classes': self._classes,
            'cm': confusion_matrix
        }

        level_dict = dict()
        level_dict[iou_level] = confusion_matrix_dict
        return level_dict

    def confusion_matrix_plot(self,
                              iou_level: float):

        cm_dict = self.confusion_matrix(iou_level)
        cm = cm_dict[iou_level]['cm']
        disp = sk_metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self._classes)
        return disp.plot(include_values=True, cmap='viridis')
    
    def compute_ap(self,
                   recall, precision, n=40):
        """
        # Arguments
            recall:    The recall curve (list).
            precision: The precision curve (list).
        # Returns
            The average precision as computed in py-faster-rcnn.
        """
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([0.0], precision, [0.0]))

        # # # compute the precision envelope
        # for i in range(mpre.size - 1, 0, -1):
        #     mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # # to calculate area under PR curve, look for points
        # # where X axis (recall) changes value
        # i = np.where(mrec[1:] != mrec[:-1])[0]

        # # and sum (\Delta recall) * prec
        # ap_all = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        
        ap_n = 0.

        if not isinstance(n, float):
            n = float(n)

        up_n = n/10.

        for t in np.arange(0., up_n, 0.1):
            if np.sum(recall >= t) == 0:
                p = 0
            else:
                p = np.max(precision[recall >= t])
            ap_n = ap_n + p / n
            
        return ap_n
    
    def precision(self, _tp_binary: torch.Tensor) -> float:

        tp = _tp_binary.sum().item()

        if len(_tp_binary) == 0:
            print("Warning: Calculating precision with an empty tensor. Set to 0.0 to prevent by zero division.")
            return 0.0
        
        precision = tp / len(_tp_binary) 
        return precision
    
    def compute_precision(self,
                            iou_level: Union[float, List[float]],
                        class_accuracy_requirements: Union[str, List[str]] = 'easy'):
        """ 
            This function computes the average precision score with the traditional method without using sklearn.

            Args:
                iou_level (Union[float, List[float]]): The IoU levels to be evaluated.
                class_accuracy_requirements (Union[str, List[str]]): The class accuracy requirements to be evaluated. Defaults to 'easy'.

            Returns:
                Dict: A dictionary that contains the mean average precision scores for the different IoU levels and class accuracy requirements.

        """
        
        if isinstance(iou_level, float):
            iou_level = [iou_level]

        if isinstance(class_accuracy_requirements, str):
            class_accuracy_requirements = [class_accuracy_requirements]

        map_level_dict = dict()
        for level in iou_level:

            class_dict = dict()
            for class_accuracy_requirement in class_accuracy_requirements:
                if not level in self.threshold_specific_results_dict.keys():
                    self.val_batch_evaluation(level)

                _threshold_specific_results_dict = self.threshold_specific_results_dict[level]
                
                tp_binary, score, y_true, y_pred = self.class_accuracy_requirement_map[class_accuracy_requirement](filtered_dict=_threshold_specific_results_dict)
                
                precision = self.precision(tp_binary)

                class_dict[class_accuracy_requirement] = round(precision, 3)

            map_level_dict[level] = class_dict

        return map_level_dict
            
    def is_of_type_kitti(self, _lidar_path: 'str') -> bool:
        """
        Example of _lidar_path:
        'data/kitti_osdar23_merge/points/000000.bin'
        """
        # Extract the file name
        file_name = _lidar_path.split('/')[-1][:-4]
        # Check if the file name is a number
        return file_name.isnumeric()   
    
####### Helper functions #######

def draw_bev_projection(key, gt_3d_bboxes, dt_3d_bboxes):
    
    # Format of the boundingboxes
    
    print("gt_3d_bboxes: ", gt_3d_bboxes)
    print("dt_3d_bboxes: ", dt_3d_bboxes)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Three horizontally aligned plots
    # fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    # ax1, ax2, ax3 = axs
    
    # # Birds eye view on ax1
    # for i in range(gt_3d_bboxes.shape[0]):
    #     gt_bbox = gt_3d_bboxes[i]
    #     corners = np.array([
    #         [-gt_bbox[3]/2, -gt_bbox[4]/2],
    #         [gt_bbox[3]/2, -gt_bbox[4]/2],
    #         [gt_bbox[3]/2, gt_bbox[4]/2],
    #         [-gt_bbox[3]/2, gt_bbox[4]/2]
    #     ])
        
    #     theta = np.radians(gt_bbox[6])
    #     R = np.array([
    #         [np.cos(theta), -np.sin(theta)],
    #         [np.sin(theta), np.cos(theta)]
    #     ])
        
    #     rotated_corners = corners.dot(R)
    #     translated_corners = np.add(rotated_corners, gt_bbox[:2].cpu().numpy())
        
    #     square = plt.Polygon(translated_corners, fill=None, edgecolor='green',  linestyle='--', closed=True)
    #     ax1.add_patch(square)
        
        
    # for i in range(dt_3d_bboxes.shape[0]):
    #     gt_bbox = dt_3d_bboxes[i]
    #     corners = np.array([
    #         [-gt_bbox[3]/2, -gt_bbox[4]/2],
    #         [gt_bbox[3]/2, -gt_bbox[4]/2],
    #         [gt_bbox[3]/2, gt_bbox[4]/2],
    #         [-gt_bbox[3]/2, gt_bbox[4]/2]
    #     ])
        
    #     theta = np.radians(gt_bbox[6])
    #     R = np.array([
    #         [np.cos(theta), -np.sin(theta)],
    #         [np.sin(theta), np.cos(theta)]
    #     ])
        
    #     rotated_corners = corners.dot(R)
    #     translated_corners = np.add(rotated_corners, gt_bbox[:2].cpu().numpy())
        
    #     square = plt.Polygon(translated_corners, fill=None, edgecolor='red', linestyle='--', closed=True)
    #     ax1.add_patch(square)
    
    # Corner points of the 3D bounding box ground truth
    for i in range(gt_3d_bboxes.shape[0]):
        gt_box_3d = gt_3d_bboxes[i]        
        corners = np.array([
            [gt_box_3d[3]/2, gt_box_3d[4]/2, 0], # B1: Bottom front left
            [gt_box_3d[3]/2, -gt_box_3d[4]/2, 0], # B2: Bottom front right
            [-gt_box_3d[3]/2, -gt_box_3d[4]/2, 0], # B3: Bottom back right
            [-gt_box_3d[3]/2, gt_box_3d[4]/2, 0], # B4: Bottom back left
            [gt_box_3d[3]/2, gt_box_3d[4]/2, gt_box_3d[5]], # T1: Top front left
            [gt_box_3d[3]/2, -gt_box_3d[4]/2, gt_box_3d[5]], # T2: Top front right
            [-gt_box_3d[3]/2, -gt_box_3d[4]/2, gt_box_3d[5]], # T3: Top back right
            [-gt_box_3d[3]/2, gt_box_3d[4]/2, gt_box_3d[5]] # T4: Top back left
        ])
        
        theta = np.radians(gt_box_3d[6])
        R = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])
        
        rotated_corners = R @ corners.T
        translated_corners = np.add(rotated_corners.T, gt_box_3d[:3].cpu().numpy())
        
        vertices = translated_corners
        edges = [
            [vertices[0], vertices[1], vertices[2], vertices[3], vertices[0]],
            [vertices[4], vertices[5], vertices[6], vertices[7], vertices[4]],
            [vertices[0], vertices[4]],
            [vertices[1], vertices[5]],
            [vertices[2], vertices[6]],
            [vertices[3], vertices[7]]
        ]
        
        for edge in edges:
            ax.add_collection3d(Poly3DCollection([edge], facecolors='springgreen', linewidths=1, edgecolors='seagreen', alpha=.25))
            
    for i in range(dt_3d_bboxes.shape[0]):
        dt_box_3d = dt_3d_bboxes[i]
        corners = np.array([
            [dt_box_3d[3]/2, dt_box_3d[4]/2, 0], # B1: Bottom front left
            [dt_box_3d[3]/2, -dt_box_3d[4]/2, 0], # B2: Bottom front right
            [-dt_box_3d[3]/2, -dt_box_3d[4]/2, 0], # B3: Bottom back right
            [-dt_box_3d[3]/2, dt_box_3d[4]/2, 0], # B4: Bottom back left
            [dt_box_3d[3]/2, dt_box_3d[4]/2, dt_box_3d[5]], # T1: Top front left
            [dt_box_3d[3]/2, -dt_box_3d[4]/2, dt_box_3d[5]], # T2: Top front right
            [-dt_box_3d[3]/2, -dt_box_3d[4]/2, dt_box_3d[5]], # T3: Top back right
            [-dt_box_3d[3]/2, dt_box_3d[4]/2, dt_box_3d[5]] # T4: Top back left
        ])
        
        theta = np.radians(dt_box_3d[6])
        R = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])
        
        rotated_corners = R @ corners.T
        translated_corners = np.add(rotated_corners.T, dt_box_3d[:3].cpu().numpy())

        vertices = translated_corners
        edges = [
            [vertices[0], vertices[1], vertices[2], vertices[3], vertices[0]],
            [vertices[4], vertices[5], vertices[6], vertices[7], vertices[4]],
            [vertices[0], vertices[4]],
            [vertices[1], vertices[5]],
            [vertices[2], vertices[6]],
            [vertices[3], vertices[7]]
        ]
        
        for edge in edges:
            ax.add_collection3d(Poly3DCollection([edge], facecolors='cornflowerblue', linewidths=1, edgecolors='royalblue', alpha=.25))
    
        
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
        
    # set title for complete figure
    fig.suptitle('Projections of key: ' + key)
    
    x_upper = 80
    x_lower = 0
    y_upper = -40
    y_lower = 40
    z_upper = 5
    z_lower = -5
    
    if not (gt_3d_bboxes.shape[0] == 0 or dt_3d_bboxes.shape[0] == 0):
        x_upper = max([gt_3d_bboxes[:, 0].max(), dt_3d_bboxes[:, 0].max()]) + 5
        x_lower = 0
        y_upper = max([gt_3d_bboxes[:, 1].max(), dt_3d_bboxes[:, 1].max()]) + 5
        y_lower = min([gt_3d_bboxes[:, 1].min(), dt_3d_bboxes[:, 1].min()]) - 5
        z_upper = 5
        z_lower = -5
    else:
        print("No bounding boxes to plot for key: ", key)
        return
    
    # Set the limits for the complete figure
    ax.set_xlim(x_lower, x_upper)
    ax.set_ylim(y_lower, y_upper)
    ax.set_zlim(z_lower, z_upper)

    # Apply the calculated aspect ratio
    ax.set_aspect('equal')
    scale = vertices.flatten()
    ax.auto_scale_xyz(scale, scale, scale)
        
    # save the plot to the save directory
    plt.show(block=True)