# Copyright (c) OpenMMLab. All rights reserved.
import tempfile
import os
from os import path as osp
from typing import Dict, List, Optional, Sequence, Tuple, Union
from abc import ABC, abstractmethod
import json

import matplotlib.pyplot as plt
import pprint
import bbox
import torch
import pandas as pd
from sklearn import metrics as sk_metrics

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
                 ann_file: Optional[str] = None, # we do not necessarily need an annotation file
                 metric: str = 'det3d',
                 pcd_limit_range: List[float] = [0, -39.68, -20, 69.12, 39.68, 20],
                 force_single_assignement: bool = False,
                 prefix: Optional[str] = None,
                 pklfile_prefix: Optional[str] = None,
                 default_cam_key: str = 'CAM2',
                 format_only: bool = False,
                 submission_prefix: Optional[str] = None,
                 collect_device: str = 'cpu',
                 save_graphics: Optional[bool] = False,
                 save_evaluation_results: Optional[bool] = False,
                 difficulty_levels: Optional[List[float]] = [0.1, 0.3, 0.5, 0.7],
                 classes: Optional[List[str]] = None,
                 output_dir: Optional[str] = None,
                 evaluation_file_name: Optional[str] = 'evaluation_results.json',
                 backend_args: Optional[dict] = None,
                 save_random_viz: Optional[bool] = False,
                save_tp_positioning: Optional[bool] = False,
                 random_viz_keys: Optional[int] = None) -> 5:
    
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
        self.num_random_keys_to_be_visualized = random_viz_keys
        self.save_random_viz = save_random_viz
        self.save_tp_positioning = save_tp_positioning

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
        
        key_intersection = set(gt_annos.keys()) & set(dt_annos.keys())
        
        # Only keep the keys that are in both gt_annos and dt_annos
        gt_annos = {key: gt_annos[key] for key in key_intersection}
        dt_annos = {key: dt_annos[key] for key in key_intersection}
        
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
        if not self.evaluator.total_gt_instances >= 1:
            print_log("The number of gt instances is less than 1, skip evaluation.")
            return {'evaluations': evaluation_results_dict, 'curves': curves_dict}
        
        if not self.evaluator.total_dt_instances >= 1:
            print_log("The number of dt instances is less than 1, skip evaluation.")
            return {'evaluations': evaluation_results_dict, 'curves': curves_dict}

        ######## Evaluation ########
        
        ad_dict = self.evaluator.options_frame_metrics(method='ap', 
                                                        iou_level=self.difficulty_levels,
                                                        class_accuracy_requirements=['easy', 'hard'])
        
        evaluation_results_dict.update(ad_dict)
        
        prec_dict = self.evaluator.options_frame_metrics(method='precision',
                                                        iou_level=self.difficulty_levels,
                                                        class_accuracy_requirements=['easy', 'hard'])
        
        evaluation_results_dict.update(prec_dict)

        recall_dict = self.evaluator.options_frame_metrics(method='recall',
                                                            iou_level=self.difficulty_levels,
                                                            class_accuracy_requirements=['easy', 'hard'])
        
        evaluation_results_dict.update(recall_dict)

        curves_dict['prec'] = self.evaluator.options_frame_curves(method='precision_recall_curve', iou_level=0.3)
        curves_dict['roc'] = self.evaluator.options_frame_curves(method='roc_curve', iou_level=0.3)
        # curves_dict['cm'] = self.evaluator.options_frame_curves(method='confusion_matrix', iou_level=0.3)

        ######## Visualization ########

        if self.save_graphics:
            self.save_plot(plot=self.evaluator.options_frame_plots(method='precision_recall_plot', iou_level=0.3), filename = 'precision_recall_plot_pointpillars_kitti.png')
            self.save_plot(plot=self.evaluator.options_frame_plots(method='roc_plot', iou_level=0.3), filename = 'roc_plot_pointpillars_kitti.png')
            self.save_plot(plot=self.evaluator.options_frame_plots(method='confusion_matrix_plot', iou_level=0.3), filename = 'confusion_matrix_plot_pointpillars_kitti.png')
            
        if self.save_tp_positioning:
            # self.evaluator.save_dt_positioning(output_dir=self.output_dir, iou_level=0.3)
            # self.evaluator.save_gt_positioning(output_dir=self.output_dir, iou_level=0.3)
            self.evaluator.save_all_results(output_dir=self.output_dir, iou_level=0.3)

        ####### Saving the evaluation results #######

        # Save the evaluation results to a .json file
        if self.save_evaluation_results:
            save_path = osp.join(self.output_dir, self.evaluation_file_name)
            with open(save_path, 'w') as f:
                json.dump(evaluation_results_dict, f, indent=4)

        ######## Visualization of random keys ########
        self.visualize_random_keys(_gt_annos=gt_annos_valid, _dt_annos=dt_annos_valid, num_keys=self.num_random_keys_to_be_visualized)
            
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
        print("Number of frames evaluated: ", len(annos.keys()))
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

                try:
                    if not isinstance(data_sample['pred_instances_3d']['labels_3d'], torch.Tensor):
                        data_sample['pred_instances_3d']['labels_3d'] = torch.from_numpy(data_sample['pred_instances_3d']['labels_3d'])

                    dt_bboxes.bboxes_3d = data_sample['pred_instances_3d']['bboxes_3d'].tensor.to('cpu')
                    dt_bboxes.labels_3d = data_sample['pred_instances_3d']['labels_3d'].to('cpu')
                    dt_bboxes.scores = data_sample['pred_instances_3d']['scores_3d'].to('cpu')
                    dt_dict[sample_idx] = dt_bboxes
                except Exception as e:
                    print("Error: ", e)
                    print("The following sample has no predictions: ", sample_idx)

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

    def visualize_random_keys(self,
                              _gt_annos,
                              _dt_annos,
                              num_keys: int = 4) -> None:
        
        if num_keys == 0 or num_keys == None:
            return
        elif num_keys > len(_gt_annos.keys()):
            print("The number of keys is larger than the number of keys in the dictionary. Only visulaizing max 20 keys.")
            num_keys = 20
        if num_keys > 20:
            print("The number of keys is larger than 20. Only visualizing 20 keys.")
            num_keys = 20
                
        keys = list(_gt_annos.keys())
        random_choice = np.random.choice(keys, num_keys, replace=False)

        for key in random_choice:
            fig = show_projections(key=key,
                                gt_3d_bboxes=_gt_annos[key].bboxes_3d,
                                dt_3d_bboxes=_dt_annos[key].bboxes_3d)
            
            if self.save_random_viz:
                
                save_dir = osp.join(self.output_dir, 'bbox_visualizations', f'random_viz_{key}.png')

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
        
        self.metrics_method = {
            'ap': self.ap,
            'precision': self.precision,
            'recall': self.recall
        }
        
        self.curves_method = {
            'precision_recall_curve': self.precision_recall_curve,
            'roc_curve': self.roc_curve,
            'confusion_matrix': self.confusion_matrix
        }
        
        self.plots_method = {
            'precision_recall_plot': self.precision_recall_plot,
            'roc_plot': self.roc_plot,
            'confusion_matrix_plot': self.confusion_matrix_plot
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

    def get_unassigned_gt_instances(self, _gt_instances, _assign_result_gt_inds) -> torch.Tensor:

        """
        
        Args:
            _gt_instances (InstanceData): The ground truth instances.
            _assign_result_gt_inds (torch.Tensor): The indices of the ground truth instances that are assigned to a detection.

        Returns:
            torch.Tensor: Binary mask, 1 if the ground truth bounding box was assigned a detection, 0 if not.
        """
        
        gt_indices = torch.arange(1, len(_gt_instances.bboxes_3d) + 1)
        indices_assigned_to_dt = _assign_result_gt_inds[_assign_result_gt_inds > 0]
        gt_instance_mask = torch.isin(gt_indices, indices_assigned_to_dt)
        return gt_instance_mask
    
    def val_batch_evaluation(self, 
                             iou_threshold: float) -> dict:
        
        iou_assigner = Max3DIoUAssigner(pos_iou_thr=iou_threshold,
                                        neg_iou_thr=iou_threshold,
                                        min_pos_iou=iou_threshold,
                                        iou_calculator=dict(type=self._iou_calculator, coordinate='lidar'))
        
        dt_labels = []                  # Labels of the detections
        dt_scores = []                  # Confidence scores of the detections
        dt_assigned_gt_indices = []     # For each detection, the index of the assigned ground truth instance
        dt_max_overlaps = []            # For each detection, the maximum overlap with a ground truth instance
        dt_assigned_labels = []         # For each detection, the assigned label
        
        gt_labels = []                  # Labels of the ground truth instances
        gt_assigned_or_not_binary = []  # For each ground truth instance, if it is assigned to a detection or not (1 for assigned, 0 for not assigned)    

        
        for i, key in enumerate(self._gt_annos_valid.keys()):
            
            gt_instance = self._gt_annos_valid[key]
            dt_instance = self._dt_annos_valid[key]
            assign_result = iou_assigner.assign(pred_instances=dt_instance, gt_instances=gt_instance, force_single_assignement=self.force_single_assignement)

            dt_labels.append(dt_instance.labels_3d)
            dt_scores.append(dt_instance.scores)
            dt_assigned_gt_indices.append(assign_result.gt_inds) # For each detection, the index of the assigned ground truth instance
            dt_max_overlaps.append(assign_result.max_overlaps)
            dt_assigned_labels.append(assign_result.labels)
            
            gt_labels.append(gt_instance.labels_3d)
            gt_assigned_or_not_binary.append(self.get_unassigned_gt_instances(_gt_instances=gt_instance, _assign_result_gt_inds=assign_result.gt_inds))

        # assert that all out the lists have the same length
        assert len(dt_labels) == len(dt_scores) == len(dt_assigned_gt_indices) == len(dt_max_overlaps) == len(dt_assigned_labels)
        assert len(gt_labels) == len(gt_assigned_or_not_binary)
        
        threshold_specific_results = {
            'dt_labels': torch.cat(dt_labels),
            'dt_scores': torch.cat(dt_scores),
            'dt_assigned_gt_indices': torch.cat(dt_assigned_gt_indices),
            'dt_max_overlaps': torch.cat(dt_max_overlaps),
            'dt_assigned_labels': torch.cat(dt_assigned_labels),
            'gt_labels': torch.cat(gt_labels),
            'gt_assigned_or_not_binary': torch.cat(gt_assigned_or_not_binary)            
        }
                            
        self.threshold_specific_results_dict[iou_threshold] = threshold_specific_results

    def filter_for_prediction_class(self,
                                    filter_dict: dict,
                                    class_idx: int) -> dict:
        
        dt_criterion = filter_dict['dt_labels'] == class_idx
        gt_criterion = filter_dict['gt_labels'] == class_idx

        filtered_dict = {
            'dt_labels': filter_dict['dt_labels'][dt_criterion],
            'dt_scores': filter_dict['dt_scores'][dt_criterion],
            'dt_assigned_gt_indices': filter_dict['dt_assigned_gt_indices'][dt_criterion],
            'dt_max_overlaps': filter_dict['dt_max_overlaps'][dt_criterion],
            'dt_assigned_labels': filter_dict['dt_assigned_labels'][dt_criterion],
            'gt_labels': filter_dict['gt_labels'][gt_criterion],
            'gt_assigned_or_not_binary': filter_dict['gt_assigned_or_not_binary'][gt_criterion]
        }
        
        assert len(filtered_dict['dt_labels']) == len(filtered_dict['dt_scores']) == len(filtered_dict['dt_assigned_gt_indices']) == len(filtered_dict['dt_max_overlaps']) == len(filtered_dict['dt_assigned_labels'])
        assert len(filtered_dict['gt_labels']) == len(filtered_dict['gt_assigned_or_not_binary'])

        return filtered_dict

    def tp_detection_and_label(self,
                               filtered_dict: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        
        Args:
            filtered_dict (dict): A dict that contains the results that were, if this is wanted, already filtered for detection labels.
            
        Returns:
            tp_binary (torch.Tensor): A tensor that indicates if a detection is a true positive or not.
            
        ***Attention***: True positives are defined here as detections, that have a valid, corresponding bounding box AND the correct label.
        
        """                   
        
        tp_criterion = filtered_dict['dt_labels'] == filtered_dict['dt_assigned_labels']
        tp_binary = torch.where(tp_criterion, torch.tensor(1), torch.tensor(0))
        return tp_binary

    def tp_detection(self,
                     filtered_dict: dict) -> torch.Tensor:
        """
        
        Args:
            filtered_dict (dict): A dict that contains the results that were, if this is wanted, already filtered for detection labels.

        Returns:
            tp_binary (torch.Tensor): A tensor that indicates if a detection is a true positive or not.
        

        ***Attention***: True positives are defined here as detections, that have a valid, corresponding bounding box (but not 
        necessarily the correct label).
            
        """
        
        tp_criterion = filtered_dict['dt_assigned_gt_indices'] > 0
        tp_binary = torch.where(tp_criterion, torch.tensor(1), torch.tensor(0))
        return tp_binary
    
    def compute_ap(self,
                   recall, precision, n=40):
        """
        
        Compute the average precision score.
        
        Args:
            recall (torch.Tensor): The recall values.
            precision (torch.Tensor): The precision values.
            n (int): The number of recall levels to be evaluated. Defaults to 40.
            
        Returns:
            float: The average precision score.
        
        """
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([0.0], precision, [0.0]))
        
        recall = recall.numpy()
        precision = precision.numpy()
        
        ap_n = 0.
        if not isinstance(n, int):
            n = float(n)

        # Assert that precision and recall have the same length
        assert len(precision) == len(recall)
        recall_levels = np.linspace(0, 1, n)
        for t in recall_levels:
            if np.sum(recall >= t) == 0:
                p = 0
            else:
                p = np.max(precision[recall >= t])
            ap_n = ap_n + p / n
            
        return ap_n
    
    def generate_precision_recall_curve(self,
                                        _dt_tp_binary: torch.Tensor,
                                        _dt_confidence: torch.Tensor,
                                        _gt_labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        """
        
        Generate the precision-recall curve.
        
        Args:
            _dt_tp_binary (torch.Tensor): A tensor that indicates if a detection is a true positive or not.
            _dt_confidence (torch.Tensor): A tensor that contains the confidence scores of the detections.
            _gt_labels (torch.Tensor): A tensor that contains the true labels of the detections.
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The precision and recall values.
        
        """
        
        # TODO: Are we sure this is ordered by confidence?
        # Sort the detections by confidence
        sorted_indices = torch.argsort(_dt_confidence, descending=True)
        tp_binary = _dt_tp_binary[sorted_indices]
        fp_binary = torch.logical_not(tp_binary)
        
        acc_tp = torch.cumsum(tp_binary, dim=0)
        acc_fp = torch.cumsum(fp_binary, dim=0)
        gt_instances_lengths = len(_gt_labels)
        num_gt = torch.tensor(gt_instances_lengths).sum()
        
        precision_curve = acc_tp / (acc_tp + acc_fp)
        recall_curve = acc_tp / num_gt
        
        return precision_curve, recall_curve
    
    ######### Methods metrics #########
    
    def ap(self,
        level: float,
        class_accuracy_requirement: str,
        class_idx: Optional[int] = None) -> float:
        
        """
        
        Average precision score.
        
        Args:
            level (float): The IoU level.
            class_accuracy_requirement (str): The class accuracy requirement.
            class_idx (Optional[int]): The class index. Defaults to None.
            
        Returns:
            float: The average precision score.
        
        """
        
        if not level in self.threshold_specific_results_dict.keys():
            self.val_batch_evaluation(level)
            
        results_dict = self.threshold_specific_results_dict[level]
        
        if class_idx is not None:
            results_dict = self.filter_for_prediction_class(filter_dict=results_dict, class_idx=class_idx)
        
        dt_tp_binary = self.class_accuracy_requirement_map[class_accuracy_requirement](filtered_dict=results_dict)
        
        # prevent case of no detections
        if dt_tp_binary.nelement() == 0 or results_dict['dt_labels'].nelement() == 0 or results_dict['gt_labels'].nelement() == 0:
            return 0.0
        
        precision, recall = self.generate_precision_recall_curve(_dt_tp_binary=dt_tp_binary, _dt_confidence=results_dict['dt_scores'], _gt_labels=results_dict['gt_labels'])
        
        ap = self.compute_ap(precision=precision, recall=recall)
        return round(ap, 3)
    
    def precision(self,
                    level: float,
                    class_accuracy_requirement: str,
                    class_idx: Optional[int] = None) -> float:
        
        """
        
        Precision score.
        
        Args:
            level (float): The IoU level.
            class_accuracy_requirement (str): The class accuracy requirement.
            class_idx (Optional[int]): The class index. Defaults to None.
            
        Returns:    
            float: The precision score.
        
        """
        
        if not level in self.threshold_specific_results_dict.keys():
            self.val_batch_evaluation(level)
            
        results_dict = self.threshold_specific_results_dict[level]
        
        if class_idx is not None:
            results_dict = self.filter_for_prediction_class(filter_dict=results_dict, class_idx=class_idx)
            
        dt_tp_binary = self.class_accuracy_requirement_map[class_accuracy_requirement](filtered_dict=results_dict)
        
        # prevent case of no detections
        if dt_tp_binary.nelement() == 0 or results_dict['dt_labels'].nelement() == 0:
            return 0.0
        
        tp = dt_tp_binary.sum().item()
        fp = len(results_dict['dt_labels']) - tp
        precision = tp / (tp + fp)
        
        return round(precision, 3)
    
    def recall(self,
                level: float,
                class_accuracy_requirement: str,
                class_idx: Optional[int] = None) -> float:
        
        """
        
        Recall score.
        
        Args:
            level (float): The IoU level.
            class_accuracy_requirement (str): The class accuracy requirement.
            class_idx (Optional[int]): The class index. Defaults to None.
            
        Returns:
            float: The recall score.
            
        """
        
        if not level in self.threshold_specific_results_dict.keys():
            self.val_batch_evaluation(level)
            
        results_dict = self.threshold_specific_results_dict[level]
        
        if class_idx is not None:
            results_dict = self.filter_for_prediction_class(filter_dict=results_dict, class_idx=class_idx)
            
        # prevent case of no detections
        if len(results_dict['gt_labels']) == 0 or results_dict['gt_assigned_or_not_binary'].nelement() == 0:
            return 0.0
        
        recall = results_dict['gt_assigned_or_not_binary'].sum().item() / len(results_dict['gt_assigned_or_not_binary'])
        
        return round(recall, 3)
    
    ######### Methods visuals #########
    
    def precision_recall_plot(self,
                              iou_level: float):
    
        _dict = self.precision_recall_curve(iou_level)
        precision = _dict['precision']
        recall = _dict['recall']
      
        disp = sk_metrics.PrecisionRecallDisplay(precision=precision, recall=recall, estimator_name='Precision-Recall Curve')
        return disp.plot()
    
    def precision_recall_curve(self,
                                iou_level: float):
        
        if not iou_level in self.threshold_specific_results_dict.keys():
            self.val_batch_evaluation(iou_level)
            
        results_dict = self.threshold_specific_results_dict[iou_level]
        dt_tp_binary = self.tp_detection_and_label(filtered_dict=results_dict)
        precision, recall = self.generate_precision_recall_curve(_dt_tp_binary=dt_tp_binary, _dt_confidence=results_dict['dt_scores'], _gt_labels=results_dict['gt_labels'])
        
        return {'precision': precision, 'recall': recall}

    def roc_curve(self,
                  iou_level: float):
        
        if not iou_level in self.threshold_specific_results_dict.keys():
            self.val_batch_evaluation(iou_level)
            
        results_dict = self.threshold_specific_results_dict[iou_level]
        dt_tp_binary = self.tp_detection(filtered_dict=results_dict)
        fpr, tpr, _ = sk_metrics.roc_curve(y_true=dt_tp_binary, y_score=results_dict['dt_scores'])
        return {'fpr': fpr, 'tpr': tpr}

    def roc_plot(self,
                 iou_level: float):
        
        _dict = self.roc_curve(iou_level)
        fpr = _dict['fpr']
        tpr = _dict['tpr']
        roc_auc = sk_metrics.auc(fpr, tpr)
        disp = sk_metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, estimator_name='ROC Curve', roc_auc=roc_auc)
        return disp.plot()
    
    def confusion_matrix(self,
                         iou_level: float):
        
        if not iou_level in self.threshold_specific_results_dict.keys():
            self.val_batch_evaluation(iou_level)
            
        results_dict = self.threshold_specific_results_dict[iou_level]
        
        relevant_inds = results_dict['dt_assigned_labels'] != -1
        _y_true = results_dict['dt_labels'][relevant_inds]
        _y_pred = results_dict['dt_assigned_labels'][relevant_inds]
        _labels = range(len(self._classes))
        
        confusion_matrix = sk_metrics.confusion_matrix(y_true=_y_true, y_pred=_y_pred, labels=_labels)
        
        return {'cm': confusion_matrix}

    def confusion_matrix_plot(self,
                                iou_level: float):
            
        _dict = self.confusion_matrix(iou_level)
        cm = _dict['cm']
        disp = sk_metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self._classes)
        return disp.plot(include_values=True, cmap='viridis')

    
    ######### Option Frames #########
    
    def options_frame_metrics(self,
        method: str,
        iou_level: Union[float, List[float]],
        class_accuracy_requirements: Union[str, List[str]] = 'easy',
        class_idx: Union[int, List[int]] = []):
        
        if isinstance(iou_level, float):
            iou_level = [iou_level]
            
        if isinstance(class_idx, int):
            class_idx = [class_idx]
            
        if isinstance(class_accuracy_requirements, str):
            class_accuracy_requirements = [class_accuracy_requirements]
            
        if not class_idx:
            class_idx = range(len(self._classes))
            
        if not class_accuracy_requirements:
            class_accuracy_requirements = self.class_accuracy_requirement_possible
            
        if not all([class_accuracy_requirement in self.class_accuracy_requirement_possible for class_accuracy_requirement in class_accuracy_requirements]):
            raise ValueError('class_accuracy_requirement should only contain the possible values: ', self.class_accuracy_requirement_possible)
        
        if not all([cls_idx in range(len(self._classes)) for cls_idx in class_idx]):
            raise ValueError('class_idx should only contain valid indices for the classes.')
        
        level_dict = dict()
        for level in iou_level:
        
            class_accuracy_requirement_dict = dict()
            for class_accuracy_requirement in class_accuracy_requirements:
                
                class_dict = dict()
                
                class_dict['avg'] = self.metrics_method[method](level=level, class_accuracy_requirement=class_accuracy_requirement)
                
                for cls_idx in class_idx:
                    class_dict[self._classes[cls_idx]] = self.metrics_method[method](level=level, class_accuracy_requirement=class_accuracy_requirement, class_idx=cls_idx)
                        
                class_accuracy_requirement_dict[class_accuracy_requirement] = class_dict
            
            level_dict[level] = class_accuracy_requirement_dict
            
        return {method: level_dict}

    def options_frame_plots(self,
        method: str,
        iou_level: float):
        return self.plots_method[method](iou_level)
    
    def options_frame_curves(self,
        method: str,
        iou_level: float):
        return self.curves_method[method](iou_level)
    
    def save_all_results(self, output_dir: str, iou_level: float):
        
        if not iou_level in self.threshold_specific_results_dict.keys():
            self.val_batch_evaluation(iou_level)
            
        results_dict = copy.deepcopy(self.threshold_specific_results_dict[iou_level])
        
        results_dict_np = {k: v.cpu().numpy() for k, v in results_dict.items()}
 
        # Part relevant for the detections
        
        dt_frame_names = []
        
        x = []
        y = []
        z = []
        l = []
        w = []
        h = []
        theta = []
                
        for key in self._dt_annos_valid.keys():
            dt_instance = self._dt_annos_valid[key]
            x.append(dt_instance.bboxes_3d[:, 0].cpu().numpy())
            y.append(dt_instance.bboxes_3d[:, 1].cpu().numpy())
            z.append(dt_instance.bboxes_3d[:, 2].cpu().numpy())
            l.append(dt_instance.bboxes_3d[:, 3].cpu().numpy())
            w.append(dt_instance.bboxes_3d[:, 4].cpu().numpy())
            h.append(dt_instance.bboxes_3d[:, 5].cpu().numpy())
            theta.append(dt_instance.bboxes_3d[:, 6].cpu().numpy())
            dt_frame_names += [key] * len(dt_instance.bboxes_3d)
            
        x = np.concatenate(x)
        y = np.concatenate(y)
        z = np.concatenate(z)
        l = np.concatenate(l)
        w = np.concatenate(w)
        h = np.concatenate(h)
        theta = np.concatenate(theta)
        
        dt_labels = results_dict_np['dt_labels']
        dt_scores = results_dict_np['dt_scores']
        dt_assigned_gt_indices = results_dict_np['dt_assigned_gt_indices']
        dt_max_overlaps = results_dict_np['dt_max_overlaps']
        dt_assigned_labels = results_dict_np['dt_assigned_labels']
        tp_binary_easy = self.tp_detection(filtered_dict=results_dict)
        tp_binary_hard = self.tp_detection_and_label(filtered_dict=results_dict)
                
        assert len(dt_frame_names) == len(dt_labels) == len(dt_scores) == len(dt_assigned_gt_indices) == len(dt_max_overlaps) == len(dt_assigned_labels) == len(tp_binary_easy) == len(tp_binary_hard)
        
        # # save in a csv file
        df = pd.DataFrame({'x': x, 'y': y, 'z': z, 'l': l, 'w': w, 'h': h, 'theta': theta, 'dt_labels': dt_labels, 'dt_scores': dt_scores, 'dt_assigned_gt_indices': dt_assigned_gt_indices, 'dt_max_overlaps': dt_max_overlaps, 'dt_assigned_labels': dt_assigned_labels, 'tp_binary_easy': tp_binary_easy, 'tp_binary_hard': tp_binary_hard, 'dt_frame_names': dt_frame_names})
        df.to_csv(osp.join(output_dir, f'dt_results_{iou_level}.csv'), index=False)

        # Part relevant for the ground truth instances
        gt_frame_names = []
        gt_labels = results_dict_np['gt_labels']
        gt_assigned_or_not_binary = results_dict_np['gt_assigned_or_not_binary']

        x = []
        y = []
        z = []
        l = []
        w = []
        h = []
        theta = []
        
        for key in self._gt_annos_valid.keys():
            gt_instance = self._gt_annos_valid[key]
            x.append(gt_instance.bboxes_3d[:, 0].cpu().numpy())
            y.append(gt_instance.bboxes_3d[:, 1].cpu().numpy())
            z.append(gt_instance.bboxes_3d[:, 2].cpu().numpy())
            l.append(gt_instance.bboxes_3d[:, 3].cpu().numpy())
            w.append(gt_instance.bboxes_3d[:, 4].cpu().numpy())
            h.append(gt_instance.bboxes_3d[:, 5].cpu().numpy())
            theta.append(gt_instance.bboxes_3d[:, 6].cpu().numpy())
            gt_frame_names += [key] * len(gt_instance.bboxes_3d)
            
        x = np.concatenate(x)
        y = np.concatenate(y)
        z = np.concatenate(z)
        l = np.concatenate(l)
        w = np.concatenate(w)
        h = np.concatenate(h)
        theta = np.concatenate(theta)
        
        assert len(gt_frame_names) == len(gt_labels) == len(gt_assigned_or_not_binary)
        
        # save in a csv file
        df_gt = pd.DataFrame({'x': x, 'y': y, 'z': z, 'l': l, 'w': w, 'h': h, 'theta': theta, 'gt_labels': gt_labels, 'gt_assigned_or_not_binary': gt_assigned_or_not_binary, 'gt_frame_names': gt_frame_names})
        df_gt.to_csv(osp.join(output_dir, f'gt_results_{iou_level}.csv'), index=False)
    
    def save_dt_positioning(self, output_dir: str, iou_level: float):
        
        if not iou_level in self.threshold_specific_results_dict.keys():
            self.val_batch_evaluation(iou_level)
            
        results_dict = copy.deepcopy(self.threshold_specific_results_dict[iou_level])
        
        dt_tp_binary = self.class_accuracy_requirement_map['easy'](filtered_dict=results_dict)
        
        # from _dt_annos_valid save the position of all detections
        
        _x = []
        _y = []
        _z = []
        
        for key in self._dt_annos_valid.keys():
            dt_instance = self._dt_annos_valid[key]
            _x.append(dt_instance.bboxes_3d[:, 0].cpu().numpy())
            _y.append(dt_instance.bboxes_3d[:, 1].cpu().numpy())
            _z.append(dt_instance.bboxes_3d[:, 2].cpu().numpy())
            
        x = np.concatenate(_x)
        y = np.concatenate(_y)
        z = np.concatenate(_z)
        
        df = pd.DataFrame({'x': x, 'y': y, 'z': z, 'tp': dt_tp_binary.cpu().numpy()})
        
        df.to_csv(osp.join(output_dir, 'dt_positioning.csv'), index=False)
        
    def save_gt_positioning(self, output_dir: str, iou_level: float):
        
        if not iou_level in self.threshold_specific_results_dict.keys():
            self.val_batch_evaluation(iou_level)
            
        results_dict = copy.deepcopy(self.threshold_specific_results_dict[iou_level])
        
        gt_assigned_or_not_binary = results_dict['gt_assigned_or_not_binary']
        
        _x = []
        _y = []
        _z = []
        
        for key in self._gt_annos_valid.keys():
            gt_instance = self._gt_annos_valid[key]
            _x.append(gt_instance.bboxes_3d[:, 0].cpu().numpy())
            _y.append(gt_instance.bboxes_3d[:, 1].cpu().numpy())
            _z.append(gt_instance.bboxes_3d[:, 2].cpu().numpy())
            
        x = np.concatenate(_x)
        y = np.concatenate(_y)
        z = np.concatenate(_z)
        
        df = pd.DataFrame({'x': x, 'y': y, 'z': z, 'assigned': gt_assigned_or_not_binary.cpu().numpy()})
        
        df.to_csv(osp.join(output_dir, 'gt_positioning.csv'), index=False)
      
    
####### Helper functions #######

def show_projections(key, gt_3d_bboxes, dt_3d_bboxes):
        
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
        
        theta = gt_box_3d[6]
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
        
        theta = dt_box_3d[6]
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

    # equal aspect rati
    ax.set_box_aspect([1, 1, 1])

    scale = vertices.flatten()
    ax.auto_scale_xyz(scale, scale, scale)
        
    # plt.show(block=True)

    return fig