# Copyright (c) OpenMMLab. All rights reserved.
import tempfile
import os
from os import path as osp
from typing import Dict, List, Optional, Sequence, Tuple, Union
from sklearn import metrics
from abc import ABC, abstractmethod

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
from mmengine.structures import InstanceData

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
                 pcd_limit_range: List[float] = [0, -39.68, -3, 69.12, 39.68, 20],
                 prefix: Optional[str] = None,
                 pklfile_prefix: Optional[str] = None,
                 default_cam_key: str = 'CAM2',
                 format_only: bool = False,
                 submission_prefix: Optional[str] = None,
                 collect_device: str = 'cpu',
                 save_graphics: Optional[bool] = False,
                 classes: Optional[List[str]] = None,
                 output_dir: Optional[str] = None,
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

        if self.format_only:
            assert submission_prefix is not None, 'submission_prefix must be '
            'not None when format_only is True, otherwise the result files '
            'will be saved to a temp directory which will be cleaned up at '
            'the end.'

        # '3d' and 'bev' stand for '3-dimensional object detction metrics' and 'bird's eye view object detection metrics'
        allowed_metrics = ['det3d', 'bev']
        # self.metrics = metric if isinstance(metric, list) else [metric]
        # For now, metric should just be one. Don't select both.
        self.metric = metric

        self.difficulty_levels = [0.1, 0.3, 0.5]
        
        # Check that difficulty levels are not empty
        if not self.difficulty_levels:
            raise ValueError('difficulty_levels should not be empty.')

        # for metric in self.metrics:
        #     if metric not in allowed_metrics:
        #         raise KeyError("metric should be one of {allowed_metrics} but got {metric}.")

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
        """Process one batch of data samples and predictions.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.


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
        for data_sample in data_samples:
            results_dict = dict()
            sample_idx = data_sample['lidar_path'].split('/')[-1][-7:-4]
            dt_bboxes = InstanceData()
            dt_bboxes.bboxes_3d = data_sample['pred_instances_3d']['bboxes_3d']
            dt_bboxes.labels_3d = data_sample['pred_instances_3d']['labels_3d']
            dt_bboxes.scores = data_sample['pred_instances_3d']['scores_3d']
            results_dict[sample_idx] = dt_bboxes

            self.results.append(results_dict)

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

        # self.classes = self.dataset_meta['classes']

        # load annotations
        ann_file = load(self.ann_file, backend_args=self.backend_args)
        gt_annos = self.convert_from_ann_file(ann_file)
        dt_annos = self.convert_from_results(results)

        # print("Debug: gt_annos: ", gt_annos)
        # print("Debug: dt_annos: ", dt_annos)

        # Assert that the keys of the gt_annos and dt_annos are the same
        assert gt_annos.keys() == dt_annos.keys(), "The keys of the gt_annos and dt_annos are not the same."

        gt_annos_valid, perc = self.filter_valid_gt_annos(gt_annos)
        print("Percentage of valid bounding boxes for ground truth: ", perc)

        dt_annos_valid, perc = self.filter_valid_dt_annos(dt_annos)
        print("Percentage of valid bounding boxes for detections: ", perc)

        evaluation_results_dict = dict() # Contains all evaluation results
        curves_dict = dict() # Contains all curves

        self.evaluator = EvaluatorMetrics(gt_annos_valid=gt_annos_valid, dt_annos_valid=dt_annos_valid, classes=self.classes, metric=self.metric)

        # Only perform the evaluation when there is at least 10 gt instances and 10 dt instances
        
        if not self.evaluator.total_gt_instances >= 10:
            print_log("The number of gt instances is less than 10, skip evaluation.")
            return {'evaluations': evaluation_results_dict, 'curves': curves_dict}
        
        if not self.evaluator.total_dt_instances >= 10:
            print_log("The number of dt instances is less than 10, skip evaluation.")
            return {'evaluations': evaluation_results_dict, 'curves': curves_dict}
        
        evaluation_results_dict['mAP'] = self.evaluator.sklearn_mean_average_precision_score(iou_level=self.difficulty_levels,
                                                                                         class_accuracy_requirements='hard')

        evaluation_results_dict['AP'] = self.evaluator.sklearn_average_precision_score(iou_level=self.difficulty_levels,
                                                                                       class_accuracy_requirements='hard',
                                                                                       class_idx=range(len(self.classes)))

        level_lower_bound = self.difficulty_levels[0]



        curves_dict['prec'] = self.evaluator.precision_recall_curve(iou_level=level_lower_bound)
        curves_dict['roc'] = self.evaluator.roc_curve(iou_level=level_lower_bound)
        curves_dict['cm'] = self.evaluator.confusion_matrix(iou_level=level_lower_bound)

        if self.save_graphics:
            self.save_plot(plot=self.evaluator.precision_recall_plot(iou_level=0.5), filename = 'precision_recall_plot.png')
            self.save_plot(plot=self.evaluator.roc_plot(iou_level=0.5), filename = 'roc_plot.png')
            self.save_plot(plot=self.evaluator.confusion_matrix_plot(iou_level=0.5), filename = 'confusion_matrix_plot.png')
            
        return {'evaluations': evaluation_results_dict, 'curves': curves_dict}
    
    def convert_from_ann_file(self, ann_file: dict) -> dict:
        """
        Convert the annotations from the ann_file to the required format for the evaluation.

        Args:
            ann_file (dict): The loaded annotation file.

        Returns:
            dict: The converted annotations.

        The ann_file has the following format:
        {'instances': [{'alpha': 0.0,
                'bbox': [-1.0, -1.0, -1.0, -1.0],
                'bbox_3d': [139.85,
                            14.59,
                            1.0699999999999998,
                            0.98,
                            1.0,
                            1.8,
                            0.0],
                'bbox_label': 0,
                'bbox_label_3d': 0,
                'difficulty': -1,
                'group_id': 0,
                'index': 0,
                'occluded': 0,
                'score': 0.0,
                'truncated': 0.0},
                {'alpha': 0.0,
                'bbox': [-1.0, -1.0, -1.0, -1.0],
                'bbox_3d': [230.78,
                            -4.51,
                            -0.08999999999999986,
                            1.6,
                            1.3,
                            11.9,
                            0.0],
                'bbox_label': 5,
                'bbox_label_3d': 5,
                'difficulty': -1,
                'group_id': 8,
                'index': 8,
                'occluded': 0,
                'score': 0.0,
                'truncated': 0.0}],
         'lidar_points': {'lidar_path': '3_fire_site_3.1_083.bin',
                          'num_pts_feats': 4},
         'sample_idx': '083.bin'},

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

        Final output format:
        {'_sample_idx_': InstanceData}

        """
        
        annos_dict = dict()
        for i, anno in enumerate(ann_file['data_list']):
            # NOTE: This should be corrected and in the OSDAR23 conversion script: Use sample_idx without the .bin
            # For now, only use the first three characters of the sample_idx
            sample_idx = anno['lidar_points']['lidar_path'][-7:-4]
            bbox_3d_lst = []
            labels_3d_lst = []
            scores_3d_lst = []
            for instance in anno['instances']:
                bbox_3d_lst.append(instance['bbox_3d'])
                labels_3d_lst.append(instance['bbox_label'])
                scores_3d_lst.append(instance['score']) # Currently, this value is unnenecessary and will just be 0.00 since it is not implemented 
           
            gt_instance = InstanceData()
            gt_instance.bboxes_3d = torch.tensor(bbox_3d_lst)
            gt_instance.labels_3d = torch.tensor(labels_3d_lst)
            gt_instance.scores = torch.tensor(scores_3d_lst)        
            annos_dict[sample_idx] = gt_instance

        return annos_dict
    
    def convert_from_results(self, results: List[dict]) -> dict:
        """
        
        Convert the results from the model to the required format for the evaluation. Since the processing is already done in 
        the process method, the results are already in the required format. Only step left is to concatinate the list into a
        single dict.

        Args:
            results (List[dict]): The processed results of the whole dataset.

        Returns:
            dict: The converted results.
        
        """
        merged_dict = {k: v for d in results for k, v in d.items()}
        return merged_dict
    
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
            _bbox_3d = annos[key].bboxes_3d
            
            # Check if any dimension is empty
            if _bbox_3d.size(0) == 0:
                continue
            elif _bbox_3d.size(1) == 0:
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
        # Devision by zero check
        if num_bbox_before == 0:
            percentage = 0.0
        else:
            percentage = round((num_bbox_after / num_bbox_before) * 100, 2)
            
        return annos_valid, percentage
    
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
            _bbox_3d = annos[key].bboxes_3d.tensor
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
        
        for key in self._gt_annos_valid.keys():
            gt_instance = self._gt_annos_valid[key]
            dt_instance = self._dt_annos_valid[key]
            assign_result = iou_assigner.assign(pred_instances=dt_instance, gt_instances=gt_instance)

            # dt_labels.append(dt_instance.labels_3d.tolist())
            # dt_scores.append(dt_instance.scores.tolist())   
            # assigned_gt_inds.append(assign_result.gt_inds.tolist())
            # assigned_max_overlaps.append(assign_result.max_overlaps.tolist())
            # assigned_labels.append(assign_result.labels.tolist())

            # Maybe something is not fully right here: Does it automatically mean that the labels correspond to the gt_labels?

            dt_labels.append(dt_instance.labels_3d)
            dt_scores.append(dt_instance.scores)
            assigned_gt_inds.append(assign_result.gt_inds)
            assigned_max_overlaps.append(assign_result.max_overlaps)
            assigned_labels.append(assign_result.labels)

        # assert that all out the lists have the same length
        assert len(dt_labels) == len(dt_scores) == len(assigned_gt_inds) == len(assigned_max_overlaps) == len(assigned_labels)

        threshold_specific_results = {
            'dt_labels': torch.cat(dt_labels),
            'confidence': torch.cat(dt_scores),
            'assigned_gt_inds': torch.cat(assigned_gt_inds),
            'assigned_max_overlaps': torch.cat(assigned_max_overlaps),
            'assigned_labels': torch.cat(assigned_labels)
        }

        # print("Debug: threshold_specific_results: ", threshold_specific_results)

        # # print("Debug: threshold_specific_results: ", threshold_specific_results)

        # # save these values into a file
        # filename = 'threshold_specific_results.txt'
        # with open(filename, 'w') as f:
        #     for value in threshold_specific_results.values():
        #         val_str = ', '.join(map(str, value.tolist()))
        #         f.write(val_str + '\n')

        # raise NotImplementedError

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
            Tuple[torch.Tensor, torch.Tensor]: A tuple that contains the true positives for the detection labels and the corresponding scores.

        ***Attention***: True positives are defined here as detections, that have a valid, corresponding bounding box AND the correct label.
        
        """
        
        tp_criterion = filtered_dict['dt_labels'] == filtered_dict['assigned_labels']
        tp = torch.where(tp_criterion, torch.tensor(1), torch.tensor(0))
        score = filtered_dict['confidence']
        
        return tp, score

    def tp_detection(self,
                     filtered_dict: dict) -> torch.Tensor:
        """
        
        Args:
            filtered_dict (dict): A dict that contains the results that were, if this is wanted, already filtered for detection labels.

        Returns:
            torch.Tensor: A tensor that contains the true positives for the detection labels.

        ***Attention***: True positives are defined here as detections, that have a valid, corresponding bounding box (but not 
        necessarily the correct label).
            
        """
        
        tp_criterion = filtered_dict['assigned_gt_inds'] > 0
        tp = torch.where(tp_criterion, torch.tensor(1), torch.tensor(0))
        score = filtered_dict['confidence']

        return tp, score    

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
                    y_pred, y_score = self.class_accuracy_requirement_map[class_accuracy_requirement](filtered_dict=_class_filtered_dict)
                    if y_pred.nelement() > 0 and y_score.nelement() > 0:
                        class_dict[self._classes[cls_idx]] = round(sk_metrics.average_precision_score(y_true=y_pred, y_score=y_score), 3)
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
                y_pred, y_score = self.class_accuracy_requirement_map[class_accuracy_requirement](filtered_dict=_threshold_specific_results_dict)
                if y_pred.nelement() > 0 and y_score.nelement() > 0:
                    class_dict[class_accuracy_requirement] = round(sk_metrics.average_precision_score(y_true=y_pred, y_score=y_score), 3)
                else:
                    class_dict[class_accuracy_requirement] = 0.0                
            level_dict[level] = class_dict

        return level_dict
    
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

            # class_dict = dict()
            # if class_idx:
            #     for cls_idx in class_idx:
            #         _class_filtered_dict = self.filter_for_prediction_class(filter_dict=_threshold_specific_results_dict, class_idx=cls_idx)
            #         y_pred, y_score = self.tp_detection(filtered_dict=_class_filtered_dict)
            #         precision, recall, _ = sk_metrics.precision_recall_curve(y_true=y_pred, probas_pred=y_score)

            #         class_dict[self._classes[cls_idx]] = {
            #             'precision': precision,
            #             'recall': recall,
            #             'y_true': y_pred,
            #             'y_probas': y_score,
            #             'labels': self._classes
            #         }
            # else:
            #     y_pred, y_score = self.tp_detection(filtered_dict=_threshold_specific_results_dict)
            #     precision, recall, _ = sk_metrics.precision_recall_curve(y_true=y_pred, probas_pred=y_score)
            #     class_dict['all_classes'] = {
            #         'precision': precision,
            #         'recall': recall,
            #         'y_true': y_pred,
            #         'y_probas': y_score,
            #         'labels': self._classes
            #     }
        
            # level_dict[level] = class_dict

            class_dict = dict()
            y_pred, y_score = self.tp_detection(filtered_dict=_threshold_specific_results_dict)
            precision, recall, _ = sk_metrics.precision_recall_curve(y_true=y_pred, probas_pred=y_score)
            class_dict['all_classes'] = {
                'precision': precision,
                'recall': recall,
                'y_true': y_pred,
                'y_probas': y_score,
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

            # class_dict = dict()
            # if class_idx:
            #     for cls_idx in class_idx:
            #         _class_filtered_dict = self.filter_for_prediction_class(filter_dict=_threshold_specific_results_dict, class_idx=cls_idx)
            #         y_pred, y_score = self.tp_detection(filtered_dict=_class_filtered_dict)
            #         fpr, tpr, _ = sk_metrics.roc_curve(y_true=y_pred, y_score=y_score)
            #         class_dict[self._classes[cls_idx]] = {
            #             'fpr': fpr,
            #             'tpr': tpr,
            #             'y_true': y_pred,
            #             'y_probas': y_score,
            #             'labels': self._classes
            #         }

            #     level_dict[level] = class_dict
            # else:
            #     y_pred, y_score = self.tp_detection(filtered_dict=_threshold_specific_results_dict)
            #     fpr, tpr, _ = sk_metrics.roc_curve(y_true=y_pred, y_score=y_score)
            #     class_dict['all_classes'] = {
            #         'fpr': fpr,
            #         'tpr': tpr,
            #         'y_true': y_pred,
            #         'y_probas': y_score,
            #         'labels': self._classes
            #     }
        
            #     level_dict[level] = class_dict

            class_dict = dict()

            y_pred, y_score = self.tp_detection(filtered_dict=_threshold_specific_results_dict)
            fpr, tpr, _ = sk_metrics.roc_curve(y_true=y_pred, y_score=y_score)
            class_dict['all_classes'] = {
                'fpr': fpr,
                'tpr': tpr,
                'y_true': y_pred,
                'y_probas': y_score,
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
