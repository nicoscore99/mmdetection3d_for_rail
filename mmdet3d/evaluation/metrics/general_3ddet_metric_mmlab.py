# Copyright (c) OpenMMLab. All rights reserved.
import tempfile
import os
from os import path as osp
from typing import Dict, List, Optional, Sequence, Tuple, Union
from sklearn import metrics
from abc import ABC, abstractmethod

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
from mmdet3d.structures.bbox_3d import LiDARInstance3DBoxes, iou3d_calculator
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
                 metric: Union[str, List[str]] = ['bev', 'det3d'],
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
        self.default_prefix = 'General 3D Det metric'
        super(General_3dDet_Metric, self).__init__(collect_device=collect_device, prefix=prefix)
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
        self.metrics = metric if isinstance(metric, list) else [metric]
        self.difficulty_levels = [0.01, 0.1, 0.5]
        
        # Check that difficulty levels are not empty
        if not self.difficulty_levels:
            raise ValueError('difficulty_levels should not be empty.')

        for metric in self.metrics:
            if metric not in allowed_metrics:
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
            result = {
                'bbox_3d': data_sample['pred_instances_3d']['bboxes_3d'].tensor,
                'labels_3d': data_sample['pred_instances_3d']['labels_3d'],
                'scores_3d': data_sample['pred_instances_3d']['scores_3d']
            }
            results_dict[sample_idx] = result
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

        # Assert that the keys of the gt_annos and dt_annos are the same
        assert gt_annos.keys() == dt_annos.keys(), "The keys of the gt_annos and dt_annos are not the same."

        gt_annos_valid, perc = self.filter_valid_annos(gt_annos)
        print("Percentage of valid bounding boxes for ground truth: ", perc)

        dt_annos_valid, perc = self.filter_valid_annos(dt_annos)
        print("Percentage of valid bounding boxes for detections: ", perc)

        metric_dict = {}

        if 'bev' in self.metrics:
            bev_metric = BevMetrics(_gt_annos=gt_annos_valid,
                                  _dt_annos=dt_annos_valid,
                                  _classes_list=self.classes,
                                  _output_dir=self.output_dir)

            results_dict = bev_metric.evaluate(save_graphics=self.save_graphics, levels=self.difficulty_levels)
            metric_dict['bev'] = results_dict

        if 'det3d' in self.metrics:
            det3d_metric = Det3DMetric(_gt_annos=gt_annos_valid,
                                     _dt_annos=dt_annos_valid,
                                     _classes_list=self.classes,
                                     _output_dir=self.output_dir)

            results_dict = det3d_metric.evaluate(save_graphics=self.save_graphics, levels=self.difficulty_levels)
            metric_dict['det3d'] = results_dict

        return metric_dict
    
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
            annos_dict[sample_idx] = {
                'bbox_3d': torch.tensor(bbox_3d_lst),
                'labels_3d': torch.tensor(labels_3d_lst),
                'scores_3d': torch.tensor(scores_3d_lst)
            }
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
    
    def filter_valid_annos(self, annos: dict) -> dict:
        """
        Filter the annotations to only include the valid ones.

        Args:
            annos (dict): The annotations to be filtered.

        Returns:
            dict: Identical dict that contains only bounding boxes that have their center point within the point cloud range.

        """

        # Copy the annos dict
        annos_valid = copy.deepcopy(annos)

        # Iterate through all keys and values in the dict annos_valid
        for key in annos_valid.keys():
            bbox_3d = annos_valid[key]['bbox_3d']
            bbox_3d_center = bbox_3d[:, :3]
            valid_inds = ((bbox_3d_center[:, 0] >= self.pcd_limit_range[0]) &
                          (bbox_3d_center[:, 0] <= self.pcd_limit_range[3]) &
                          (bbox_3d_center[:, 1] >= self.pcd_limit_range[1]) &
                          (bbox_3d_center[:, 1] <= self.pcd_limit_range[4]) &
                          (bbox_3d_center[:, 2] >= self.pcd_limit_range[2]) &
                          (bbox_3d_center[:, 2] <= self.pcd_limit_range[5]))

            # Valid mask
            annos_valid[key]['bbox_3d'] = annos_valid[key]['bbox_3d'][valid_inds]
            annos_valid[key]['labels_3d'] = annos_valid[key]['labels_3d'][valid_inds]
            annos_valid[key]['scores_3d'] = annos_valid[key]['scores_3d'][valid_inds]

        # Number of bounding boxes before filtering
        num_bbox_before = sum([len(annos[key]['bbox_3d']) for key in annos.keys()])
        # Number of bounding boxes after filtering
        num_bbox_after = sum([len(annos_valid[key]['bbox_3d']) for key in annos_valid.keys()])

        percentage = round((num_bbox_after / num_bbox_before) * 100, 2)

        return annos_valid, percentage
    
class MetricEvaluation(ABC):
    def __init__(self,
                 _gt_annos: dict,
                 _dt_annos: dict,
                 _classes_list: List[str],
                 _output_dir: str):
        
        """
        
        Args:
            _gt_annos (dict): The ground truth annotations.
            _dt_annos (dict): The detected annotations.
            _classes_list (List[str]): The list of class names.
            _output_dir (str): The output directory for the evaluation results.

        """
        
        self.gt_annos = _gt_annos
        self.dt_annos = _dt_annos
        self.classes_list = _classes_list
        self.output_dir = _output_dir

    @abstractmethod
    def generate_correspondence_matrices(self):
        """
        Abstract class that implements a criterion to generate the correspondence matrices for the ground truth and the detected annotations.
        """

        pass


    def evaluate(self,
                 save_graphics: bool = False,
                 levels: List[float] = [0.01, 50.0, 70.0]):
        """

        This function evaluates the defined metric at any of the difficulty levels.

        Args:
            levels (List[float]): The difficulty levels at which the metric should be evaluated.

        Returns:
            Dict[str, float]: The results of the evaluation at the specified difficulty levels.

        """

        # Check that the levels are not empty
        if not levels:
            raise ValueError('levels should not be empty.')
        
        # Initialize the results dictionary
        results_class_unspecific = dict()
        results_class_specific = dict()

        try:
            # generate correspondence matrices
            self.generate_correspondence_matrices()

            # generate evaluation table
            evaluation_table = self.generate_evaluation_table()

            tp_curves = []
            fp_curves = []
            precision_curves = []
            recall_curves = []
            
            for level in levels:
                # Class unspecific evaluation metrics
                precision_curve, recall_curve, tp, fp = self.class_unspecific_evaluation(eval_tab=evaluation_table, _iou_threshold=level)
                tp_curves.append(tp)
                fp_curves.append(fp)
                precision_curves.append(precision_curve)
                recall_curves.append(recall_curve)     
                results_class_unspecific[f'AP_{level}'] = self.calculate_ap(recall_curve, precision_curve)

                # Class specific evaluation metrics
                map_list = []
                ap_per_class = dict()
                for cls_idx in range(len(self.classes_list)):
                    precision_curve, recall_curve, tp, fp = self.class_specific_evaluation(eval_tab=evaluation_table, _cls_idx=cls_idx, _iou_threshold=level)
                    ap = self.calculate_ap(recall_curve, precision_curve)
                    map_list.append(ap)
                    ap_per_class[f'class_{self.classes_list[cls_idx]}'] = ap

                results_class_specific[f'AP_{level}'] = ap_per_class
                results_class_specific[f'mAP_{level}'] = self.map(map_list)

            if save_graphics:
                self.save_roc_precrec_curves(y_true_positive=np.array(tp_curves), 
                                            y_score=np.array(evaluation_table['confidence']),
                                            levels=levels)
                self.save_confusion_matrix(gt_class=evaluation_table['gt_labels'], dt_class=evaluation_table['dt_labels'])

        except Exception as e:
            print(f"An error occurred during the evaluation: {e}")

        results_dict = {
            'class_unspecific': results_class_unspecific,
            'class_specific': results_class_specific
        }

        return results_dict

    def generate_evaluation_table(self):
        """
        Generate the precision-recall table for the detections.

        Returns:
            pd.DataFrame: The precision-recall table for the detections.

        Format of the evaluation table table:
        +--------+-----------+------------+------------+---------+-----------+-----------+
        | sample | detection | confidence | corr_gt_id | max_iou | dt_labels | gt_labels |
        +--------+-----------+------------+------------+---------+-----------+-----------+
        |  000   |     0     |    0.95    |     1      |  0.75   |     0     |     1     |
        """

        evaluation_table = pd.DataFrame(columns=['sample', 'detection', 'confidence', 'corr_gt_idx', 'max_iou', 'dt_labels', 'gt_labels'])
        sample = []
        detection = []
        confidence = []
        corr_gt_idx = []
        max_iou = []
        dt_labels = []
        gt_labels = []

        for key in self.dt_annos.keys():
            corr_mat = self.dt_annos[key]['corr_mat']
            sample_num_dt_annos = corr_mat.shape[0]
            sample_num_gt_annos = corr_mat.shape[1]

            # If there are multiple detections with a positive IoU with a single ground truth annotation, the maximum IoU is taken 
            # (i.e. for each column, we set all values to 0 except for the maximum value in the column)
            if not (sample_num_dt_annos == 0 or sample_num_gt_annos == 0):
                max_indices_per_column = np.argmax(corr_mat, axis=0)
                mask = np.zeros_like(corr_mat, dtype=bool)
                mask[max_indices_per_column, np.arange(corr_mat.shape[1])] = True
                corr_mat = np.where(max_indices_per_column, corr_mat, 0.0)
            
            # NOTE: Case where all detections are wrong (i.e. the row of the corr_mat is all zeros), or where all detections have an identical IoU (i.e. the row of the corr_mat is all the same value)
            # are deliberately not considered here. The IoU in the dataframe will be 0 and it will count as a FP in the precision-recall calculation.

            for i in range(sample_num_dt_annos):
                # Handle case where there is no ground truth annotation (i.e. the shape of bbox3d_gt is (0, #dt))
                if sample_num_gt_annos == 0:
                    max_iou_idx = -1
                    max_iou_value = -1.0
                else:
                    max_iou_idx = np.argmax(corr_mat[i, :])
                    max_iou_value = corr_mat[i, max_iou_idx]
                sample.append(key)
                detection.append(i)
                confidence.append(self.dt_annos[key]['scores_3d'][i].item())
                corr_gt_idx.append(max_iou_idx)
                max_iou.append(max_iou_value)
                dt_labels.append(self.dt_annos[key]['labels_3d'][i].item())

                if max_iou_idx == -1:
                    # No matching ground truth annotation
                    gt_labels.append(-1)
                else:
                    # Matching ground truth annotation
                    gt_labels.append(self.gt_annos[key]['labels_3d'][max_iou_idx].item())

        evaluation_table['sample'] = sample
        evaluation_table['detection'] = detection
        evaluation_table['confidence'] = confidence
        evaluation_table['corr_gt_idx'] = corr_gt_idx
        evaluation_table['max_iou'] = max_iou
        evaluation_table['dt_labels'] = dt_labels
        evaluation_table['gt_labels'] = gt_labels

        return evaluation_table
    
    def _acc(self, 
             lst: List[int]):
        acc_list = []
        temp_sum = 0
        for i in lst:
            temp_sum += i
            acc_list.append(temp_sum)
        return acc_list
    
    def _prec(self,
              _acc_tp: List[int], 
              _acc_fp: List[int]):
        temp_list = []
        for i in range(len(_acc_tp)):
            temp_list.append(_acc_tp[i] / (i+1))
        return np.array(temp_list)

    def _rec (self, 
              _acc_tp: List[int],
              _acc_fp: List[int]):
        return np.array(_acc_tp) / len(_acc_tp)

    def class_unspecific_evaluation(self,
                                    eval_tab: pd.DataFrame,
                                    _iou_threshold: float):
        """
        Evaluates the evaluation table for all classes and a specific IoU threshold (correct classification is not a criterium for the evaluation).
        
        Args:
            _iou_threshold (float): The IoU threshold to be used for the evaluation.
        
        """

        # _iou_threshold needs to be float
        if not isinstance(_iou_threshold, float):
            raise ValueError('The IoU threshold has to be a float.')

        # copy evaluation table
        evaluation_table = eval_tab.copy()

        above_threshold = evaluation_table['max_iou'] > _iou_threshold
        correct_predictions = above_threshold
        false_predictions = np.logical_not(above_threshold)

        accumulated_correct_predictions = self._acc(correct_predictions)
        accumulated_false_predictions = self._acc(false_predictions)

        precision_curve = self._prec(accumulated_correct_predictions, accumulated_false_predictions)
        recall_curve = self._rec(accumulated_correct_predictions, accumulated_false_predictions)

        return precision_curve, recall_curve, correct_predictions, false_predictions

    def class_specific_evaluation(self, 
                                  eval_tab: pd.DataFrame,
                                  _cls_idx: int,
                                  _iou_threshold: float):
        """
        Evaluates the evaluation table for a specific class and a specific IoU threshold.
        
        Args:
            _cls_idx (int): The class index to be evaluated.
            _iou_threshold (float): The IoU threshold to be used for the evaluation.
        
        """

        # _iou_threshold needs to be float, and _cls_idx needs to be int
        if not isinstance(_iou_threshold, float):
            raise ValueError('The IoU threshold has to be a float.')
        if not isinstance(_cls_idx, int):
            raise ValueError('The class index has to be an integer.')

        # copy evaluation table
        evaluation_table = eval_tab.copy()
        evaluation_table = evaluation_table[(evaluation_table['dt_labels'] == _cls_idx) | (evaluation_table['gt_labels'] == _cls_idx)]

        above_threshold = evaluation_table['max_iou'] > _iou_threshold
        correctly_classified = (evaluation_table['gt_labels'] == _cls_idx) & (evaluation_table['dt_labels'] == _cls_idx)
        correct_predictions = above_threshold & correctly_classified
        false_predictions = above_threshold & np.logical_not(correctly_classified)

        accumulated_correct_predictions = self._acc(correct_predictions)
        accumulated_false_predictions = self._acc(false_predictions)

        precision_curve = self._prec(accumulated_correct_predictions, accumulated_false_predictions)
        recall_curve = self._rec(accumulated_correct_predictions, accumulated_false_predictions)

        return precision_curve, recall_curve, correct_predictions, false_predictions
    
    def calculate_ap(self, recall_curve, precision_curve):

        # Check type of recall and precision curve is list of type float
        if not isinstance(recall_curve, np.ndarray):
            raise ValueError('The recall curve has to be a numpy array.')
        if not isinstance(precision_curve, np.ndarray):
            raise ValueError('The precision curve has to be a numpy array.')

        # first append sentinel values at the end
        mrec = np.concatenate(([0.0], recall_curve, [1.0]))
        mpre = np.concatenate(([0.0], precision_curve, [0.0]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap

    def map(self, ap_list: List[float]) -> float:
        """
        Calculate the mean average precision (mAP) from a list of average precisions (AP).

        Args:
            ap_list (List[float]): The list of average precisions.

        Returns:
            float: The mean average precision (mAP).
        """

        # Ensure correct class list lenght
        if len(ap_list) != len(self.classes_list):
            raise ValueError('The length of the average precision list has to be the same as the length of the classes list.')

        return np.mean(ap_list)

    def save_roc_precrec_curves(self,
                                y_true_positive: np.ndarray,
                                y_score: np.ndarray,
                                levels: List[float]):
        
        """

        Generates a ROC curve and a precision-recall curve for the detections at every difficulty level (class unspecific).

        Args:
            true_positives (np.ndarray): The true positives for the detections.
            false_positives (np.ndarray): The false positives for the detections.
            recall_curves (np.ndarray): The recall curves for the detections.
            precision_curves (np.ndarray): The precision curves for the detections.
            levels (List[float]): The difficulty levels at which the curves should be generated.

        """

        # assert that the lengths of y_true_positive and y_score are the same
        if y_true_positive.shape[1] != len(y_score):
            raise ValueError('The lengths of the true positives and the scores are not the same.')

        # subplot with two plots
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        axs[0].set_title('ROC Curve')
        axs[0].set_xlabel('False Positive Rate')
        axs[0].set_ylabel('True Positive Rate')
        axs[1].set_title('Precision-Recall Curve')
        axs[1].set_xlabel('Recall')
        axs[1].set_ylabel('Precision')

        for level in range(len(levels)):

            fpr, tpr, _ = metrics.roc_curve(y_true=y_true_positive[level], y_score=y_score)
            auc = metrics.roc_auc_score(y_true=y_true_positive[level], y_score=y_score)
            precision, recall, _ = metrics.precision_recall_curve(y_true=y_true_positive[level], probas_pred=y_score)

            axs[0].plot(fpr, tpr, label=f'ROC IoU {levels[level]} (AUC = {auc:.2f})')
            axs[1].plot(recall, precision, label=f'Precision-Recall IoU {levels[level]}')

        axs[0].legend()
        axs[1].legend()

        plt.savefig(osp.join(self.output_dir, self.token + '_roc_precrec_curves.png'))      

    def save_confusion_matrix(self,
                              gt_class: List[int],
                              dt_class: List[int]):
        """
        Generate the confusion matrix for the detections.

        Args:
            iou_threshold (float): The IoU threshold to be used for the calculation.

        Returns:
            np.ndarray: The confusion matrix for the detections.
        """

        # Check that the classes have equal length
        if len(gt_class) != len(dt_class):
            raise ValueError('The length of the ground truth classes and the detected classes is not the same.')
        
        # Eliminate all dt_classes or gt_classes that are -1 (i.e. no matching ground truth annotation)
        gt_class = np.array(gt_class)
        dt_class = np.array(dt_class)
        mask = np.logical_and(gt_class != -1, dt_class != -1)
        gt_class = gt_class[mask]
        dt_class = dt_class[mask]

        gt_class = [self.classes_list[i] for i in gt_class]
        dt_class = [self.classes_list[i] for i in dt_class]

        # Create the confusion matrix
        confusion_matrix_display = metrics.ConfusionMatrixDisplay.from_predictions(y_true=gt_class, y_pred=dt_class)
        # save the confusion matrix
        plot = confusion_matrix_display.plot(include_values=True, cmap='viridis', ax=None, xticks_rotation='horizontal')
        plt.savefig(osp.join(self.output_dir, self.token + '_confusion_matrix.png'))

class Det3DMetric(MetricEvaluation):
    def __init__(self,
                 _gt_annos: dict,
                 _dt_annos: dict,
                 _classes_list: List[str],
                 _output_dir: str):
        
        super(Det3DMetric, self).__init__(_gt_annos, _dt_annos, _classes_list, _output_dir)
        self.token = 'det3d'

    def generate_correspondence_matrices(self):
        """
        Generate the correspondence matrices for the ground truth and the detected annotations.
        """
        
        # Iterate through the detections
        for key in self.dt_annos.keys():
            
            bbox3d_dt = self.dt_annos[key]['bbox_3d']
            bbox3d_gt = self.gt_annos[key]['bbox_3d']

            sample_num_dt_annos = bbox3d_dt.shape[0]
            sample_num_gt_annos = bbox3d_gt.shape[0]

            # Create a matrix with the number of detected annotations as rows and the number of ground truth annotations as columns
            correspondence_matrix = np.zeros((sample_num_dt_annos, sample_num_gt_annos))

            # Iterate through the detected annotations
            for i in range(sample_num_dt_annos):
                
                # NOTE: Not completely certain that this is correct
                dt_bbox = bbox.BBox3D(x=bbox3d_dt[i, 0], y=bbox3d_dt[i, 1], z=bbox3d_dt[i, 2],
                                      height=bbox3d_dt[i, 3], width=bbox3d_dt[i, 4], length=bbox3d_dt[i, 5],
                                      euler_angles=[0,0, bbox3d_dt[i, 6]], is_center=True)

                # Iterate through the ground truth annotations
                for j in range(sample_num_gt_annos):
                    
                    gt_bbox = bbox.BBox3D(x=bbox3d_gt[j, 0], y=bbox3d_gt[j, 1], z=bbox3d_gt[j, 2],
                                            height=bbox3d_gt[j, 3], width=bbox3d_gt[j, 4], length=bbox3d_gt[j, 5],
                                            euler_angles=[0,0, bbox3d_gt[j, 6]], is_center=True)

                    _3d_jaccard = bbox.metrics.iou_3d(dt_bbox, gt_bbox)

                    correspondence_matrix[i, j] = _3d_jaccard

            self.dt_annos[key]['corr_mat'] = correspondence_matrix

class BevMetrics(MetricEvaluation):
    def __init__(self,
                 _gt_annos: dict,
                 _dt_annos: dict,
                 _classes_list: List[str],
                 _output_dir: str):
        
        super(BevMetrics, self).__init__(_gt_annos, _dt_annos, _classes_list, _output_dir)
        self.token = 'bev'

    def generate_correspondence_matrices(self):
        """
        Generate the correspondence matrices for the ground truth and the detected annotations.
        """
        
        # Iterate through the detections
        for key in self.dt_annos.keys():
            
            bbox3d_dt = self.dt_annos[key]['bbox_3d']
            bbox3d_gt = self.gt_annos[key]['bbox_3d']

            sample_num_dt_annos = bbox3d_dt.shape[0]
            sample_num_gt_annos = bbox3d_gt.shape[0]

            # Create a matrix with the number of detected annotations as rows and the number of ground truth annotations as columns
            correspondence_matrix = np.zeros((sample_num_dt_annos, sample_num_gt_annos))

            # Iterate through the detected annotations
            for i in range(sample_num_dt_annos):

                dt_xywh = (bbox3d_dt[i, 0], bbox3d_dt[i, 1], bbox3d_dt[i, 3], bbox3d_dt[i, 4])
                dt_bbox = bbox.BBox2D(dt_xywh, mode=0)

                # Iterate through the ground truth annotations
                for j in range(sample_num_gt_annos):
                    
                    gt_xywh = (bbox3d_gt[j, 0], bbox3d_gt[j, 1], bbox3d_gt[j, 3], bbox3d_gt[j, 4])
                    gt_bbox = bbox.BBox2D(gt_xywh, mode=0)

                    _2d_jaccard = bbox.metrics.iou_2d(dt_bbox, gt_bbox)

                    correspondence_matrix[i, j] = _2d_jaccard

            self.dt_annos[key]['corr_mat'] = correspondence_matrix

