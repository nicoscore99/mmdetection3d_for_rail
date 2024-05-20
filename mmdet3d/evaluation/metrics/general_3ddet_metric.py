# Copyright (c) OpenMMLab. All rights reserved.
import tempfile
from os import path as osp
from typing import Dict, List, Optional, Sequence, Tuple, Union

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

@METRICS.register_module()
class General_3dDet_Metric(BaseMetric):
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
                 metric: Union[str, List[str]] = [''],
                 pcd_limit_range: List[float] = [0, -40, -3, 70.4, 40, 0.0],
                 prefix: Optional[str] = None,
                 pklfile_prefix: Optional[str] = None,
                 default_cam_key: str = 'CAM2',
                 format_only: bool = False,
                 submission_prefix: Optional[str] = None,
                 collect_device: str = 'cpu',
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

        if self.format_only:
            assert submission_prefix is not None, 'submission_prefix must be '
            'not None when format_only is True, otherwise the result files '
            'will be saved to a temp directory which will be cleaned up at '
            'the end.'

        # '3d' and 'bev' stand for '3-dimensional object detction metrics' and 'bird's eye view object detection metrics'
        allowed_metrics = ['det3d', 'bev']
        self.metrics = metric if isinstance(metric, list) else [metric]
        self.difficulty_levels = [0.01, 50.0, 70.0]
        
        # Check that difficulty levels are not empty
        if not self.difficulty_levels:
            raise ValueError('difficulty_levels should not be empty.')

        for metric in self.metrics:
            if metric not in allowed_metrics:
                raise KeyError("metric should be one of 'bbox', 'img_bbox', "
                               f'but got {metric}.')

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

        gt_annos_valid = self.filter_valid_annos(gt_annos)
        dt_annos_valid = self.filter_valid_annos(dt_annos)

        metric_dict = {}

        if 'bev' in self.metrics:
            bev_dict = dict()
            bev_metrics = BevMetrics(gt_annos_valid, dt_annos_valid)
            bev_metrics.generate_correspondence_matrices()
            bev_metrics.generate_prec_rec_table()

            for difficulty in self.difficulty_levels:
                bev_dict[f'bev_{difficulty}'] = bev_metrics.calculate_pascal_voc_ap_by_aoc(difficulty)

            metric_dict['bev'] = bev_dict

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

        return annos_valid

class BevMetrics():
    def __init__(self,
                 _gt_annos,
                 _dt_annos):
        
        self.gt_annos = _gt_annos
        self.dt_annos = _dt_annos

        self.generate_correspondence_matrices()
        self.prec_rec_table = self.generate_prec_rec_table()


    def generate_correspondence_matrices(self):
        """
        Generate the correspondence matrices for the ground truth and the detected annotations.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The correspondence matrices for the ground truth and the detected annotations.
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

    def generate_prec_rec_table(self):
        """
        Generate the precision-recall table for the detections.

        Returns:
            pd.DataFrame: The precision-recall table for the detections.


        Format of the precision-recall table:
        +--------+-----------+------------+------------+---------+
        | sample | detection | confidence | corr_gt_id | max_iou |
        +--------+-----------+------------+------------+---------+
        |  000   |     0     |    0.95    |     1      |  0.75   |
        """

        prec_rec_table = pd.DataFrame(columns=['sample', 'detection', 'confidence', 'corr_gt_idx', 'max_iou'])
        sample = []
        detection = []
        confidence = []
        corr_gt_idx = []
        max_iou = []

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
                confidence.append(self.dt_annos[key]['scores_3d'][i])
                corr_gt_idx.append(max_iou_idx)
                max_iou.append(max_iou_value)

        prec_rec_table['sample'] = sample
        prec_rec_table['detection'] = detection
        prec_rec_table['confidence'] = confidence
        prec_rec_table['corr_gt_idx'] = corr_gt_idx
        prec_rec_table['max_iou'] = max_iou

        return prec_rec_table
    
    def _acc(self, list):
        acc_list = []
        temp_sum = 0
        for i in list:
            temp_sum += i
            acc_list.append(temp_sum)
        return acc_list
    
    def _prec(self, _acc_tp, _acc_fp):
        temp_list = []
        for i in range(len(_acc_tp)):
            temp_list.append(_acc_tp[i] / (i+1))
        return temp_list

    def _rec (self, _acc_tp, _acc_fp):
        return _acc_tp / len(_acc_fp)
    
    def add_tp_fp_acctp_accfp_prec_rec(self, iou_threshold):

        if 'tp'+str(iou_threshold) in self.prec_rec_table.columns:
            raise ValueError(f'The columns tp{str(iou_threshold)} and fp{str(iou_threshold)} already exist in the precision-recall table.')

        # True positive list is list that contains 1 where max_iou is above the iou_threshold and 0 otherwise
        tp_list = self.prec_rec_table['max_iou'] > iou_threshold
        fp_list = np.logical_not(tp_list)

        # Add the true positive and false positive lists to the precision-recall table
        self.prec_rec_table['tp'+str(iou_threshold)] = tp_list
        self.prec_rec_table['fp'+str(iou_threshold)] = fp_list

        # Sort table by confidence
        self.prec_rec_table = self.prec_rec_table.sort_values(by='confidence', ascending=False)

        # Add the accumulated true positive and false positive lists to the precision-recall table
        self.prec_rec_table['acc_tp'+str(iou_threshold)] = self._acc(tp_list)
        self.prec_rec_table['acc_fp'+str(iou_threshold)] = self._acc(fp_list)

        # Calculate the precision and recall
        self.prec_rec_table['prec'+str(iou_threshold)] = self._prec(self.prec_rec_table['acc_tp'+str(iou_threshold)], self.prec_rec_table['acc_fp'+str(iou_threshold)])
        self.prec_rec_table['rec'+str(iou_threshold)] = self._rec(self.prec_rec_table['acc_tp'+str(iou_threshold)], self.prec_rec_table['acc_fp'+str(iou_threshold)])

    def calculate_pascal_voc_ap_by_aoc(self, iou_threshold):
        """
        Calculate the Pascal VOC AP for the detections.

        PASCAL VOC 2012 challenge uses the interpolated average precision. It tries to summarize the shape of the Precision x Recall curve by averaging the precision at 
        a set of eleven equally spaced recall levels [0, 0.1, 0.2, … , 1]. This was later extended to 40 equally spaced recall levels [0, 0.025, 0.05, … , 1].

        Instead of using interpolated AP as suggested for VOC originally, we will use the area over the curve (AOC) to calculate the AP. Advantages: More precision, capability to
        compage methods with low AP. Methode copied from: https://github.com/wang-tf/pascal_voc_tools/blob/master/pascal_voc_tools/Evaluater/tools.py

        Args:
            iou_threshold (float): The IoU threshold to be used for the calculation.

        Returns:
            float: The Pascal VOC AP for the detections.
        """

        # Add the true positive and false positive lists if it does not yet exist
        if 'tp'+str(iou_threshold) not in self.prec_rec_table.columns:
            self.add_tp_fp_acctp_accfp_prec_rec(iou_threshold)
            
        # first append sentinel values at the end
        mrec = np.concatenate(([0.0], self.prec_rec_table['rec'+str(iou_threshold)], [1.0]))
        mpre = np.concatenate(([0.0], self.prec_rec_table['prec'+str(iou_threshold)], [0.0]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap
    
    def calculate_pascal_voc_ap_by_interp(self, iou_threshold):
        raise NotImplementedError("This method is not yet implemented.")
    
    def calculate_pascal_voc_map(self):
        raise NotImplementedError("This method is not yet implemented. Use calculate_pascal_voc_ap_by_aoc instead.")
    
class Det3dMetrics():
    def __init__(self):
        pass

    def generate_correspondence_matrices(self):
        raise NotImplementedError("This method is not yet implemented.")

