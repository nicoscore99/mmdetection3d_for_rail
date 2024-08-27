# MMDetection3D_for_rail

Welcome to the MMDetection3D_for_rail package. This package was forked from the original [MMDetection3D](https://github.com/open-mmlab/mmdetection3d) package developped by [OpenMMLab](https://github.com/open-mmlab). The fork was created with the intention of testing and deploying LiDAR point cloud-based object detection algrorithms that perform well in railway-specific environments, with the vision in mind of using ML algorithms for detection objects in the context of automatic train operations.

## Environment Setup

The changes that were made during this project to the original MMDetection3D package were developed and tested in the follwing environment:

* __Operating System:__ Ubuntu 22.04
* __NVIDIA Driver Version:__ 535.183.01
* __CUDA Toolbox:__ cuda_11.8
* __Pytorch:__ 2.3.1+cu118
* __Torchvision:__ 0.18.1+cu118

The [MMDetection3D](https://github.com/open-mmlab/mmdetection3d) package is closely integrated with [MMEngine](https://github.com/open-mmlab/mmdetection3d), which acts as a foundational library for model development and training. This project required to make some changes to the MMengine library as well. Thus, to make this package work you need to use my fork of the MMengine library, which can be found here: [MMEngine_for_rail](https://github.com/nicoscore99/mmengine_for_rail)


## Relevant Changes

The following is a list of the relevant changes that I have made on top of the [MMDetection3D](https://github.com/open-mmlab/mmdetection3d) package:

### Custom Datasets
For this project, the **Open Sensor Data For Rail 2023** (subsequently *OSDaR23*) dataset was integrated as a custom dataset. Find more information on the OS[OSDaR23 dataset](https://data.fid-move.de/dataset/osdar23). This happens by conversion of the OSDaR23 data fromat into the format of the KITTI dataset such that for all subsequent tasks (visualizations, training, testing, etc.) the standard MMDetection3D functionality can be used. See here how OSDaR23 can be integrated.

The same process can be used for any 3D data labeled with the web-based labelling service [Segments.ai](https://segments.ai/). Using the MMDetection3D package functionality on self-labeled data from Segments.ai required converting the .json release-file into KITTI type labels. See this script to do so.

### 3D Detection Evaluation Metric
The 3D evaluation metrics provided by MMDetection3D could not be adopted for the custom datasets used during this project. Thus, for this project a new evaluation script that is dataset-independent and compatible with both the per-default integrated datasets as well as any custom datasets. The [General_3dDet_Metric_MMLab](mmdet3d/evaluation/metrics/general_3ddet_metric_mmlab.py) class is set up to evaluate Mean Average Precision (mAP), Precision and Recall over the complete test set. Per default, it evaluates the mAP at 40 reall positions, as suggested by the KITTI dataset challenge.

### New WandB Logger Hook
To gain oversight over the trained models and the training process, a [Weights&Biases](https://wandb.ai) integration for MMDetection3D was created. This [script](/mmdet3d/engine/hooks/wandb_logger_hook.py) implements a registered hook that logs the training progress per iteration and also per epoch. Addtionally, the WandB logger hook also logs the metrics from the 3D-detection evaluation metrics described above along with the graphs created by that script. In order to access the validation losses, a few changes were made to the MMEngine foundational library. These are elaborated in the corresponding package ([MMEngine_for_rail](https://github.com/nicoscore99/mmengine_for_rail))

### Pointcloud Classification
I developed a pipeline for the classification of 3D point clouds. The pipeline uses the grund truth data that is generated when any dataset is converted using the [dataset conversion script](tools/dataset_converters/create_gt_database.py). The ground truth label file can be generated using another conversion script, that takes the .pkl files for the training sets and the ground truth dataset as an input (see [create_gt_dbinfos.py](/home/cws-ml-lab/mmdetection3d_for_rail/tools/dataset_converters/create_gt_dbinfos.py)).

The point cloud classification pipeline is based on the default implementation of Pointnet++ provided by MMDetection. Instead of an Encoder-Decoder Network, only the encoder is used and a classification head is added ontop of the encoder.