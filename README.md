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
For this project, the **Open Sensor Data For Rail 2023** (subsequently *OSDaR23*) dataset was integrated as a custom dataset. Find more information on the [OSDaR23 dataset](https://data.fid-move.de/dataset/osdar23). This happens by conversion of the OSDaR23 data fromat into the format of the KITTI dataset such that for all subsequent tasks (visualizations, training, testing, etc.) the standard MMDetection3D functionality can be used. See here how OSDaR23 can be integrated.

The same process can be used for any 3D data labeled with the web-based labelling service [Segments.ai](https://segments.ai/). Using the MMDetection3D package functionality on self-labeled data from Segments.ai required converting the .json release-file into KITTI type labels. See this [conversion script](/home/cws-ml-lab/mmdetection3d_for_rail/tools/dataset_converters/robosense_m1_plus_sequences_kitti_castor.py) to do so.

Here's an example of how to transform the OSDaR23 dataset in a KITTI compatible format:

```bash
# Transfer file structure and create label-files
python3 tools/dataset_converters/osdar23_kitti_castor.py ./data/osdar23 ./data/osdar23

# Generate annotation .pkl files
python3 tools/create_data.py osdar23 --root-path ./data/osdar23 --out-dir ./data/osdar23

# Visualize and see in transform was successful
python3 tools/misc/browse_dataset.py configs/base/datasets/osdar23-3d-3class.py --task lidar_det
```

### 3D Detection Evaluation Metric
The 3D evaluation metrics provided by MMDetection3D could not be adopted for the custom datasets used during this project. Thus, for this project a new evaluation script that is dataset-independent and compatible with both the per-default integrated datasets as well as any custom datasets. The [General_3dDet_Metric_MMLab](mmdet3d/evaluation/metrics/general_3ddet_metric_mmlab.py) class is set up to evaluate Mean Average Precision (mAP), Precision and Recall over the complete test set. Per default, it evaluates the mAP at 40 reall positions, as suggested by the KITTI dataset challenge.

The evaluation metric can be loaded into a configuration file as a custom import (see e.g. [CenterPoint Generic Config](/home/cws-ml-lab/mmdetection3d_for_rail/configs/centerpoint/centerpoint_voxel01_second_secfpn_generic.py)):

```
val_evaluator = dict(
    type='General_3dDet_Metric_MMLab',
    metric='det3d',
    classes=['Pedestrian', 'Cyclist', 'RoadVehicle', 'Train'],
    pcd_limit_range=[0, -40, -3.0, 80, 40, 1.0], 
    output_dir='/home/cws-ml-lab/mmdetection3d_for_rail/checkpoints/rtx4090_cp_run8_mix_kitti_osdar23_4class_80m',
    save_evaluation_results = True,
    save_random_viz = False,
    random_viz_keys = None)
```

### New WandB Logger Hook
To gain oversight over the trained models and the training process, a [Weights&Biases](https://wandb.ai) integration for MMDetection3D was created. This [script](/mmdet3d/engine/hooks/wandb_logger_hook.py) implements a registered hook that logs the training progress per iteration and also per epoch. Addtionally, the WandB logger hook also logs the metrics from the 3D-detection evaluation metrics described above along with the graphs created by that script. In order to access the validation losses, a few changes were made to the MMEngine foundational library. These are elaborated in the corresponding package ([MMEngine_for_rail](https://github.com/nicoscore99/mmengine_for_rail))

The WandB logger hook is implemnted as custom hook and can be added to the configuration file like this:

```
custom_hooks = [
    dict(type='WandbLoggerHook', 
         save_dir='/home/cws-ml-lab/mmdetection3d_for_rail/checkpoints',
         yaml_config_path='wandb_auth.yaml',
         log_artifact=True,
         init_kwargs={
             'entity': 'railsensing',
             'project': 'centerpoint',
             'name': 'rtx4090_cp_run8_mix_kitti_osdar23_4class_80m',
             })
]
```

### Pointcloud Classification
I developed a pipeline for the classification of 3D point clouds. The pipeline uses the grund truth data that is generated when any dataset is converted using the [dataset conversion script](tools/dataset_converters/create_gt_database.py). The ground truth label file can be generated using another conversion script, that takes the .pkl files for the training sets and the ground truth dataset as an input (see [create_gt_dbinfos.py](/home/cws-ml-lab/mmdetection3d_for_rail/tools/dataset_converters/create_gt_dbinfos.py)).

The point cloud classification pipeline is based on the default implementation of Pointnet++ provided by MMDetection. Instead of the full Encoder-Decoder Network, only the [encoder](mmdet3d/models/backbones/pointnetpp_sa_ssg_torch_impl.py) is used and a [classification head](mmdet3d/models/decode_heads/pointnet2_cls_ssg.py) is added ontop of the encoder.

### 3D Evaluation with Clustering and Cluster-Classification
This package allows a one-to-one comparison of the end-to-end detection methods as defined in the config-folder and a hybrid approach that combines a traditional clustering technique with classification of the clusters. The approach requires to load the ***Depth Clustering*** algorithm as implemented in the ***depth_clustering_for_rail*** package. Since the depth clustering algorithm is implemented in C++ code, running the evaluation pipeline requires including a Pybind11 module that wraps the depth cluster algorithm (see [ClusterPointCloud](mmdet3d/datasets/transforms/cluster_points_transform.py)). 
