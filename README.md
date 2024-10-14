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
For this project, the **Open Sensor Data For Rail 2023** (subsequently *OSDaR23*) dataset was integrated as a custom dataset. Find more information on the [OSDaR23 dataset](https://data.fid-move.de/dataset/osdar23). This happens by conversion of the OSDaR23 data fromat into the format of the KITTI dataset such that for all subsequent tasks (visualizations, training, testing, etc.) the standard MMDetection3D functionality can be used.

The same process can be used for any 3D data labeled with the web-based labelling service [Segments.ai](https://segments.ai/). Using the MMDetection3D package functionality on self-labeled data from Segments.ai required converting the .json release-file into KITTI type labels. See this [conversion script](/home/cws-ml-lab/mmdetection3d_for_rail/tools/dataset_converters/robosense_m1_plus_sequences_kitti_castor.py) to do so.

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

Note that the 'General_3dDet_Metric_MMLab' evaluator is configured to load the authentification key for the WandB account to log to from the [wandb_auth.yaml](/home/cws-ml-lab/mmdetection3d_for_rail/mmdet3d/engine/hooks/wandb_auth.yaml) if the device has not already been configured with the key. Hence, make sure to set the correct WandB key in the authentification file. 

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

## WIKI

### How to: Install

The [MMDetection3d documentation](https://mmdetection3d.readthedocs.io/en/latest/get_started.html) suggests installing MMDetection3D inside a conda environment using [openmim](https://openmim.readthedocs.io/en/latest/). However, this complicates using MMDetection3D with ROS2. 

I suggest to install MMDetection3D and MMEngine natively wit pip:

```bash
cd mmdetection3d_for_rail/
pip3 install .
```

### How to: Train a model

To train an object detection model, follow the subsequent steps.

1. Generate a configuration file.

    [Here](/home/cws-ml-lab/mmdetection3d_for_rail/configs/centerpoint/centerpoint_voxel01_second_secfpn_generic.py) provides a good example. It is advisable to generate a single, combined file instead of handeling the model, dataset and schedulers in separate files.

    Also make sure to have the validation pipeline with the custom metrics and the [WandB Logger Hook](/home/cws-ml-lab/mmdetection3d_for_rail/mmdet3d/engine/hooks/wandb_logger_hook.py) properly linked. 

2. Run the training script.

    ```bash
    cd mmdetection3d_for_rail/
    python3 tools/train.py ./configs/name_of_configuration_file.py
    ```

### How to: Test a model

Models can be tested using the [test script](/home/cws-ml-lab/mmdetection3d_for_rail/tools/test.py). Run the following command:

```bash
cd mmdetection3d_for_rail/
python3 tools/test.py ./configs/name_of_configuration_file.py ./checkpoints/path_to_model_to_test.pth
```

The test script will run the 'test_dataloader' and 'test_evaluator'. Make sure these are defined in the configuration file. Additionally, it is paramount that the model configuration in the configuration file is exactly identical with the model architecture that was used during training, such that the exact weights are loaded.


### How to: Create a Custom Datast

A relatively comprehensive description on how to customize MMDetection3D for new datasets is provided [here](https://mmdetection3d.readthedocs.io/en/dev/tutorials/customize_dataset.html).

Generally, the following steps need to be followed:

1. Register a new dataset. Examples for datasets can be found [here](/home/cws-ml-lab/mmdetection3d_for_rail/mmdet3d/datasets).
2. Bring your data into the required structure. A good approach is to just use the exact same structure as the KITTI dataset uses - but this can be configured.
3. Adapt the follwing two scripts as needed:

    - [create_gt_database.py](tools/dataset_converters/create_gt_database.py) This script defined the data loading process for the creation of the ground truth database. The dataloading process needs to match with the structure of the custom dataset. 
    - [update_infos_to_v2.py](tools/dataset_converters/update_infos_to_v2.py) This is a scirpt that updates the data structure to v1.1.0 of mmdetection3d. Basically you can follow the KITTI structure here and ignore all irrelevante information (e.g. such as 'center_2d' when only LiDAR is used). You should also adjust the METAINFO to the relevant set of classes in your dataset.

        ```
        METAINFO = {
            'classes': ('Pedestrian', 'Cyclist', 'RoadVehicle', 'Train'),
        }
        ```

4. Use your own [conversion script](/home/cws-ml-lab/mmdetection3d_for_rail/tools/dataset_converters/osdar23_kitti_castor.py) and the script [create_data.py](/home/cws-ml-lab/mmdetection3d_for_rail/tools/create_data.py) to convert the data into a KITTI-compatible format.

    Here's an example of how to transform the OSDaR23 dataset in a KITTI compatible format:

    ```bash
    # Transfer file structure and create label-files
    python3 tools/dataset_converters/osdar23_kitti_castor.py ./data/osdar23 ./data/osdar23

    # Generate annotation .pkl files
    python3 tools/create_data.py osdar23 --root-path ./data/osdar23 --out-dir ./data/osdar23

    # Visualize and see in transform was successful
    python3 tools/misc/browse_dataset.py configs/base/datasets/osdar23-3d-3class.py --task lidar_det
    ```