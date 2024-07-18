# MMDetection3D_for_rail

Welcome to the MMDetection3D_for_rail package. This package was forked from the original [MMDetection3D](https://github.com/open-mmlab/mmdetection3d) package developped by [OpenMMLab](https://github.com/open-mmlab). The fork was created with the intention of testing and deploying LiDAR point cloud-based object detection algrorithms that perform well in railway-specific environments, with the vision in mind of using ML algorithms for detection objects in the context of automatic train operations.

## Environment Setup

The changes that were made during this project to the original MMDetection3D package were developed and tested in the follwing environment:

* __Operating System:__ Ubuntu 22.04
* __NVIDIA Driver Versin:__ 535.183.01
* __CUDA Toolbox:__ cuda_11.8
* __Pytorch:__ 2.3.1+cu118
* __Torchvision:__ 0.18.1+cu118

The [MMDetection3D](https://github.com/open-mmlab/mmdetection3d) package is closely integrated with [MMEngine](https://github.com/open-mmlab/mmdetection3d), which acts as a foundational library for model development and training. This project required to make some changes to the MMengine library as well. Thus, to make this package work you need to use my fork of the MMengine library, which can be found here: [MMEngine_for_rail](https://github.com/nicoscore99/mmengine_for_rail)


## Relevant Changes

The following is a list of the relevant changes that I have made on top of the [MMDetection3D](https://github.com/open-mmlab/mmdetection3d) package:

