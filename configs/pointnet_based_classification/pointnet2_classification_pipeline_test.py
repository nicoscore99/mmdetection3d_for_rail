_base_ = [
    'pointnet_model_large.py',
]

custom_imports = dict(imports=['mmdet3d.datasets.robosense_m1_plus_dataset', 
                               'mmdet3d.engine.hooks.wandb_logger_hook',
                               'mmdet3d.evaluation.metrics.general_3ddet_metric_mmlab_classification_version'], allow_failed_imports=False)             

# from mmdet3d.utils import register_all_modules
# register_all_modules()


#### Generic loader information ####

class_names = ['Pedestrian', 'Cyclist', 'RoadVehicle', 'Train']
point_cloud_range = [0, -40, -3, 80, 40, 10]
input_modality = dict(use_lidar=True, use_camera=False)
metainfo = dict(classes=class_names)

points_loader = dict(
    type='LoadPointsFromFile',
    coord_type='LIDAR',
    load_dim=4,
    use_dim=4,
    backend_args=None)

generic_eval_pipeline = [
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=None),
    dict(type='Pack3DDetInputs', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d']),
    dict(type='ClusterPointCloud'),
    dict(type='Open3DBallPivotingSequence',
         min_points=50,
         num_pts_sample=256),
]

############# Robosense Specific Config #############

robosense_dataroot = 'data/robosense_cls'
robosense_dataset_type = 'ROBOSENSE_M1_PLUS'
robosense_m1_plus_dataset = dict(type='ROBOSENSE_M1_PLUS')

robosense_val_dataset = dict(
    type=robosense_dataset_type,
    # indices=10,
    data_root=robosense_dataroot,
    data_prefix=dict(pts='points'),
    ann_file='kitti_infos_val.pkl',
    pipeline=generic_eval_pipeline,
    modality=input_modality,
    test_mode=True,
    metainfo=metainfo,
    backend_args=None)

robosense_test_dataset = robosense_val_dataset

############# Dataloader Config #############

val_dataloader = dict(
    batch_size=1, # Batch size has to be 1 for in order to not create problems with the evaluation
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=robosense_val_dataset
)

test_dataloader = val_dataloader    

# For the val and test evaluator, we do not need to specify the annotation files
val_evaluator = dict(
    type='General_3dDet_Metric_MMLab_Classification_Version',
    metric='det3d',
    classes=class_names,
    pcd_limit_range=point_cloud_range,
    output_dir='/home/cws-ml-lab/mmdetection3d_for_rail/experiments/cluster_classification/rtx4090_pointnetpp_cls_all_data_256pts_yanx27_with_upsampling_normalized_size_correct_sampling/evaluation',
    save_graphics = False,
    save_evaluation_results = True,
    save_random_viz = False,
    random_viz_keys = None)

test_evaluator = val_evaluator

############# Model Config #############

# as defined in pointnet_model_large.py

### runtime settings ###
val_cfg = dict()
test_cfg = dict()

######### Default Runtime Config #############

default_scope = 'mmdet3d'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=4, by_epoch=True),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='Det3DVisualizationHook'))

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

log_level = 'INFO'
load_from = None
resume = False

############# Work Directory #############
work_dir = '/home/cws-ml-lab/mmdetection3d_for_rail/experiments/cluster_classification/rtx4090_pointnetpp_cls_all_data_256pts_yanx27_with_upsampling_normalized_size_correct_sampling/evaluation'
