_base_ = [
    '../../../configs/_base_/models/pointpillars_hv_secfpn_kitti.py',
    '../../../configs/_base_/datasets/kitti-3d-3class.py',
    '../../../configs/_base_/schedules/cyclic-40e.py', '../../../configs/_base_/default_runtime.py'
]

custom_imports = dict(
    imports=['pcd_ros2_pipeline'], allow_failed_imports=False)


point_cloud_range = [0, -39.68, -3, 69.12, 39.68, 1]
# dataset settings
data_root = 'data/kitti/'
class_names = ['Pedestrian', 'Cyclist', 'Car']
metainfo = dict(classes=class_names)
backend_args = None

# PointPillars adopted a different sampling strategies among classes
db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'kitti_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(Car=5, Pedestrian=5, Cyclist=5)),
    classes=class_names,
    sample_groups=dict(Car=15, Pedestrian=15, Cyclist=15),
    points_loader=dict(
        type='LoadPointsFromDict',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    backend_args=backend_args)

test_pipeline = [
    dict(
        type='LoadPointsFromPointcloud2',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(
                type='PointsRangeFilter', point_cloud_range=point_cloud_range)
        ]),
    dict(type='Pack3DDetInputs', keys=['points'])
]

test_dataloader = dict(dataset=dict(pipeline=test_pipeline, metainfo=metainfo))
# In practice PointPillars also uses a different schedule
test_cfg = dict()
