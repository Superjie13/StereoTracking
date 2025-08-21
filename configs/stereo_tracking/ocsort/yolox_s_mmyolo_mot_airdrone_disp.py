
_base_ = ['../../_base_/default_runtime.py',
          '../../_base_/yolox_s_8x8_mmyolo.py',]

data_root = 'data/AirSim_drone/'

DEPTH_RANGE = 80

img_scale = (720, 1280)
num_classes = 1
classes = ['drone',]

disp_thr_h = 1000  # remove
disp_thr_l = 0

deepen_factor = 0.33
widen_factor = 0.5

save_epoch_intervals = 5
train_batch_size_per_gpu = 8
train_num_workers = 16
val_batch_size_per_gpu = 1
val_num_workers = 2

max_epochs = 50
num_last_epochs = 5


model = dict(
    type='OCSORT_Disparity',
    data_preprocessor=dict(
        type='TrackDataPreprocessor_Disparity_V1',
        pad_size_divisor=32,
        batch_augments=[
        ]
        ),
    detector=dict(
        type='mmtrack.YOLODetector_Disparity_V1',
        backbone=dict(type='mmtrack.YOLOXCSPDarknet_Disparity_V1_MMYOLO',
                      input_channels=3),
        bbox_head=dict(head_module=dict(num_classes=num_classes)),
        test_cfg=dict(score_thr=0.01, nms=dict(type='nms', iou_threshold=0.5)),
        init_cfg=dict(
            type='ColorPretrained',
            checkpoint=  # noqa: E251
            'https://download.openmmlab.com/mmyolo/v0/yolox/yolox_s_8xb8-300e_coco/yolox_s_8xb8-300e_coco_20220917_030738-d7e60cb2.pth'
        )),
    motion=dict(type='KalmanFilter'),
    tracker=dict(
        type='OCSORTTracker_Disparity',
        obj_score_thr=0.3,
        init_track_thr=0.7,
        weight_iou_with_det_scores=False,
        match_iou_thr=0.1,
        num_tentatives=3,
        vel_consist_weight=0.2,
        vel_delta_t=3,
        num_frames_retain=30))

pre_transform = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadTrackAnnotations'),
    dict(type='LoadDisparityFromFile', to_3channel=True,
         post_processing=dict(disp_thr_h=disp_thr_h, disp_thr_l=disp_thr_l)),
]

train_pipeline_stage1 = [
    *pre_transform,
    dict(type='Resize_Disparity', scale=img_scale, keep_ratio=True),
    dict(
        type='YOLOXMixUp_Disparity',
        img_scale=img_scale,
        ratio_range=(0.8, 1.6),
        pad_val=114.0,
        pre_transform=pre_transform),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(type='RandomFlip_Disparity', prob=0.5),
    dict(type='mmdet.FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(
        type='PackTrackInputs_Disparity', pack_single_img=True,
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                   'flip_direction'))
]

train_pipeline_stage2 = [
    *pre_transform,
    dict(type='Resize_Disparity', scale=img_scale, keep_ratio=True),
    dict(
        type='Pad_Disparity',
        pad_to_square=False,
        size_divisor=32,
        # If the image is three-channel, the pad value needs
        # to be set separately for each channel.
        pad_val=dict(img=(114.0, 114.0, 114.0), disp=0, disp_mask=0)),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(type='RandomFlip_Disparity', prob=0.5),
    dict(
        type='mmdet.FilterAnnotations',
        min_gt_bbox_wh=(1, 1),
        keep_empty=False),
    dict(type='PackTrackInputs_Disparity', pack_single_img=True,)
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadDisparityFromFile', to_3channel=True,
         post_processing=dict(disp_thr_h=disp_thr_h, disp_thr_l=disp_thr_l)),
    dict(type='Resize_Disparity', scale=img_scale, keep_ratio=True),
    dict(type='Pad_Disparity',
         size_divisor=32,
         pad_val=dict(img=(114.0, 114.0, 114.0), disp=0, disp_mask=0)),
    dict(
        type='PackTrackInputs_Disparity', pack_single_img=True,
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CocoDispDataset',
        data_root=data_root,
        ann_file=f'annotations/train_cocoformat_{DEPTH_RANGE}.json',
        data_prefix=dict(img_path='train/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        metainfo=dict(CLASSES=classes),
        pipeline=train_pipeline_stage1))

val_dataloader = dict(
    batch_size=val_batch_size_per_gpu,
    num_workers=val_num_workers,
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='VideoSampler'),
    dataset=dict(
        type='MOTDispDataset',
        data_root=data_root,
        ann_file=f'annotations/val_cocoformat_{DEPTH_RANGE}.json',
        data_prefix=dict(img_path='val/'),
        depth_dir_name='depth',
        metainfo=dict(CLASSES=classes),
        ref_img_sampler=None,
        load_as_video=True,
        test_mode=True,
        pipeline=test_pipeline))
test_dataloader = val_dataloader

# optimizer
base_lr = 0.001 / 8 * train_batch_size_per_gpu / 1

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='SGD', lr=base_lr, momentum=0.9, weight_decay=5e-4, nesterov=True),
    paramwise_cfg=dict(norm_decay_mult=0., bias_decay_mult=0.))

# learning policy
param_scheduler = [
    dict(
        type='mmdet.QuadraticWarmupLR',
        by_epoch=True,
        begin=0,
        end=2,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=2,
        T_max=max_epochs - num_last_epochs,
        end=max_epochs - num_last_epochs,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='mmdet.ConstantLR',
        by_epoch=True,
        factor=1,
        begin=max_epochs - num_last_epochs,
        end=max_epochs,
    )
]

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
    checkpoint=dict(
        type='CheckpointHook', interval=save_epoch_intervals, max_keep_ckpts=3, save_best='auto'))

custom_hooks = [
    dict(
        type='YOLOXModeSwitchHook_mmyolox',
        num_last_epochs=num_last_epochs,
        new_train_pipeline=train_pipeline_stage2,
        priority=48),
    dict(
        type='mmyolo.EMAHook',
        ema_type='mmyolo.ExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        priority=49)
]

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=save_epoch_intervals)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# evaluator
val_evaluator = [
    dict(
    type='mmdet.CocoMetric',
    ann_file=data_root + f'annotations/val_cocoformat_{DEPTH_RANGE}.json',
    metric='bbox',
    format_only=False),

    # do not use during training
    # dict(
    #     type='MOTDroneMetrics',
    #     metric=['HOTA', 'CLEAR', 'Identity'],
    #     depth_thr=DEPTH_RANGE,
    #     ignore_depth=False,
    #     postprocess_tracklet_cfg=[
    #         # dict(type='InterpolateTracklets', min_num_frames=5, max_num_frames=20)
    #     ]
    # )
]
test_evaluator = val_evaluator
