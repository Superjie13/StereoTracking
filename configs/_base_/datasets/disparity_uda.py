# dataset settings
target_val_dataset_type = 'MOTKittiDataset'
target_train_dataset_type = 'CocoDispDataset'
source_dataset_type = 'SELMA_Coco_Dataset'
target_data_root = '/home/sijie/Documents/datasets/MOT/kitti_tracking/MOT_Kitti/'
source_data_root = '/home/sijie/Documents/datasets/MOT/SELMA/'

file_client_args = dict(backend='disk')

target_scale_det = (384, 1280)
target_scale_disp = (384, 1280)
source_scale_disp = (640, 1280)  # when resize, keep ratio
num_classes = 2
classes = ['car', 'pedestrian',]

branch_field = ['source_disp', 'target_disp', 'target_sup_det']

# pipeline used to augment syn data for depth completion,
# which will be source domain for supervised training disparity.
source_disp_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadDisparityFromFile', to_3channel=True,
         post_processing = dict(disp_thr_h=1200, disp_thr_l=10)),
    dict(type='LoadDepthFromFile', to_3channel=False, inv_depth=True,
         post_processing = dict(thr_h=2000, thr_l=0)),
    dict(type='Resize_Disparity', scale=source_scale_disp, keep_ratio=True),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(type='RandomFlip_Disparity', prob=0.5),
    dict(
        type='mmdet.MultiBranch',
        branch_field=branch_field,
        source_disp=dict(
            type='PackTrackInputs_Disparity', pack_single_img=True,
            meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                       'flip_direction', 'depth_postp')),
    )

]

# pipeline used to augment real data for depth completion,
# which will be target domain for supervised training disparity.
target_disp_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadDisparityFromFile', to_3channel=True,
         post_processing = dict(disp_thr_h=1200, disp_thr_l=10)),
    dict(type='Resize_Disparity', scale=target_scale_disp, keep_ratio=True),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(type='RandomFlip_Disparity', prob=0.5),
    dict(type='PackTrackInputs_Disparity', pack_single_img=True,
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                    'flip_direction')),
]

# pipeline used to augment real data for detection,
# which will be target domain for supervised training disparity.
pre_transform = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadTrackAnnotations'),
    dict(type='LoadDisparityFromFile', to_3channel=True,
         post_processing=dict(disp_thr_h=1200, disp_thr_l=10)),
]
target_sup_det_pipeline = [
    *pre_transform,
    dict(type='Resize_Disparity', scale=target_scale_det, keep_ratio=True),
    dict(
        type='YOLOXMixUp_Disparity',
        img_scale=target_scale_det,
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

target_pipeline = [
    dict(
        type='mmdet.MultiBranch',
        branch_field=branch_field,
        target_disp=target_disp_pipeline,
        target_sup_det=target_sup_det_pipeline,
    )
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadDisparityFromFile', to_3channel=True,),
    # dict(type='Disp2ColorImg'),
    # dict(type='TLBRCrop', crop_size=crop_size),  # -1 denotes bottom-right of original image
    dict(type='Resize_Disparity', scale=target_scale_det, keep_ratio=True),
    dict(type='Pad_Disparity',
         size_divisor=32,
         pad_val=dict(img=(114.0, 114.0, 114.0), disp=0, disp_mask=0)),
    dict(
        type='PackTrackInputs_Disparity', pack_single_img=True,
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]


train_batch_size_per_gpu = 8
train_num_workers = 16

val_batch_size_per_gpu = 1
val_num_workers = 2

source_dataset = dict(
    type=source_dataset_type,
    data_root=source_data_root,
    ann_file='annotations/half-train_cocoformat.json',
    data_prefix=dict(img_path='train/'),
    pipeline=source_disp_pipeline)

target_dataset = dict(
    type=target_train_dataset_type,
    data_root=target_data_root,
    ann_file='annotations/train_cocoformat.json',
    data_prefix=dict(img_path='train/'),
    metainfo=dict(CLASSES=classes),
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=target_pipeline)

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(
        type='mmdet.MultiSourceSampler',
        batch_size=train_batch_size_per_gpu,
        source_ratio=[1, 1],
        shuffle=True),
    collate_fn=dict(type='multi_dataset_collate'),
    dataset=dict(
        # ConcatDataset use the Datainfo of the first dataset, so put target dataset in the first.
        type='ConcatDataset',
        ignore_keys='CLASSES',  # classes in dataset1 and dataset2 can be unique.
        datasets=[target_dataset, source_dataset])
)

val_dataloader = dict(
    batch_size=val_batch_size_per_gpu,
    num_workers=val_num_workers,
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='VideoSampler'),
    dataset=dict(
        type=target_val_dataset_type,
        data_root=target_data_root,
        ann_file='annotations/val_cocoformat.json',
        data_prefix=dict(img_path='train'),
        metainfo=dict(CLASSES=classes),
        ref_img_sampler=None,
        load_as_video=True,
        test_mode=True,
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='mmdet.CocoMetric',
    ann_file=target_data_root + 'annotations/val_cocoformat.json',
    metric='bbox',
    format_only=False)
test_evaluator = val_evaluator

