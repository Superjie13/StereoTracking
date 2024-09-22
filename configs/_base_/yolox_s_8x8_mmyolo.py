img_scale = (640, 640)  # height, width
deepen_factor = 0.33
widen_factor = 0.5


model = dict(
    data_preprocessor=dict(
        type='mmdet.DetDataPreprocessor',
        pad_size_divisor=32,
        batch_augments=[
            dict(
                type='mmdet.BatchSyncRandomResize',
                random_size_range=(480, 800),
                size_divisor=32,
                interval=10)
        ]),
    detector=dict(
        _scope_='mmyolo',
        type='YOLODetector',
        # use_syncbn=False,
        backbone=dict(
            type='YOLOXCSPDarknet',
            deepen_factor=deepen_factor,
            widen_factor=widen_factor,
            out_indices=(2, 3, 4),
            spp_kernal_sizes=(5, 9, 13),
            norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg=dict(type='SiLU', inplace=True),
        ),
        neck=dict(
            type='YOLOXPAFPN',
            deepen_factor=deepen_factor,
            widen_factor=widen_factor,
            in_channels=[256, 512, 1024],
            out_channels=256,
            norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg=dict(type='SiLU', inplace=True)),
        bbox_head=dict(
            type='YOLOXHead',
            head_module=dict(
                type='YOLOXHeadModule',
                num_classes=80,
                in_channels=256,
                feat_channels=256,
                widen_factor=widen_factor,
                stacked_convs=2,
                featmap_strides=(8, 16, 32),
                use_depthwise=False,
                norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                act_cfg=dict(type='SiLU', inplace=True),
            ),
            loss_cls=dict(
                type='mmdet.CrossEntropyLoss',
                use_sigmoid=True,
                reduction='sum',
                loss_weight=1.0),
            loss_bbox=dict(
                type='mmdet.IoULoss',
                mode='square',
                eps=1e-16,
                reduction='sum',
                loss_weight=5.0),
            loss_obj=dict(
                type='mmdet.CrossEntropyLoss',
                use_sigmoid=True,
                reduction='sum',
                loss_weight=1.0),
            loss_bbox_aux=dict(
                type='mmdet.L1Loss', reduction='sum', loss_weight=1.0)),
        train_cfg=dict(
            assigner=dict(
                type='mmdet.SimOTAAssigner',
                center_radius=2.5,
                iou_calculator=dict(type='mmdet.BboxOverlaps2D'))),
        test_cfg=dict(
            yolox_style=True,  # better
            multi_label=True,  # 40.5 -> 40.7
            score_thr=0.001,
            max_per_img=300,
            nms=dict(type='nms', iou_threshold=0.65)
        )
    )
)
