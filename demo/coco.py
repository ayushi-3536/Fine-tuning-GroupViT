# -------------------------------------------------------------------------
# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License.
# To view a copy of this license, visit
# https://github.com/NVlabs/GroupViT/blob/main/LICENSE
#
# Written by Jiarui Xu
# -------------------------------------------------------------------------

_base_ = ['../custom_import.py']
# dataset settings
dataset_type = 'COCOObjectDataset'
data_root = '/misc/lmbraid21/sharmaa/coco_stuff164k'
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        #img_scale=(2048, 512),
        img_scale=(2048, 448),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/val2017',
        ann_dir='annotations/val2017',
        pipeline=test_pipeline))

# test_cfg = dict(bg_thresh=.95, mode='whole')

#For stride 384, crop 448, Vary bg_thresh
#test_cfg = dict(bg_thresh=.95, mode='slide', stride=(384, 384), crop_size=(448, 448))
#test_cfg = dict(bg_thresh=.9, mode='slide', stride=(384, 384), crop_size=(448, 448))
#test_cfg = dict(bg_thresh=.8, mode='slide', stride=(384, 384), crop_size=(448, 448))

#For stride 448, crop 448, Vary bg_thresh
#test_cfg = dict(bg_thresh=.95, mode='slide', stride=(448, 448), crop_size=(448, 448))
#currently here
#test_cfg = dict(bg_thresh=.9, mode='slide', stride=(448, 448), crop_size=(448, 448))
#test_cfg = dict(bg_thresh=.8, mode='slide', stride=(448, 448), crop_size=(448, 448))

test_cfg = dict(bg_thresh=.8, mode='slide', stride=(224, 224), crop_size=(448, 448))
#test_cfg = dict(bg_thresh=.9, mode='slide', stride=(224, 224), crop_size=(448, 448))
#test_cfg = dict(bg_thresh=.95, mode='slide', stride=(224, 224), crop_size=(448, 448))
