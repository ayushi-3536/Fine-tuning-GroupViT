# -------------------------------------------------------------------------
# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License.
# To view a copy of this license, visit
# https://github.com/NVlabs/GroupViT/blob/main/LICENSE
#
# Written by Jiarui Xu
# -------------------------------------------------------------------------

import mmcv
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.datasets.pipelines import Compose
from omegaconf import OmegaConf
from utils import build_dataset_class_tokens

from .group_vit_seg import GroupViTSegInference


def build_seg_dataset(config):
    """Build a dataset from config."""
    cfg = mmcv.Config.fromfile(config.cfg)
    print("cfg.data.test", cfg.data.test)
    dataset = build_dataset(cfg.data.test)
    return dataset


def build_seg_dataloader(dataset):

    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=1,
        dist=True,
        shuffle=False,
        persistent_workers=True,
        pin_memory=False)
    return data_loader

def build_train_dataloader_with_annotations(dataset):
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=1,
        dist=True,
        shuffle=False,
        persistent_workers=True,
        pin_memory=False)
    return data_loader


def build_seg_inference(model, dataset, text_transform, config, add_synonyms=True):
    cfg = mmcv.Config.fromfile(config.cfg)
    print("dataset", dataset)
    if len(config.opts):
        cfg.merge_from_dict(OmegaConf.to_container(OmegaConf.from_dotlist(OmegaConf.to_container(config.opts))))
    with_bg = dataset.CLASSES[0] == 'background'
    if with_bg:
        classnames = dataset.CLASSES[1:]
    else:
        classnames = dataset.CLASSES

    #if add_synonyms and dataset.synonyms is not None then add for every unique value in dataset.synonyms dict a new class
    # if add_synonyms and dataset.SYNONYMS is not None:
    #     for key, value in dataset.SYNONYMS.items():
    #         if value not in classnames:
    #             #change list to tuple
    #             value = tuple(value)
    #             classnames = classnames + value
    # #keep only unique values
    # classnames = list(set(classnames))
    
    # print("classnames", classnames)

    text_tokens = build_dataset_class_tokens(text_transform, config.template, classnames)
    text_embedding = model.build_text_embedding(text_tokens)
    kwargs = dict(with_bg=with_bg)
    if hasattr(cfg, 'test_cfg'):
        kwargs['test_cfg'] = cfg.test_cfg
    seg_model = GroupViTSegInference(model, text_embedding, **kwargs)

    seg_model.CLASSES = dataset.CLASSES
    seg_model.PALETTE = dataset.PALETTE

    return seg_model

#This methos takes custome set of classes while keeping the evaluation configs same as the dataset provided
#This was done to evaluate the performance of the model on unseen classes without changing the evaluation configs
def build_seg_inference_for_customclasses(model, dataset, text_transform, config, texts):
    cfg = mmcv.Config.fromfile(config.cfg)
    print("dataset", dataset)
    if len(config.opts):
        cfg.merge_from_dict(OmegaConf.to_container(OmegaConf.from_dotlist(OmegaConf.to_container(config.opts))))
    with_bg = dataset.CLASSES[0] == 'background'
    classnames = tuple(texts)

    text_tokens = build_dataset_class_tokens(text_transform, config.template, classnames)
    text_embedding = model.build_text_embedding(text_tokens)
    kwargs = dict(with_bg=0)
    if hasattr(cfg, 'test_cfg'):
        kwargs['test_cfg'] = cfg.test_cfg
    seg_model = GroupViTSegInference(model, text_embedding, **kwargs)
    PALETTE = [[180, 120, 120], [6, 230, 230], [80, 50, 50], [4, 200, 3],
               [120, 120, 80], [140, 140, 140], [204, 5, 255], [230, 230, 230],
               [4, 250, 7], [224, 5, 255], [235, 255, 7], [150, 5, 61],
               [120, 120, 70], [8, 255, 51], [255, 6, 82], [143, 255, 140],
               [204, 255, 4], [255, 51, 7], [204, 70, 3], [0, 102, 200],
               [61, 230, 250], [255, 6, 51], [11, 102, 255], [255, 7, 71],
               [255, 9, 224], [9, 7, 230], [220, 220, 220], [255, 9, 92],
               [112, 9, 255], [8, 255, 214], [7, 255, 224], [255, 184, 6],
               [10, 255, 71], [255, 41, 10], [7, 255, 255], [224, 255, 8],
               [102, 8, 255], [255, 61, 6], [255, 194, 7], [255, 122, 8],
               [0, 255, 20], [255, 8, 41], [255, 5, 153], [6, 51, 255],
               [235, 12, 255], [160, 150, 20], [0, 163, 255], [140, 140, 140],
               [250, 10, 15], [20, 255, 0], [31, 255, 0], [255, 31, 0],
               [255, 224, 0], [153, 255, 0], [0, 0, 255], [255, 71, 0],
               [0, 235, 255], [0, 173, 255], [31, 0, 255]]  
    seg_model.CLASSES = tuple(texts)
    seg_model.PALETTE = PALETTE

    return seg_model




class LoadImage:
    """A simple pipeline to load image."""

    def __call__(self, results):
        """Call function to load images into results.

        Args:
            results (dict): A result dict contains the file name
                of the image to be read.

        Returns:
            dict: ``results`` will be returned containing loaded image.
        """

        if isinstance(results['img'], str):
            results['filename'] = results['img']
            results['ori_filename'] = results['img']
        else:
            results['filename'] = None
            results['ori_filename'] = None
        img = mmcv.imread(results['img'])
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results

class LoadTrainImage:
    """A simple pipeline to load image."""

    def __call__(self, results):
        """Call function to load images into results.

        Args:
            results (dict): A result dict contains the file name
                of the image to be read.

        Returns:
            dict: ``results`` will be returned containing loaded image.
        """

        if isinstance(results['img'], str):
            results['filename'] = results['img']
            results['ori_filename'] = results['img']
        else:
            results['filename'] = None
            results['ori_filename'] = None
        #img = mmcv.imread(results['img'])
        img = results['img'][0].numpy()
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results


def build_seg_demo_pipeline():
    """Build a demo pipeline from config."""
    img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
    print("build_seg_demo_pipeline")
    test_pipeline = Compose([
        LoadImage(),
        dict(
            type='MultiScaleFlipAug',
            
            #img_scale=(2048, 512),
            img_scale=(2048, 448),
            flip=False,
            transforms=[
                dict(type='Resize', keep_ratio=True),
                dict(type='RandomFlip'),
                dict(type='Normalize', **img_norm_cfg),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img']),
            ])
    ])
    return test_pipeline

def build_train_assessment_pipeline():
    """Build a demo pipeline from config."""
    img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
    test_pipeline = Compose([
        LoadTrainImage(),
        dict(
            type='MultiScaleFlipAug',
            img_scale=(2048, 448),
            flip=False,
            transforms=[
                dict(type='Resize', keep_ratio=True),
                dict(type='RandomFlip'),
                dict(type='Normalize', **img_norm_cfg),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img']),
            ])
    ])
    return test_pipeline
