# -------------------------------------------------------------------------
# # Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License.
# To view a copy of this license, visit
# https://github.com/NVlabs/GroupViT/blob/main/LICENSE
#
# Written by Jiarui Xu
# -------------------------------------------------------------------------

import torch
import argparse
import os.path as osp
import sys
from omegaconf import OmegaConf
parentdir = osp.dirname(osp.dirname(__file__))
sys.path.insert(0, parentdir)
import mmcv
import json
import tqdm
import os
from datasets import build_dataloader, build_analysis_dataloader, build_text_transform, imagenet_classes

from datasets import build_text_transform
from mmcv.cnn.utils import revert_sync_batchnorm
from mmcv.image import tensor2imgs
from mmcv.parallel import collate, scatter
from models import build_model
from omegaconf import read_write
from segmentation.datasets import COCOObjectDataset, PascalContextDataset, PascalVOCDataset
from segmentation.evaluation import build_seg_demo_pipeline, build_train_assessment_pipeline, build_seg_inference
from utils import get_config, load_checkpoint
import numpy as np
from segmentation.evaluation import GroupViTTrainAssessment
from datasets import build_loader_sync, build_train_assessment_dataset
from pathlib import Path
from pycocotools.coco import COCO
import cv2
# import debugpy

# 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
# debugpy.listen(2122)
# print("Waiting for debugger attach")
# debugpy.wait_for_client()
# debugpy.breakpoint()
# print('break on this line')
def parse_args():
    parser = argparse.ArgumentParser('GroupViT demo')
    parser.add_argument(
        '--cfg',
        type=str,
        required=True,
        help='path to config file',
    )
    parser.add_argument(
        '--opts',
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument(
        '--vis',
        help='Specify the visualization mode, '
        'could be a list, support "input", "pred", "input_pred", "all_groups", "first_group", "final_group", "input_pred_label"',
        default=None,
        nargs='+')

    parser.add_argument('--device', default='cpu', help='Device used for inference')
    # parser.add_argument(
    #     '--dataset', default='voc', choices=['voc', 'coco', 'context'], help='dataset classes for visualization')

    #parser.add_argument('--output_dir', type=str, help='output dir')
    parser.add_argument('--allow_shape_change', default=True, type=bool,  help='path to config file')
    
    args = parser.parse_args()
    args.local_rank = 0  # compatible with config

    return args

def val_assessment(args, config):
    model = build_model(config.model)
    model = revert_sync_batchnorm(model)
    model.to(args.device)
    model.eval()

    load_checkpoint(config, model, None, None, args.allow_shape_change)

    
    #Choosing one of the classes to get the palette
    dataset = PascalContextDataset
    # data = build_train_assessment_dataset(is_train=True, config=config.data)
    # print("data", data)
    model = GroupViTTrainAssessment(model)
    model.PALETTE = dataset.PALETTE
    text_transform = build_text_transform(False, config.data.text_aug, with_dc=False)
    seg_model = build_seg_inference(model_without_ddp, data_loader.dataset, text_transform, config.evaluate.seg)


    test_pipeline = build_seg_demo_pipeline()

    annotation_file = '/misc/lmbraid21/sharmaa/coco_stuff164k/annotations/instances_val2017.json'
    images_dir = '/misc/lmbraid21/sharmaa/coco_stuff164k/annotations/val2017/'

    # Initialize the COCO API for loading annotations
    # coco = COCO(annotation_file)

    # # Get the image IDs for the validation split
    # image_ids = coco.getImgIds()
    # print("image_ids", len(image_ids))
    coco_annotation_path = '/misc/lmbraid21/sharmaa/coco_stuff164k/images/val2017'
    #get all filename in the directory
    image_ids = os.listdir(coco_annotation_path)
    sorted_file_names = sorted(image_ids)
    print("sorted_file_names", sorted_file_names[:10])
    json_file = '/misc/lmbraid21/sharmaa/fann_seqnewval_analysis_files/'
    #check if dir exit else create it
    Path(json_file).mkdir(parents=True, exist_ok=True)
    for i, file in enumerate(sorted_file_names):
            print("i", i)
            
            # Load the image corresponding to the current image ID
            # image_info = coco.loadImgs(image_id)[0]
            img_path = coco_annotation_path + '/' + file
            print("imag_path", img_path)
            data = dict(img=img_path)
            data = test_pipeline(data)
            data = collate([data], samples_per_gpu=1)
            result_dict = vis_seg(model, data)
            grouped_img_token = result_dict['grouped_img_tokens']
            group_features = result_dict['group_feat']
            grouped_image = result_dict['grouped_image']
       
            result_dict = {'filename':str(img_path),
                        'grouped_image':grouped_image.tolist(),
                        'group_features':group_features.detach().cpu().numpy().tolist(),
                        'grouped_img_token':grouped_img_token.detach().cpu().numpy().tolist()
                       
                       }
            with open(json_file + f'/data_{file}.json', 'w') as f:
                json.dump(result_dict, f)
def vis_seg(seg_model, data):
    device = next(seg_model.parameters()).device
    if next(seg_model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        data['img_metas'] = [i.data[0] for i in data['img_metas']]
    #print("data", data)
    img_tensor = data['img'][0]
    img_metas = data['img_metas'][0]
    imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
    assert len(imgs) == len(img_metas)
    for img, img_meta in zip(imgs, img_metas):
        h, w, _ = img_meta['img_shape']
        img_show = img[:h, :w, :]
        ori_h, ori_w = img_meta['ori_shape'][:-1]
        img_show = mmcv.imresize(img_show, (ori_w, ori_h))
        result_dict = seg_model.get_all_feat(img_tensor.to(device), img_show)
        #print("result_dict", result_dict)   
        return result_dict
        

def main():
    args = parse_args()
    cfg = get_config(args)

    with read_write(cfg):
        cfg.evaluate.eval_only = True

    val_assessment(args, cfg)


if __name__ == '__main__':
    main()
