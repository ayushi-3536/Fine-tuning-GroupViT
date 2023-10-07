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
import os
import pickle
from datasets import build_dataloader, build_analysis_dataloader, build_text_transform, imagenet_classes

from datasets import build_text_transform
from mmcv.cnn.utils import revert_sync_batchnorm
from mmcv.image import tensor2imgs
from mmcv.parallel import collate, scatter
from models import build_model
from omegaconf import read_write
from segmentation.datasets import COCOObjectDataset, PascalContextDataset, PascalVOCDataset
from segmentation.evaluation import build_seg_demo_pipeline, build_train_assessment_pipeline
from utils import get_config, load_checkpoint
import numpy as np
from segmentation.evaluation import GroupViTTrainAssessment
from datasets import build_loader_sync, build_train_assessment_dataset
from pathlib import Path
from pycocotools.coco import COCO

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

def normalize_dict_values(dictionary):
    # # Get the maximum value in the dictionary
    max_value = sum(dictionary.values())
    min_value = min(dictionary.values())

    # Normalize the values by dividing each value by the maximum value
    normalized_dict = {key: value / max_value for key, value in dictionary.items()}

    return normalized_dict



def find_category_for_feature(arr1, arr2, category_mapping):
    val_arr1 = {}
    unique_values = np.unique(arr1)

    # Iterate over each unique value in arr1
    for value in unique_values:
        val_arr1[int(value)] = {}
        # Find the corresponding values in arr2 at the same positions
        positions = np.where(arr1 == value)
        corresponding_values = arr2[positions]

        # Calculate the frequency of each corresponding value
        unique_corr_values, corr_value_counts = np.unique(corresponding_values, return_counts=True)
        freq_dict = dict(zip(unique_corr_values, corr_value_counts))
        
        # Normalize the frequency values
        normalized_freq_dict = normalize_dict_values(freq_dict)

        # Filter the normalized frequency values by keeping values greater than 0.1
        normalized_freq_dict = {key: value for key, value in normalized_freq_dict.items() if value > 0.1}

        # Map the key values with category IDs and get the labels
        mapped_dict = {category_mapping.get(int(key), 'background'): value for key, value in normalized_freq_dict.items()}

        # Update the val_arr1 dictionary with the mapped and filtered values
        val_arr1[int(value)] = mapped_dict

    return val_arr1

def train_assessment(args, config):
    model = build_model(config.model)
    model = revert_sync_batchnorm(model)
    model.to(args.device)
    model.eval()

    load_checkpoint(config, model, None, None, args.allow_shape_change)

    text_transform = build_text_transform(True, config.data.text_aug, with_dc=False,  train_assessment=True)
    
    #Choosing one of the classes to get the palette
    dataset = COCOObjectDataset
    # data = build_train_assessment_dataset(is_train=True, config=config.data)
    # print("data", data)
    model = GroupViTTrainAssessment(model)
    model.PALETTE = dataset.PALETTE

    data_loader_train = build_analysis_dataloader(config.data)
    test_pipeline = build_seg_demo_pipeline()
    #/misc/lmbraid21/sharmaa/coco_stuff164k/annotations/instances_train2017.json
    annotation_file = '/misc/lmbraid21/sharmaa/coco_stuff164k/annotations/instances_train2017.json'
    coco = COCO(annotation_file)
    json_objects = []
    #data_loader_seg = build_seg_dataloader(build_seg_dataset(cfg.evaluate.seg))
    #put a tqdm here
    # Get the category mapping from category ID to label name
    categories = coco.loadCats(coco.getCatIds())
    # print("categories",categories)
    # print("len categories",len(categories))
    category_mapping = {category['id']: category['name'] for category in categories}
    #print("category mapping",category_mapping)


    # Create a mapping dictionary for category ID to label
    coco_labels = {}
    for category in categories:
        category_id = category['id']
        category_label = category['name']
        coco_labels[category_id] = category_label
    json_file = '/misc/lmbraid21/sharmaa/analysis_files_pickle/'
    #check if dir exist else create it

    if not os.path.exists(os.path.dirname(json_file)):
        os.makedirs(os.path.dirname(json_file))
    for idx, samples in enumerate(data_loader_train):
        # if idx > 10:
        #     break
        print("idx", idx)
        #covert string to pathlib.Path
        img_path = Path(samples['filename'][0])
        data = dict(img=img_path)
        data = test_pipeline(data)
        data = collate([data], samples_per_gpu=1)
        result_dict = vis_seg(model, data)
        # Convert train2017 directory path to val2017 directory path
        val_directory = img_path.parent.parent.parent/'annotations' / 'train2017'

        # Get the filename without extension
        filename_without_extension = img_path.stem

        # Create the corresponding annotation file path in val2017
        val_annotation_file = val_directory / f'{filename_without_extension}.png'
        
        image_id = str(filename_without_extension).replace('0', '')
        # print("image_id", image_id)
        # print("image_id type", type(image_id))
        # Get the image IDs for the training split
        image_ids = coco.getImgIds()
        
        # image_id = None
        file=f'{filename_without_extension}.jpg'
        for id in image_ids:
            img_info = coco.loadImgs(id)[0]
            # print("file",file)
            # print("img_info", img_info['file_name'])
            if img_info['file_name'] == file:
                image_id = id
                break
        # print("cal image_id", image_id)
        # print("cal image_id type", type(image_id))

        # Load the annotations for the current image ID
        annotation_ids = coco.getAnnIds(imgIds=image_id)
        #print("annotation_ids", annotation_ids)
        annotations = coco.loadAnns(annotation_ids)
        #print("annotations", annotations)

        image_info = coco.loadImgs(image_id)[0]
        # Create an empty mask for the image
        mask = np.zeros((image_info['height'], image_info['width']), dtype=np.uint8)
        ####print("pre mask",mask.shape)
        for annotation in annotations:
            category_id = annotation['category_id']
            
            mask_temp = coco.annToMask(annotation)
            #print("mask_temp",mask_temp.shape)
            #print("category_id",category_id)
            # Fill the mask with the category ID value for the corresponding pixels
            mask = np.where(mask_temp == 1, category_id, mask)
            #print("after mask",mask.shape)
            
        #print("mask",mask)    
        #save mask as text file
        #np.savetxt('mask.txt', mask, fmt='%d')
        # print("mask shape",mask.shape)
        #print("mask type", type(mask))
        grouped_image = result_dict['grouped_image']
        # print("grouped_image",grouped_image.shape)
        #print("grouped_image type", type(grouped_image))
        grouped_img_token = result_dict['grouped_img_tokens']
        # print("grouped_img_token",grouped_img_token.shape)
        #print("grouped_img_token type", type(grouped_img_token))
        group_features = result_dict['group_feat']
        # print("group_features",group_features.shape)
        #print("group_features type", type(group_features))

        #create a dictionary to store the results
        result_dict = {'filename':str(img_path),
                       'grouped_image':grouped_image.tolist(),
                       'grouped_img_token':grouped_img_token.detach().cpu().numpy().tolist(),
                       'group_features':group_features.detach().cpu().numpy().tolist(),
                       'mask':mask.tolist(),
                       'soft_label': find_category_for_feature(np.array(grouped_image), np.array(mask), category_mapping)}
        json_objects.append(result_dict)
        
        if idx % 100 == 0:
            with open(json_file + f'/data_{idx}.pkl', 'wb') as f:
                pickle.dump(json_objects, f)
            json_objects = []
                

        #print("result dict", result_dict)
    with open(json_file + f'/data_{idx}.pkl', 'wb') as f:
        pickle.dump(json_objects, f)
    #json_file = '/misc/lmbraid21/sharmaa/analysis/image_annotations_10k.json'
    #     if idx % 1000 == 0:
    #         json_file_idx = f'json_file_{idx}.json'
    #         with open(json_file_idx, 'w') as f:
    #             json.dump(json_objects, f)
    #         print(f"JSON file saved for itr: {idx}")
    #         json_objects = []
    #     #break

    # with open(json_file, 'w') as f:
    #     json.dump(json_objects, f)
    # print(f"JSON file saved ")

    
    # 
    # from itertools import islice   
    # for i, sample in enumerate(islice(data, 0, 200)):
    #     for key, value in sample.items():
    #         print(key, repr(value))
    #         # # # prepare data
    #         if key is 'image':
    #             img = np.array(value)
    #             print("image shape", img.shape)
    #             print("type image", type(img))
    #             data = dict(img=img)
    #             data = test_pipeline(data)
    #             data = collate([data], samples_per_gpu=1)
    #     vis_seg(model, data, args.output_dir + '/' + str(i), args.vis)


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

    train_assessment(args, cfg)


if __name__ == '__main__':
    main()
