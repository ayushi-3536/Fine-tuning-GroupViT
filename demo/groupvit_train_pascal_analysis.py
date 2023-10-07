import os
import cv2
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
import xmltodict
import os
import pickle
from PIL import Image
from datasets import build_dataloader, build_analysis_dataloader, build_text_transform, imagenet_classes

from datasets import build_text_transform
from mmcv.cnn.utils import revert_sync_batchnorm
from mmcv.image import tensor2imgs
from mmcv.parallel import collate, scatter
from models import build_model
from omegaconf import read_write
from segmentation.datasets import PascalVOCDataset
from segmentation.evaluation import build_seg_demo_pipeline, build_train_assessment_pipeline
from utils import get_config, load_checkpoint
import numpy as np
from segmentation.evaluation import GroupViTTrainAssessment
from datasets import build_loader_sync, build_train_assessment_dataset
from pathlib import Path
def get_voc_colormap():
    voc_colormap = {
        0: (0, 0, 0),      # background: black
        1: (128, 0, 0),    # aeroplane: maroon
        2: (0, 128, 0),    # bicycle: green
        3: (128, 128, 0),  # bird: olive
        4: (0, 0, 128),    # boat: navy
        5: (128, 0, 128),  # bottle: purple
        6: (0, 128, 128),  # bus: teal
        7: (128, 128, 128),# car: gray
        8: (64, 0, 0),     # cat: dark red
        9: (192, 0, 0),    # chair: red
        10: (64, 128, 0),  # cow: dark green
        11: (192, 128, 0), # diningtable: orange
        12: (64, 0, 128),  # dog: dark purple
        13: (192, 0, 128), # horse: pink
        14: (64, 128, 128),# motorbike: dark teal
        15: (192, 128, 128), # person: light pink
        16: (0, 64, 0),    # pottedplant: dark green
        17: (128, 64, 0),  # sheep: brown
        18: (0, 192, 0),   # sofa: bright green
        19: (128, 192, 0), # train: yellow
        20: (0, 64, 128),  # tvmonitor: dark blue
    }
    return voc_colormap
class_name_to_id = {
        'background': 0,
        'aeroplane': 1,
        'bicycle': 2,
        'bird': 3,
        'boat': 4,
        'bottle': 5,
        'bus': 6,
        'car': 7,
        'cat': 8,
        'chair': 9,
        'cow': 10,
        'table': 11,
        'dog': 12,
        'horse': 13,
        'motorbike': 14,
        'person': 15,
        'plant': 16,
        'sheep': 17,
        'sofa': 18,
        'train': 19,
        'monitor': 20,
    }
class_id_to_name = {
        0: 'background',
        1: 'aeroplane',
        2: 'bicycle',
        3: 'bird',
        4: 'boat',
        5: 'bottle',
        6: 'bus',
        7: 'car',
        8: 'cat',
        9: 'chair',
        10: 'cow',
        11: 'table',
        12: 'dog',
        13: 'horse',
        14: 'motorbike',
        15: 'person',
        16: 'plant',
        17: 'sheep',
        18: 'sofa',
        19: 'train',
        20: 'monitor',
    }

def colorize_mask(mask_array):
    # Define a colormap to assign colors to different classes
    colormap = get_voc_colormap()
    height, width = mask_array.shape
    colored_mask = np.zeros((height, width, 3), dtype=np.uint8)

    for class_id, color in colormap.items():
        colored_mask[mask_array == class_id] = color

    return colored_mask
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
        
def generate_segmentation_mask_from_voc(xml_path, image_size):
    with open(xml_path) as f:
        annotations = xmltodict.parse(f.read())

    mask_array = np.zeros(image_size, dtype=np.uint8)

    objects = annotations['annotation']['object']
    if not isinstance(objects, list):
        objects = [objects]

    for obj in objects:
        class_name = obj['name']
        if class_name not in class_name_to_id:
            continue  # Skip unknown class names
        class_id = class_name_to_id[class_name]
        class_mask = int(obj['bndbox']['xmin']), int(obj['bndbox']['ymin']), int(obj['bndbox']['xmax']), int(obj['bndbox']['ymax'])
        mask_array[class_mask[1]:class_mask[3], class_mask[0]:class_mask[2]] = class_id

    return mask_array

def normalize_dict_values(dictionary):
    # # Get the maximum value in the dictionary
    max_value = sum(dictionary.values())

    # Normalize the values by dividing each value by the maximum value
    normalized_dict = {key: value / max_value for key, value in dictionary.items()}

    return normalized_dict

def find_category_for_feature(arr1, arr2, category_mapping):
    val_arr1 = {}
    unique_values = np.unique(arr1)
    print("unique_values", unique_values)

    # Iterate over each unique value in arr1
    for value in unique_values:
        print("value", value)
        val_arr1[int(value)] = {}

        positions = np.where(arr1 == value)
        corresponding_values = arr2[positions]
        print("corresponding_values", np.unique(corresponding_values))

        # Calculate the frequency of each corresponding value
        unique_corr_values, corr_value_counts = np.unique(corresponding_values, return_counts=True)
        print("unique_corr_values", unique_corr_values)
        print("corr_value_counts", corr_value_counts)
        freq_dict = dict(zip(unique_corr_values, corr_value_counts))
        
        # Normalize the frequency values
        normalized_freq_dict = normalize_dict_values(freq_dict)

        # Filter the normalized frequency values by keeping values greater than 0.1
        normalized_freq_dict = {key: value for key, value in normalized_freq_dict.items() if value > 0.1}
        print("normalized_freq_dict", normalized_freq_dict)

        # Map the key values with category IDs and get the labels
        mapped_dict = {class_id_to_name.get(int(key), 'background'): value for key, value in normalized_freq_dict.items()}
        print("mapped_dict", mapped_dict)

        # Update the val_arr1 dictionary with the mapped and filtered values
        val_arr1[int(value)] = mapped_dict
    print("val_arr1", val_arr1)
    return val_arr1

def train_assessment(args, config):
    model = build_model(config.model)
    model = revert_sync_batchnorm(model)
    model.to(args.device)
    model.eval()

    load_checkpoint(config, model, None, None, args.allow_shape_change)

    #text_transform = build_text_transform(True, config.data.text_aug, with_dc=False,  train_assessment=True)
    
    #Choosing one of the classes to get the palette
    dataset = PascalVOCDataset
    model = GroupViTTrainAssessment(model)
    model.PALETTE = dataset.PALETTE

    #data_loader_train = build_analysis_dataloader(config.data)
    test_pipeline = build_seg_demo_pipeline()
    #/misc/lmbraid21/sharmaa/coco_stuff164k/annotations/instances_train2017.json

    json_file = '/misc/lmbraid21/sharmaa/pascal_analysis_files_pickle/'
    #check if dir exist else create it
    if not os.path.exists(os.path.dirname(json_file)):
        os.makedirs(os.path.dirname(json_file))

    # Create a mapping dictionary for category ID to label
    data_dir = '/misc/lmbraid21/sharmaa/VOCdevkit/VOC2012'

    # Get the list of image IDs from the training set
    train_dir = os.path.join(data_dir, 'ImageSets', 'Segmentation')
    train_file = os.path.join(train_dir, 'train.txt')

    with open(train_file, 'r') as f:
        train_ids = f.read().strip().split()
    
    print("train_ids", len(train_ids))
    json_objects = []

    # Loop through each image ID and get the corresponding image and mask
    for idx, img_id in enumerate(train_ids):
        print("img_id", img_id)
        img_file = os.path.join(data_dir, 'JPEGImages', f"{img_id}.jpg")
        #mask_file
        mask_file = os.path.join(data_dir, 'SegmentationClass', f"{img_id}.png")
        
        # from PIL import Image
        mask = np.array(Image.open(mask_file))
        
        # print("mask", mask.shape)
        # #save mask as txt file
        # np.savetxt(json_file + f'/mask_pil{idx}.txt', mask, fmt='%d')
        # mask  = mmcv.imread(mask_file, 'grayscale')
        # print("mask", mask.shape)
        # #save mask as txt file
        # np.savetxt(json_file + f'/mask_{idx}.txt', mask, fmt='%d')

        xml_path = os.path.join(data_dir, 'Annotations', f"{img_id}.xml")
        print("idx", idx)
        img_path = Path(img_file)
        data = dict(img=img_path)
        data = test_pipeline(data)
        data = collate([data], samples_per_gpu=1)
        print("data", data['img'][0].shape)
        print("data", data)

        result_dict = vis_seg(model, data)

        grouped_image = result_dict['grouped_image']

        print("grouped_image",grouped_image.shape)
        print("grouped_image type", type(grouped_image))
        grouped_img_token = result_dict['grouped_img_tokens']
        print("grouped_img_token",grouped_img_token.shape)
        print("grouped_img_token type", type(grouped_img_token))
        group_features = result_dict['group_feat']
        print("group_features",group_features.shape)
        print("group_features type", type(group_features))
        # mask = generate_segmentation_mask_from_voc(xml_path, grouped_image.shape[:2])
        # print("mask_array", mask.shape)
        # print("mask_array type", type(mask))
        # #save mask as txt file
        # np.savetxt(json_file + f'/mask_arr{idx}.txt', mask, fmt='%d')

        soft_label = find_category_for_feature(np.array(grouped_image), np.array(mask), class_name_to_id)
        print("soft_label", soft_label)
        colored_mask = colorize_mask(mask)

        # Save the colored mask as an image
        colored_mask_image = Image.fromarray(colored_mask)
        colored_mask_image.save('demo/colored_mask.png')

        colored_mask = colorize_mask(grouped_image)

        # Save the colored mask as an image
        colored_mask_image = Image.fromarray(colored_mask)
        colored_mask_image.save('demo/groupedimg.png')
        #create a dictionary to store the results
        result_dict = {'filename':str(img_path),
                       'grouped_image':grouped_image.tolist(),
                       'grouped_img_token':grouped_img_token.detach().cpu().numpy().tolist(),
                       'group_features':group_features.detach().cpu().numpy().tolist(),
                       'mask':mask.tolist(),
                       'soft_label': soft_label
                    }
        
        json_objects.append(result_dict)
        
        if idx % 200 == 0:
            with open(json_file + f'/data_{idx}.pkl', 'wb') as f:
                pickle.dump(json_objects, f)
            json_objects = []
                

        #print("result dict", result_dict)
    with open(json_file + f'/data_{idx}.pkl', 'wb') as f:
        pickle.dump(json_objects, f)

def main():
    args = parse_args()
    cfg = get_config(args)

    with read_write(cfg):
        cfg.evaluate.eval_only = True

    train_assessment(args, cfg)


if __name__ == '__main__':
    main()
