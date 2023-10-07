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
import xmltodict
import os
import pickle
from PIL import Image
from mmcv.cnn.utils import revert_sync_batchnorm
from mmcv.image import tensor2imgs
from mmcv.parallel import collate, scatter
from models import build_model
from omegaconf import read_write
from einops import rearrange
from segmentation.datasets import PascalVOCDataset
from segmentation.evaluation import build_seg_demo_pipeline, build_train_assessment_pipeline
from utils import get_config, load_checkpoint
import numpy as np
from segmentation.evaluation import GroupViTTrainAssessment
from pathlib import Path
from extractors import ViTExtractor
from torch.nn import functional as F
from sd_dino_utils.utils_correspondence import resize
from sklearn.cluster import KMeans

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


feature_map = []
label_map = []

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
    parser.add_argument('--device', default='cpu', help='Device used for inference')
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
        normalized_freq_dict = {key: value for key, value in normalized_freq_dict.items() if value > 0.05}
        print("normalized_freq_dict", normalized_freq_dict)

        # Map the key values with category IDs and get the labels
        mapped_dict = {class_id_to_name.get(int(key), 'background'): value for key, value in normalized_freq_dict.items()}
        print("mapped_dict", mapped_dict)

        # Update the val_arr1 dictionary with the mapped and filtered values
        val_arr1[int(value)] = mapped_dict
    print("val_arr1", val_arr1)
    return val_arr1

def find_channels(anno, arr1 ):
    # Step 1: Find all unique values from anno
    arr1 = arr1.cpu().numpy()
    unique_classes = np.unique(anno)
    print("unique_values_arr2", unique_classes)

    trainfeat = {}
    
    for value in unique_classes:
        if value == 0 or value == 255:
            continue
        print("value", value)
        position = np.where(anno == value)

        # #define an array to store the result
        # # Step 3: Find channels at corresponding positions in arr1  
        channels = [arr1[row, col, :] for row,col in zip(position[0], position[1])] 
        #print("len channels", len(channels))
        channels = np.array(channels)
        print("channels", channels.shape)
        #randomly select 20 channels
        random_indices = np.random.choice(channels.shape[0], 20, replace=False)
        channels = channels[random_indices]
        print("channels after filtering", channels.shape)
        feature_map.append(channels)
        #create a zero tensor of shape [20, 21]
        label = np.zeros((20, 21), dtype=np.uint8)
        #fill the positions in the result_array with the value
        label[:, value] = 1
        label_map.append(label)
        print("label", label_map[-1].shape)
        print("feature_map", feature_map[-1].shape)


        # mean_channels = np.mean(channels, axis=0)
        # #find argmax index
        # argmax_index = np.argmax(mean_channels)
        # #print("argmax_index", argmax_index)
        # variance = np.var(channels, axis=0)
        # var[value] = variance[value]
        # #print("mean_channels", mean_channels.shape)
        # #print("var shape",variance.shape)
        # #print("var", variance)
        # #fill the positions in the result_array with the value from argmax_index
        # for row,col in zip(position[0], position[1]):
        #     result_array[row, col] = np.uint8(argmax_index)
    #return result_array, var


def get_array_from_softlabel(soft_label, num_clusters):
        # Iterate through each item in the list
        
        # Create an empty NumPy array with shape (8, 81)
        arr = np.zeros((num_clusters, 21))
        # Iterate through each key-value pair in the item dictionary
        for key, inner_dict in soft_label.items():
            # Get the group feature index
            index = int(key)
            # Iterate through each key-value pair in the inner dictionary
            for label, value in inner_dict.items():
                #find index of label in categories
                label_index =  class_name_to_id[label] 
                # Set the value in the array at the corresponding indices
                arr[index, label_index] = value
        
        return arr

def train_assessment(args):

    extractor = ViTExtractor('dino_vits16', 16, device=args.device)
    json_file = '/misc/lmbraid21/sharmaa/dino_pvoc_k8_layer8_nbic/'
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

    feature_map = []
    label_map = []
    num_clusters=8
    
    # Loop through each image ID and get the corresponding image and mask
    for idx, img_id in enumerate(train_ids):
        print("img_id", img_id)
        if idx>1:
            break
        img_file = os.path.join(data_dir, 'JPEGImages', f"{img_id}.jpg")
        #mask_file
        mask_file = os.path.join(data_dir, 'SegmentationClass', f"{img_id}.png")
        # from PIL import Image
        mask = np.array(Image.open(mask_file))

        print("idx", idx)
        img_path = Path(img_file)
        oriimg = mmcv.imread(img_path)
        print("oriimg", oriimg.shape)
        img = extractor.preprocess_pil(oriimg)
        img_desc_dino = extractor.extract_descriptors(img, 9, 'key').squeeze(0)
        #img_desc_dino = F.normalize(img_desc_dino, dim=-1)

        aggregated_features = np.zeros((num_clusters, 384))
        kmeans = KMeans(n_clusters=num_clusters, random_state=0)
        clustered_label = kmeans.fit(img_desc_dino).labels_
        #clustered_feat = np.dot(clustered_label.T, img_desc_dino)
        for cluster_idx in range(num_clusters):
            # Find indices where cluster label matches the current cluster index
            indices = np.where(clustered_label == cluster_idx)[0]
            # Extract corresponding descriptors from img_desc_dino
            cluster_descriptors = img_desc_dino[indices]
            # Aggregate the descriptors for this cluster
            aggregated_features[cluster_idx] = torch.mean(cluster_descriptors, dim=0)
        #[8, 384]
        print("aggregated_features", aggregated_features.shape)
        #take cosine similarity bwn (196, 384) and (8, 384) get(8,196) and then interpolate
        #clustered_label = np.dot(aggregated_features, img_desc_dino.T)

        one_hot_array = np.zeros((196,8))
        one_hot_array[np.arange(196), clustered_label] = 1
        print("one_hot_array", one_hot_array.shape)
        
        #np.savetxt(img_id+'_obtaineclusteredlabel.txt', clustered_label, fmt='%d')
        clustered_label = np.reshape(one_hot_array, (196,8))
        print("one_hot_array", clustered_label)
        #save clustered array in a text file
        #np.savetxt(img_id+'_onehotclusteredlabel.txt', clustered_label, fmt='%d')


        feat_ori = rearrange(clustered_label, '(h w) g -> g h w', h=14, w=14)
        print("feat ori", feat_ori.shape)
        feat_ori = torch.tensor(feat_ori).unsqueeze(0)
        print("ori image", oriimg.shape[:-1])
        #feat_ori_inter = F.interpolate(feat_ori, size=oriimg.shape[:-1], mode='nearest').squeeze(0)
        feat_ori_inter = F.interpolate(feat_ori, size=oriimg.shape[:-1], mode='bicubic', 
                                  align_corners=False).squeeze(0)
        
        print("feat_ori_inter", feat_ori_inter.shape)
        grouped_image = np.argmax(feat_ori_inter,axis=0)
        
        print("grouped image", grouped_image.shape)
        
        
        #interpolate (196,384) first, get (384, H,W), then cosine similarity (H, W, 8) and now argmax
        # feat_ori = rearrange(img_desc_dino, '(h w) d -> d h w', h=14, w=14)
        # print("feat ori", np.array(feat_ori).shape)
        # print("ori image", oriimg.shape[:-1])
        # feat_ori = torch.tensor(feat_ori).unsqueeze(0)
        # print("featori", feat_ori.shape)
        # feat_ori_inter = F.interpolate(feat_ori, size=oriimg.shape[:-1], mode='bicubic', 
        #                           align_corners=False)
        # print("feat_ori_inter", feat_ori_inter.shape)
        # #.squeeze(0)
        # # Reshape tensor_2 to (384, H*W)
        # reshaped_tensor_2 = feat_ori_inter.reshape(384, -1)

        # # Compute the dot product between tensor_1 and reshaped_tensor_2
        # dot_product = np.dot(aggregated_features, reshaped_tensor_2)
        # print("dot_product", dot_product.shape)

        # dot_product = dot_product.reshape(8, feat_ori_inter.shape[2], feat_ori_inter.shape[3])
        # print("dot_product", dot_product.shape)
        
        # grouped_image = np.argmax(dot_product,axis=0)


        
        # print("feat ori", feat_ori.shape)
        # feat_ori = torch.tensor(feat_ori).unsqueeze(0)
        # print("ori image", oriimg.shape[:-1])
        # #feat_ori_inter = F.interpolate(feat_ori, size=oriimg.shape[:-1], mode='nearest').squeeze(0)
        # feat_ori_inter = F.interpolate(feat_ori, size=oriimg.shape[:-1], mode='bicubic', 
        #                           align_corners=False).squeeze(0)
        
        # print("feat_ori_inter", feat_ori_inter.shape)
        colored_mask = colorize_mask(grouped_image)
        # Save the colored mask as an image
        colored_mask_image = Image.fromarray(colored_mask)
        #colored_mask_image.save(json_file+f'groupedimg_{idx}.png')
        colored_mask_image.save(f'demo/locafteronehotimgbicub_{idx}.png')

        soft_label = find_category_for_feature(np.array(grouped_image), np.array(mask), class_name_to_id)
        print("soft_label", soft_label)

        #labels = find_channels(mask, grouped_image) 
        labels = get_array_from_softlabel(soft_label, num_clusters)
        print("labels", labels.shape)

        feature_map.append(aggregated_features)
        label_map.append(labels)

        if len(feature_map) % 100 == 0:
            with open(json_file + f'/feat_{idx}.pkl', 'wb') as f:
               pickle.dump(feature_map, f)
            with open(json_file + f'/label_{idx}.pkl', 'wb') as f:
               pickle.dump(label_map, f)
            feature_map.clear()
            label_map.clear()

    with open(json_file + f'/feat_{idx}.pkl', 'wb') as f:
            pickle.dump(feature_map, f)
    with open(json_file + f'/label_{idx}.pkl', 'wb') as f:
            pickle.dump(label_map, f)




    #     colored_mask = colorize_mask(mask)
    #     # Save the colored mask as an image
    #     colored_mask_image = Image.fromarray(colored_mask)
    #     colored_mask_image.save('demo/colored_mask.png')

    #             feat_ori = rearrange(img_desc_dino, 'b (h w) d -> b d h w', h=14, w=14)
    #     print("feat ori", feat_ori.shape)
    #     oriimg = rearrange(oriimg, 'h w c  -> c h w')
    #     feat_ori_inter = F.interpolate(feat_ori, size=oriimg.shape[-2:], mode='bilinear', 
    #                              align_corners=False)
    #     print("clustered_feat", clustered_feat.shape)
    #     #Iterate through each cluster's labels and corresponding descriptors
    #     for cluster_idx, cluster_labels in enumerate(clustered_label.T):
    #         # Find indices where cluster label matches the current cluster index
    #         indices = np.where(cluster_labels == cluster_idx)[0]
            
    #         # Extract corresponding descriptors from img_desc_dino
    #         cluster_descriptors = img_desc_dino[indices]
    #         print("cluster_descriptors", cluster_descriptors.shape)
            
    #         # Aggregate the descriptors for this cluster
    #         aggregated_features[cluster_idx] = np.mean(cluster_descriptors, axis=0)

    #     # Now aggregated_features contains the aggregated features for each cluster
    #     clustered_feat = aggregated_features.T
        
    #     soft_label = find_category_for_feature(np.array(clustered_feat), np.array(mask), class_name_to_id)
    #     print("soft_label", soft_label)


    #     colored_mask = colorize_mask(clustered_feat)
    #     # Save the colored mask as an image
    #     colored_mask_image = Image.fromarray(colored_mask)
    #     colored_mask_image.save('demo/groupedimg.png')
    #     feat_ori = rearrange(img_desc_dino, 'b (h w) d -> b d h w', h=14, w=14)
    #     print("feat ori", feat_ori.shape)
    #     oriimg = rearrange(oriimg, 'h w c  -> c h w')
    #     feat_ori_inter = F.interpolate(feat_ori, size=oriimg.shape[-2:], mode='bilinear', 
    #                              align_corners=False)
        
    #     feat_ori_inter = rearrange(feat_ori_inter, 'b d h w  -> b h w d').squeeze(0)
    #     print("after interpolation", feat_ori_inter.shape)
    #     find_channels(mask, feat_ori_inter) 
    #     print("len feature_map", len(feature_map))
    #     print("len label_map", len(label_map))
        
   
        
    #     mask = generate_segmentation_mask_from_voc(xml_path, grouped_image.shape[:2])
    #     print("mask_array", mask.shape)
    #     print("mask_array type", type(mask))
    #     #save mask as txt file
    #     np.savetxt(json_file + f'/mask_arr{idx}.txt', mask, fmt='%d')

    #     soft_label = find_category_for_feature(np.array(grouped_image), np.array(mask), class_name_to_id)
    #     print("soft_label", soft_label)
    #     colored_mask = colorize_mask(mask)

    #     # Save the colored mask as an image
    #     colored_mask_image = Image.fromarray(colored_mask)
    #     colored_mask_image.save('demo/colored_mask.png')

    #     colored_mask = colorize_mask(grouped_image)

    #     # Save the colored mask as an image
    #     colored_mask_image = Image.fromarray(colored_mask)
    #     colored_mask_image.save('demo/groupedimg.png')
    #     #create a dictionary to store the results

    #     json_objects.append(result_dict)
        

    #         json_objects = []
                

    #     print("result dict", result_dict)
    # with open(json_file + f'/data_{idx}.pkl', 'wb') as f:
    #     pickle.dump(json_objects, f)

def main():
    args = parse_args()

    train_assessment(args)


if __name__ == '__main__':
    main()
