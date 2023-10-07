import json
import torch
import numpy as np
import os
import pickle
import mmcv
from PIL import Image
import os.path as osp
import sys
parentdir = osp.dirname(osp.dirname(__file__))
sys.path.insert(0, parentdir)
from mmseg.datasets import build_dataset
from pycocotools.coco import COCO
from sklearn.neighbors import NearestNeighbors
from concurrent.futures import ThreadPoolExecutor
import wandb
from pathlib import Path
from extractors import ViTExtractor
from torch.nn import functional as F
from einops import rearrange
from omegaconf import OmegaConf, read_write
from extractors import ViTExtractor
from sklearn.cluster import KMeans

name='dinopascalfeatwithlabel>0.5_kmean4'
output_dir = f'/misc/lmbraid21/sharmaa/dinopascal_bg1kfeat_elsebal_nn1/{name}/'
os.makedirs(output_dir, exist_ok=True)

wandb.init(
        project='group_vit',
        name=osp.join('analysis',name),
        dir=f'{output_dir}wandb'
)
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
# Load the COCO dataset annotations file
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

def load_json_file(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Loaded file: {file_path}")
    return data

def colorize_mask(mask_array):
    # Define a colormap to assign colors to different classes
    colormap = get_voc_colormap()
    height, width = mask_array.shape
    colored_mask = np.zeros((height, width, 3), dtype=np.uint8)

    for class_id, color in colormap.items():
        colored_mask[mask_array == class_id] = color

    return colored_mask

def get_file_names_from_val_file(val_file_path):
    with open(val_file_path, 'r') as file:
        file_names = [line.strip() for line in file.readlines()]
    return file_names

def find_channels(anno, arr1 ):
    # Step 1: Find all unique values from anno
    arr1 = arr1.cpu().numpy()
    unique_classes = np.unique(anno)
    print("unique_values_arr2", unique_classes)
    feature_label = {}
    
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
        if channels.shape[0] < 20:
            continue
        random_indices = np.random.choice(channels.shape[0], 20, replace=False)
        channels = channels[random_indices]
        print("channels after filtering", channels.shape)
        feature_label[value] = channels
    
    return feature_label

def get_metric(val_seg_map):
    cfg = mmcv.Config.fromfile(filename='/misc/student/sharmaa/groupvit/GroupViT/segmentation/configs/_base_/datasets/pascal_voc12.py')
    dataset = build_dataset(cfg.data.test)
    metrics = [dataset.evaluate(val_seg_map, metric='mIoU')]
    print("metrics", metrics)
    return metrics

def val_assessment(soft_label):
        
    extractor = ViTExtractor('dino_vits16', 16, device='cpu')

    json_file = '/misc/lmbraid21/sharmaa/dino_val_pascal_analysis/'
    #check if dir exist else create it
    if not os.path.exists(os.path.dirname(json_file)):
        os.makedirs(os.path.dirname(json_file))
    val_file_path = '/misc/lmbraid21/sharmaa/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt'
    file_names_list = get_file_names_from_val_file(val_file_path)

    # Print the list of file names (for demonstration purposes)
    print(len(file_names_list))
    
    sorted_file_names = sorted(file_names_list)
    # Loop through each image ID and get the corresponding image and mask
    annotation_path = '/misc/lmbraid21/sharmaa/VOCdevkit/VOC2012/JPEGImages'
    
    val_annotation_path = '/misc/lmbraid21/sharmaa/VOCdevkit/VOC2012'
    result_masks_val = []
    num_clusters = 8
    for idx, img_file in enumerate(sorted_file_names):
        print("img_id", idx)
        img_path = annotation_path + '/' + img_file + '.jpg'
            
        img_path = Path(img_path)
        oriimg = mmcv.imread(img_path)
        print("oriimg", oriimg.shape)
        img = extractor.preprocess_pil(oriimg)
        img_desc_dino = extractor.extract_descriptors(img, 9, 'key').squeeze(0)
        aggregated_features = np.zeros((8, 384))
        kmeans = KMeans(n_clusters=8, random_state=0)
        clustered_label = kmeans.fit(img_desc_dino).labels_
        #clustered_feat = np.dot(clustered_label.T, img_desc_dino)
        for cluster_idx in range(num_clusters):
            # Find indices where cluster label matches the current cluster index
            indices = np.where(clustered_label == cluster_idx)[0]
            # Extract corresponding descriptors from img_desc_dino
            cluster_descriptors = img_desc_dino[indices]
            # Aggregate the descriptors for this cluster
            aggregated_features[cluster_idx] = torch.mean(cluster_descriptors, dim=0)
        
        print("aggregated_features", aggregated_features.shape)
        clustered_label = np.dot(aggregated_features, img_desc_dino.T)


        feat_ori = rearrange(clustered_label, 'g (h w) -> g h w', h=14, w=14)
        print("feat ori", feat_ori.shape)
        feat_ori = torch.tensor(feat_ori).unsqueeze(0)
        print("ori image", oriimg.shape[:-1])
        
        feat_ori_inter = F.interpolate(feat_ori, size=oriimg.shape[:-1], mode='bilinear', 
                                  align_corners=False).squeeze(0)
        
        print("feat_ori_inter", feat_ori_inter.shape)
        group_image = np.argmax(feat_ori_inter,axis=0)
        print("grouped image", group_image.shape)
        colored_mask = colorize_mask(group_image)
        # Save the colored mask as an image
        colored_mask_image = Image.fromarray(colored_mask)
        colored_mask_image.save(f'demo/groupedimg_{idx}.png')

        #colored_mask_image.save(json_file+f'groupedimg_{idx}.png')


        distances, indices = nbrs.kneighbors(aggregated_features)
        print("distances", distances.shape)
        print("indices", indices.shape)
        matched_soft_labels = []
        for i in range(indices.shape[0]):
            matched_indices = indices[i]
            matched_soft_labels.append([soft_label[j] for j in matched_indices])

        matched_soft_labels = np.array(matched_soft_labels)
        matched_soft_labels = np.array(matched_soft_labels)
        print("matched_soft_labels shape", matched_soft_labels.shape)
        mean_soft_labels = np.sum(matched_soft_labels, axis=1)
        print("mean_soft_labels", mean_soft_labels.shape)
        max_label_index = np.argmax(mean_soft_labels[:,:], axis=-1)
        print("max_label_index shape", max_label_index.shape)
        print("max_label_index", max_label_index)


        unique_values = np.unique(group_image)
        print("unique_values", unique_values)

        #get values at the indices of max_label which are present in unique_values
        unique_labels = {i:max_label_index[i] for i in unique_values if max_label_index[i] != 0}
        print("unique_labels", unique_labels)

        #initialize an np zero array of shape group_img
        arr = np.zeros(group_image.shape)

        for value, replacement in unique_labels.items():
            # Find indices where value is present
            indices = np.where(group_image == value)
            
            # Replace values with the corresponding replacement
            arr[indices] = replacement

        print("arr", arr.shape)
        result_masks_val.append(arr)

    print("result_masks", len(result_masks_val))
    mIoU = get_metric(result_masks_val)
    print("mIoU", mIoU)

            


        # find_channels(mask, feat_ori_inter) 
        # print("len feature_map", len(feature_map))
        # print("len label_map", len(label_map))
        # if len(feature_map) % 100 == 0:
        #     with open(json_file + f'/feat_{idx}.pkl', 'wb') as f:
        #        pickle.dump(feature_map, f)
        #     with open(json_file + f'/label_{idx}.pkl', 'wb') as f:
        #        pickle.dump(label_map, f)
        #     feature_map.clear()
        #     label_map.clear()

    # with open(json_file + f'/feat_{idx}.pkl', 'wb') as f:
    #         pickle.dump(feature_map, f)
    # with open(json_file + f'/label_{idx}.pkl', 'wb') as f:
    #         pickle.dump(label_map, f)

def load_json_files(directory):
    print("loading pickle files")


    feat_files = [f for f in os.listdir(directory) if f.startswith('feat')]
    label_files = [f for f in os.listdir(directory) if f.startswith('label')]
    #load json file having certain prefix

    feat = None
    labels = None

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(load_json_file, os.path.join(directory, file)) for file in feat_files]

        for future in futures:
            data = future.result()
            print("feat data loaded", np.array(data).shape)
            if feat is None:
                feat = np.array(data)
            else:
                feat = np.concatenate((feat, np.array(data)), axis=0)
            # Clear up memory by deleting the loaded data
            del data
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(load_json_file, os.path.join(directory, file)) for file in label_files]
        for future in futures:
            data = future.result()
            print("label data loaded", data[-1].shape)
            if labels is None:
                labels = np.array(data)
            else:
                labels = np.concatenate((labels, np.array(data)), axis=0)
            # Clear up memory by deleting the loaded data
            del data

    print("loaded pickle files")
    print("size of feat", len(feat))
    print("size of labels", len(labels))
    return feat, labels

def load_val_json_files(directory):
    json_files = [f for f in os.listdir(directory) if f.endswith('.json')]
    json_files = sorted(json_files)
    grouped_feat = []
    filenames = []
    grouped_img = []
    for file in json_files:
        file_path = os.path.join(directory, file)
        with open(file_path, 'r') as f:
            data = json.load(f)
            grouped_feat.append(data['group_features'])
            grouped_img.append(data['grouped_image'])
            filenames.append(data['filename'])
            del data
    return grouped_feat, grouped_img, filenames

def get_array_from_softlabel(soft_label):
    result_dict = []
    # Iterate through each item in the list
    
    for i, item in enumerate(soft_label):
        # Create an empty NumPy array with shape (8, 81)
        arr = np.zeros((8, 21))
        # Iterate through each key-value pair in the item dictionary
        for key, inner_dict in item.items():
            # Get the group feature index
            index = int(key)
            # Iterate through each key-value pair in the inner dictionary
            for label, value in inner_dict.items():
                #find index of label in categories
                label_index =  class_name_to_id[label] 
                # Set the value in the array at the corresponding indices
                arr[index, label_index] = value
        result_dict.append(arr)
    return result_dict

def get_val_seg_map(grouped_img, max_label_index):

    val_seg_map = []
    for i, (group_img, max_label) in enumerate(zip(grouped_img, max_label_index)):
            group_img = np.array(group_img)
            print("group_img", group_img.shape)
            # colored_mask = colorize_mask(group_img)

            # # Save the colored mask as an image
            # colored_mask_image = Image.fromarray(colored_mask)
            # colored_mask_image.save('demo/colored_mask_gi_val.png')
            print("group_img", group_img.shape)
            print("max_label", max_label)
            non_zero_indices = np.nonzero(max_label)[0]
            print("features which belong to a class", non_zero_indices)
            list_of_non_zero_indices = list(non_zero_indices)
            print("list_of_non_zero_indices", list_of_non_zero_indices)

            #initialize an np zero array of shape group_img
            arr = np.zeros(group_img.shape)

            for value in list_of_non_zero_indices:
                print("value", value)
                # Find indices where value is present
                indices = np.where(group_img == value)
                print("indices", indices)
                print("replaceemnet value", max_label[value])
                
                # Replace values with the corresponding replacement
                arr[indices] = max_label[value] 

            colored_mask = colorize_mask(arr)

            # Save the colored mask as an image
            colored_mask_image = Image.fromarray(colored_mask)
            #check if output dir exists else create it
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            colored_mask_image.save(output_dir + f'colored_mask_gi_val{i}.png')

            val_seg_map.append(arr)
    return val_seg_map

def get_metric(val_seg_map):
    cfg = mmcv.Config.fromfile(filename='/misc/student/sharmaa/groupvit/GroupViT/segmentation/configs/_base_/datasets/pascal_voc12.py')
    dataset = build_dataset(cfg.data.test)
    metrics = [dataset.evaluate(val_seg_map, metric='mIoU')]
    print("metrics", metrics)
    return metrics

def balance_data(features, labels):
    num_classes = labels.shape[1]
    print("num_classes", num_classes)
    max_indices = np.argmax(labels, axis=1)
    one_hot_array = np.zeros_like(labels, dtype=np.int)
    one_hot_array[np.arange(len(max_indices)), max_indices] = 1
    print("one_hot_array", one_hot_array)
    print("one_hot_array shape", one_hot_array.shape)
    
    num_features_per_class = np.sum(one_hot_array, axis=0)

    print("num_features_per_class", num_features_per_class)
    
    # Step 2: Find the class with the minimum number of features
    min_num_features = np.min(num_features_per_class)
    print("min_num_features", min_num_features)
    
    min_num_features = 1000
    # Step 3: Sample the same number of features for every class
    balanced_features = []
    balanced_labels = []
    for class_idx in range(num_classes):
        print("class_idx", class_idx)
        if num_features_per_class[class_idx] == 0:
            continue
        class_indices =  np.where(labels[:, class_idx] > 0.5)[0]
        #class_indices = np.random.choice(labels, size=feat_needed, replace=True)
        print("class_indices", len(class_indices))
        min_num_features = min(min_num_features, len(class_indices))
        sampled_indices = np.random.choice(class_indices, size=min_num_features, replace=False)
        
        balanced_features.extend(features[sampled_indices])
        balanced_labels.extend(labels[sampled_indices])
    # for class_idx in range(num_classes):
    #     print("class_idx", class_idx)
    #     if num_features_per_class[class_idx] == 0:
    #         continue
    #     onehotclass_indices = np.where(one_hot_array[:, class_idx] > 0)[0]
    #     soft_label_indices = []
    #     if min_num_features > len(onehotclass_indices):
    #         feat_needed = min_num_features - len(onehotclass_indices)
    #         print("feat_needed", feat_needed)
    #         soft_label_indices =  np.where(labels[:, class_idx] > 0.5)[0]
    #     #class_indices = np.random.choice(labels, size=feat_needed, replace=True)
    #     class_indices = onehotclass_indices.tolist() + soft_label_indices.tolist()
    #     print("class_indices", len(class_indices))
    #     min_num_features = min(min_num_features, len(class_indices))
    #     sampled_indices = np.random.choice(class_indices, size=min_num_features, replace=False)
        
    #     balanced_features.extend(features[sampled_indices])
    #     balanced_labels.extend(labels[sampled_indices])
    
    # Convert the balanced data to numpy arrays
    balanced_features = np.array(balanced_features)
    balanced_labels = np.array(balanced_labels)
    
    # Step 4: Shuffle the balanced data
    indices = np.arange(len(balanced_features))
    print("indices", indices)
    #np.random.shuffle(indices)
    balanced_features = balanced_features[indices]
    balanced_labels = balanced_labels[indices]
    
    return balanced_features, balanced_labels

# Specify the directory containing the JSON files
#directory = '/misc/lmbraid21/sharmaa/dino_pascal_analysis_kmeans/'
directory = '/misc/lmbraid21/sharmaa/dino_pascal_analysis_kmeans_4/'
# Load JSON files into a list of dictionaries
feat, label = load_json_files(directory)
print("group_feat", len(feat))

feat = np.array(feat)
print("feat", feat.shape)

label = np.array(label)
print("label", label.shape)


#change group feat from [B, N, D] to [B*N, D]
feat = feat.reshape(-1, feat.shape[-1])
print("feat", feat.shape)
#change soft label from [B, N, C] to [B*N, C]
label = label.reshape(-1, label.shape[-1])
print("label", label.shape)

feat, label = balance_data(feat, label)
# print("feat", feat.shape)
# #soft_label = soft_label[:, 1:]

# print("soft_label", soft_label.shape)
nbrs = NearestNeighbors(n_neighbors=25).fit(feat)
print("model fitted")
feat = val_assessment(label)
# #now load json from valid dataset and find 5 nn
# directory = '/misc/lmbraid21/sharmaa/pascal_val_analysis_files/'
# group_feat, grouped_img, filename = load_val_json_files(directory)

# grouped_img = np.array(grouped_img)
# print("group_img", grouped_img.shape)

# group_feat = np.array(group_feat)
# print("group_feat", group_feat.shape)

# #change group feat from [B, N, D] to [B*N, D]
# group_feat = group_feat.reshape(-1, group_feat.shape[-1])
# print("group_feat", group_feat.shape)

# distances, indices = nbrs.kneighbors(group_feat)
# print("distances", distances.shape)
# print("indices", indices.shape)

# matched_soft_labels = []
# for i in range(indices.shape[0]):
#     matched_indices = indices[i]
#     print("matched_indices", matched_indices)
#     matched_soft_labels.append([soft_label[j] for j in matched_indices])
#     print("len matched_soft_labels", len(matched_soft_labels))

# matched_soft_labels = np.array(matched_soft_labels)
# print("matched_soft_labels shape", matched_soft_labels.shape)
# print("matched_soft_labels", matched_soft_labels)
# mean_soft_labels = np.sum(matched_soft_labels, axis=1)
# print("mean_soft_labels", mean_soft_labels.shape)
# print("mean_soft_labels", mean_soft_labels)
# max_label_index = np.argmax(mean_soft_labels[:,:], axis=-1)
# print("max_label_index shape", max_label_index.shape)
# print("max_label_index", max_label_index)

# # change size of max_label_index from [B*N,] to [B, N, 1]
# max_label_index = max_label_index.reshape(-1, 8).tolist()
# print("max_label_index", max_label_index)
# print("len max_label_index", len(max_label_index))

# val_seg_map = get_val_seg_map(grouped_img, max_label_index)
# print("val_seg_map", len(val_seg_map))
# mIoU = get_metric(val_seg_map)
# print("mIoU", mIoU)



    # #save arr as text file
    # print("saving arr")
    # np.savetxt(f"mask_new{i}.txt",arr,fmt="%d")
    # #save group_img as text file
    # np.savetxt(f"group_img{i}.txt",group_img,fmt="%d")
    #
    # for i, label in enumerate(max_label):
    #     print("label", label)
    #     group_img[i] = coco_labels[label]
    # print("group_img", group_img)
    #break   





            
