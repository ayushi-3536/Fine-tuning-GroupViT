import json
import torch
import numpy as np
import os
import pickle
import mmcv
import os.path as osp
from mmseg.datasets import build_dataset
from pycocotools.coco import COCO
from sklearn.neighbors import NearestNeighbors
from concurrent.futures import ThreadPoolExecutor
import wandb
from omegaconf import OmegaConf, read_write
name='cocoval_300files_knn5'
wandb.init(
        project='group_vit',
        name=osp.join('analysis',name),
        dir=f'/misc/lmbraid21/sharmaa/resultsanalysis/{name}/wandb'
)
# Load the COCO dataset annotations file
annotation_file = '/misc/lmbraid21/sharmaa/coco_stuff164k/annotations/instances_val2017.json'
coco = COCO(annotation_file)
categories = coco.loadCats(coco.getCatIds())
coco_labels = {}
coco_labels['background'] = 0

for idx, category in enumerate(categories):
    category_label = category['name']
    coco_labels[category_label] = idx +1

def load_json_file(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Loaded file: {file_path}")
    return data

def load_json_files(directory):
    print("loading pickle files")
    json_files = [f for f in os.listdir(directory) if f.endswith('.pkl')][:100]
    grouped_feat = []
    soft_labels = []

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(load_json_file, os.path.join(directory, file)) for file in json_files]

        for future in futures:
            data = future.result()
            if isinstance(data, list):
                for item in data:
                    grouped_feat.append(item['group_features'])
                    soft_labels.append(item['soft_label'])
            else:
                grouped_feat.append(data['group_features'])
                soft_labels.append(data['soft_label'])
            
            # Clear up memory by deleting the loaded data
            del data

    return grouped_feat, soft_labels

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
        arr = np.zeros((8, 81))
        # Iterate through each key-value pair in the item dictionary
        for key, inner_dict in item.items():
            # Get the group feature index
            index = int(key)
            # Iterate through each key-value pair in the inner dictionary
            for label, value in inner_dict.items():
                #find index of label in categories
                label_index =  coco_labels[label] 
                # Set the value in the array at the corresponding indices
                arr[index, label_index] = value
        result_dict.append(arr)
    return result_dict

def get_val_seg_map(grouped_img, max_label_index):
    val_seg_map = []
    for group_img, max_label in zip(grouped_img, max_label_index):
            group_img = np.array(group_img)
            print("group_img", group_img.shape)
            print("max_label", max_label)
            non_zero_indices = np.nonzero(group_img)
            print("non_zero_indices", non_zero_indices)
            non_zero_values = group_img[non_zero_indices]
            unique_values = np.unique(non_zero_values)
            print("unique_values", unique_values)

            #get values at the indices of max_label which are present in unique_values
            unique_labels = {i:max_label[i] for i in unique_values if max_label[i] != 0}
            print("unique_labels", unique_labels)

            #initialize an np zero array of shape group_img
            arr = np.zeros(group_img.shape)

            for value, replacement in unique_labels.items():
                # Find indices where value is present
                indices = np.where(group_img == value)
                
                # Replace values with the corresponding replacement
                arr[indices] = replacement


            val_seg_map.append(arr)
    return val_seg_map

def get_metric(val_seg_map):
    cfg = mmcv.Config.fromfile(filename='demo/coco.py')
    dataset = build_dataset(cfg.data.test)
    metrics = [dataset.evaluate(val_seg_map, metric='mIoU')]
    print("metrics", metrics)
    return metrics


# Specify the directory containing the JSON files
directory = '/misc/lmbraid21/sharmaa/analysis_files_pickle/'

# Load JSON files into a list of dictionaries
group_feat, soft_label = load_json_files(directory)
print("group_feat", len(group_feat))

group_feat = np.array(group_feat)
print("group_feat", group_feat.shape)

soft_label = np.array(get_array_from_softlabel(soft_label))
print("soft_label", soft_label.shape)

#change group feat from [B, N, D] to [B*N, D]
group_feat = group_feat.reshape(-1, group_feat.shape[-1])
print("group_feat", group_feat.shape)
#change soft label from [B, N, C] to [B*N, C]
soft_label = soft_label.reshape(-1, soft_label.shape[-1])
print("soft_label", soft_label.shape)

nbrs = NearestNeighbors(n_neighbors=5).fit(group_feat)
print("model fitted")
#now load json from valid dataset and find 5 nn
directory = '/misc/lmbraid21/sharmaa/fann_seqnewval_analysis_files/'
group_feat, grouped_img, filename = load_val_json_files(directory)

grouped_img = np.array(grouped_img)
print("group_img", grouped_img.shape)

group_feat = np.array(group_feat)
print("group_feat", group_feat.shape)

#change group feat from [B, N, D] to [B*N, D]
group_feat = group_feat.reshape(-1, group_feat.shape[-1])
print("group_feat", group_feat.shape)

distances, indices = nbrs.kneighbors(group_feat)

matched_soft_labels = []
for i in range(indices.shape[0]):
    matched_indices = indices[i]
    matched_soft_labels.append([soft_label[j] for j in matched_indices])

matched_soft_labels = np.array(matched_soft_labels)
print("matched_soft_labels", matched_soft_labels.shape)
print("matched_soft_labels", matched_soft_labels)
mean_soft_labels = np.sum(matched_soft_labels, axis=1)
print("mean_soft_labels", mean_soft_labels.shape)
print("mean_soft_labels", mean_soft_labels)
max_label_index = np.argmax(mean_soft_labels[:,1:], axis=-1)
print("max_label_index", max_label_index.shape)
print("max_label_index", max_label_index)

# change size of max_label_index from [B*N,] to [B, N, 1]
max_label_index = max_label_index.reshape(-1, 8).tolist()
print("max_label_index", max_label_index)
print("max_label_index", len(max_label_index))

val_seg_map = get_val_seg_map(grouped_img, max_label_index)
print("val_seg_map", len(val_seg_map))
mIoU = get_metric(val_seg_map)
print("mIoU", mIoU)



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





            
