import json
import torch
import numpy as np
import os
from pycocotools.coco import COCO
from sklearn.neighbors import NearestNeighbors
# Load the COCO dataset annotations file
annotation_file = '/misc/lmbraid21/sharmaa/coco_stuff164k/annotations/instances_val2017.json'
coco = COCO(annotation_file)
# Get the category mapping from category ID to label name
categories = coco.loadCats(coco.getCatIds())
# print("categories",categories)
# # print("len categories",len(categories))
# category_mapping = {category['name']: i for i, category in enumerate(categories)}
# category_mapping['background'] = 0
# print("category mapping",category_mapping)


# Create a mapping dictionary for category ID to label
coco_labels = {}
coco_labels['background'] = 0
for idx, category in enumerate(categories):
    category_label = category['name']
    coco_labels[category_label] = idx +1

# def normalize_dict_values(dictionary):
#     # Get the maximum value in the dictionary
#     max_value = max(dictionary.values())

#     # Normalize the values by dividing each value by the maximum value
#     normalized_dict = {key: value / max_value for key, value in dictionary.items()}

#     return normalized_dict

# def find_category_for_feature(arr1, arr2):
#     val_arr1 = {}
#     unique_values = np.unique(arr1)

#     # Iterate over each unique value in arr1
#     for value in unique_values:
#         val_arr1[value] = {}

#         # Find the corresponding values in arr2 at the same positions
#         positions = np.where(arr1 == value)
#         corresponding_values = arr2[positions]

#         # Calculate the frequency of each corresponding value
#         unique_corr_values, corr_value_counts = np.unique(corresponding_values, return_counts=True)
#         freq_dict = dict(zip(unique_corr_values, corr_value_counts))
        
#         # Normalize the frequency values
#         normalized_freq_dict = normalize_dict_values(freq_dict)

#         # Filter the normalized frequency values by keeping values greater than 0.1
#         normalized_freq_dict = {key: value for key, value in normalized_freq_dict.items() if value > 0.1}

#         # Map the key values with category IDs and get the labels
#         mapped_dict = {category_mapping.get(key, 'background'): value for key, value in normalized_freq_dict.items()}

#         # Update the val_arr1 dictionary with the mapped and filtered values
#         val_arr1[value] = mapped_dict

        

#     print(val_arr1)
#     return val_arr1

# # Load the JSON file
# json_file = '/misc/student/sharmaa/groupvit/GroupViT/image_annotations_test10.json'
# with open(json_file, 'r') as f:
#     data = json.load(f)
# print("data", len(data))
# # Iterate through the items in the dictionary
# for item in data:
#     print(type(item))
#     # for key, value in item.items():
#     #     print(key)
#     #     print(type(value))
#     #     if key == 'grouped_image' or key == 'group_features' or key == 'grouped_img_token' or key=='mask':
#     #         value = np.array(value)
#     #         item[key] = value
    
#     if 'soft_label' not in item.keys():
#         img = item['grouped_image']
#         mask = item['mask']
#         mapping = find_category_for_feature(img, mask)
#         print("mapping", mapping)
#         item['soft_label'] = mapping
#         print("items after adding soft labels", item)

# #load valid dataset find, group feat and see 5 nn
# Nearest_Neighbours = 5
# specific_key = 'group_features'
# values = np.array([item[specific_key] for item in data])
# print("values", values.shape)

def load_json_files(directory):
    json_files = [f for f in os.listdir(directory) if f.endswith('.json')][:10]
    grouped_feat = []
    soft_labels = []
    for file in json_files:

        file_path = os.path.join(directory, file)
        with open(file_path, 'r') as f:
            data = json.load(f)
            grouped_feat.append(data['group_features'])
            soft_labels.append(data['soft_label'])
    return grouped_feat, soft_labels

def load_val_json_files(directory):
    json_files = [f for f in os.listdir(directory) if f.endswith('.json')]
    json_files = sorted(json_files)[:10]
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
    return grouped_feat, grouped_img, filenames
# Specify the directory containing the JSON files
directory = '/misc/lmbraid21/sharmaa/analysis_files/'

# Load JSON files into a list of dictionaries
group_feat, soft_label = load_json_files(directory)
print("group_feat", len(group_feat))

group_feat = np.array(group_feat)

result_dict = []

# Iterate through each item in the list
for i, item in enumerate(soft_label):
    # Create an empty NumPy array with shape (8, 81)
    arr = np.zeros((8, 81))
    # Iterate through each key-value pair in the item dictionary
    for key, inner_dict in item.items():
        # Get the index corresponding to the key
        index = int(key)
        
        # Iterate through each key-value pair in the inner dictionary
        for label, value in inner_dict.items():
            # Get the index corresponding to the label (add 1 to offset the 'background' class)
            #find index of label in categories

            label_index =  coco_labels[label] 
            
            # Set the value in the array at the corresponding indices
            arr[index, label_index] = value
    
    # Add the array to the result dictionary using the index as the key
    
    #break
    result_dict.append(arr)

print("result_dict", len(result_dict))
soft_label = np.array(result_dict)
print("soft_label", soft_label.shape)

print("group_feat", group_feat.shape)

#change group feat from [B, N, D] to [B*N, D]
group_feat = group_feat.reshape(-1, group_feat.shape[-1])
print("group_feat", group_feat.shape)
#change soft label from [B, N, C] to [B*N, C]
soft_label = soft_label.reshape(-1, soft_label.shape[-1])
print("soft_label", soft_label.shape)
nbrs = NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(group_feat)
print("model fitted")
#now load json from valid dataset and find 5 nn
directory = '/misc/lmbraid21/sharmaa/seqnewval_analysis_files/'
group_feat, grouped_img, filename = load_val_json_files(directory)

grouped_img = np.array(grouped_img)
print("group_img", grouped_img.shape)

group_feat = np.array(group_feat)
print("group_feat", group_feat.shape)
#change group feat from [B, N, D] to [B*N, D]
group_feat = group_feat.reshape(-1, group_feat.shape[-1])
print("group_feat", group_feat.shape)
distances, indices = nbrs.kneighbors(group_feat)
print("distances", distances.shape)
print("indices", indices.shape)
#print("indices", indices)
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

    print("arr", arr.shape)

    #just a dummy to get miou
    import mmcv
    
    from mmseg.datasets import build_dataset
    #from segmentation.evaluation import build_seg_dataset
    #cfg = mmcv.Config.fromfile(filename='/misc/student/sharmaa/groupvit/GroupViT/segmentation/configs/_base_/datasets/coco.py')
    cfg = mmcv.Config.fromfile(filename='demo/coco.py')
    print("filename", type(filename))
    print("filename", filename[0])
   
    print("cfg.data.test", cfg.data.test)
    dataset = build_dataset(cfg.data.test)
    metrics = [dataset.evaluate([arr], metric='mIoU')]
    print("metrics", metrics)
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
    break   





            
