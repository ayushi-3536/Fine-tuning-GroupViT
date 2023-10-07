import json
import torch
import numpy as np
import os
import pickle
import mmcv
from PIL import Image
import os.path as osp
from mmseg.datasets import build_dataset
from pycocotools.coco import COCO
from sklearn.neighbors import NearestNeighbors
from concurrent.futures import ThreadPoolExecutor
import wandb
from omegaconf import OmegaConf, read_write
name='pascalval_nn10_nobal_l0.0withbg'
output_dir = f'/misc/lmbraid21/sharmaa/pascal_visualgrouping/{name}/'
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
def load_json_files(directory):
    print("loading pickle files")
    json_files = [f for f in os.listdir(directory) if f.endswith('.pkl')]
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
        class_indices = np.where(labels[:, class_idx] > 0)[0]
        print("class_indices", len(class_indices))
        min_num_features = min(min_num_features, len(class_indices))
        sampled_indices = np.random.choice(class_indices, size=min_num_features, replace=False)
        balanced_features.extend(features[sampled_indices])
        balanced_labels.extend(labels[sampled_indices])
    
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
directory = '/misc/lmbraid21/sharmaa/pascal_analysis_files_pickle/'

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

#group_feat, soft_label = balance_data(group_feat, soft_label)
print("group_feat", group_feat.shape)
#soft_label = soft_label[:, 1:]

print("soft_label", soft_label.shape)
nbrs = NearestNeighbors(n_neighbors=10).fit(group_feat)
print("model fitted")
#now load json from valid dataset and find 5 nn
directory = '/misc/lmbraid21/sharmaa/pascal_val_analysis_files/'
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

matched_soft_labels = []
for i in range(indices.shape[0]):
    matched_indices = indices[i]
    print("matched_indices", matched_indices)
    matched_soft_labels.append([soft_label[j] for j in matched_indices])
    print("len matched_soft_labels", len(matched_soft_labels))

matched_soft_labels = np.array(matched_soft_labels)
print("matched_soft_labels shape", matched_soft_labels.shape)
print("matched_soft_labels", matched_soft_labels)
mean_soft_labels = np.sum(matched_soft_labels, axis=1)
print("mean_soft_labels", mean_soft_labels.shape)
print("mean_soft_labels", mean_soft_labels)
max_label_index = np.argmax(mean_soft_labels[:,:], axis=-1)
print("max_label_index shape", max_label_index.shape)
print("max_label_index", max_label_index)

# change size of max_label_index from [B*N,] to [B, N, 1]
max_label_index = max_label_index.reshape(-1, 8).tolist()
print("max_label_index", max_label_index)
print("len max_label_index", len(max_label_index))

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





            
