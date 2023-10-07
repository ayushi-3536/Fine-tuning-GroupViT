import os
import cv2
import numpy as np
from segmentation.datasets import COCOObjectDataset, PascalContextDataset

import matplotlib.pyplot as plt
from PIL import Image
# Directory containing annotation masks
#annotation_dir = '/misc/lmbraid21/sharmaa/coco_stuff164k/annotations/val2017'
annotation_dir = '/misc/lmbraid21/sharmaa/VOCdevkit/VOC2010/SegmentationClassContext'
#dataset = COCOObjectDataset
dataset = PascalContextDataset
colormap = dataset.PALETTE
print("colormap", colormap)

# List all files in the annotation directory
mask_files = os.listdir(annotation_dir)
#mask_files = [f for f in mask_files if f.endswith('1425_instanceTrainIds.png')]
def colorize_mask(mask_array):
    # Define a colormap to assign colors to different classes
    colormap = dataset.PALETTE
    height, width = mask_array.shape
    colored_mask = np.zeros((height, width, 3), dtype=np.uint8)
    print("mask array", np.unique(mask_array))

    for class_id, color in enumerate(colormap):
        colored_mask[mask_array == class_id] = color

    return colored_mask
# Iterate through mask files
mask_files = sorted(mask_files)
for idx, mask_file in enumerate(mask_files):
    # Load annotation mask image
    #load the mask file
    print("mask file", mask_file)
    mask_path = os.path.join(annotation_dir, mask_file)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    print("mask shape", mask.shape)
    
    save_path = './outputs/masks/'
    os.makedirs(save_path, exist_ok=True)
    save_path = os.path.join(save_path, mask_file)

    if mask is not None:
        print("mask shape:", mask.shape)

        # Create a colormap for visualization (you can customize the colormap)
        colormap = plt.get_cmap('viridis')  # You can change 'viridis' to any other colormap
        
        plt.figure()
        plt.imshow(mask)
        plt.colorbar()

        # Save the figure as an image without displaying it
        output_image_path = save_path + 'colored_mask_{}.png'
        plt.savefig(output_image_path)

        print("Colored mask saved as", output_image_path.format(idx))
        # Apply the colormap to the mask image
        # colored_mask = colormap(mask)

        # # Display the colored mask
        # plt.imshow(colored_mask)
        # plt.colorbar()
        # plt.savefig('./outputs/colored_masks/'+mask_file)
    else:
        print("Failed to load the mask image.")
    #save the mask file as a text file
    #save the mask file as an image
    # Save the colored mask as an image
    mask_image = Image.fromarray(mask)
    mask_image.save(save_path)
    
    colored_mask = colorize_mask(mask)
    save_path = './outputs/colored_masks/'


    #save_path = '/misc/student/sharmaa/groupvit/GroupViT'
    #save_path = '/misc/lmbraid21/sharmaa/context_coloredmask/'
    os.makedirs(save_path, exist_ok=True)
    save_path = os.path.join(save_path, mask_file)
    #save the colored mask with each pixel having the class number as an image

    # Save the colored mask as an image
    colored_mask_image = Image.fromarray(colored_mask)
    colored_mask_image.save(save_path)
    if idx == 10 :
        break
        

