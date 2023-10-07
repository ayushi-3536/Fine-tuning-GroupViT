from pycocotools.coco import COCO
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Set the paths to the COCO annotation file and the corresponding images directory
annotation_file = '/misc/lmbraid21/sharmaa/coco_stuff164k/annotations/instances_val2017.json'
images_dir = '/misc/lmbraid21/sharmaa/coco_stuff164k/annotations/val2017/'

# Initialize the COCO API for loading annotations
coco = COCO(annotation_file)

# Get the image IDs for the validation split
image_ids = coco.getImgIds()

# Get the category mapping from category ID to label name
categories = coco.loadCats(coco.getCatIds())
category_mapping = {category['id']: category['name'] for category in categories}

# Create a color map for each category
color_map = {}
for category_id in category_mapping:
    color_map[category_id] = np.random.randint(0, 256, size=(3,), dtype=np.uint8)

# Iterate over the image IDs
for image_id in image_ids:
    try:
        # Load the image corresponding to the current image ID
        image_info = coco.loadImgs(image_id)[0]
        image_path = images_dir + image_info['file_name'].replace("jpg","png")
        image = cv2.imread(image_path)

        if image is None:
            raise Exception(f"Failed to load image: {image_path}")

        # Load the annotations for the current image ID
        annotation_ids = coco.getAnnIds(imgIds=image_id)
        annotations = coco.loadAnns(annotation_ids)

        # Create an empty mask for the image
        mask = np.zeros((image_info['height'], image_info['width']), dtype=np.uint8)
        print("pre mask",mask.shape)

        # Iterate over the annotations and assign labels to corresponding pixels in the mask
        for annotation in annotations:
            category_id = annotation['category_id']
            #segmentation = annotation['segmentation']
            mask_temp = coco.annToMask(annotation)
            #print("mask_temp",mask_temp.shape)
            print("category_id",category_id)
            # Fill the mask with the category ID value for the corresponding pixels
            mask = np.where(mask_temp == 1, category_id, mask)
            #print("after mask",mask.shape)
            
        print("mask",mask)    
        #save the mask
        np.save(f"mask_new{image_id}.npy",mask)
        #save the mask as tx1
        np.savetxt(f"mask_new{image_id}.txt",mask,fmt="%d")
        # Convert the mask to grayscale
        mask_rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
        for category_id in category_mapping:
            mask_rgb[mask == category_id] = color_map[category_id]
            # Get the label for the category ID
            

        print("mask rgb", mask_rgb.shape)



        # Display the image and the corresponding mask
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title('Image')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(mask_rgb)
        plt.title('Mask')
        plt.axis('off')


        plt.savefig(f"greyimage_new{image_id}.png")
        plt.close()
        break

    except Exception as e:
        print(f"Error processing image ID {image_id}: {str(e)}")
