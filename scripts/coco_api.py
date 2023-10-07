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
print("len",len(image_ids))

# Get the category mapping from category ID to label name
categories = coco.loadCats(coco.getCatIds())
print("categories",categories)
print("len categories",len(categories))
category_mapping = {category['id']: category['name'] for category in categories}
print("category mapping",category_mapping)


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
        print("len",len(annotations))
        print("ann",annotations)

        # Create an empty mask for the image
        mask = np.zeros((image_info['height'], image_info['width']), dtype=np.uint8)

        # Iterate over the annotations and assign labels to corresponding pixels in the mask
        for annotation in annotations:
            category_id = annotation['category_id']
            print("category",category_id)
            segmentation = annotation['segmentation']
            mask_temp = coco.annToMask(annotation)

            # Fill the mask with the category ID value for the corresponding pixels
            mask = np.where(mask_temp == 1, category_id, mask)

            # Get the label for the category ID
            label = category_mapping[category_id]
            print("label",label)

            # Get the bounding box coordinates
            bbox = annotation['bbox']
            x, y, w, h = bbox
            
            # Display the label within the mask
            contours, _ = cv2.findContours(mask_temp.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(mask, contours, -1, (255, 255, 255), thickness=cv2.FILLED)
            
            # Display the label along with the bounding box
            cv2.putText(mask, label, (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        print("mask",mask)
        # Display the image and the corresponding mask
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title('Image')
        plt.axis('off')

    

        plt.subplot(1, 2, 2)
        plt.imshow(mask)
        plt.title('Mask')
        plt.axis('off')

        plt.savefig(f"image_new{image_id}.png")
        plt.close()
        break

    except Exception as e:
        print(f"Error processing image ID {image_id}: {str(e)}")

#/misc/lmbraid21/sharmaa/coco_stuff164k/annotations/val2017/000000054592.png
# # Initialize the COCO API for loading annotations
# coco = COCO(annotation_file)

# # Get the image IDs for the validation split
# image_ids = coco.getImgIds()

# # Iterate over the image IDs
# for image_id in image_ids:
#     # Load the image corresponding to the current image ID
#     image_info = coco.loadImgs(image_id)[0]
#     image_path = images_dir + image_info['file_name']
#     print(image_path)
#     image = cv2.imread(image_path)
#     print(image.shape)

#     # Load the annotations for the current image ID
#     annotation_ids = coco.getAnnIds(imgIds=image_id)
#     annotations = coco.loadAnns(annotation_ids)

#     # Create an empty mask for the image
#     mask = np.zeros((image_info['height'], image_info['width']), dtype=np.uint8)

#     # Iterate over the annotations and assign labels to corresponding pixels in the mask
#     for annotation in annotations:
#         category_id = annotation['category_id']
#         segmentation = annotation['segmentation']
#         coco.annToMask(annotation)

#         # Fill the mask with the category ID value for the corresponding pixels
#         mask = np.where(mask == 1, category_id, mask)

#     # Display the image and the corresponding mask
#     plt.subplot(1, 2, 1)
#     plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#     plt.title('Image')
#     plt.axis('off')

#     plt.subplot(1, 2, 2)
#     plt.imshow(mask)
#     plt.title('Mask')
#     plt.axis('off')

#     plt.savefig('coco_mask.png')
#     break
