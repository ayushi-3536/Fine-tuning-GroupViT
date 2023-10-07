from pycocotools.coco import COCO
import random
import json

# Path to the COCO annotations file for the validation set
annotations_file = '/misc/student/sharmaa/annotations/instances_val2017.json'

# Initialize COCO API for the dataset
coco = COCO(annotations_file)

# Get all image IDs present in the dataset
image_ids = coco.getImgIds()

# List to store the data for each image
data_list = []

# Iterate over each image ID
for image_id in image_ids:
    # Load the image information for the current image ID
    image_info = coco.loadImgs(image_id)[0]
    filename = image_info['file_name']

    # Get all annotations for the current image ID
    annotations_ids = coco.getAnnIds(imgIds=image_id)
    annotations = coco.loadAnns(annotations_ids)
    
    # Get the unique label names for the annotations
    label_names = list(set([coco.loadCats(ann['category_id'])[0]['name'] for ann in annotations]))


    # Generate a random caption by combining the labels
    caption = f"a photo of {' and some '.join(label_names)}"

    # Create a dictionary with the filename and caption
    data = {
        "filename": filename,
        "caption": caption
    }

    # Append the data to the list
    data_list.append(data)

# Save the data as JSON
output_file = 'unique_output_new.json'
with open(output_file, 'w') as f:
    json.dump(data_list, f)

print("Data saved to", output_file)

