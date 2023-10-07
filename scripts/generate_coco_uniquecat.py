from pycocotools.coco import COCO
import random
import json

# Path to the COCO annotations file for the dataset
annotations_file = '/misc/student/sharmaa/annotations/instances_val2017.json'

# Initialize COCO API for the dataset
coco = COCO(annotations_file)

# Get all image IDs present in the dataset
image_ids = coco.getImgIds()

# Iterate over each image ID
for image_id in image_ids:
    # Load the image information for the current image ID
    image_info = coco.loadImgs(image_id)[0]
    filename = image_info['file_name']

    # Get all annotations for the current image ID
    annotations_ids = coco.getAnnIds(imgIds=image_id)
    annotations = coco.loadAnns(annotations_ids)

    # Get the unique category names for the annotations
    category_ids = set([ann['category_id'] for ann in annotations])
    category_names = [coco.loadCats(category_id)[0]['name'] for category_id in category_ids]

    # Generate a random caption by combining the labels
    caption = f"a photo of {' and some '.join(category_names)}"

    # Create a dictionary with the filename and caption
    data = {
        "filename": filename,
        "caption": caption
    }

    # Save the data as a JSON object
    with open('coco_val_captions.json', 'a') as f:
        json.dump(data, f)
        f.write('\n')

print("Data saved to output.json")
