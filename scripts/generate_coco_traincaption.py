import json
from pycocotools.coco import COCO

# Load COCO annotations
coco = COCO('/misc/student/sharmaa/annotations/instances_train2017.json')

# Read data from JSON file
with open('/misc/student/sharmaa/groupvit/GroupViT/scripts/input_coco_caption.json', 'r') as file:
    data = json.load(file)

output = []

for image_id, item in data.items():
    # Get the image information from COCO annotations
    image_info = coco.loadImgs(int(image_id))[0]
    filename = image_info['file_name']
    caption = item["caption"]

    output_item = {
        "filename": filename,
        "caption": caption
    }

    output.append(output_item)

with open("new_coco_train_captions.json", "w") as outfile:
    for item in output:
        json.dump(item, outfile)
        outfile.write('\n')