import json
import random

def process_json_file(input_file, output_file):
    with open(input_file, 'r') as f:
        data = [json.loads(line) for line in f]

    filename_counts = {}
    filtered_data = []

    for item in data:
        filename = item['filename']
        if filename in filename_counts:
            filename_counts[filename].append(item)
        else:
            filename_counts[filename] = [item]

    for filename, items in filename_counts.items():
        if len(items) == 1:
            filtered_data.append(items[0])
        else:
            selected_item = random.choice(items)
            filtered_data.append(selected_item)

    with open(output_file, 'w') as f:
        for item in filtered_data:
            f.write(json.dumps(item) + '\n')

# Usage example:
input_file = '/misc/student/sharmaa/groupvit/OVSegmentor/scripts/coco_meta.json'
output_file = '/misc/student/sharmaa/groupvit/GroupViT/scripts/coco_meta_new.json'
process_json_file(input_file, output_file)
