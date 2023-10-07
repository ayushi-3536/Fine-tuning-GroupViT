from datasets import clip_dataset
import json
TOP_100 = clip_dataset.TOP_CLASSES_1
syn_dict = clip_dataset.syn_dict
UNIQUE_CLASSES = clip_dataset.TOP_UNIQUE_CLASSES

filename_dict = {}

with open('coco_meta_new.json', 'r') as file:
    for line in file:
        data = json.loads(line)
        filename = data['filename']
        caption = data['caption']
        
        for entity in TOP_100:
            if entity in caption:
                value = syn_dict.get(entity, entity)
                print(f"entity:{entity}, value: {value}")
                if value in UNIQUE_CLASSES:
                    index = UNIQUE_CLASSES.index(value)
                    print(f"index:{index}, value: {value}")
                    filename_dict[filename] = str(index)
                    break

with open('output_file_new.json', 'w') as output_file:
    json.dump(filename_dict, output_file)


