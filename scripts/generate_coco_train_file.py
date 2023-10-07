import json

input_file = "/misc/student/sharmaa/groupvit/GroupViT/caption_of_coco_instances_train_2017.json"
output_file = "/misc/student/sharmaa/groupvit/GroupViT/coco_train_allinstances.txt"

with open(input_file, "r") as file:
    data = json.load(file)

with open(output_file, "w") as file:
    for image_id, info in data.items():
        filename = str(info["image_id"]).zfill(12) + ".jpg"
        caption = info["caption"]
        output = {"filename": filename, "caption": caption}
        file.write(json.dumps(output) + "\n")
