# import json

# # Read the file with filename-index pairs
# with open('class_label_new.json', 'r') as file:
#     filename_index_data = json.load(file)

# # Read the file with filename-caption pairs
# with open('coco_meta_new.json', 'r') as file:
#     filename_caption_data = json.load(file)

# # Create a dictionary to store the mapping of indices to filenames and captions
# index_mapping = {}

# # Iterate through each filename-index pair in the first file
# for filename, index in filename_index_data.items():
#     index = str(index)  # Convert index to string for consistency
#     print(f"filename: {filename}, index: {index}")
#     # Get the corresponding caption for the filename from the second file
#     caption = filename_caption_data.get(filename, None)
#     print(f"caption: {caption}")
#     # If the caption exists, add the filename and caption to the index_mapping dictionary
#     if caption is not None:
#         if index in index_mapping:
#             index_mapping[index].append([filename, caption])
#         else:
#             index_mapping[index] = [[filename, caption]]
#     print(f"index_mapping: {index_mapping[index]}")

# # Convert the index_mapping dictionary to the desired output format
# output_data = {}
# for index, pairs in index_mapping.items():
#     output_data[index] = pairs

# # Write the output_data to a new JSON file
# with open('sample_list.json', 'w') as file:
#     json.dump(output_data, file)

import json

# Read the file with filename-index pairs
with open('class_label_new.json', 'r') as file:
    filename_index_data = json.load(file)

# Read the file with filename-caption pairs
filename_caption_data = {}
with open('coco_meta_new.json', 'r') as file:
    for line in file:
        data = json.loads(line)
        filename = data['filename']
        caption = data['caption']
        if filename in filename_caption_data:
            filename_caption_data[filename].append(caption)
        else:
            filename_caption_data[filename] = [caption]

# Create a dictionary to store the mapping of indices to filenames and captions
index_mapping = {}

# Iterate through each filename-index pair in the first file
for filename, index in filename_index_data.items():
    index = str(index)  # Convert index to string for consistency
    print(f"filename: {filename}, index: {index}")
    # Get the corresponding captions for the filename from the second file
    captions = filename_caption_data.get(filename, [])
    print(f"captions: {captions}")
    # If there are captions, add the filename and captions to the index_mapping dictionary
    if captions:
        if index in index_mapping:
            index_mapping[index].append([filename, captions])
        else:
            index_mapping[index] = [[filename, captions]]
    #print(f"index_mapping: {index_mapping[index]}")
    # Write the index_mapping dictionary to a new JSON file
    with open(f'/misc/lmbraid21/sharmaa/sample_list_{index}.json', 'w') as file:
        json.dump(index_mapping, file)
