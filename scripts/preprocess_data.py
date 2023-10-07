import json
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag

# Download the required NLTK resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def extract_nouns(caption):
    nouns = []
    sentences = sent_tokenize(caption)
    for sentence in sentences:
        words = word_tokenize(sentence)
        tagged_words = pos_tag(words)
        for word, tag in tagged_words:
            if tag.startswith('NN'):  # Select only nouns
                nouns.append(word)
    return nouns

# Read the JSON file
with open('/misc/student/sharmaa/groupvit/OVSegmentor/scripts/coco_meta.json', 'r') as file:
    json_data = json.load(file)

# Process and update each JSON object
updated_json_data = []
for data in json_data:
    caption = data['caption']
    nouns = extract_nouns(caption)
    data['nouns'] = nouns
    updated_json_data.append(data)

# Write the updated JSON objects to a file
with open('output_preprocessed_cococaption.json', 'w') as file:
    json.dump(updated_json_data, file, indent=4)
