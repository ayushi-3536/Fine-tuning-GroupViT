import json
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from collections import defaultdict

def extract_nouns(caption):
    tagged_words = pos_tag(word_tokenize(caption))
    nouns = [word for word, pos in tagged_words if pos in ['NN', 'NNS', 'NNP']]
    return nouns

def process_json_file(input_file, output_files):
    noun_frequency = defaultdict(int)
    with open(input_file, 'r') as file:
        for line in file:
            entry = json.loads(line)
            caption = entry['caption']
            nouns = extract_nouns(caption)
            for noun in nouns:
                noun_frequency[noun] += 1

    for freq_threshold, output_file in output_files:
        filtered_noun_frequency = {noun: freq for noun, freq in noun_frequency.items() if freq > freq_threshold}
        with open(output_file, 'w') as file:
            json.dump(filtered_noun_frequency, file, indent=4)

# Example usage
input_file = '/misc/student/sharmaa/groupvit/OVSegmentor/scripts/coco_meta.json'
output_files = [
    (100, 'noun_frequency_gt_100.json'),
    (500, 'noun_frequency_gt_500.json'),
    (1000, 'noun_frequency_gt_1000.json'),
    (5000, 'noun_frequency_gt_5000.json'),
    (2500, 'noun_frequency_gt_2500.json')
]
process_json_file(input_file, output_files)
