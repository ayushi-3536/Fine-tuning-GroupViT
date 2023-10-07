import json
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

def extract_nouns(caption):
    tagged_words = pos_tag(word_tokenize(caption))
    nouns = [word for word, pos in tagged_words if pos in ['NN', 'NNS', 'NNP']]
    return nouns

def process_json_file(input_file, output_file):
    noun_synonyms = {}
    with open(input_file, 'r') as file:
        for line in file:
            entry = json.loads(line)
            caption = entry['caption']
            nouns = extract_nouns(caption)
            for noun in nouns:
                if noun not in noun_synonyms:
                    noun_synonyms[noun] = []

    with open(output_file, 'w') as file:
        json.dump(noun_synonyms, file, indent=4)

# Example usage
input_file = '/misc/student/sharmaa/groupvit/OVSegmentor/scripts/coco_meta.json'
output_file = 'extract_nouns_fromcc.json'
process_json_file(input_file, output_file)