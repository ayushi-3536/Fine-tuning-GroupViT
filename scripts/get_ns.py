import json
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

def get_synonyms(word):
    synonyms = set()
    synsets = wordnet.synsets(word)
    if synsets:
        target_synset = synsets[0]  # Consider the first synset as the target synset
        for syn in synsets[1:]:
            similarity = target_synset.wup_similarity(syn)
            if similarity is not None:
                synonyms.add(syn.lemmas()[0].name())
    return list(synonyms)

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
                    synonyms = get_synonyms(noun)
                    noun_synonyms[noun] = synonyms

    with open(output_file, 'w') as file:
        json.dump(noun_synonyms, file, indent=4)
# Example usage
input_file = '/misc/student/sharmaa/groupvit/OVSegmentor/scripts/coco_meta.json'
output_file = 'top5_ws_only_noun_synonyms_unique.json'
process_json_file(input_file, output_file)
