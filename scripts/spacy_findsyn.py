import json
import spacy

nlp = spacy.load('en_core_web_md')

def get_synonyms(word):
    synonyms = []
    token = nlp(word)
    if token.has_vector:
        word_vector = token.vector
        most_similar = sorted(nlp.vocab, key=lambda x: word_vector.dot(x.vector), reverse=True)
        for similar_word in most_similar:
            if similar_word.text != word:
                synonyms.append(similar_word.text)
            if len(synonyms) == 5:
                break
    return synonyms

def extract_nouns(sentence):
    doc = nlp(sentence)
    nouns = [token.text for token in doc if token.pos_ in ['NOUN', 'PROPN']]
    return nouns

def process_file(input_file, output_file):
    data = []
    with open(input_file, 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                entry = json.loads(line)
                filename = entry['filename']
                caption = entry['caption']
                nouns = extract_nouns(caption)
                noun_synonyms = {noun: get_synonyms(noun) for noun in nouns}
                data.append({filename: noun_synonyms})

    with open(output_file, 'w') as file:
        json.dump(data, file, indent=4)

input_file = '/misc/student/sharmaa/groupvit/OVSegmentor/scripts/coco_meta.json'
output_file = 'top5_synonyms_spacy.json'
process_file(input_file, output_file)
