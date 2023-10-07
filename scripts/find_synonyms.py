# import json
# from nltk.corpus import wordnet
# from nltk.tokenize import word_tokenize
# from nltk.tag import pos_tag

# def extract_nouns(sentence):
#     tagged_words = pos_tag(word_tokenize(sentence))
#     nouns = [word for word, pos in tagged_words if pos in ['NN', 'NNS', 'NNP']]
#     return nouns

# def get_synonyms(word):
#     synonyms = []
#     for syn in wordnet.synsets(word):
#         for lemma in syn.lemmas():
#             synonym = lemma.name()
#             if synonym not in synonyms:
#                 synonyms.append(synonym)
#     return synonyms

# def process_file(input_file, output_file):
#     data = []
#     with open(input_file, 'r') as file:
#         for line in file:
#             line = line.strip()
#             if line:
#                 entry = json.loads(line)
#                 filename = entry['filename']
#                 caption = entry['caption']
#                 nouns = extract_nouns(caption)
#                 noun_synonyms = {noun: get_synonyms(noun) for noun in nouns}
#                 data.append({filename: noun_synonyms})

#     with open(output_file, 'w') as file:
#         json.dump(data, file, indent=4)

# # Example usage
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
import json
from operator import itemgetter

# def get_synonyms(word, pos):
#     synonyms = set()
#     synsets = wordnet.synsets(word, pos=pos)
#     if synsets:
#         target_synset = synsets[0]  # Consider the first synset as the target synset
#         for syn in synsets[1:]:
#             similarity = target_synset.wup_similarity(syn)
#             if similarity is not None and similarity > 0.7:  # Adjust the similarity threshold as desired
#                 synonyms.add(syn.lemmas()[0].name())
#     return list(synonyms)

#Get only top 5 synonyms
#Using Wordnet
# def get_synonyms(word, pos):
#     synonyms = []
#     synsets = wordnet.synsets(word, pos=pos)
#     if synsets:
#         target_synset = synsets[0]  # Consider the first synset as the target synset
#         similarity_scores = []
#         for syn in synsets[1:]:
#             similarity = target_synset.wup_similarity(syn)
#             if similarity is not None:
#                 similarity_scores.append((syn.lemmas()[0].name(), similarity))
#         similarity_scores.sort(key=itemgetter(1), reverse=True)
#         synonyms = [syn for syn, _ in similarity_scores[:5]]  # Keep only top 5 synonyms
#     return synonyms
# import spacy
def get_synonyms(word, pos):
    synonyms = set()  # Use a set to store unique synonyms
    synsets = wordnet.synsets(word, pos=pos)
    if synsets:
        target_synset = synsets[0]  # Consider the first synset as the target synset
        similarity_scores = []
        for syn in synsets[1:]:
            similarity = target_synset.wup_similarity(syn)
            if similarity is not None:
                similarity_scores.append((syn.lemmas()[0].name(), similarity))
        similarity_scores.sort(key=itemgetter(1), reverse=True)
        for syn, _ in similarity_scores:
            if syn not in synonyms:
                synonyms.add(syn)
            if len(synonyms) == 5:  # Keep only 5 unique synonyms
                break
    return list(synonyms)
# Load the pre-trained word vectors

#Using Spacy
# import spacy
# nlp = spacy.load('en_core_web_md')

# def get_synonyms(word):
#     synonyms = []
#     token = nlp(word)
#     if token.has_vector:
#         word_vector = token.vector
#         most_similar = nlp.vocab.vectors.most_similar(word_vector, n=5)
#         for idx, score in most_similar:
#             similar_word = nlp.vocab[idx].text
#             synonyms.append(similar_word)
#     return synonyms
def extract_nouns(sentence):
    tagged_words = pos_tag(word_tokenize(sentence))
    nouns = [word for word, pos in tagged_words if pos in ['NN', 'NNS', 'NNP']]
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
                noun_synonyms = {noun: get_synonyms(noun, 'n') for noun in nouns}
                data.append({filename: noun_synonyms})

    with open(output_file, 'w') as file:
        json.dump(data, file, indent=4)

# Example usage
# input_file = 'captions.txt'
# output_file = 'synonyms.json'
# process_file(input_file, output_file)

input_file = '/misc/student/sharmaa/groupvit/OVSegmentor/scripts/coco_meta.json'
output_file = 'top5_synonyms_unique.json'
process_file(input_file, output_file)
