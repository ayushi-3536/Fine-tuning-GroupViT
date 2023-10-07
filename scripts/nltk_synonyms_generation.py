from nltk.corpus import wordnet

def generate_synonyms(word):
    synonyms = set()
    for synset in wordnet.synsets(word):
        for lemma in synset.lemmas():
            synonym = lemma.name()
            if synonym != word:
                synonyms.add(synonym)
    return list(synonyms)

# Example usage
word = "man"
synonyms = generate_synonyms(word)

print(f"Synonyms for '{word}':")
for synonym in synonyms:
    print(synonym)
