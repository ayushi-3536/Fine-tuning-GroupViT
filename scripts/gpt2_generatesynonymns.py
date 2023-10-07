from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

def generate_synonyms(word):
    input_text = f"Synonyms for the word '{word}':"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    output = model.generate(input_ids, max_length=77, num_return_sequences=1, early_stopping=True)

    print(f"Output:{output}")
    synonyms = [tokenizer.decode(sequence, skip_special_tokens=True) for sequence in output]
    
    print(f"Synonyms:{synonyms}")
    return synonyms

# Example usage
word = "happy"
synonyms = generate_synonyms(word)

print(f"Synonyms for '{word}':")
for synonym in synonyms:
    print(synonym)
