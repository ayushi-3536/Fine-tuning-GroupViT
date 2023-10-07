import torch
import json
import pickle


#sample example

# {'license': 4, 'file_name': '000000178175.jpg', 'coco_url': 'http://images.cocodataset.org/train2017/000000178175.jpg',
#   'height': 464, 'width': 640, 'date_captured': '2013-11-18 11:22:12', 
#   'flickr_url': 'http://farm4.staticflickr.com/3084/2894606072_c7b0791aaf_z.jpg', 'id': 178175, 
#   'captions': ['lots of backpacks and hats lined up along a walkway',
#                'a group of back packs on a bridge walkway over a stream', 
#                'luggage and backpacks sitting on a wooden walkway to a beach',
#                'suitcases, duffle bags, and backpacks are sitting along a wooden walkway', 
#                'backpacks line a boardwalk to a beach surrounded by trees'],
#   'poc': [{'caption': 'lots of backpacks and hats lined up along a walkway',
#             'poc': [{'nph': 'a walkway', 'nouns': ['walkway'], 'ncomp': []}]}, 
#           {'caption': 'a group of back packs on a bridge walkway over a stream', 
#             'poc': [{'nph': 'a group of back packs', 'nouns': ['packs', 'group'], 'ncomp': ['back']},
#                      {'nph': 'back packs', 'nouns': ['packs'], 'ncomp': ['back']},
#                      {'nph': 'a group on a bridge', 'nouns': ['bridge', 'group'], 'ncomp': ['on']},
#                      {'nph': 'a group', 'nouns': ['group'], 'ncomp': []}, 
#                      {'nph': 'a bridge', 'nouns': ['bridge'], 'ncomp': []}, 
#                      {'nph': 'a stream', 'nouns': ['stream'], 'ncomp': []},
#                      {'nph': 'a group walkway over a stream', 'nouns': ['stream', 'group'], 'ncomp': ['over', 'walkway']}]},
#             {'caption': 'luggage and backpacks sitting on a wooden walkway to a beach',
#               'poc': [{'nph': 'backpacks sitting on a wooden walkway', 'nouns': ['backpacks', 'walkway'], 'ncomp': ['wooden', 'sitting', 'on']},
#                       {'nph': 'backpacks to a beach', 'nouns': ['beach', 'backpacks'], 'ncomp': []},
#                       {'nph': 'a wooden walkway', 'nouns': ['walkway'], 'ncomp': ['wooden']},
#                       {'nph': 'a beach', 'nouns': ['beach'], 'ncomp': []}]},
#              {'caption': 'suitcases, duffle bags, and backpacks are sitting along a wooden walkway',
#                'poc': [{'nph': 'duffle bags', 'nouns': ['bags', 'duffle'], 'ncomp': []},
#                        {'nph': 'backpacks along a wooden walkway', 'nouns': ['backpacks', 'walkway'], 'ncomp': ['wooden', 'along']},
#                        {'nph': 'a wooden walkway', 'nouns': ['walkway'], 'ncomp': ['wooden']},
#                        {'nph': 'suitcases along a wooden walkway', 'nouns': ['suitcases', 'walkway'], 'ncomp': ['wooden', 'along']}]},
#              {'caption': 'backpacks line a boardwalk to a beach surrounded by trees',
#                'poc': [{'nph': 'backpacks line a boardwalk', 'nouns': ['backpacks', 'boardwalk'], 'ncomp': ['line']},
#                        {'nph': 'a boardwalk', 'nouns': ['boardwalk'], 'ncomp': []},
#                        {'nph': 'a boardwalk to a beach', 'nouns': ['beach', 'boardwalk'], 'ncomp': []}, 
#                        {'nph': 'a beach', 'nouns': ['beach'], 'ncomp': []}]}]}
def load_nouns():
    # Open the JSON file and read its contents
    with open('/misc/lmbraid21/sharmaa/poc_captions/poc_captions_train2017.json', 'r') as file:
        data = json.load(file)
    pre_computed_nouns = {}
    # Access the contents of the JSON file
    print(data.keys())
    print(len(data['images']))
    print(type(data['images']))
    for item in data['images']:
        print(item.keys())
        for data in item['poc']:
            caption = data['caption']
            print(f"Caption: {caption}")
            nouns = []
            noun_phrases = []
            
            for phrase in data['poc']:
                all_nph = phrase['nph']
                all_nouns = phrase['nouns']
                
                print(f"  Noun Phrase: {all_nph}")
                print(f"  Nouns: {all_nouns}")
                nouns = nouns + all_nouns
                noun_phrases = noun_phrases + [all_nph]

            print(f"  Noun Phrase: {noun_phrases}")
            nouns = list(set(nouns)) + noun_phrases
            print(f"  Nouns: {nouns}")
            pre_computed_nouns.update(caption, nouns)
    return pre_computed_nouns
def generate_nouns():
    # Open the JSON file and read its contents
    with open('/misc/lmbraid21/sharmaa/poc_captions/poc_captions_train2017.json', 'r') as file:
        data = json.load(file)
    pre_computed_nouns = {}
    # Access the contents of the JSON file
    print(data.keys())
    print(len(data['images']))
    print(type(data['images']))
    for item in data['images']:
        print(item.keys())
        for data in item['poc']:
            caption = data['caption']
            print(f"Caption: {caption}")
            nouns = []
            noun_phrases = []
            
            for phrase in data['poc']:
                all_nph = phrase['nph']
                all_nouns = phrase['nouns']
                
                print(f"  Noun Phrase: {all_nph}")
                print(f"  Nouns: {all_nouns}")
                nouns = nouns + all_nouns
                noun_phrases = noun_phrases + [all_nph]

            print(f"  Noun Phrase: {noun_phrases}")
            nouns = list(set(nouns)) + noun_phrases
            print(f"  Nouns: {nouns}")
            pre_computed_nouns[caption] = nouns
    
    with open('/misc/lmbraid21/sharmaa/poc_captions/poc_extract_noun_coco2017.pickle', 'wb') as file:
        pickle.dump(pre_computed_nouns, file)

#generate_nouns()

#print(data['key2'])


# embeddings = generate_text_embeddings() # your generated embeddings

# torch.save(embeddings, 'embeddings.pt')
