import json

# Load JSON data from file
with open('/misc/student/sharmaa/groupvit/GroupViT/noun_frequency.json', 'r') as json_file:
    data = json.load(json_file)

# PASCAL SYNONYMS dictionary
# SYNONYMS = {
#     "background": ["background"],
#     "aeroplane": ["aeroplane","airplane","plane","aircraft","jet","airliner","jetliner","jetplane","airbus"],
#     "bicycle": ["bicycle","bike","cycle","pedal bike"],
#     "bird": ["bird","avian","fowl"],
#     "boat": ["boat","vessel", "watercraft", "ship", "yacht", "canoe","sailboat"],
#     "bottle": ["bottle","bottleful","flask","flagon","jar","jug"],
#     "bus": ["bus"],
#     "car": ["car","auto","automobile","motorcar","vehicle"],
#     "cat": ["cat","feline","felid","kitten"],
#     "chair": ["chair","seat"],
#     "cow": ["cow","bovine","cattle","kine","ox","oxen","bison","buffalo"],
#     "table": ["diningtable","dining table","dining-room table","kitchen table","table"],
#     "dog": ["dog","canine","canid","puppy","hound"],
#     "horse": ["horse","equine","pony"],
#     "motorbike": ["motorbike","motorcycle","motor cycle","moto","moped","mop"],
#     "person": ["person","man","woman","adult","people","someone","kid","guy","boy","girl","baby","child","human"],
#     "plant": ["pottedplant","houseplant","plant","potted plant"],
#     "sheep": ["sheep","lamb","ewe","ram","wether","hogget","mutton"],
#     "sofa": ["sofa","couch","lounge","lounge chair","lounge suite","sofa bed","sofabed"],
#     "train": ["train","tram","subway","metro"],
#     "monitor": ["tvmonitor","tv","tv screen","television"]
# }

# COCO Object Category Synonyms
SYNONYMS = {
    "background": ["background"],
    "airplane": ["airplane", "aeroplane", "plane", "aircraft", "jet"],
    "bicycle": ["bicycle", "bike", "cycle"],
    "bird": ["bird"],
    "boat": ["boat", "ship", "vessel", "watercraft"],
    "bottle": ["bottle"],
    "bus": ["bus"],
    "car": ["car", "auto", "automobile", "motorcar", "vehicle"],
    "cat": ["cat"],
    "chair": ["chair"],
    "cow": ["cow"],
    "table": ["table", "dining table"],
    "dog": ["dog"],
    "horse": ["horse"],
    "motorbike": ["motorbike"],
    "person": ["person", "man", "woman", "adult"],
    "plant": ["plant", "potted plant"],
    "sheep": ["sheep"],
    "sofa": ["sofa", "couch"],
    "train": ["train"],
    "tvmonitor": ["tvmonitor", "tv"],
    # Additional COCO-specific categories from COCO dataset
    "traffic light": ["traffic light", "signal"],
    "fire hydrant": ["fire hydrant", "hydrant"],
    "stop sign": ["stop sign"],
    "parking meter": ["parking meter", "meter"],
    "bench": ["bench"],
    "elephant": ["elephant"],
    "zebra": ["zebra"],
    "giraffe": ["giraffe"],
    "backpack": ["backpack"],
    "umbrella": ["umbrella"],
    "handbag": ["handbag", "purse"],
    "tie": ["tie"],
    "suitcase": ["suitcase"],
    "frisbee": ["frisbee"],
    "skis": ["skis"],
    "snowboard": ["snowboard"],
    "sports ball": ["sports ball", "ball"],
    "kite": ["kite"],
    "baseball bat": ["baseball bat"],
    "baseball glove": ["baseball glove", "glove"],
    "skateboard": ["skateboard"],
    "surfboard": ["surfboard"],
    "tennis racket": ["tennis racket", "racket"],
    "wine glass": ["wine glass"],
    "cup": ["cup"],
    "fork": ["fork"],
    "knife": ["knife"],
    "spoon": ["spoon"],
    "banana": ["banana"],
    "apple": ["apple"],
    "sandwich": ["sandwich"],
    "orange": ["orange"],
    "broccoli": ["broccoli"],
    "carrot": ["carrot"],
    "hot dog": ["hot dog"],
    "pizza": ["pizza"],
    "donut": ["donut"],
    "cake": ["cake"],
    "couch": ["couch"],
    "bed": ["bed"],
    "dining table": ["dining table"],
    "toilet": ["toilet"],
    "laptop": ["laptop"],
    "mouse": ["mouse", "computer mouse"],
    "remote": ["remote", "remote control"],
    "keyboard": ["keyboard"],
    "cell phone": ["cell phone", "mobile phone"],
    "microwave": ["microwave"],
    "oven": ["oven"],
    "toaster": ["toaster"],
    "sink": ["sink"],
    "refrigerator": ["refrigerator", "fridge"],
    "book": ["book"],
    "clock": ["clock"],
    "vase": ["vase"],
    "scissors": ["scissors"],
    "teddy bear": ["teddy bear"],
    "hair drier": ["hair drier", "hair dryer"],
    "toothbrush": ["toothbrush"],
    "bench":["bench"],
}


  

# Initialize a dictionary to store key frequencies
key_frequencies = {key: 0 for key in SYNONYMS}

# Initialize a variable to store the sum of corresponding values
sum_of_corresponding_values = 0

# Iterate through the data and calculate key frequencies
for key, frequency in data.items():
    # Find the corresponding synonyms for the key
    for synonyms, values in SYNONYMS.items():
        if key in values:
            key_frequencies[synonyms] += frequency
            break  # Break the loop once the synonym is found
    # Sum up the corresponding values
    sum_of_corresponding_values += frequency

# Prepare the final results as a dictionary
final_results = {
    "key_frequencies": key_frequencies,
    "sum_of_corresponding_values": sum_of_corresponding_values
}

# Write the final results to a JSON file
with open('freqmap_cococategorykeys_traincococaption.json', 'w') as output_file:
    json.dump(final_results, output_file, indent=4)

print("Results written to freqmap_pascalkeys_traincococaption.json")
