# read .pt file and get the feature vector for each label
import torch
import pickle

#load pickle file
with open('label_feature_dict_originalcheckpoint.pickle', 'rb') as handle:
    label_feature_dict = pickle.load(handle) #.to(torch.device('cpu'))
with open('feature_label_dict_oc.pickle', 'rb') as handle:
    feature_label_dict = pickle.load(handle) #.to(torch.device('cpu'))

print("label_feature_dict", label_feature_dict)
print("feature_label_dict_oc", feature_label_dict)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

data = feature_label_dict

# Extract features and labels
features = feature_label_dict.keys()
#chage features ndarray
features = np.array(list(features))
labels = np.array(list(feature_label_dict.values()))
print("features", features)
print("labels", labels)

# Perform t-SNE dimensionality reduction
tsne = TSNE(n_components=2, random_state=42)
projected_features = tsne.fit_transform(features)
print("projected_features", projected_features)

# Visualize projected features with same label in the same color
# Generate 20 colors for 20 labels
unique_labels = np.unique(labels)
num_labels = len(unique_labels)
print("unique_labels", unique_labels)
print("num_labels", num_labels)
colors = ['red', 'lime', 'blue', 'yellow', 'magenta', 'cyan', 'maroon', 'green', 'navy', 'olive', 'purple', 'teal', 'orange', 'brown', 'pink', 'gray', 'silver', 'maroon', 'fuchsia', 'black']

for label, color in zip(unique_labels, colors):
    print("label", label)
    indices = np.where(labels == label)
    print("indices", indices)
    print("label color indices", label, color, indices)

    plt.scatter(projected_features[indices, 0], projected_features[indices, 1], c=color, label=label)

plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')

plt.legend()
# Save the plot as a PDF file
plt.savefig('tsne_plot.pdf', bbox_inches='tight')
plt.show()

