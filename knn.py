import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import torch
import pickle

with open('feature_label_dict_oc.pickle', 'rb') as handle:
    feature_label_dict = pickle.load(handle) #.to(torch.device('cpu'))

print("feature_label_dict_oc", feature_label_dict)

data = feature_label_dict

# Extract features and labels
features = feature_label_dict.keys()
#chage features ndarray
features = np.array(list(features))
labels = np.array(list(feature_label_dict.values()))
print("labels", labels)

unique_labels = np.unique(labels)
print("unique_labels", unique_labels)
num_labels = len(unique_labels)
colors = plt.cm.get_cmap('tab20', num_labels)




# Filter 20 features for each label in a sequential manner
filtered_features = {}
for label in np.unique(labels):
    label_indices = np.where(labels == label)[0]
    filtered_features[label] = features[label_indices][:25]


features_array = []
labels_array = []

for key, values in filtered_features.items():
    features_array.extend(values)
    labels_array.extend([key] * len(values))

print("Values List:", features_array)
print("Keys List:",  labels_array)

# Find 15 nearest neighbors for each feature
n_neighbors = 25
knn = NearestNeighbors(n_neighbors=n_neighbors)
knn.fit(features_array)
distances, indices = knn.kneighbors(features_array)
print("indices", indices)   

# Plot features and their neighbors
num_rows = 10
num_cols = 10

            # Define the figure and axis objects
fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(50, 50))

for i, feature in enumerate(features_array):
    ax = axs[i // num_cols, i % num_cols]
                
    if i > 10:
        break
    neighbors = [features_array[idx] for idx in indices[i] if idx != 0]
    neighbor_labels = [labels_array[idx] for idx in indices[i] if idx != 0]
  
    print("neighbors", neighbor_labels)
    label = labels_array[i]
    #get index of label in unique_labels
    label_index = np.where(unique_labels == label)[0][0]
    print("label_index", label_index)
    color = colors(label_index)
    plt.scatter(distances[i][0], 0, color=color, label=label)

    for j, neighbor in enumerate(neighbors):
        neighbor_label = neighbor_labels[j]
        #get index of label in unique_labels
        neighbor_label_index = np.where(unique_labels == neighbor_label)[0][0]
        print("neighbor_label_index", neighbor_label_index)
        color = colors(neighbor_label_index)
        plt.scatter(distances[i][j], 0, color=color, label=neighbor_label)
        

    ax.set_xlabel('Distance1')
    ax.set_ylabel('Distance 2')
    
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.savefig(f'knn_plot_{i}.pdf', bbox_inches='tight')



    plt.close()





