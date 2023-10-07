import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pickle

# output_dir = '/misc/student/sharmaa/groupvit/GroupViT/finetune_groupentropy/'
# with open(output_dir +'feature_label_dict_fT_123_GE_pascalvoc.pickle', 'rb') as handle:
#     feature_label_dict = pickle.load(handle) #.to(torch.device('cpu'))

# output_dir = '/misc/student/sharmaa/groupvit/GroupViT/finetune_aftercleanup_pascal/'
# with open(output_dir +'feature_label_dict_fT_123_11_pascalvoc.pickle', 'rb') as handle:
#     feature_label_dict = pickle.load(handle) #.to(torch.device('cpu'))

output_dir = '/misc/student/sharmaa/groupvit/GroupViT/originalckpt_pascalanalysis/'
with open(output_dir +'feature_label_dict_oc.pickle', 'rb') as handle:
    feature_label_dict = pickle.load(handle) #.to(torch.device('cpu'))

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

# Generate 20 colors for 20 labels
unique_labels = np.unique(labels)
num_labels = len(unique_labels)
colors = plt.cm.get_cmap('tab20', num_labels)

# Visualize projected features with same label in the same color
for i, label in enumerate(unique_labels):
    color = colors(i)
    indices = np.where(labels == label)
    plt.scatter(projected_features[indices, 0], projected_features[indices, 1], color=color, label=label)

plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
   

# Save the plot as a PDF file
plt.savefig(output_dir + 'tsne_plot_with_labels.pdf', bbox_inches='tight')
plt.show()

# Create a separate plot for each label
for i, label in enumerate(unique_labels):
    plt.figure()
    color = colors(i)
    indices = np.where(labels == label)
    plt.scatter(projected_features[indices, 0], projected_features[indices, 1], color=color, label=label)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend()

    # Save each label plot as a separate PDF file
    plt.savefig(output_dir + f'label_plot_{label}.pdf')
    plt.close()
