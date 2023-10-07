import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pickle

output_dir = '/misc/student/sharmaa/groupvit/GroupViT/finetune_groupentropy/'
with open(output_dir +'feature_label_dict_fT_123_GE_coco.pickle', 'rb') as handle:
    feature_ft_ge = pickle.load(handle) #.to(torch.device('cpu'))

output_dir = '/misc/student/sharmaa/groupvit/GroupViT/finetune_aftercleanup_pascal/'
with open(output_dir +'feature_label_dict_fT_123_coco.pickle', 'rb') as handle:
    feature_ft = pickle.load(handle) #.to(torch.device('cpu'))

output_dir = '/misc/student/sharmaa/groupvit/GroupViT/originalckpt_pascalanalysis/'
with open(output_dir +'feature_label_dict_oc_coco.pickle', 'rb') as handle:
    feature_originalckpt = pickle.load(handle) #.to(torch.device('cpu'))




# Extract features and labels
features_originalckpt = np.array(list(feature_originalckpt.keys()))
labels_originalckpt = np.array(list(feature_originalckpt.values()))
features_ft = np.array(list(feature_ft.keys()))
labels_ft = np.array(list(feature_ft.values()))
features_ft_ge = np.array(list(feature_ft_ge.keys()))
labels_ft_ge = np.array(list(feature_ft_ge.values()))
print("features_originalckpt", features_originalckpt.shape)
print("labels_originalckpt", labels_originalckpt.shape)
print("features_ft", features_ft.shape)
print("labels_ft", labels_ft.shape)
print("features_ft_ge", features_ft_ge.shape)
print("labels_ft_ge", labels_ft_ge.shape)
#chage features ndarray
stacked_feature = np.concatenate((features_originalckpt, features_ft, features_ft_ge), axis=0)
print("stacked features", stacked_feature.shape)
stacked_labels = np.concatenate((labels_originalckpt, labels_ft, labels_ft_ge), axis=0)
print("stacked labels", stacked_labels.shape)

# Perform t-SNE dimensionality reduction
tsne = TSNE(n_components=2, random_state=42)
projected_features = tsne.fit_transform(stacked_feature)
print("projected_features", projected_features.shape)

# Generate 20 colors for 20 labels
unique_labels = np.unique(stacked_labels)
num_labels = len(unique_labels)
print("unique_labels", num_labels)
import matplotlib.colors as mcolors

# Get a list of 80 distinct colors
colors = ["Red", "Blue", "Green", "Yellow", "Orange", "Purple", "Pink", "Brown", "Black", "White", "Gray", "Turquoise", "Cyan", "Magenta", "Lavender", "Indigo", "Maroon", "Olive", "Gold", "Silver", "Beige", "Teal", "Navy", "Coral", "Salmon", "Lime", "Aqua", "Violet", "Orchid", "Mint", "Slate", "Ruby", "Pearl", "Ivory", "Emerald", "Amber", "Crimson", "Saffron", "Cobalt", "Plum", "Burgundy", "Lilac", "Mauve", "Sienna", "Charcoal", "Magenta", "Bronze", "Peach", "Rose", "Sunset", "Honeydew", "Sapphire", "Cerulean", "Cyan", "Chartreuse", "Moccasin", "Periwinkle", "Cornsilk", "Tomato", "Thistle", "Sky Blue", "Slate Gray", "Chocolate", "Ghost White", "Khaki", "Lawn Green", "Midnight Blue", "Misty Rose", "Olive Drab", "Powder Blue", "Rosy Brown", "Sea Green", "Steel Blue", "Tan", "Wheat", "Yellow Green", "Dark Cyan", "Dark Slate Gray", "Hot Pink", "Indian Red", "Light Salmon", "Medium Orchid", "Pale Green", "Royal Blue", "Sandy Brown"]
print("colors", len(colors))
#colors = plt.cm.get_cmap('tab20', num_labels)


projfeatures_originalckpt = projected_features[:features_originalckpt.shape[0], :]
print("projfeatures_originalckpt", projfeatures_originalckpt.shape)
projfeatures_ft = projected_features[features_originalckpt.shape[0]:features_originalckpt.shape[0]+features_ft.shape[0], :]
print("projfeatures_ft", projfeatures_ft.shape)
projfeatures_ft_ge = projected_features[features_originalckpt.shape[0]+features_ft.shape[0]:, :]
print("projfeatures_ft_ge", projfeatures_ft_ge.shape)


fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

# # Visualize projected features with same label in the same color
for i, label in enumerate(unique_labels):
    
    color = colors[i]
    indices = np.where(labels_originalckpt == label)
    axes[0].scatter(projfeatures_originalckpt[indices, 0], projfeatures_originalckpt[indices, 1], color=color, label=label)
    
    indices = np.where(labels_ft == label)
    axes[1].scatter(projfeatures_ft[indices, 0], projfeatures_ft[indices, 1], color=color, label=label)

    indices = np.where(labels_ft_ge == label)
    axes[2].scatter(projfeatures_ft_ge[indices, 0], projfeatures_ft_ge[indices, 1], color=color, label=label)
    # Set titles and legends for each subplot
    axes[0].set_title('Projected Original CKPT')
    axes[1].set_title('Projected FT')
    axes[2].set_title('Projected FT_GE')

plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
   

# # Save the plot as a PDF file
plt.savefig('all_feat_tsne_plot_coco.pdf', bbox_inches='tight')
# plt.show()
plt.close()


# fig, axes = plt.subplots(nrows=5, ncols=4, figsize=(25, 25))
# # # Create a separate plot for each label
# for i, label in enumerate(unique_labels):
#     color = colors(i)
#     print("i", i)
#     indices = np.where(labels_originalckpt == label)
#     axes[i // 4, i % 4].scatter(projfeatures_originalckpt[indices, 0], projfeatures_originalckpt[indices, 1], color=color, label=label)
#     axes[i // 4, i % 4].set_title(label)
# # Remove empty subplots
# if num_labels < len(axes.flat):
#     for j in range(num_labels, len(axes.flat)):
#         fig.delaxes(axes.flatten()[j])
# fig.text(0.5, 0.04, 'Dimension 1', ha='center')
# fig.text(0.04, 0.5, 'Dimension 2', va='center', rotation='vertical')

# plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
# plt.savefig('proj_feat_origckpt_tsne_plot_coco.pdf', bbox_inches='tight')
# plt.close()

# fig, axes = plt.subplots(nrows=5, ncols=4, figsize=(25, 25))
# # # Create a separate plot for each label
# for i, label in enumerate(unique_labels):
#     color = colors(i)

#     indices = np.where(labels_ft == label)
#     axes[i // 4, i % 4].scatter(projfeatures_ft[indices, 0], projfeatures_ft[indices, 1], color=color, label=label)
#     axes[i // 4, i % 4].set_title(label)
# # Remove empty subplots
# if num_labels < len(axes.flat):
#     for j in range(num_labels, len(axes.flat)):
#         fig.delaxes(axes.flatten()[j])
# fig.text(0.5, 0.04, 'Dimension 1', ha='center')
# fig.text(0.04, 0.5, 'Dimension 2', va='center', rotation='vertical')
# plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
# plt.savefig('proj_feat_ft_tsne_plot_coco.pdf', bbox_inches='tight')
# plt.close()

# fig, axes = plt.subplots(nrows=5, ncols=4, figsize=(25, 25))
# # # Create a separate plot for each label
# for i, label in enumerate(unique_labels):
#     color = colors(i)

#     indices = np.where(labels_ft_ge == label)
#     axes[i // 4, i % 4].scatter(projfeatures_ft_ge[indices, 0], projfeatures_ft_ge[indices, 1], color=color, label=label)
#     axes[i // 4, i % 4].set_title(label)
# # Remove empty subplots
# if num_labels < len(axes.flat):
#     for j in range(num_labels, len(axes.flat)):
#         fig.delaxes(axes.flatten()[j])
# fig.text(0.5, 0.04, 'Dimension 1', ha='center')
# fig.text(0.04, 0.5, 'Dimension 2', va='center', rotation='vertical')
# plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
# plt.savefig('proj_feat_ftge_tsne_plot_coco.pdf', bbox_inches='tight')
# plt.close()