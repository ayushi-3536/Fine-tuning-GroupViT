import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pickle

# output_dir = '/misc/student/sharmaa/groupvit/GroupViT/finetune_groupentropy/'
# with open(output_dir +'feature_label_dict_fT_123_GE_pascalvoc.pickle', 'rb') as handle:
#     feature_ft_ge = pickle.load(handle) #.to(torch.device('cpu'))
# output_dir = '/misc/student/sharmaa/groupvit/GroupViT/feat_analysis_tsne/originalckpt/'
# with open(output_dir +'feature_label_dict_neworgckpt_pascal.pickle', 'rb') as handle:
#     feature_originalckpt = pickle.load(handle) #.to(torch.device('cpu'))

# output_dir = '/misc/student/sharmaa/groupvit/GroupViT/feat_analysis_tsne/nonnoisy_feat_tsne/'
# with open(output_dir +'feature_label_dict_newnonnoisy_pascal.pickle', 'rb') as handle:
#     feature_ft = pickle.load(handle) #.to(torch.device('cpu'))


output_dir = '/misc/student/sharmaa/groupvit/GroupViT/feat_analysis_tsne/originalckpt/'
with open(output_dir +'feature_label_dict_oc_coco.pickle', 'rb') as handle:
    feature_originalckpt = pickle.load(handle) #.to(torch.device('cpu'))

output_dir = '/misc/student/sharmaa/groupvit/GroupViT/feat_analysis_tsne/nonnoisy_feat_tsne/'
with open(output_dir +'feature_label_dict_newnonnoisy_coco.pickle', 'rb') as handle:
    feature_ft = pickle.load(handle) #.to(torch.device('cpu'))


# Extract features and labels
features_originalckpt = np.array(list(feature_originalckpt.keys()))
labels_originalckpt = np.array(list(feature_originalckpt.values()))
features_ft = np.array(list(feature_ft.keys()))
labels_ft = np.array(list(feature_ft.values()))
# features_ft_ge = np.array(list(feature_ft_ge.keys()))
# labels_ft_ge = np.array(list(feature_ft_ge.values()))
print("features_originalckpt", features_originalckpt.shape)
print("labels_originalckpt", labels_originalckpt.shape)
print("features_ft", features_ft.shape)
print("labels_ft", labels_ft.shape)
# print("features_ft_ge", features_ft_ge.shape)
# print("labels_ft_ge", labels_ft_ge.shape)
#chage features ndarray
stacked_feature = np.concatenate((features_originalckpt, features_ft),axis=0)#, features_ft_ge), axis=0)
print("stacked features", stacked_feature.shape)
stacked_labels = np.concatenate((labels_originalckpt, labels_ft),axis=0) #, labels_ft_ge), axis=0)
print("stacked labels", stacked_labels.shape)

# Perform t-SNE dimensionality reduction
tsne = TSNE(n_components=2, random_state=42)
projected_features = tsne.fit_transform(stacked_feature)
print("projected_features", projected_features.shape)

# Generate 20 colors for 20 labels
unique_labels = np.unique(stacked_labels)
#remove background label not by index
low_mIoU_class = [
    "background","person","umbrella","backpack", "train", "traffic_light", "tie", "bench", "scissors", "vase", "teddybear","clock","sink","toaster","tv","microwave"
]

unique_labels = list(unique_labels)
if 'background' in unique_labels:
    unique_labels.remove('background')
for i in low_mIoU_class:
    if i in unique_labels:
        unique_labels.remove(i)
print("unique_labels", unique_labels)
num_labels = len(unique_labels)
print("num_labels", num_labels)
colors = plt.cm.get_cmap('tab20', len(low_mIoU_class))


projfeatures_originalckpt = projected_features[:features_originalckpt.shape[0], :]
print("projfeatures_originalckpt", projfeatures_originalckpt.shape)
projfeatures_ft = projected_features[features_originalckpt.shape[0]:features_originalckpt.shape[0]+features_ft.shape[0], :]
print("projfeatures_ft", projfeatures_ft.shape)
# projfeatures_ft_ge = projected_features[features_originalckpt.shape[0]+features_ft.shape[0]:, :]
# print("projfeatures_ft_ge", projfeatures_ft_ge.shape)


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

# # Visualize projected features with same label in the same color
#for i, label in enumerate(unique_labels):
for i, label in enumerate(low_mIoU_class):
    print("i", i)

    color = colors(i)
    indices = np.where(labels_originalckpt == label)
    axes[0].scatter(projfeatures_originalckpt[indices, 0], projfeatures_originalckpt[indices, 1], color=color, label=label)
    
    indices = np.where(labels_ft == label)
    axes[1].scatter(projfeatures_ft[indices, 0], projfeatures_ft[indices, 1], color=color, label=label)

    # indices = np.where(labels_ft_ge == label)
    # axes[2].scatter(projfeatures_ft_ge[indices, 0], projfeatures_ft_ge[indices, 1], color=color, label=label)
    # # Set titles and legends for each subplot
    #axes[0].set_title('Pretrained Model')
    #axes[1].set_title('')
    #axes[2].set_title('Finetuned with Group Entropy Loss')

# plt.xlabel('Dimension 1')
# plt.ylabel('Dimension 2')
plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
   

# # Save the plot as a PDF file
plt.savefig('all_feat_tsne_plot_org_baseline_coco_lowmIoU.pdf', bbox_inches='tight')
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
# plt.savefig('proj_feat_origckpt_tsne_plot.pdf', bbox_inches='tight')
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
# plt.savefig('proj_feat_baseline_tsne_plot.pdf', bbox_inches='tight')
# plt.close()

# # fig, axes = plt.subplots(nrows=5, ncols=4, figsize=(25, 25))
# # # # Create a separate plot for each label
# # for i, label in enumerate(unique_labels):
# #     color = colors(i)

# #     indices = np.where(labels_ft_ge == label)
# #     axes[i // 4, i % 4].scatter(projfeatures_ft_ge[indices, 0], projfeatures_ft_ge[indices, 1], color=color, label=label)
# #     axes[i // 4, i % 4].set_title(label)
# # # Remove empty subplots
# # if num_labels < len(axes.flat):
# #     for j in range(num_labels, len(axes.flat)):
# #         fig.delaxes(axes.flatten()[j])
# # fig.text(0.5, 0.04, 'Dimension 1', ha='center')
# # fig.text(0.04, 0.5, 'Dimension 2', va='center', rotation='vertical')
# # plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
# # plt.savefig('proj_feat_ftge_tsne_plot.pdf', bbox_inches='tight')
# # plt.close()