import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read the mIoU values for two runs from CSV files
run1_df = pd.read_csv('/misc/student/sharmaa/groupvit/GroupViT/mIoU_results/nonnoisy_baeline/COCOObjectDataset_per_class_iou.csv')

run2_df = pd.read_csv('/misc/student/sharmaa/groupvit/GroupViT/mIoU_results/original_ckpt/PascalVOCDataset_per_class_iou.csv')

# Merge the two dataframes based on the class name column
merged_df = pd.merge(run1_df, run2_df, on='Class Name', suffixes=('_run1', '_run2'))

# Calculate the absolute differences between the two runs
merged_df['abs_diff'] = merged_df['IoU Score_run1'] - merged_df['IoU Score_run2']
print("merged df", merged_df)

# Sort the classes in descending order based on the absolute differences
sorted_classes = merged_df.sort_values('abs_diff', ascending=False)['Class Name'].values

# Get the top 5 classes with the maximum differences
top_classes = sorted_classes[:5]

# Get the bottom 5 classes with the minimum differences
bottom_classes = sorted_classes[-5:]

# Create a horizontal bar chart to plot the mIoU differences
fig, ax = plt.subplots(figsize=(10, 15))

# Set the x-axis labels to the COCO class names
class_names = merged_df['Class Name'].values
ax.set_yticks(range(len(class_names)))
ax.set_yticklabels(class_names, rotation=0, ha='right')

# Plot the mIoU differences for the first run
ax.barh(range(len(class_names)), merged_df['abs_diff'], color='blue')

# Plot the mIoU differences for the second run as negative values
#ax.barh(range(len(class_names)), -merged_df['IoU Score_run2'], color='red')

# Set the x-axis limit to the maximum absolute difference
ax.set_xlim([0, max(merged_df['abs_diff'])])

# Add labels for the top and bottom classes
# for i in range(len(class_names)):
#     if class_names[i] in top_classes:
#         ax.text(merged_df['IoU Score_run1'][i] + 1, i, 'Top ' + str(top_classes.tolist().index(class_names[i])+1),
#                 ha='left', va='center', color='blue', fontweight='bold')
#     elif class_names[i] in bottom_classes:
#         ax.text(-merged_df['IoU Score_run2'][i] - 1, i, 'Bottom ' + str(bottom_classes.tolist().index(class_names[i])+1),
#                 ha='right', va='center', color='red', fontweight='bold')


# Set the title and labels for the plot
#ax.set_title('mIoU Differences between Two Runs')
ax.set_xlabel('mIoU Value')
ax.set_ylabel('COCO Class Name')

# Save the plot as a PNG image
fig.savefig('miou_differences_org_baseline.png')

# Save the plot as a PDF file
fig.savefig('miou_differences_org_baseline.pdf')

# Show the plot
plt.show()
