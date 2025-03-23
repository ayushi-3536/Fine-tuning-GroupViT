# Fine-tuning Vision Language Models for Downstream Tasks
Master Thesis Work at the Computer Vision Group under Prof. Thomas Brox, University of Freiburg

### Abstract
In the evolving realm of deep learning for scene understanding, the traditional
dichotomy between grouping and recognition has blurred, thanks to integrated end-
to-end training systems. Yet, GroupViT reshapes this landscape by reintroducing
explicit grouping in deep networks. This novel bottom-up approach rekindles the
essence of semantic segmentation in contrasts with traditional top-down methods.
GroupViT, trained on weak supervisory signals from text, showcases impressive results
across various datasets, highlighting its efficacy for open-vocabulary segmentation.
Within this thesis, we undertake a dissection of the pretrained GroupViT model,
isolating two pivotal components for open-vocabulary segmentation: Visual Grouping
and Vision-Text Alignment. Based on our analysis, it becomes evident that there is
substantial room for improvement, particularly in Visual Grouping. By leveraging
GroupViT’s pretrained model and further training on a cleaner, smaller MSCOCO
dataset, we observed promising enhancements on the dataset. However, there are
trade-offs, compromising performance on downstream dataset such as PASCAL VOC
and PASCAL Context.
To address these challenges, this work introduces strategic enhancements. First,
we incorporate entropy regularization techniques to improve semantic grouping and
visual-text alignment. Second, we propose a non-noisy contrastive loss, countering
limitations of training on a smaller dataset and leading to more robust, accurate
results.
These systematic refinements furnish a framework to further train the pretrained
model of GroupViT on smaller and cleaner datasets for segmentation, while upholding
the performance across datasets.

This repository implements **vanilla fine-tuning** with an improved extraction of text tokens from image captions. The model architecture is based on **GroupViT**, which utilizes multiple tokens to encode information from distinct image segments.

### Method
- **Fine-Tuning:** Standard vanilla fine-tuning approach with refined text token extraction.
- **Group Entropy Regularization (GE):** Reduces entropy, allowing the model to confidently assign image pixels to their most probable segments.
- **Non-Noisy Contrastive Loss + GE:**
  - Trains using noise-free contrastive loss by matching multiple positive images representing the same entity in a batch.
  - Incorporates GE to further refine segment assignments.
- **High-Resolution Training:** All models are trained at **384×384 resolution**.


### Results:
Here, Fine-tuning is vanilla fine-tuning with
refined extraction of text tokens from image captions. GE denotes Group Entropy Regularization baseline, GroupVit model architecture has multiple tokens to contain information of segments emerging from images.Reducing entropy helps the model to assign image pixels to the segment it is most confident about.
Non-noisy CL + GE denotes model trained with noise-free contrastive
loss and Group Entropy Regularization. Here noise-free contrastive loss indication matching with multiple positive image representing same entity in the batch. All the models are trained on
the resolution of 384.
![image](https://github.com/user-attachments/assets/289e2555-0322-40bd-b9d9-9a780154d952)

### Qualitative Analysis
![image](https://github.com/user-attachments/assets/da616fdb-6f3f-4ee2-8a48-045ceb23d6eb)

![image](https://github.com/user-attachments/assets/7c659e78-ffd2-4d73-a3ff-3658d347d440)


![image](https://github.com/user-attachments/assets/5b55b38d-88a2-4754-9cda-94b45e68e517)

### Full Thesis: 

## Acknowledgments
- [GroupViT: Semantic Segmentation Emerges from Text Supervision](https://arxiv.org/abs/2202.11094)
