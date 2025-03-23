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


### Method
- **Fine-Tuning:** Standard vanilla fine-tuning approach with refined text token extraction.
- **Group Entropy Regularization (GE):** Reduces entropy, allowing the model to confidently assign image pixels to their most probable segments.
- **Non-Noisy Contrastive Loss + GE:**
  - Trains using noise-free contrastive loss by matching multiple positive images representing the same entity in a batch.
  - Incorporates GE to further refine segment assignments.
- **High-Resolution Training:** All models are trained at **384×384 resolution**.


### Results: 
Our results show that incorporating non-noisy contrastive loss and group entropy regularization while fine-tuning GroupViT on MSCOCO helps in improving performance while performing consistently on other datasets in zero-shot.

![image](https://github.com/user-attachments/assets/289e2555-0322-40bd-b9d9-9a780154d952)

### Qualitative Analysis:
Visualization of segmentation outputs of the vanilla fine-tuning vs fine-tuning with proposed method on Pascal VOC.
![image](https://github.com/user-attachments/assets/87fc17c4-1b0c-47e3-9de0-04275397f683)


### Full Thesis: 

## Acknowledgments
- [GroupViT: Semantic Segmentation Emerges from Text Supervision](https://arxiv.org/abs/2202.11094)
