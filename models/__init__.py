# -------------------------------------------------------------------------
# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License.
# To view a copy of this license, visit
# https://github.com/NVlabs/GroupViT/blob/main/LICENSE
#
# Written by Jiarui Xu
# -------------------------------------------------------------------------

from .builder import build_model
from .group_vit import GroupViT
from .dino_group_vit import DINOGroupViT
from .dino_group_vit_featdistill import DINOFeatDistillGroupViT
from .dino_group_vit_withgs2 import DINOGS2_GroupViT
from .dino_group_vit_notraining import DINO_NOTRAIN_GroupViT
from .group_vit_pacl import GroupViT_PACL
from .multi_label_contrastive import MultiLabelContrastive
from .multi_label_contrastive2 import MultiLabelContrastive2
from .multi_label_contrastive_entropy import MultiLabelContrastiveEntropy
from .multi_label_contrastive_pacl import MultiLabelContrastive_PACL
from .multi_label_contrastive_refined import MultiLabelContrastiveRefined
from .dino_gvit_feature_extract import GVIT_DINO
from .group_vit_simloss import GroupViT_SimLoss
from .multi_label_contrastive_celoss import MultiLabelContrastiveCELoss
from .multi_label_contrastive_simloss import MultiLabelContrastiveSimLoss
from .clip_groupvit_multi_label_contrastive import CLIPMultiLabelContrastive
from .multi_label_contrastive_dino_distill import MultiLabelContrastiveDINO
from .transformer import TextTransformer
from .cliptransformer import CLIPTextTransformer

__all__ = ['build_model', 'MultiLabelContrastiveCELoss', 'GroupViT_SimLoss', 'MultiLabelContrastiveSimLoss', 'GVIT_DINO', 'MultiLabelContrastiveDINO', 'DINOFeatDistillGroupViT', 'DINO_NOTRAIN_GroupViT', 'DINOGS2_GroupViT', 'MultiLabelContrastive', 'MultiLabelContrastiveRefined', 'GroupViT', 'TextTransformer', 'CLIPMultiLabelContrastive',
            'CLIPTextTransformer', 'DINOGroupViT', 'MultiLabelContrastive2', 'MultiLabelContrastiveEntropy', 'MultiLabelContrastive_PACL', 'GroupViT_PACL']
