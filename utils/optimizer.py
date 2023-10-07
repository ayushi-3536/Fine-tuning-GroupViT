# -------------------------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE
#
# Written by Ze Liu, Zhenda Xie
# Modified by Jiarui Xu
# -------------------------------------------------------------------------

from torch import optim as optim
from utils import get_logger
def check_items_in_string(items, key):
    for item in items:
        if item in key:
            return True
    return False

def set_gradient(model, cfg):
    #logger = get_logger()
    text_encoder_keys = ['text_encoder', 'text_projector']
    #Cross Attention module is added to take the cross attention between the group tokens
    #and text token before projecting group tokens into the multi-modal space for TACL(Token aligned loss)
    grouping_layer_key = ['.downsample', 'cross_attention.']
    grouping_layer2_key = ['layers.1.downsample']
    projectors_only = ['text_projector', 'img_projector']
    
    img_projector_only = ['img_projector']
    

    print("cfg for finetuning", cfg)
    #assert that only one of cfg.only_grouping and cfg.only_grouping2 is true
    assert not (cfg.only_grouping and cfg.only_grouping2), "only one of cfg.only_grouping and cfg.only_grouping2 can be true"
    
    #assert if cfg.freeze_text_encoder is true then all other flags are false
    assert not (cfg.freeze_text_encoder and cfg.only_grouping), "if cfg.freeze_text_encoder is true then cfg.only_grouping should be false"
    assert not (cfg.freeze_text_encoder and cfg.only_grouping2), "if cfg.freeze_text_encoder is true then cfg.only_grouping2 should be false"
    
    #assert only one of projectors_only and img_projector_only is true
    assert not (cfg.only_mlp_projectors and cfg.only_img_projector), "only one of cfg.only_mlp_projectors and cfg.only_img_projector can be true"
    
    for name, param in model.named_parameters():
        #logger.info(f'key: {name}')
        if not param.requires_grad:
            continue  # frozen weights
        
        #Finetune entire Visual Encoder and Visual Projector
        if cfg.freeze_text_encoder:
            if check_items_in_string(text_encoder_keys, name):
                #logger.info(f'setting {name} as untrainable')
                param.requires_grad=False
        #Finetune only the grouping layer
        elif cfg.only_grouping and not cfg.only_mlp_projectors and not cfg.only_img_projector:
            if not check_items_in_string(grouping_layer_key, name):
                param.requires_grad=False
        #Finetune only the grouping layer and the mlp projectors of both the encoders        
        elif cfg.only_grouping and cfg.only_mlp_projectors:
            if not check_items_in_string(grouping_layer_key, name) and not check_items_in_string(projectors_only, name):
                param.requires_grad=False
        #Finetune only the grouping layer and the img projector of the visual encoder
        elif cfg.only_grouping and cfg.only_img_projector:
            if not check_items_in_string(grouping_layer_key, name) and not check_items_in_string(img_projector_only, name):
                param.requires_grad=False
        #Finetune only the grouping2 layer
        elif cfg.only_grouping2 and not cfg.only_mlp_projectors and not cfg.only_img_projector:
            if not check_items_in_string(grouping_layer2_key, name):
                param.requires_grad=False
        #Finetune only the second grouping layer and the mlp projectors of both the encoders
        elif cfg.only_grouping2 and cfg.only_mlp_projectors:
            if not check_items_in_string(grouping_layer2_key, name) and not check_items_in_string(projectors_only, name):
                param.requires_grad=False
        #Finetune only the second grouping layer and the img projector of the visual encoder
        elif cfg.only_grouping2 and cfg.only_img_projector:
            if not check_items_in_string(grouping_layer2_key, name) and not check_items_in_string(img_projector_only, name):
                param.requires_grad=False
        #Finetune only the mlp projectors of both the encoders
        elif cfg.only_mlp_projectors:
            if not check_items_in_string(projectors_only, name):
                param.requires_grad=False
        #Finetune only the img projector of the visual encoder
        elif cfg.only_img_projector:
            if not check_items_in_string(img_projector_only, name):
                param.requires_grad=False


    return model

def build_optimizer(config, model):
    """Build optimizer, set weight decay of normalization to 0 by default."""
    
    #logger = get_logger('optimizer')
    if config.finetune:
        #logger.info('Setting untrainable parameters')
        model = set_gradient(model, config.finetune)

    parameters = set_weight_decay(model, {}, {})
    
    for name, param in model.named_parameters():
        if  param.requires_grad:
            print(f'Tunable param name::{name}')

    opt_name = config.optimizer.name
    optimizer = None
    if opt_name == 'adamw':
        optimizer = optim.AdamW(
            parameters,
            eps=config.optimizer.eps,
            betas=config.optimizer.betas,
            lr=config.base_lr,
            weight_decay=config.weight_decay)
    else:
        raise ValueError(f'Unsupported optimizer: {opt_name}')

    return optimizer


def set_weight_decay(model, skip_list=(), skip_keywords=()):
    has_decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith('.bias') or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):
            no_decay.append(param)
            # print(f"{name} has no weight decay")
        else:
            has_decay.append(param)
    return [{'params': has_decay}, {'params': no_decay, 'weight_decay': 0.}]


def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin
