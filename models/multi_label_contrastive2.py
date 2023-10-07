66666# -------------------------------------------------------------------------
# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License.
# To view a copy of this license, visit
# https://github.com/NVlabs/GroupViT/blob/main/LICENSE
#
# Written by Jiarui Xu
# -------------------------------------------------------------------------

import diffdist.functional as diff_dist
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from timm.loss import SoftTargetCrossEntropy

from .builder import MODELS
from .misc import Result


def dist_collect(x):
    """ collect all tensor from all GPUs
    args:
        x: shape (mini_batch, ...)
    returns:
        shape (mini_batch * num_gpu, ...)
    """
    x = x.contiguous()
    out_list = [torch.zeros_like(x, device=x.device, dtype=x.dtype).contiguous() for _ in range(dist.get_world_size())]
    out_list = diff_dist.all_gather(out_list, x)
    return torch.cat(out_list, dim=0).contiguous()

def dist_collect_list_of_str(x):
    """collect all list of str from all GPUs"""
    x = x.contiguous()
    # Perform all_gather to collect the encoded lists from different GPUs
    world_size = dist.get_world_size()
    gathered_lists = [torch.zeros_like(x, device=x.device, dtype=x.dtype).contiguous() for _ in range(world_size)]
    gathered_lists = diff_dist.all_gather(gathered_lists, x)
    # Convert the gathered encoded lists back to a list of strings
    output_list = ["".join([chr(char) for char in gathered]) for gathered_list in gathered_lists for gathered in gathered_list]
    return output_list

# def dist_collect_list(x):
#     """collect all list from all GPUs"""
#     encoded_list = [[ord(char) for char in string] for string in x]

#     # Perform all_gather to collect the encoded lists from different GPUs
#     world_size = dist.get_world_size()
#     gathered_lists = [torch.zeros_like(encoded_list) for _ in range(world_size)]
#     dist.all_gather(gathered_lists, torch.tensor(encoded_list))

#     # Convert the gathered encoded lists back to a list of strings
#     output_list = ["".join([chr(char) for char in gathered]) for gathered_list in gathered_lists for gathered in gathered_list]

#     # Print the collected list of strings
#     print(output_list)

    # Assuming x is the list of strings on each GPU
    # Convert the list of strings to a list of tensors on each GPU
    # tensor_list = [torch.tensor([ord(char) for char in string]) for string in x]

    # # Perform all_gather to collect the tensors from different GPUs
    # world_size = dist.get_world_size()
    # gathered_tensors = [torch.empty_like(tensor) for tensor in tensor_list]
    # dist.all_gather(gathered_tensors, tensor_list)

    # # Convert the gathered tensors back to a list of strings
    # output_list = ["".join([chr(char) for char in tensor]) for gathered in gathered_tensors for tensor in gathered]

    # # Print the collected list of strings
    # print(output_list)

    # Assuming x is the list tensor on each GPU
    
    # x = x.contiguous()

    # # Gather the tensors from all processes
    # out_list = [torch.zeros_like(x, device=x.device, dtype=x.dtype).contiguous() for _ in range(dist.get_world_size())]
    # out_list = dist.all_gather(out_list, x)

    # # Concatenate the gathered tensors along the specified dimension
    # result = torch.cat(out_list, dim=0).contiguous()
    # all_lists = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
    # dist.all_gather(all_lists, ddp_model.module.list)
    # out_list = [torch.zeros_like(x).contiguous() for _ in range(dist.get_world_size())]
    # out_list = diff_dist.all_gather(out_list, x)
    # return out_list


class ProjectMLP(nn.Module):

    def __init__(self, in_dim=256, inner_dim=4096, out_dim=256, num_layers=2):
        super(ProjectMLP, self).__init__()
        # hidden layers
        linear_hidden = []
        for i in range(num_layers - 1):
            linear_hidden.append(nn.Conv1d(in_dim if i == 0 else inner_dim, inner_dim, kernel_size=1))
            linear_hidden.append(nn.BatchNorm1d(inner_dim))
            linear_hidden.append(nn.ReLU(inplace=True))
        self.linear_hidden = nn.Sequential(*linear_hidden)

        self.linear_out = nn.Conv1d(
            in_dim if num_layers == 1 else inner_dim, out_dim, kernel_size=1) if num_layers >= 1 else nn.Identity()

    def forward(self, x):
        """

        Args:
            x (torch.Tensor): output of transformers, shape [B, L, C]

        Returns:

        """
        assert x.ndim in [2, 3], x.ndim
        add_dim = False
        if x.ndim == 2:
            # [B, C] -> [B, L, C]
            x = x.unsqueeze(1)
            add_dim = True

        x = rearrange(x, 'b l c -> b c l')
        x = self.linear_hidden(x)
        x = self.linear_out(x)
        x = rearrange(x, 'b c l -> b l c')

        if add_dim:
            x = x.squeeze(1)

        return x


@MODELS.register_module()
class MultiLabelContrastive2(nn.Module):

    def __init__(self,
                 img_encoder,
                 text_encoder,
                 output_dim=256,
                 contrast_temperature=0.07,
                 proj_num_layers=2,
                 multi_label=0,
                 share_temperature=False,
                 multi_label_loss_weight=1.0,
                 debugging=False,
                 use_tiered_entropy_loss=False,
                 use_group_token_entropy_loss=False,
                 use_label_entropy_loss=False,
                 use_pad_token=False,):
        super().__init__()

        self.img_encoder = MODELS.build(img_encoder)
        self.text_encoder = MODELS.build(text_encoder)


        self.contrast_temperature = contrast_temperature
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / contrast_temperature))
        self.cross_entropy = nn.CrossEntropyLoss()
        self.soft_cross_entropy = SoftTargetCrossEntropy()
        
        self.proj_num_layers = proj_num_layers
        self.multi_label = multi_label
        if proj_num_layers > 0:
            self.img_projector = ProjectMLP(
                in_dim=self.img_encoder.width, num_layers=proj_num_layers, out_dim=output_dim)
            self.text_projector = ProjectMLP(
                in_dim=self.text_encoder.width, num_layers=proj_num_layers, out_dim=output_dim)
        else:
            self.img_projector = nn.Identity()
            self.text_projector = nn.Identity()
        self.debugging = debugging
        self.with_tier_entropy_loss = use_tiered_entropy_loss
        self.with_gt_entropy_loss = use_group_token_entropy_loss
        self.with_label_entropy_loss = use_label_entropy_loss
        

        self.use_pad_token = use_pad_token
        self.padtoken = None
        
        self.share_temperature = share_temperature
        if self.with_multi_label and not self.share_temperature:
            self.multi_label_logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / contrast_temperature))
        self.multi_label_loss_weight = multi_label_loss_weight
        

    @property
    def with_multi_label(self):
        return self.multi_label > 0

    def loss(self, image_x, text_x):

        batch_size = image_x.shape[0]
        #print("current batch size", batch_size)
        # get label globally
        labels = torch.arange(batch_size, dtype=torch.long, device=image_x.device) + batch_size * dist.get_rank()
        #print("labels",labels)
        # [B, C]
        image_x = F.normalize(image_x, dim=-1)
        text_x = F.normalize(text_x, dim=-1)

        logits_per_img = image_x @ dist_collect(text_x).t()
        logits_per_text = text_x @ dist_collect(image_x).t()
        # print("logits per image", logits_per_img.shape)
        # print("logits per text", logits_per_text.shape)

        logit_scale = torch.clamp(self.logit_scale.exp(), max=100)
        loss_img = self.cross_entropy(logits_per_img * logit_scale, labels)
        loss_text = self.cross_entropy(logits_per_text * logit_scale, labels)
        
        loss = 0.5 * (loss_img + loss_text)

        return loss

    def loss_sync(self, image_x, text_x):

        batch_size = image_x.shape[0]
        
        # get label globally
        labels = torch.arange(batch_size, dtype=torch.long, device=image_x.device) + batch_size * 0 #dist.get_rank()
        # [B, C]
        image_x = F.normalize(image_x, dim=-1)
        text_x = F.normalize(text_x, dim=-1)

        logits_per_img = image_x @ text_x.t()
        logits_per_text = text_x @ image_x.t()

        logit_scale = torch.clamp(self.logit_scale.exp(), max=100)
        loss_img = self.cross_entropy(logits_per_img * logit_scale, labels)
        loss_text = self.cross_entropy(logits_per_text * logit_scale, labels)
        
        loss = 0.5 * (loss_img + loss_text)

        return loss

    def multi_label_loss(self, image_feat, text_feat):
        """

        Args:
            image_feat (torch.Tensor): shape [B, L1, C]
            text_feat (torch.Tensor): shape [B, L2, C]

        Returns:

        """
        # [B, L1, C], L1 = 1
        image_feat = F.normalize(image_feat, dim=-1)
        # [B, L2, C]
        text_feat = F.normalize(text_feat, dim=-1)

        # [B, L1, L2]
        dist_per_img = image_feat @ rearrange(text_feat, 'b l c -> b c l')
        # [B, L2, L1]
        dist_per_text = text_feat @ rearrange(image_feat, 'b l c -> b c l')

        if self.share_temperature:
            logit_scale = torch.clamp(self.logit_scale.exp(), max=100)
        else:
            logit_scale = torch.clamp(self.multi_label_logit_scale.exp(), max=100)

        batch = image_feat.shape[0]
        img_len = image_feat.shape[1]
        text_len = text_feat.shape[1]
        # [B, L1, L2]
        pos_labels_batch_img = rearrange(torch.ones_like(dist_per_text) / dist_per_text.size(1), 'b l2 l1 -> b l1 l2')
        # [B, L2, L1]
        pos_labels_batch_text = rearrange(torch.ones_like(dist_per_img) / dist_per_img.size(1), 'b l1 l2 -> b l2 l1')

        image_x = rearrange(image_feat, 'b l c -> (b l) c')
        text_x = rearrange(text_feat, 'b l c -> (b l) c')
        
        logits_per_img = image_x @ dist_collect(text_x).t()
        logits_per_text = text_x @ dist_collect(image_x).t()
        
        # get label globally
        # [B, L1, B, L2, W]
        labels_per_img = F.one_hot(
            torch.ones(batch, img_len, batch, text_len, dtype=torch.long, device=image_x.device) * dist.get_rank(),
            num_classes=dist.get_world_size()).to(image_x.dtype)
        labels_per_img *= rearrange(pos_labels_batch_img, 'b l1 l2 -> b l1 1 l2 1') * repeat(
            torch.eye(batch, dtype=image_x.dtype, device=image_x.device), 'b1 b2 -> b1 1 b2 1 1')
        # [BxL1, WxBxL2]
        labels_per_img = rearrange(labels_per_img, 'b1 l1 b2 l2 w -> (b1 l1) (w b2 l2)')
        # [B, L2, B, L1, W]
        labels_per_text = F.one_hot(
            torch.ones(batch, text_len, batch, img_len, dtype=torch.long, device=text_x.device) * dist.get_rank(),
            num_classes=dist.get_world_size()).to(text_x.dtype)
        labels_per_text *= rearrange(pos_labels_batch_text, 'b l2 l1 -> b l2 1 l1 1') * repeat(
            torch.eye(batch, dtype=text_x.dtype, device=image_x.device), 'b2 b1 -> b2 1 b1 1 1')
        # [BxL2, WxBxL1]
        labels_per_text = rearrange(labels_per_text, 'b2 l2 b1 l1 w -> (b2 l2) (w b1 l1)')

        loss_img = self.soft_cross_entropy(logits_per_img * logit_scale, labels_per_img)
        loss_text = self.soft_cross_entropy(logits_per_text * logit_scale, labels_per_text)

        loss = 0.5 * (loss_img + loss_text)

        return loss
    
    def refined_multi_label_loss(self, image_feat, text_feat, text_meta):
        """

        Args:
            image_feat (torch.Tensor): shape [B, L1, C]
            text_feat (torch.Tensor): shape [B, L2, C]

        Returns:

        """
        # [B, L1, C], L1 = 1
        image_feat = F.normalize(image_feat, dim=-1)
        # [B, L2, C]
        text_feat = F.normalize(text_feat, dim=-1)
        # [B, L1, L2]
        dist_per_img = image_feat @ rearrange(text_feat, 'b l c -> b c l')
        # [B, L2, L1]
        dist_per_text = text_feat @ rearrange(image_feat, 'b l c -> b c l')


        if self.share_temperature:
            logit_scale = torch.clamp(self.logit_scale.exp(), max=100)
        else:
            logit_scale = torch.clamp(self.multi_label_logit_scale.exp(), max=100)

        batch = image_feat.shape[0]
        img_len = image_feat.shape[1]
        text_len = text_feat.shape[1]

        pos_labels_batch_img_onehot = rearrange(torch.ones_like(dist_per_text), 'b l2 l1 -> b l1 l2')
        
        pos_labels_batch_text_onehot = rearrange(torch.ones_like(dist_per_img), 'b l1 l2 -> b l2 l1')
        
        image_x = rearrange(image_feat, 'b l c -> (b l) c')
        text_x = rearrange(text_feat, 'b l c -> (b l) c')
        
        logits_per_img = image_x @ dist_collect(text_x).t()
        logits_per_text = text_x @ dist_collect(image_x).t()

        #print shape of logits
        print('logits_per_text', logits_per_text.shape)
        print('logits_per_img', logits_per_img.shape)
        print('logits_per_text', logits_per_text)
        print('logits_per_img', logits_per_img)

        labels_per_img_onehot = F.one_hot(
            torch.ones(batch, img_len, batch, text_len, dtype=torch.long, device=image_x.device) * dist.get_rank(), 
            num_classes=dist.get_world_size()).to(image_x.dtype)
        labels_per_img_onehot *= rearrange(pos_labels_batch_img_onehot, 'b l1 l2 -> b l1 1 l2 1') * repeat(
            torch.eye(batch, dtype=image_x.dtype, device=image_x.device), 'b1 b2 -> b1 1 b2 1 1')
        labels_per_img_onehot = rearrange(labels_per_img_onehot, 'b1 l1 b2 l2 w -> (b1 l1) (w b2 l2)')

        text_meta = dist_collect(text_meta)

        text_meta_flatten = rearrange(text_meta, 'b l2 w -> (b l2) w')
        true_postive_labels_per_img = []
        for img_map in labels_per_img_onehot:
            texts = [text_meta_flatten[i] for i, val in enumerate(img_map) if (val == 1 and not torch.equal(text_meta_flatten[i], self.padtoken))]
            text_positive = [0] * len(text_meta_flatten)
            for text in texts:
                for i, meta in enumerate(text_meta_flatten):
                    if torch.equal(meta, text):
                        text_positive[i] = 1
            if sum(text_positive) > 0:
                text_positive = [round(x/sum(text_positive), 4) for x in text_positive]
            true_postive_labels_per_img.append(text_positive)
        true_postive_labels_per_img = torch.tensor(true_postive_labels_per_img, dtype=torch.float32).to(image_x.device)

        # [B, L2, B, L1, W]

        labels_per_text_onehot = F.one_hot(
            torch.ones(batch, text_len, batch, img_len, dtype=torch.long, device=text_x.device) * dist.get_rank(),
            num_classes=dist.get_world_size()).to(text_x.dtype)
        labels_per_text_onehot *= rearrange(pos_labels_batch_text_onehot, 'b l2 l1 -> b l2 1 l1 1') * repeat(
            torch.eye(batch, dtype=text_x.dtype, device=image_x.device), 'b2 b1 -> b2 1 b1 1 1')
        labels_per_text_onehot = rearrange(labels_per_text_onehot, 'b2 l2 b1 l1 w -> (b2 l2) (w b1 l1)')
        
        # true_postive_img_per_label = []
        # for idx, text_map in enumerate(labels_per_text_onehot):
        #     text = text_meta_flatten[idx]
        #     if torch.equal(text, self.padtoken):
        #         true_postive_img_per_label.append([0]*batch*dist.get_world_size())
        #         continue

        #     #get all the indices in text_meta_flatten which has the same text as text
        #     matched_texts_indices = [i for i, t in enumerate(text_meta_flatten) if torch.equal(t, text)]

        #     img_map_for_text = [0]*batch*dist.get_world_size()
            
        #     for index in matched_texts_indices:
        #         img_map_for_text[int(index/self.multi_label)] = 1
        #     if sum(img_map_for_text) > 0:
        #         img_positive = [round(x/sum(img_map_for_text), 4) for x in img_map_for_text]
        #     true_postive_img_per_label.append(img_positive)
        # true_postive_img_per_label = torch.tensor(true_postive_img_per_label, dtype=torch.float32).to(text_x.device)
        true_positive_img_per_label = []
        for idx, text_map in enumerate(labels_per_text_onehot):
            text = text_meta_flatten[idx]
            if torch.equal(text, self.padtoken):
                true_positive_img_per_label.append(torch.zeros(batch * dist.get_world_size(), dtype=torch.float32))
                continue

            matched_texts_indices = torch.nonzero(torch.all(text_meta_flatten == text, dim=1)).flatten()
            img_map_for_text = torch.zeros(batch * dist.get_world_size(), dtype=torch.float32)

            img_map_for_text[matched_texts_indices // self.multi_label] = 1
            if img_map_for_text.sum() > 0:
                img_positive = img_map_for_text / img_map_for_text.sum()
            else:
                img_positive = img_map_for_text

            true_positive_img_per_label.append(img_positive)

        true_positive_img_per_label = torch.stack(true_positive_img_per_label).to(text_x.device)


        loss_img = self.soft_cross_entropy(logits_per_img * logit_scale, true_postive_labels_per_img)
        loss_text = self.soft_cross_entropy(logits_per_text * logit_scale, true_positive_img_per_label)

        loss = 0.5 * (loss_img + loss_text)

        return loss
    
    def entropy_loss(self, image_feat, text_feat, weighted=True):
        """

        Args:
            image_feat (torch.Tensor): shape [B, L1, C]
            text_feat (torch.Tensor): shape [B, L2, C]

        Returns:

        """
        # [B, L1, C], L1 = 8
        image_feat = F.normalize(image_feat, dim=-1)
        # [B, L2, C]
        text_feat = F.normalize(text_feat, dim=-1)

        # [B, L1, L2]
        simi_mat = image_feat @ rearrange(text_feat, 'b l c -> b c l')

        dist_img = torch.softmax(simi_mat, dim=-1)
        dist_text = torch.softmax(simi_mat, dim=-2)

        loss_img = -torch.mean(torch.sum(dist_img * torch.log(dist_img + 1e-8), dim=-1))
        loss_text = -torch.mean(torch.sum(dist_text * torch.log(dist_text + 1e-8), dim=-2))

        if weighted:
            loss = 0.2 *  loss_img + 0.1 * loss_text
        else:
            loss = 0.5 * ( loss_img + loss_text)

        return loss
    
    def multi_label_loss_sync(self, image_feat, text_feat):
        """

        Args:
            image_feat (torch.Tensor): shape [B, L1, C]
            text_feat (torch.Tensor): shape [B, L2, C]

        Returns:

        """
        # [B, L1, C], L1 = 1
        image_feat = F.normalize(image_feat, dim=-1)
        # [B, L2, C]
        text_feat = F.normalize(text_feat, dim=-1)

        # [B, L1, L2]
        dist_per_img = image_feat @ rearrange(text_feat, 'b l c -> b c l')
        # [B, L2, L1]
        dist_per_text = text_feat @ rearrange(image_feat, 'b l c -> b c l')

        if self.share_temperature:
            logit_scale = torch.clamp(self.logit_scale.exp(), max=100)
        else:
            logit_scale = torch.clamp(self.multi_label_logit_scale.exp(), max=100)

        batch = image_feat.shape[0]
        img_len = image_feat.shape[1]
        text_len = text_feat.shape[1]
        # [B, L1, L2]
        pos_labels_batch_img = rearrange(torch.ones_like(dist_per_text) / dist_per_text.size(1), 'b l2 l1 -> b l1 l2')
        # [B, L2, L1]
        pos_labels_batch_text = rearrange(torch.ones_like(dist_per_img) / dist_per_img.size(1), 'b l1 l2 -> b l2 l1')

        image_x = rearrange(image_feat, 'b l c -> (b l) c')
        text_x = rearrange(text_feat, 'b l c -> (b l) c')
        
        logits_per_img = image_x @ text_x.t()
        logits_per_text =  text_x @ image_x.t()
        
        # get label globally
        # [B, L1, B, L2, W]
        labels_per_img = F.one_hot(
            torch.ones(batch, img_len, batch, text_len, dtype=torch.long, device=image_x.device) * 0,
            num_classes=1).to(image_x.dtype)
        labels_per_img *= rearrange(pos_labels_batch_img, 'b l1 l2 -> b l1 1 l2 1') * repeat(
            torch.eye(batch, dtype=image_x.dtype, device=image_x.device), 'b1 b2 -> b1 1 b2 1 1')
        # [BxL1, WxBxL2]
        labels_per_img = rearrange(labels_per_img, 'b1 l1 b2 l2 w -> (b1 l1) (w b2 l2)')
        # [B, L2, B, L1, W]
        labels_per_text = F.one_hot(
            torch.ones(batch, text_len, batch, img_len, dtype=torch.long, device=text_x.device) * 0,
            num_classes=1).to(text_x.dtype)
        labels_per_text *= rearrange(pos_labels_batch_text, 'b l2 l1 -> b l2 1 l1 1') * repeat(
            torch.eye(batch, dtype=text_x.dtype, device=image_x.device), 'b2 b1 -> b2 1 b1 1 1')
        # [BxL2, WxBxL1]
        labels_per_text = rearrange(labels_per_text, 'b2 l2 b1 l1 w -> (b2 l2) (w b1 l1)')

        loss_img = self.soft_cross_entropy(logits_per_img * logit_scale, labels_per_img)
        loss_text = self.soft_cross_entropy(logits_per_text * logit_scale, labels_per_text)

        loss = 0.5 * (loss_img + loss_text)

        return loss

    def encode_image(self, image, *, return_feat=False, return_attn = True, as_dict=False):
        outs = Result(as_dict)
        img_outs = self.img_encoder(image, return_feat=return_feat, return_attn=True, as_dict=True)
        outs.append(self.img_projector(img_outs['x']), 'image_x')
        if return_feat:
            outs.append(self.img_projector(img_outs['feat']), 'image_feat')
        if return_attn:
            outs.append(img_outs['attn_dicts'], 'image_attn')
        return outs.as_return()

    def encode_text(self, text, *, as_dict=False):
        assert text.ndim in [2, 3], text.ndim
        squeeze_dim = False
        num_text = 1
        if text.ndim == 3:
            num_text = text.shape[1]
            text = rearrange(text, 'b n l -> (b n) l', n=num_text)
            squeeze_dim = True

        outs = Result(as_dict=as_dict)
        # [B, C]
        x = self.text_encoder(text)
        text_x = self.text_projector(x)
        outs.append(text_x, 'text_x')
        if squeeze_dim:
            text_x = rearrange(text_x, '(b n) c -> b n c', n=num_text)
            text_multi_label_x = text_x[:, 1:]
            text_x = text_x[:, 0]
            outs.update(text_x=text_x, text_multi_label_x=text_multi_label_x)

        return outs.as_return()

    def get_attn_maps(self, attn_dicts):
            attn_maps = []
            prev_attn_masks = None
            for idx, attn_dict in enumerate(attn_dicts):
                if attn_dict is None:
                    assert idx == len(attn_dicts) - 1, 'only last layer can be None'
                    continue
                # [B, G, HxW]
                # B: batch size (1), nH: number of heads, G: number of group token
                attn_masks = attn_dict['soft']
                # [B, nH, G, HxW] -> [B, nH, HxW, G]
                attn_masks = rearrange(attn_masks, 'b h g n -> b h n g')
                if prev_attn_masks is None:
                    prev_attn_masks = attn_masks
                else:
                    prev_attn_masks = prev_attn_masks @ attn_masks
                
                attn_maps.append(prev_attn_masks)

            for i in range(len(attn_maps)):
                attn_map = attn_maps[i]
                # [B, nh, H, W, G]
                assert attn_map.shape[1] == 1
                # [B, H, W, G]
                attn_map = attn_map.squeeze(1)
                attn_maps[i] = attn_map
            return attn_maps

    def forward_train(self, image, text, text_meta):
        image_outs = self.encode_image(image,  return_feat=True, return_attn=True, as_dict=True)

        # [B, C]
        #print("text meta", text_meta)
        #text_meta = text_meta.to(image.device)
        image_x = image_outs['image_x']
        if self.padtoken is not None:
            #print("text before", text)
            mask_tensor = text.eq(self.padtoken).all(dim=-1)
            #print("mask tensor", mask_tensor)
            binary_mask = mask_tensor.unsqueeze(-1).repeat(1, 1, text.shape[-1])
            # Apply the mask to the text tensor
            #print("binary mask", binary_mask)
            
            #print("text before", text)
            text.masked_fill_(binary_mask, 0)

        text_outs = self.encode_text(text, as_dict=True)
        # [B, C]
        text_x = text_outs['text_x']
        
        # [B, G, C]
        grouped_img_tokens = image_outs['image_feat']
         #normalize group image tokens
        grouped_img_tokens = F.normalize(grouped_img_tokens, dim=-1)
        
        losses = self.loss(image_x, text_x)

        losses_dict = dict(loss=losses)

        if self.with_gt_entropy_loss or self.with_label_entropy_loss:
            attn_maps = self.get_attn_maps(image_outs['image_attn'])
            attn_map = attn_maps[-1]
            attn_map = F.softmax(attn_map, dim=-1)
            entropy = -torch.mean(attn_map * torch.log(attn_map), dim=-1)
            entropy_loss = torch.mean(entropy)
            losses_dict['gt_entropy_loss'] = entropy_loss

            if self.with_label_entropy_loss:
                image_feat = F.normalize(grouped_img_tokens, dim=-1)
                text_feat = F.normalize(text_outs['text_multi_label_x'], dim=-1)
                simi_mat = image_feat @ rearrange(text_feat, 'b l c -> b c l') #[8,256] *[256,5]
                dist_img = torch.softmax(simi_mat, dim=-1)
                label_dist = torch.softmax(attn_map @ dist_img, dim=-1) #[28,28,8] * [8,5] = [28,28,5]
                entropy = -torch.mean(label_dist * torch.log(label_dist), dim=-1)
                entropy_label_loss = torch.mean(entropy)
                losses_dict['label_entropy_loss'] = entropy_label_loss

        if self.with_tier_entropy_loss:
            losses_dict['entropy_loss'] = self.entropy_loss(grouped_img_tokens, text_outs['text_x'].unsqueeze(1))
            if self.with_multi_label:
                losses_dict['multi_entropy_loss'] = self.entropy_loss(grouped_img_tokens, text_outs['text_multi_label_x'])
        if self.with_multi_label:
            image_multi_label_x = image_x.unsqueeze(1)
            text_multi_label_x = text_outs['text_multi_label_x']
            text_multi_label_x_masked = text_multi_label_x.masked_scatter_(binary_mask[:,1:].sum(dim=-1).bool().unsqueeze(-1), self.PAD_EMBEDDING.expand(text_multi_label_x.shape[0], text_multi_label_x.shape[1], self.PAD_EMBEDDING.shape[-1]).to(text_multi_label_x.dtype))
            # print("text multi labelbefore", text_multi_label_x)
            
            # losses_dict['multi_label_loss'] = self.multi_label_loss(image_multi_label_x,
            #                                                         text_multi_label_x_masked) * self.multi_label_loss_weight
                        
            losses_dict['multi_label_loss'] = self.refined_multi_label_loss(image_multi_label_x, text_multi_label_x_masked, text_meta=text_meta) * self.multi_label_loss_weight
                                                                            
            # losses_dict['multi_label_loss'] = self.multi_label_loss(image_multi_label_x,
            #                                                         text_multi_label_x) * self.multi_label_loss_weight
        return losses_dict
    
    def forward_train_sync(self, image, text):
        image_outs = self.encode_image(image,  return_feat=True, return_attn=True, as_dict=True)
        # [B, C]
        image_x = image_outs['image_x']
        if self.padtoken is not None:
            self.padtoken = torch.tensor(self.padtoken).to(text.device)

            mask_tensor = text.eq(self.padtoken).all(dim=-1)
            #print("mask tensor", mask_tensor)
            binary_mask = mask_tensor.unsqueeze(-1).repeat(1, 1, text.shape[-1])
            # Apply the mask to the text tensor
            #print("binary mask", binary_mask)
            
            #print("text before", text)
            text.masked_fill_(binary_mask, 0)
            #print("text after", text)

        text_outs = self.encode_text(text, as_dict=True)
        # [B, C]
        text_x = text_outs['text_x']
        
        # [B, G, C]
        grouped_img_tokens = image_outs['image_feat']
         #normalize group image tokens
        grouped_img_tokens = F.normalize(grouped_img_tokens, dim=-1)

        losses = self.loss_sync(image_x, text_x)

        losses_dict = dict(loss=losses)
        
        if self.with_gt_entropy_loss or self.with_label_entropy_loss:
            attn_maps = self.get_attn_maps(image_outs['image_attn'])
            attn_map = attn_maps[-1]
            attn_map = F.softmax(attn_map, dim=-1)
            entropy = -torch.mean(attn_map * torch.log(attn_map), dim=-1)
            entropy_loss = torch.mean(entropy)
            losses_dict['gt_entropy_loss'] = entropy_loss
            if self.with_label_entropy_loss:
                image_feat = F.normalize(grouped_img_tokens, dim=-1)
                text_feat = F.normalize(text_outs['text_multi_label_x'], dim=-1)
                simi_mat = image_feat @ rearrange(text_feat, 'b l c -> b c l')
                dist_img = torch.softmax(simi_mat, dim=-1)
                label_dist = torch.softmax(attn_map @ dist_img, dim=-1)
                entropy = -torch.mean(label_dist * torch.log(label_dist), dim=-1)
                entropy_label_loss = torch.mean(entropy)
                losses_dict['label_entropy_loss'] = entropy_label_loss

        if self.with_entropy_loss:
            losses_dict['entropy_loss'] = self.entropy_loss(grouped_img_tokens, text_outs['text_x'].unsqueeze(1))
            if self.with_multi_label:
                losses_dict['multi_entropy_loss'] = self.entropy_loss(grouped_img_tokens, text_outs['text_multi_label_x'])
        if self.with_multi_label:
            image_multi_label_x = image_x.unsqueeze(1)
            text_multi_label_x = text_outs['text_multi_label_x']
            text_multi_label_x_masked = text_multi_label_x.masked_scatter_(binary_mask[:,1:].sum(dim=-1).bool().unsqueeze(-1), self.PAD_EMBEDDING.expand(text_multi_label_x.shape[0], text_multi_label_x.shape[1], self.PAD_EMBEDDING.shape[-1]))
            losses_dict['multi_label_loss'] = self.multi_label_loss_sync(image_multi_label_x,
                                                                    text_multi_label_x_masked) * self.multi_label_loss_weight
            
            # losses_dict['multi_label_loss'] = self.multi_label_loss_sync(image_multi_label_x,
            #                                                         text_multi_label_x) * self.multi_label_loss_weight

        return losses_dict
        # image_outs = self.encode_image(image, as_dict=True)
        # # [B, C]
        # image_x = image_outs['image_x']

        # text_outs = self.encode_text(text, as_dict=True)
        # # [B, C]
        # text_x = text_outs['text_x']

        # losses = self.loss_sync(image_x, text_x)

        # losses_dict = dict(loss=losses)
        # if self.with_multi_label:
        #     image_multi_label_x = image_x.unsqueeze(1)
        #     text_multi_label_x = text_outs['text_multi_label_x']
        #     losses_dict['multi_label_loss'] = self.multi_label_loss_sync(image_multi_label_x,
        #                                                             text_multi_label_x) * self.multi_label_loss_weight

        # return losses_dict
    def forward_test(self, image, text):
        return self.zero_shot_pred(image, text)

    def forward(self, image, text, text_meta=None):
        if self.training and not self.debugging:
            return self.forward_train(image, text, text_meta)
        elif self.debugging:
            return self.forward_train_sync(image, text)
        else:
            return self.forward_test(image, text)
    
    @torch.no_grad()
    def build_padtoken_embedding(self, padtoken):
        """
        Args: padtoken (torch.Tensor): [1, CONTEXT_LENGTH]

        """
        #check if device is cuda or cpu
        #initialize device with cuda id cuda is available else cpu
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #device = cuda if torch.cuda.is_available() else cpu
        self.padtoken = padtoken.to(device)
        #print("padtoken", self.padtoken)
        
        self.PAD_EMBEDDING = torch.zeros(1, self.text_encoder.width, dtype=torch.float32).to(device)


    @torch.no_grad()
    def build_text_embedding(self, text):
        """

        Args:
            text (torch.Tensor): [NUM_CLASSES, NUM_TEMPLATES, CONTEXT_LENGTH]

        Returns:

        """
        text = text.to(next(self.parameters()).device)
        num_classes, num_templates = text.shape[:2]
        text = rearrange(text, 'n t l -> (n t) l', n=num_classes, t=num_templates)
        text_tokens = self.encode_text(text)
        # [N, T, C]
        text_tokens = rearrange(text_tokens, '(n t) c -> n t c', n=num_classes, t=num_templates)
        # [N, C]
        text_tokens = text_tokens.mean(dim=1)
        text_tokens = F.normalize(text_tokens, dim=-1)

        return text_tokens

    @torch.no_grad()
    def zero_shot_pred(self, image, text):
        # [B, C]
        image_features = self.encode_image(image)
        image_features = F.normalize(image_features, dim=-1)

        # cosine similarity as logits
        logits_per_image = image_features @ text.t()

        return logits_per_image
