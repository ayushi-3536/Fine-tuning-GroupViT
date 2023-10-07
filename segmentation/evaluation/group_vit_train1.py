# -------------------------------------------------------------------------
# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License.
# To view a copy of this license, visit
# https://github.com/NVlabs/GroupViT/blob/main/LICENSE
#
# Written by Jiarui Xu
# -------------------------------------------------------------------------

import os
import os.path as osp
import matplotlib.pyplot as plt
import mmcv
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from mmseg.models import EncoderDecoder
from PIL import Image
from utils import get_logger

GROUP_PALETTE = np.loadtxt(osp.join(osp.dirname(osp.abspath(__file__)), 'group_palette.txt'), dtype=np.uint8)[:, ::-1]


def resize_attn_map(attentions, h, w, align_corners=False):
    """

    Args:
        attentions: shape [B, num_head, H*W, groups]
        h:
        w:

    Returns:

        attentions: shape [B, num_head, h, w, groups]


    """
    scale = (h * w // attentions.shape[2])**0.5
    if h > w:
        w_featmap = w // int(np.round(scale))
        h_featmap = attentions.shape[2] // w_featmap
    else:
        h_featmap = h // int(np.round(scale))
        w_featmap = attentions.shape[2] // h_featmap
    assert attentions.shape[
        2] == h_featmap * w_featmap, f'{attentions.shape[2]} = {h_featmap} x {w_featmap}, h={h}, w={w}'

    bs = attentions.shape[0]
    nh = attentions.shape[1]  # number of head
    groups = attentions.shape[3]  # number of group token
    # [bs, nh, h*w, groups] -> [bs*nh, groups, h, w]
    attentions = rearrange(
        attentions, 'bs nh (h w) c -> (bs nh) c h w', bs=bs, nh=nh, h=h_featmap, w=w_featmap, c=groups)
    attentions = F.interpolate(attentions, size=(h, w), mode='bilinear', align_corners=align_corners)
    #  [bs*nh, groups, h, w] -> [bs, nh, h*w, groups]
    attentions = rearrange(attentions, '(bs nh) c h w -> bs nh h w c', bs=bs, nh=nh, h=h, w=w, c=groups)

    return attentions


def top_groups(attn_map, k):
    """
    Args:
        attn_map: (B, H, W, G)
        k: int

    Return:
        (B, H, W, k)
    """

    attn_map = attn_map.clone()

    for i in range(attn_map.size(0)):
        # [H*W, G]
        flatten_map = rearrange(attn_map[i], 'h w g -> (h w) g')
        kept_mat = torch.zeros(flatten_map.shape[0], device=flatten_map.device, dtype=torch.bool)
        area_per_group = flatten_map.sum(dim=0)
        top_group_idx = area_per_group.topk(k=k).indices.cpu().numpy().tolist()
        for group_idx in top_group_idx:
            kept_mat[flatten_map.argmax(dim=-1) == group_idx] = True
        # [H, W, 2]
        coords = torch.stack(
            torch.meshgrid(
                torch.arange(attn_map[i].shape[0], device=attn_map[i].device, dtype=attn_map[i].dtype),
                torch.arange(attn_map[i].shape[1], device=attn_map[i].device, dtype=attn_map[i].dtype)),
            dim=-1)
        coords = rearrange(coords, 'h w c -> (h w) c')

        # calculate distance between each pair of points
        # [non_kept, kept]
        dist_mat = torch.sum((coords[~kept_mat].unsqueeze(1) - coords[kept_mat].unsqueeze(0))**2, dim=-1)

        flatten_map[~kept_mat] = flatten_map[kept_mat.nonzero(as_tuple=True)[0][dist_mat.argmin(dim=-1)]]

        attn_map[i] = flatten_map.reshape_as(attn_map[i])

    return attn_map


def seg2coord(seg_map):
    """
    Args:
        seg_map (np.ndarray): (H, W)

    Return:
        dict(group_id -> (x, y))
    """
    h, w = seg_map.shape
    # [h ,w, 2]
    coords = np.stack(np.meshgrid(np.arange(h), np.arange(w), indexing='ij'), axis=-1)
    labels = np.unique(seg_map)
    coord_map = {}
    for label in labels:
        coord_map[label] = coords[seg_map == label].mean(axis=0)
    return coord_map


class GroupViTTrainAssessment(EncoderDecoder):

    def __init__(self, model, test_cfg=dict(mode='whole', bg_thresh=.95), with_bg=False):
        super(EncoderDecoder, self).__init__()
        if not isinstance(test_cfg, mmcv.Config):
            test_cfg = mmcv.Config(test_cfg)
        self.test_cfg = test_cfg
        
        self.model = model
        self.with_bg = with_bg
        self.bg_thresh = test_cfg['bg_thresh']
        self.align_corners = False
        self.PALETTE = GROUP_PALETTE[64:73]
        ##logger = get_logger()
        #logger.info(
        #    f'Building GroupViTSegInference with {self.num_classes} classes, test_cfg={test_cfg}, with_bg={with_bg}')

    def forward_train(self, img, img_metas, gt_semantic_seg):
        raise NotImplementedError

    def get_attn_maps(self, img, return_onehot=False, rescale=False):
        """
        Args:
            img: [B, C, H, W]

        Returns:
            attn_maps: list[Tensor], attention map of shape [B, H, W, groups]
        """
        #logger = get_logger()
        results = self.model.img_encoder(img, return_attn=True, as_dict=True)

        attn_maps = []
        with torch.no_grad():
            prev_attn_masks = None
            for idx, attn_dict in enumerate(results['attn_dicts']):
                if attn_dict is None:
                    assert idx == len(results['attn_dicts']) - 1, 'only last layer can be None'
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
                # [B, nH, HxW, G] -> [B, nH, H, W, G]
                attn_maps.append(resize_attn_map(prev_attn_masks, *img.shape[-2:]))

        for i in range(len(attn_maps)):
            attn_map = attn_maps[i]
            # [B, nh, H, W, G]
            assert attn_map.shape[1] == 1
            # [B, H, W, G]
            attn_map = attn_map.squeeze(1)

            if rescale:
                attn_map = rearrange(attn_map, 'b h w g -> b g h w')
                attn_map = F.interpolate(
                    attn_map, size=img.shape[2:], mode='bilinear', align_corners=self.align_corners)
                attn_map = rearrange(attn_map, 'b g h w -> b h w g')

            if return_onehot:
                # [B, H, W, G]
                attn_map = F.one_hot(attn_map.argmax(dim=-1), num_classes=attn_map.shape[-1]).to(dtype=attn_map.dtype)

            attn_maps[i] = attn_map

        return attn_maps
    
    def set_text_tokens(self, text):
        self.text_tokens = text
        if self.with_bg:
            self.num_classes = len(self.text_tokens) + 1
        else:
            self.num_classes = len(self.text_tokens)
        

    def encode_decode(self,img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""

        assert img.shape[0] == 1, 'batch size must be 1'

        # [B, C, H, W], get the last one only
        attn_map = self.get_attn_maps(img, rescale=True)[-1]
        # [H, W, G], select batch idx 0
        attn_map = attn_map[0]

        # [H, W, G]
        onehot_attn_map = F.one_hot(attn_map.argmax(dim=-1), num_classes=attn_map.shape[-1]).to(dtype=attn_map.dtype)

        text_outs = self.model.encode_text(self.text_tokens, as_dict=True)
        # [B, C]
        text_embeddings = text_outs['text_x']
        text_tokens = text_embeddings


        num_labels = text_embeddings.shape[0]
        class_offset = 1 if self.with_bg else 0
        num_labels = num_labels + class_offset

        logit_scale = torch.clamp(self.model.logit_scale.exp(), max=100)

        img_outs = self.model.encode_image(img, return_feat=True, as_dict=True) #, cross_attn=False)
        
        # [B, L, C] -> [L, C]
        grouped_img_tokens = img_outs['image_feat'].squeeze(0)

        img_avg_feat = img_outs['image_x']
        
        # [G, C]
        grouped_img_tokens = F.normalize(grouped_img_tokens, dim=-1)
        img_avg_feat = F.normalize(img_avg_feat, dim=-1)

        # [G, N]
        group_affinity_mat = (grouped_img_tokens @ text_tokens.T) * logit_scale
        print("group affinity map shape", group_affinity_mat.shape)
        print("group affinity map", group_affinity_mat)
        pre_group_affinity_mat = F.softmax(group_affinity_mat, dim=-1)
        print("after softmax group affinity map", pre_group_affinity_mat)
        
        text_img_affinity_mat = F.softmax(group_affinity_mat, dim=0)
        print("text_img_affinity_mat", text_img_affinity_mat)
        text_img_affinity_mat = F.softmax(group_affinity_mat, dim=0)
        
        print("text_img_affinity_mat", text_img_affinity_mat)
        text_img_affinity_mat = text_img_affinity_mat.argmax(dim=0)
        print("text_img_affinity_mat argma", text_img_affinity_mat)
        avg_affinity_mat = (img_avg_feat @ text_tokens.T) * logit_scale
        avg_affinity_mat = F.softmax(avg_affinity_mat, dim=-1)
        affinity_mask = torch.zeros_like(avg_affinity_mat)
        avg_affinity_topk = avg_affinity_mat.topk(dim=-1, k=num_labels) #min(5, num_labels))
        
        print("avg_affinity_mat affinity map", avg_affinity_mat)
        print("avg_affinity_mataffinity map shape", avg_affinity_mat.shape)
        
        print("affinity mask", affinity_mask)
        affinity_mask.scatter_add_(
            dim=-1, index=avg_affinity_topk.indices, src=torch.ones_like(avg_affinity_topk.values))
        group_affinity_mat.masked_fill_(~affinity_mask.bool(), float('-inf'))

        print("group affinity map masked filled", group_affinity_mat)
        group_affinity_mat = F.softmax(group_affinity_mat, dim=-1)

        print("group affinity map masked filled softmax", group_affinity_mat)
        # TODO: check if necessary
        group_affinity_mat *= pre_group_affinity_mat

        print("pregroup affinity map masked filled softmax", group_affinity_mat)
        pred_logits = torch.zeros(num_labels, *attn_map.shape[:2], device=img.device, dtype=img.dtype)

        print("pred logits shape", pred_logits.shape)
        
        print("pred logits", pred_logits)
        pred_logits[class_offset:] = rearrange(onehot_attn_map @ group_affinity_mat, 'h w c -> c h w')
        print("after pred logits shape", pred_logits.shape)
        
        print("after pred logits", pred_logits)
        if self.with_bg:
            bg_thresh = min(self.bg_thresh, group_affinity_mat.max().item())
            pred_logits[0, (onehot_attn_map @ group_affinity_mat).max(dim=-1).values < bg_thresh] = 1

        return pred_logits.unsqueeze(0)
    
    def get_text_gt_affinity(self, img):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""

        assert img.shape[0] == 1, 'batch size must be 1'

        # [B, C, H, W], get the last one only
        attn_map = self.get_attn_maps(img, rescale=True)[-1]
        # [H, W, G], select batch idx 0
        attn_map = attn_map[0]

        text_outs = self.model.encode_text(self.text_tokens, as_dict=True)
        # [B, C]
        text_embeddings = text_outs['text_x']
        text_tokens = text_embeddings


        num_labels = text_embeddings.shape[0]
        class_offset = 1 if self.with_bg else 0
        num_labels = num_labels + class_offset

        logit_scale = torch.clamp(self.model.logit_scale.exp(), max=100)

        img_outs = self.model.encode_image(img, return_feat=True, as_dict=True) #, cross_attn=False)
        
        # [B, L, C] -> [L, C]
        grouped_img_tokens = img_outs['image_feat'].squeeze(0)

        # [G, C]
        grouped_img_tokens = F.normalize(grouped_img_tokens, dim=-1)

        # [G, N]
        group_affinity_mat = (grouped_img_tokens @ text_tokens.T) * logit_scale
        print("group affinity map shape", group_affinity_mat.shape)
        print("group affinity map", group_affinity_mat)
        text_img_affinity_mat = F.softmax(group_affinity_mat, dim=0)
        print("text_img_affinity_mat", text_img_affinity_mat)
        text_img_affinity_mat = F.softmax(group_affinity_mat, dim=0)
        
        print("text_img_affinity_mat", text_img_affinity_mat)
        text_img_affinity_mat = text_img_affinity_mat.argmax(dim=0)
        return text_img_affinity_mat
    
    def blend_result(self, img, result, only_label=None, palette=None, out_file=None, opacity=0.5, with_bg=False):
        img = mmcv.imread(img)
        img = img.copy()
        seg = result[0]
        if palette is None:
            palette = self.PALETTE
        palette = np.array(palette)
        assert palette.shape[1] == 3, palette.shape
        assert len(palette.shape) == 2
        assert 0 < opacity <= 1.0
        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
        for label, color in enumerate(palette):
            # print("label in blend", label)
            # print("color in blend", color)
            if only_label is not None and label != only_label:
                #print(f"label:{label} is not equal to only_label:{only_label}")
                continue
            color_seg[seg == label, :] = color
        # convert to BGR
        color_seg = color_seg[..., ::-1]

        if with_bg:
            fg_mask = seg != 0
            img[fg_mask] = img[fg_mask] * (1 - opacity) + color_seg[fg_mask] * opacity
        else:
            img = img * (1 - opacity) + color_seg * opacity
        img = img.astype(np.uint8)

        if out_file is not None:
            mmcv.imwrite(img, out_file)

        return img

    def show_result(self, img_show, img_tensor, result, out_file, vis_mode='input'):
        assert vis_mode in [
            'input', 'pred', 'input_pred', 'all_groups', 'second_group', 'first_group',
             'final_group', 'input_pred_label', 'input_pred_distinct_labels', 'final_group_pred'
        ], vis_mode
        #print("palette", self.PALETTE)
        if vis_mode == 'input':
            mmcv.imwrite(img_show, out_file)
        elif vis_mode == 'pred':
            output = Image.fromarray(result[0].astype(np.uint8)).convert('P')
            output.putpalette(np.array(self.PALETTE).astype(np.uint8))
            mmcv.mkdir_or_exist(osp.dirname(out_file))
            output.save(out_file.replace('.jpg', '.png'))
        elif vis_mode == 'input_pred':
            self.blend_result(img=img_show, result=result, out_file=out_file, opacity=0.75, with_bg=self.with_bg)
        elif vis_mode == 'input_pred_label':
            labels = np.unique(result[0])
            meta_info = list(labels) + self.CLASSES
            print("labels", meta_info)
            from pathlib import Path
            path = Path(out_file)
            parent_dir = path.parent
            print("path", path)
            print("parent_dir", parent_dir)
            #np.savetxt(str(parent_dir) +'/labels.txt', np.array(meta_info), delimiter=',', fmt='%s')
            coord_map = seg2coord(result[0])
            # reference: https://github.com/open-mmlab/mmdetection/blob/ff9bc39913cb3ff5dde79d3933add7dc2561bab7/mmdet/models/detectors/base.py#L271 # noqa
            blended_img = self.blend_result(
                img=img_show, result=result, out_file=None, opacity=0.75, with_bg=self.with_bg)
            blended_img = mmcv.bgr2rgb(blended_img)
            width, height = img_show.shape[1], img_show.shape[0]
            EPS = 1e-2
            fig = plt.figure(frameon=False)
            canvas = fig.canvas
            dpi = fig.get_dpi()
            fig.set_size_inches((width + EPS) / dpi, (height + EPS) / dpi)
            # remove white edges by set subplot margin
            plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
            ax = plt.gca()
            ax.axis('off')
            for i, label in enumerate(labels):
                if self.with_bg and label == 0:
                    continue
                center = coord_map[label].astype(np.int32)
                label_text = self.CLASSES[label]
                ax.text(
                    center[1],
                    center[0],
                    f'{label_text}',
                    bbox={
                        'facecolor': 'black',
                        'alpha': 0.5,
                        'pad': 0.7,
                        'edgecolor': 'none'
                    },
                    color='orangered',
                    fontsize=16,
                    verticalalignment='top',
                    horizontalalignment='left')
            plt.imshow(blended_img)
            stream, _ = canvas.print_to_buffer()
            buffer = np.frombuffer(stream, dtype='uint8')
            img_rgba = buffer.reshape(height, width, 4)
            rgb, alpha = np.split(img_rgba, [3], axis=2)
            img = rgb.astype('uint8')
            img = mmcv.rgb2bgr(img)
            mmcv.imwrite(img, out_file)
            plt.close()
        elif vis_mode == 'input_pred_distinct_labels':
            labels = np.unique(result[0])
            meta_info = list(labels) + self.CLASSES
            print("labels", meta_info)
            from pathlib import Path
            path = Path(out_file)
            parent_dir = path.parent
            print("path", path)
            print("parent_dir", parent_dir)
            os.makedirs(parent_dir) if not os.path.exists(parent_dir) else None
            #np.savetxt(str(parent_dir) +'/labels.txt', np.array(meta_info), delimiter=',', fmt='%s')
            coord_map = seg2coord(result[0])
            print("coord map", coord_map)
            # reference: https://github.com/open-mmlab/mmdetection/blob/ff9bc39913cb3ff5dde79d3933add7dc2561bab7/mmdet/models/detectors/base.py#L271 # noqa
            for i, label in enumerate(labels):
                blended_img = self.blend_result(img=img_show, result=result, only_label=label,
                                                 out_file=None, opacity=0.75, with_bg=self.with_bg)
                blended_img = mmcv.bgr2rgb(blended_img)
                width, height = img_show.shape[1], img_show.shape[0]
                EPS = 1e-2
                fig = plt.figure(frameon=False)
                canvas = fig.canvas
                dpi = fig.get_dpi()
                fig.set_size_inches((width + EPS) / dpi, (height + EPS) / dpi)

                # remove white edges by set subplot margin
                plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
                ax = plt.gca()
                ax.axis('off')
                if self.with_bg and label == 0:
                    continue
                center = coord_map[label].astype(np.int32)
                label_text = self.CLASSES[label]
                ax.text(
                    center[1],
                    center[0],
                    f'{label_text}',
                    bbox={
                        'facecolor': 'black',
                        'alpha': 0.5,
                        'pad': 0.7,
                        'edgecolor': 'none'
                    },
                    color='orangered',
                    fontsize=16,
                    verticalalignment='top',
                    horizontalalignment='left')
                plt.imshow(blended_img)
                stream, _ = canvas.print_to_buffer()
                buffer = np.frombuffer(stream, dtype='uint8')
                img_rgba = buffer.reshape(height, width, 4)
                rgb, alpha = np.split(img_rgba, [3], axis=2)
                img = rgb.astype('uint8')
                img = mmcv.rgb2bgr(img)
                mmcv.imwrite(img, str(parent_dir)+'/'+str(label)+'.jpg')
                plt.close()
        elif vis_mode == 'final_group_pred':
            from pathlib import Path
            path = Path(out_file)
            parent_dir = path.parent
            print("path", path)
            print("parent_dir", parent_dir)
            os.makedirs(parent_dir) if not os.path.exists(parent_dir) else None
            print("get attention map")
            attn_map_list = self.get_attn_maps(img_tensor)
            
            print("got attention map")
            text_gt_affinity = self.get_text_gt_affinity(img_tensor)
            print("text group token affinity", text_gt_affinity)
            assert len(attn_map_list) in [1, 2, 3]
            num_groups = [attn_map_list[layer_idx].shape[-1] for layer_idx in range(len(attn_map_list))]
            attn_map = attn_map_list[-1]
            attn_map = rearrange(attn_map, 'b h w g -> b g h w')
            attn_map = F.interpolate(
                    attn_map, size=img_show.shape[:2], mode='bilinear', align_corners=self.align_corners)
            print("attn map after", attn_map.shape)
            pre_attention_map = attn_map.squeeze(0)
            for i in range(pre_attention_map.shape[0]):
                group_atten_map = pre_attention_map[i, :, :]
                print("group_attenmap", group_atten_map.shape)
                labels = np.unique(group_atten_map)
                print("labels", labels)
            group_result = attn_map.argmax(dim=1).cpu().numpy()
            print("group results ahpe", group_result.shape)
            print("group results", group_result)
            patches = np.unique(group_result)
            print("patches", patches)
            print("text group token affinity", text_gt_affinity)
            print("classes", self.CLASSES)
            layer_out_file = out_file
            self.blend_result(img=img_show, result=group_result, 
                    palette=GROUP_PALETTE[sum(num_groups[:1]):sum(num_groups[:2])],
                    out_file=layer_out_file,
                    opacity=0.75)
            coord_map = seg2coord(group_result.squeeze(0))
            # reference: https://github.com/open-mmlab/mmdetection/blob/ff9bc39913cb3ff5dde79d3933add7dc2561bab7/mmdet/models/detectors/base.py#L271 # noqa
            for i, patch in enumerate(patches):
                patch_group_result = group_result.copy()
                print("current pacth", patch)
                
                print("patch_group_result before", patch_group_result)
                patch_group_result[patch_group_result != patch] = 0
                print("patch group result after", patch_group_result)
                print("patch group results", patch_group_result.shape)
                blended_img = self.blend_result(img=img_show, result=patch_group_result, 
                                                palette=GROUP_PALETTE[sum(num_groups[:1]):sum(num_groups[:2])],
                                                out_file=str(parent_dir)+'/'+str(patch)+'.jpg',
                                                opacity=0.75)
                blended_img = mmcv.bgr2rgb(blended_img)
                label_indexes = np.where(text_gt_affinity == patch)[0]
                center = coord_map[patch].astype(np.int32)
                print("type", type(label_indexes))

                if label_indexes.size != 0:
                    for label in label_indexes:
                        if self.with_bg and label == 0:
                            continue
                        width, height = img_show.shape[1], img_show.shape[0]
                        EPS = 1e-2
                        fig = plt.figure(frameon=False)
                        canvas = fig.canvas
                        dpi = fig.get_dpi()
                        fig.set_size_inches((width + EPS) / dpi, (height + EPS) / dpi)

                        # remove white edges by set subplot margin
                        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
                        ax = plt.gca()
                        ax.axis('off')
                        label_blended_img = blended_img.copy()
                        label_text = self.CLASSES[label]
                        ax.text(
                            center[1],
                            center[0],
                            f'{label_text}',
                            bbox={
                                'facecolor': 'black',
                                'alpha': 0.5,
                                'pad': 0.7,
                                'edgecolor': 'none'
                            },
                            color='orangered',
                            fontsize=16,
                            verticalalignment='top',
                            horizontalalignment='left')
                        plt.imshow(label_blended_img)
                        stream, _ = canvas.print_to_buffer()
                        buffer = np.frombuffer(stream, dtype='uint8')
                        img_rgba = buffer.reshape(height, width, 4)
                        rgb, alpha = np.split(img_rgba, [3], axis=2)
                        img = rgb.astype('uint8')
                        img = mmcv.rgb2bgr(img)
                        mmcv.imwrite(img, str(parent_dir)+'/'+str(patch)+'_'+str(label)+'.jpg')
                        plt.close()
        elif vis_mode == 'all_groups' or vis_mode == 'final_group' or vis_mode == 'first_group' or vis_mode == 'second_group':
            attn_map_list = self.get_attn_maps(img_tensor)
            assert len(attn_map_list) in [1, 2, 3]
            # only show 16 groups for the first stage
            # if len(attn_map_list) == 1:
            #     attn_map_list[0] = top_groups(attn_map_list[0], k=5)

            num_groups = [attn_map_list[layer_idx].shape[-1] for layer_idx in range(len(attn_map_list))]
            for layer_idx, attn_map in enumerate(attn_map_list):
                if vis_mode == 'first_group' and layer_idx != 0:
                    continue
                if vis_mode == 'second_group' and layer_idx != 1:
                    continue
                if vis_mode == 'final_group' and layer_idx != len(attn_map_list) - 1:
                    continue
                attn_map = rearrange(attn_map, 'b h w g -> b g h w')
                attn_map = F.interpolate(
                    attn_map, size=img_show.shape[:2], mode='bilinear', align_corners=self.align_corners)
                group_result = attn_map.argmax(dim=1).cpu().numpy()
                print("group result", group_result)
                if vis_mode == 'all_groups':
                    layer_out_file = out_file.replace(
                        osp.splitext(out_file)[-1], f'_layer{layer_idx}{osp.splitext(out_file)[-1]}')
                else:
                    layer_out_file = out_file
                self.blend_result(
                    img=img_show,
                    result=group_result,
                    palette=GROUP_PALETTE[sum(num_groups[:layer_idx]):sum(num_groups[:layer_idx + 1])],
                    out_file=layer_out_file,
                    opacity=0.75)
        else:
            raise ValueError(f'Unknown vis_type: {vis_mode}')
