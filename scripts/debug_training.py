# -------------------------------------------------------------------------
# # Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License.
# To view a copy of this license, visit
# https://github.com/NVlabs/GroupViT/blob/main/LICENSE
#
# Written by Jiarui Xu
# -------------------------------------------------------------------------

import argparse
import os.path as osp
import sys
from mmcv.runner import get_dist_info, init_dist, set_random_seed
parentdir = osp.dirname(osp.dirname(__file__))
sys.path.insert(0, parentdir)
import mmcv
from mmseg.apis import multi_gpu_test
import torch
from datasets import build_text_transform
from mmcv.cnn.utils import revert_sync_batchnorm
from mmcv.image import tensor2imgs
from mmcv.parallel import collate, scatter
from models import build_model
from einops import rearrange
from mmcv.parallel import MMDistributedDataParallel
from omegaconf import read_write
from segmentation.datasets import COCOObjectDataset, PascalContextDataset, PascalVOCDataset
from segmentation.evaluation import build_seg_demo_pipeline, build_seg_inference, build_seg_dataloader, build_seg_dataset
from utils import get_config, load_checkpoint
#from main_group_vit import validate_seg

from utils import (auto_resume_helper, build_dataset_class_tokens, build_optimizer, build_scheduler, data2cuda,
                   get_config, get_grad_norm, get_logger, load_checkpoint, parse_losses, reduce_tensor, save_checkpoint)
from datasets import build_loader, build_loader_sync, build_text_transform, imagenet_classes
# 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
# debugpy.listen(2122)
# print("Waiting for debugger attach")
# debugpy.wait_for_client()
# debugpy.breakpoint()
# print('break on this line')
def parse_args():
    parser = argparse.ArgumentParser('GroupViT demo')
    parser.add_argument(
        '--cfg',
        type=str,
        required=True,
        help='path to config file',
    )
    parser.add_argument(
        '--opts',
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument(
        '--vis',
        help='Specify the visualization mode, '
        'could be a list, support "input", "pred", "input_pred", "all_groups", "first_group", "final_group", "input_pred_label"',
        default=None,
        nargs='+')

    parser.add_argument('--device', default='cpu', help='Device used for inference')
    parser.add_argument(
        '--dataset', default='voc', choices=['voc', 'coco', 'context'], help='dataset classes for visualization')

    parser.add_argument('--input', type=str, help='input image path')
    parser.add_argument('--output_dir', type=str, help='output dir')
    parser.add_argument('--allow_shape_change', default=True, type=bool,  help='path to config file')
    
    args = parser.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.local_rank = 0  # compatible with config

    return args


def inference(args, cfg):
    model = build_model(cfg.model)
    model = revert_sync_batchnorm(model)
    model.to(args.device)
    model.eval()

    load_checkpoint(cfg, model, None, None, args.allow_shape_change)

    text_transform = build_text_transform(False, cfg.data.text_aug, with_dc=False)
    if args.dataset == 'voc':
        dataset_class = PascalVOCDataset
        seg_cfg = 'segmentation/configs/_base_/datasets/pascal_voc12.py'
    elif args.dataset == 'coco':
        dataset_class = COCOObjectDataset
        seg_cfg = 'segmentation/configs/_base_/datasets/coco.py'
    elif args.dataset == 'context':
        dataset_class = PascalContextDataset
        seg_cfg = 'segmentation/configs/_base_/datasets/pascal_context.py'
    else:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))

    with read_write(cfg):
        cfg.evaluate.seg.cfg = seg_cfg
        cfg.evaluate.seg.opts = ['test_cfg.mode=whole']

    seg_model = build_seg_inference(model, dataset_class, text_transform, cfg.evaluate.seg)

    vis_seg(seg_model, args.input, args.output_dir, args.vis)


def vis_seg(seg_model, input_img, output_dir, vis_modes):
    device = next(seg_model.parameters()).device
    test_pipeline = build_seg_demo_pipeline()
    # prepare data
    data = dict(img=input_img)
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    if next(seg_model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        data['img_metas'] = [i.data[0] for i in data['img_metas']]
    with torch.no_grad():
        result = seg_model(return_loss=False, rescale=True, **data)

    img_tensor = data['img'][0]
    img_metas = data['img_metas'][0]
    imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
    assert len(imgs) == len(img_metas)

    for img, img_meta in zip(imgs, img_metas):
        h, w, _ = img_meta['img_shape']
        img_show = img[:h, :w, :]

        ori_h, ori_w = img_meta['ori_shape'][:-1]
        img_show = mmcv.imresize(img_show, (ori_w, ori_h))

        for vis_mode in vis_modes:
            out_file = osp.join(output_dir, 'vis_imgs', vis_mode, f'{vis_mode}.jpg')
            seg_model.show_result(img_show, img_tensor.to(device), result, out_file, vis_mode)
@torch.no_grad()
def validate_seg(config, data_loader, model):
    model.eval()

    if hasattr(model, 'module'):
        model_without_ddp = model.module
    else:
        model_without_ddp = model

    text_transform = build_text_transform(False, config.data.text_aug, with_dc=False)
    seg_model = build_seg_inference(model_without_ddp, data_loader.dataset, text_transform, config.evaluate.seg)
    dataset = data_loader.dataset
    loader_indices = data_loader.batch_sampler
    results=[]
    for batch_indices, data in zip(loader_indices, data_loader):
        with torch.no_grad():
            result = model(**data)
            results.extend(result)

    metric = [data_loader.dataset.evaluate(results, metric='mIoU')]
    
    miou_result = metric[0]['mIoU'] * 100

    torch.cuda.empty_cache()
    print(f'Eval Seg mIoU {miou_result:.2f}')
    return miou_result


def train_one_epoch(config, model, data_loader, optimizer, epoch, lr_scheduler):
    #logger = get_logger()
    model.train()
    optimizer.zero_grad()
    num_steps = len(data_loader)
    for idx, samples in enumerate(data_loader):
        if idx == 1:
            break
        

        #logger.info(f'idx:{idx}')
        #print(data_loader.index_sampler.batch_indices)
        samples['image'] = samples['image'].data[0]
        
        samples['text'] = samples['text'].data[0]
        text = samples['text']
        print("text: ", text)

        #Below is not working as matching on context l
        #rearrange text from [B, N, len] to [(B*N), len]
        # sample_dict={}
        # text = rearrange(text, 'b n l -> (b n) l')
        # unique_samples, indices = torch.unique(text, dim=0, return_inverse=True)

        # # Get the indices where the samples are the same
        # same_samples = [torch.where(indices == i)[0] for i in range(len(unique_samples)) if torch.count_nonzero(indices == i) > 1]
        # print("same_samples: ", same_samples)

        batch_size = config.data.batch_size
        losses = model(**samples)

        loss, log_vars = parse_losses(losses)
        print("loss: ", loss)
        loss.backward()
        grad_norm = get_grad_norm(model.parameters())
        optimizer.step()
        lr_scheduler.step_update(epoch * num_steps + idx)


def train(cfg):
    import time
    #logger = get_logger()
    dataset_train, dataset_val, \
        data_loader_train, data_loader_val = build_loader_sync(cfg.data)
    #loader_indices = data_loader_train.batch_sampler
    #data_loader_seg = build_seg_dataloader(build_seg_dataset(cfg.evaluate.seg))
    model = build_model(cfg.model)
    model = model.cuda()
    text_transform = build_text_transform(False, cfg.data.text_aug, with_dc=False)
    # if cfg.data.precompute_pad_mask:
    #     padword_tokenized = text_transform(cfg.data.pad_word)
    #     model.build_padtoken_embedding(padword_tokenized)
    optimizer = build_optimizer(cfg.train, model)
    lr_scheduler = build_scheduler(cfg.train, optimizer, len(data_loader_train))
    # dict_token_to_text = text_transform_train.token_to_text
    # text_to_subtext = text_transform_train.text_to_subtext
    for epoch in range(cfg.train.start_epoch, cfg.train.epochs):
        train_one_epoch(cfg, model, data_loader_train, optimizer, epoch, lr_scheduler)
        break # train for just one epoch to get all text embeddings
    
    
    #miou = validate_seg(cfg, data_loader_seg, model)
    #logger.info(f'mIoU of the network on the {len(data_loader_seg.dataset)} test images: {miou:.2f}%')
        


def main():
    args = parse_args()
    cfg = get_config(args)

    train(cfg)
    
    
    #dataset_train, dataset_val,  data_loader_train, data_loader_val = build_loader_sync(cfg.data)
    # with read_write(cfg):
    #     cfg.evaluate.eval_only = True

    # inference(args, cfg)


if __name__ == '__main__':
    main()
