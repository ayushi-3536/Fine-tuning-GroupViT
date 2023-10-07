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

import argparse
import datetime
import os
import os.path as osp
import time
from collections import defaultdict, OrderedDict
import torch

from sklearn.cluster import KMeans
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
from datasets import build_loader, build_text_transform, imagenet_classes
from mmcv.parallel import MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, set_random_seed
from mmcv.utils import collect_env, get_git_hash
from mmseg.apis import multi_gpu_test
from models import build_model
import copy
from omegaconf import OmegaConf, read_write
from segmentation.evaluation import build_seg_dataloader, build_seg_dataset, build_seg_inference
from timm.utils import AverageMeter, accuracy
from utils import (auto_resume_helper, build_dataset_class_tokens, build_optimizer, build_scheduler, data2cuda,
                   get_config, get_grad_norm, get_logger, load_checkpoint, parse_losses, reduce_tensor, save_checkpoint)
try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None




def parse_args():
    parser = argparse.ArgumentParser('GroupViT training and evaluation script')
    parser.add_argument('--cfg', type=str, required=True, help='path to config file')
    parser.add_argument('--opts', help="Modify config options by adding 'KEY=VALUE' list. ", default=None, nargs='+')
    # easy config modification
    parser.add_argument('--batch-size', type=int, help='batch size for single GPU')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument(
        '--amp-opt-level',
        type=str,
        default='O1',
        choices=['O0', 'O1', 'O2'],
        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument(
        '--output', type=str,default='/misc/student/sharmaa/groupvit/GroupViT/outputs_dino', help='root of output folder, '
        'the full path is <output>/<model_name>/<tag>')
    parser.add_argument('--tag', type=str, help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--wandb', action='store_true', help='Use W&B to log experiments')
    parser.add_argument('--keep', type=int, help='Maximum checkpoint to keep')

    # distributed training
    parser.add_argument('--local_rank', type=int, required=True, help='local rank for DistributedDataParallel')

    args = parser.parse_args()

    return args

def plot_heatmap(matrix):
    import matplotlib.pyplot as plt
    #remove any previous plots
    plt.clf()
    plt.imshow(matrix, cmap='jet', interpolation='nearest')
    plt.colorbar()
    plt.title("Cosine Similarity Matrix")
    return plt


def perform_clustering(features, n_clusters=64):
    # Normalize features
    features = F.normalize(features, p=2, dim=1)
    # Convert the features to float32
    features = features.cpu().detach().numpy().astype('float32')
    # Initialize a k-means clustering index with the desired number of clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    # Train the k-means index with the features
    kmeans.fit(features)
    # Assign the features to their nearest cluster
    labels = kmeans.predict(features)
    return labels


def chunk_cosine_sim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """ Computes cosine similarity between all possible pairs in two sets of vectors.
    Operates on chunks so no large amount of GPU RAM is required.
    :param x: an tensor of descriptors of shape Bx1x(t_x)xd' where d' is the dimensionality of the descriptors and t_x
    is the number of tokens in x.
    :param y: a tensor of descriptors of shape Bx1x(t_y)xd' where d' is the dimensionality of the descriptors and t_y
    is the number of tokens in y.
    :return: cosine similarity between all descriptors in x and all descriptors in y. Has shape of Bx1x(t_x)x(t_y) """
    result_list = []
    num_token_x = x.shape[1]
    for token_idx in range(num_token_x):
        token = x[:,token_idx, :].unsqueeze(dim=1)  # Bx1x1xd'
        result_list.append(torch.nn.CosineSimilarity(dim=2)(token, y))  # Bx1xt
    return torch.stack(result_list, dim=2)  # Bx1x(t_x)x(t_y)

def train(cfg):
    logger = get_logger()
    dist.barrier()
    dataset_train, dataset_val, \
        data_loader_train, data_loader_val = build_loader(cfg.data)
    data_loader_seg = build_seg_dataloader(build_seg_dataset(cfg.evaluate.seg))

    logger.info(f'Creating model:{cfg.model.type}/{cfg.model_name}')
    model = build_model(cfg.model)
    model.cuda()
        
    logger.info(str(model))

    optimizer = build_optimizer(cfg.train, model)
    if cfg.train.amp_opt_level != 'O0':
        model, optimizer = amp.initialize(model, optimizer, opt_level=cfg.train.amp_opt_level)
    # model = MMDistributedDataParallel(model, device_ids=[torch.cuda.current_device()], broadcast_buffers=False)
    # model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'number of params: {n_parameters}')
    lr_scheduler = build_scheduler(cfg.train, optimizer, len(data_loader_train))

    if cfg.checkpoint.auto_resume:
        resume_file = auto_resume_helper(cfg.output)
        if resume_file:
            if cfg.checkpoint.resume:
                logger.warning(f'auto-resume changing resume file from {cfg.checkpoint.resume} to {resume_file}')
            with read_write(cfg):
                cfg.checkpoint.resume = resume_file
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {cfg.output}, ignoring auto resume')

    max_accuracy = max_miou = 0.0
    max_metrics = {'max_accuracy': max_accuracy, 'max_miou': max_miou}
   
    if cfg.checkpoint.resume:
        max_metrics = load_checkpoint(cfg, model, optimizer, lr_scheduler)
        max_accuracy, max_miou = max_metrics['max_accuracy'], max_metrics['max_miou']
        if cfg.evaluate.eval_only:
            return


    #write_model_iteration(cfg, data_loader_train, model)

    
    logger.info('Start training')
    start_time = time.time()

    #default `log_dir` is "runs" - we'll be more specific here
    
    data_loader = copy.deepcopy(data_loader_train)
    dataiter = iter(data_loader)
    samples = next(dataiter)
    torch.cuda.empty_cache()
    print("samples images", samples['image'])
    print("samples image 0", samples['image'].data[0])
    image_outs = model.img_encoder.get_features(samples['image'].data[0].cuda().half(),  return_feat=True, return_attn=True, as_dict=True)
    gvit_feat = image_outs['gvit_feat']

    dinos_feat = image_outs['dino_feat']
    soft_feat = image_outs['attn_dicts'][0]['soft']
    print("soft_feat", soft_feat)
    print("soft_feat shape", soft_feat.shape)
    soft_feat = soft_feat.squeeze(1)


    for i in range(gvit_feat.shape[0]):
        if i> 1:
            break
        gvit_feat_0 = gvit_feat[i]
        dinos_feat_0 = dinos_feat[i]
        soft_feat_0 = soft_feat[i]
        print("gvit_feat_0", gvit_feat_0.unsqueeze(0).shape)
        print("dinos_feat_0", dinos_feat_0.unsqueeze(0).shape)
        print("soft_feat_0", soft_feat_0.unsqueeze(0).shape)
        break
        # gvit_dino_cosine_mat = chunk_cosine_sim(gvit_feat_0.unsqueeze(0), dinos_feat_0.unsqueeze(0)).squeeze(0)
        # gvit_dino_cosine_mat = gvit_dino_cosine_mat.cpu().numpy()
        # plt = plot_heatmap(gvit_dino_cosine_mat)
        # plt.savefig("cos_simmat_dino_gvit.png")

        gvit_cosine_mat = chunk_cosine_sim(gvit_feat_0.unsqueeze(0), gvit_feat_0.unsqueeze(0)).squeeze(0)
        gvit_cosine_mat = gvit_cosine_mat.cpu().numpy()

        
        torch.save(gvit_cosine_mat, 'saved_gvit_cosine.pt')
        plt = plot_heatmap(gvit_cosine_mat)
        plt.savefig(f"cos_simmat_gvit_{i}.png")

        gvit_cosine_mat = gvit_cosine_mat[0,:]
        gvit_cosine_mat = gvit_cosine_mat.reshape((14,14))
        #print("after reshaping", curr_similarities.shape)
        plt.imshow(gvit_cosine_mat, cmap='jet')
        plt.savefig(f'gvit_sim_{i}.png')

        dino_cosine_mat = chunk_cosine_sim(dinos_feat_0.unsqueeze(0), dinos_feat_0.unsqueeze(0)).squeeze(0)
        dino_cosine_mat = dino_cosine_mat.cpu().numpy()
        
        torch.save(dino_cosine_mat, 'saved_dino_cosine.pt')
        plt = plot_heatmap(dino_cosine_mat)
        plt.savefig(f"cos_simmat_dino_{i}.png")

        dino_cosine_mat = dino_cosine_mat[0,:]
        dino_cosine_mat = dino_cosine_mat.reshape((14,14))
        #print("after reshaping", curr_similarities.shape)
        plt.imshow(dino_cosine_mat, cmap='jet')
        plt.savefig(f'dino_sim_{i}.png')

        soft_feat_0 = soft_feat[i].detach()

        torch.save(soft_feat_0, 'saved_soft_attn.pt')
        #


        
        soft_feat = soft_feat_0.cpu()
        soft_feat = soft_feat.numpy()
        plt = plot_heatmap(soft_feat)
        plt.savefig(f"grouping_soft_{i}.png")

        # soft_feat_mat = soft_feat[0, :]
        # soft_feat_mat = soft_feat_mat.reshape((14,14))
        # plt.imshow(soft_feat_mat, cmap='jet')
        # plt.savefig(f'grouping_soft_{i}.png')






def main():
    args = parse_args()
    cfg = get_config(args)
    print("cfg", cfg)

    if cfg.train.amp_opt_level != 'O0':
        assert amp is not None, 'amp not installed!'

    # start faster ref: https://github.com/open-mmlab/mmdetection/pull/7036
    mp.set_start_method('fork', force=True)
    init_dist('pytorch')
    rank, world_size = get_dist_info()
    print(f'RANK and WORLD_SIZE in environ: {rank}/{world_size}')

    dist.barrier()

    set_random_seed(cfg.seed, use_rank_shift=True)
    cudnn.benchmark = True

    os.makedirs(cfg.output, exist_ok=True)
    logger = get_logger(cfg)

    # linear scale the learning rate according to total batch size, may not be optimal
    if cfg.train.lr_scaling > 0:
        linear_scaled_lr = cfg.train.base_lr * cfg.train.lr_scaling  #0.25 #cfg.data.batch_size * world_size / 4096.0
        linear_scaled_warmup_lr = cfg.train.warmup_lr * cfg.train.lr_scaling #0.01 #0.25 # cfg.data.batch_size * world_size / 4096.0
        linear_scaled_min_lr = cfg.train.min_lr * cfg.train.lr_scaling #0.25 # cfg.data.batch_size * world_size / 4096.0
    else:
        linear_scaled_lr = cfg.train.base_lr * cfg.data.batch_size * world_size / 4096.0
        linear_scaled_warmup_lr = cfg.train.warmup_lr * cfg.data.batch_size * world_size / 4096.0
        linear_scaled_min_lr = cfg.train.min_lr * cfg.data.batch_size * world_size / 4096.0


    # gradient accumulation also need to scale the learning rate
    if cfg.train.accumulation_steps > 1:
        linear_scaled_lr = linear_scaled_lr * cfg.train.accumulation_steps
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * cfg.train.accumulation_steps
        linear_scaled_min_lr = linear_scaled_min_lr * cfg.train.accumulation_steps

    with read_write(cfg):
        logger.info(f'Scale base_lr from {cfg.train.base_lr} to {linear_scaled_lr}')
        logger.info(f'Scale warmup_lr from {cfg.train.warmup_lr} to {linear_scaled_warmup_lr}')
        logger.info(f'Scale min_lr from {cfg.train.min_lr} to {linear_scaled_min_lr}')
        cfg.train.base_lr = linear_scaled_lr
        cfg.train.warmup_lr = linear_scaled_warmup_lr
        cfg.train.min_lr = linear_scaled_min_lr

    if dist.get_rank() == 0:
        path = os.path.join(cfg.output, 'config.json')
        OmegaConf.save(cfg, path)
        logger.info(f'Full config saved to {path}')

    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' + dash_line)

    logger.info(f'Git hash: {get_git_hash(digits=7)}')

    # print config
    logger.debug(OmegaConf.to_yaml(cfg))

    train(cfg)
    dist.barrier()


if __name__ == '__main__':
    import random
    import numpy as np

    seed = 123

    # Set the random seed for Python's random module
    random.seed(seed)

    # Set the random seed for numpy
    np.random.seed(seed)

    # Set the random seed for PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    main()