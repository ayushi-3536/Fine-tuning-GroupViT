# ------------------------------------------------------------------------------
# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License.
# To view a copy of this license, visit
# https://github.com/NVlabs/GroupViT/blob/main/LICENSE
#
# Written by Jiarui Xu
# ------------------------------------------------------------------------------

import argparse
import os
import os.path as osp
import mmcv
import torch
from mmcv.runner import get_dist_info, init_dist, set_random_seed
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from datasets import build_text_transform, build_dataloader
from main_group_vit import validate_seg
from mmcv.image import tensor2imgs
from mmcv.parallel import MMDistributedDataParallel
from mmcv.runner import set_random_seed
from models import build_model
from omegaconf import OmegaConf, read_write
from segmentation.evaluation import build_seg_dataset, build_train_dataloader_with_annotations, build_seg_inference
from utils import get_config, get_logger, load_checkpoint

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None


def parse_args():
    parser = argparse.ArgumentParser('GroupViT segmentation evaluation and visualization')
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
        '--output', type=str, help='root of output folder, '
        'the full path is <output>/<model_name>/<tag>')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument(
        '--vis',
        help='Specify the visualization mode, '
        'could be a list, support input, pred, input_seg, input_pred_seg_label, all_groups, first_group, last_group',
        default=['input', 'pred', 'input_pred', 'all_groups', 'second_group', 'first_group',
             'final_group', 'input_pred_label', 'input_pred_distinct_labels'],
        nargs='+')

    # distributed training
    parser.add_argument('--local_rank', type=int, required=True, help='local rank for DistributedDataParallel')

    args = parser.parse_args()

    return args


def analysis(cfg):
    logger = get_logger()
    dataset = build_seg_dataset(cfg.data.analysis)
    #iterate on dataset
    for i in range(len(dataset)):
        print("dataset[i]", dataset[i])
        break
    print("dataset", dataset)


    data_loader_train = build_train_dataloader_with_annotations(dataset)
    logger.info(f'Creating model:{cfg.model.type}/{cfg.model_name}')
    model = build_model(cfg.model)
    model.cuda()
    #if cfg.train.amp_opt_level != 'O0':
    model = amp.initialize(model, None, opt_level=cfg.train.amp_opt_level)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'number of params: {n_parameters}')

    load_checkpoint(cfg, model, None, None)

    #if 'seg' in cfg.evaluate.task:
    # miou = validate_seg(cfg, data_loader, model)
    # logger.info(f'mIoU of the network on the {len(data_loader.dataset)} test images: {miou:.2f}%')
    # # else:
    #     logger.info('No segmentation evaluation specified')
    for batch in data_loader_train:

        # logger.info(f'idx:{idx}')
        # logger.info(f'samples:{samples}')
        print("images", batch['image'].shape)
        
        print("lABELS", batch['mask'].shape)
        break

def main():
    args = parse_args()
    cfg = get_config(args)

    # if cfg.train.amp_opt_level != 'O0':
    #     assert amp is not None, 'amp not installed!'

    with read_write(cfg):
        cfg.evaluate.eval_only = True
    print("check cuda", torch.cuda.is_available())

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f'RANK and WORLD_SIZE in environ: {rank}/{world_size}')
    else:
        rank = -1
        world_size = -1
    print("local rank",cfg.local_rank)
    torch.cuda.set_device(cfg.local_rank)
    init_dist('pytorch')
    #dist.init_process_group(backend='nlcc', init_method='env://', world_size=world_size, rank=rank)

    dist.barrier()

    set_random_seed(cfg.seed, use_rank_shift=True)
    cudnn.benchmark = True

    os.makedirs(cfg.output, exist_ok=True)
    logger = get_logger(cfg)

    if dist.get_rank() == 0:
        path = os.path.join(cfg.output, 'config.json')
        OmegaConf.save(cfg, path)
        logger.info(f'Full config saved to {path}')

    # print config
    logger.info(OmegaConf.to_yaml(cfg))

    analysis(cfg)
    dist.barrier()


if __name__ == '__main__':
    main()
