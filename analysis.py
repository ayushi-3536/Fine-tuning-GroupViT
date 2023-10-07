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
from datasets import build_text_transform
from main_group_vit import validate_seg
from mmcv.image import tensor2imgs
from mmcv.parallel import MMDistributedDataParallel
from mmcv.runner import set_random_seed
from models import build_model
from omegaconf import OmegaConf, read_write
from segmentation.evaluation import build_seg_dataloader, build_seg_dataset, build_seg_inference
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


def inference(cfg):
    logger = get_logger()
    data_loader = build_seg_dataloader(build_seg_dataset(cfg.evaluate.seg))
    dataset = data_loader.dataset

    logger.info(f'Evaluating dataset: {dataset}')

    logger.info(f'Creating model:{cfg.model.type}/{cfg.model_name}')
    model = build_model(cfg.model)
    model.cuda()
    logger.info(str(model))

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

    if cfg.vis:
        vis_seg(cfg, data_loader, model, cfg.vis)


@torch.no_grad()
def vis_seg(config, data_loader, model, vis_modes):
    dist.barrier()
    model.eval()

    if hasattr(model, 'module'):
        model_without_ddp = model.module
    else:
        model_without_ddp = model

    text_transform = build_text_transform(False, config.data.text_aug, with_dc=False)
    seg_model = build_seg_inference(model_without_ddp, data_loader.dataset, text_transform, config.evaluate.seg)

    mmddp_model = MMDistributedDataParallel(
        seg_model, device_ids=[torch.cuda.current_device()], broadcast_buffers=False)
    mmddp_model.eval()
    model = mmddp_model.module
    device = next(model.parameters()).device
    dataset = data_loader.dataset
    # create a dictionary to save label and list of all the features
    label_feature_dict = {}

    #create a dictionary to save feature and corresponding label
    feature_label_dict = {}

    if dist.get_rank() == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    loader_indices = data_loader.batch_sampler
    for batch_indices, data in zip(loader_indices, data_loader):
        print("batch_indices", batch_indices)
        # if batch_indices[0] == 5:
        #     break
        assert len(batch_indices) == 1
        out_file = osp.join(config.output, 'results', f'{batch_indices[0]:04d}')
        model.set_output_dir(out_file)
        with torch.no_grad():
            result = mmddp_model(return_loss=False, **data)
        #print("result", result[0])
        labels_res = result[0]
        # print("labels_res", labels_res)
        # print("labels_res.shape", labels_res.shape)
        import numpy as np
        unique_labels = np.unique(labels_res) 
        #print("unique_labels", unique_labels)
        img_tensor = data['img'][0]
        img_meta = data['img_metas'][0].data[0]
        #print("img_meta", img_meta)
        img = tensor2imgs(img_tensor, **img_meta[0]['img_norm_cfg'])
        assert len(img) == len(img_meta)
        # print("len(img)", len(img))
        # print("len(img_meta)", len(img_meta))
        img_tensor = img_tensor[0]
        #print("img_tensor.shape", img_tensor.shape)
        img_meta = img_meta[0]
        img = img[0]
        h, w, _ = img_meta['img_shape']
        img_show = img[:h, :w, :]

        ori_h, ori_w = img_meta['ori_shape'][:-1]
        img_show = mmcv.imresize(img_show, (ori_w, ori_h))

        feat_label = model.analysis(img_tensor)
        # print("feat_label", feat_label)
        # print("feat_label.shape", feat_label.shape)
        #label has indexing from 0 to #CLASSES and 0 is background
        for label in unique_labels:
            if label == 0:
                continue
            # print("label", label)
            # print("label text", model.CLASSES[label-1]) 
            # print("feat_label[label-1]", feat_label[label-1])
            class_name = model.CLASSES[label]
            feat = tuple(feat_label[label-1].cpu().numpy())
            feature_label_dict[feat] = class_name
            if class_name not in label_feature_dict.keys():
                label_feature_dict[class_name] = [feat]
            else:
                # print("label feat dict for classname", label_feature_dict[class_name])
                # print("feat", label_feature_dict)
                label_feature_dict[class_name].append(feat)
                #print("label_feature_dict", label_feature_dict)
        #print("label_feature_dict", label_feature_dict)
    #save the label_feature_dict and feature_label_dict in a pickle file and json file
    import pickle
    with open('label_feature_dict_nenonnoisy_coco.pickle', 'wb') as handle:
        pickle.dump(label_feature_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('feature_label_dict_newnonnoisy_coco.pickle', 'wb') as handle:
        pickle.dump(feature_label_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    

          
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

    inference(cfg)
    dist.barrier()


if __name__ == '__main__':
    main()
