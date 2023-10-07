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
from PIL import Image
from mmseg.datasets import build_dataset
import numpy as np
try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None
import wandb

name='textpascalvalwithvar'
wandb.init(
        project='group_vit',
        name=osp.join('analysis',name),
        dir=f'/misc/lmbraid21/sharmaa/resultsanalysis/{name}/wandb'
)
#Load the COCO dataset annotations file
def get_voc_colormap():
    voc_colormap = {
        0: (0, 0, 0),      # background: black
        1: (128, 0, 0),    # aeroplane: maroon
        2: (0, 128, 0),    # bicycle: green
        3: (128, 128, 0),  # bird: olive
        4: (0, 0, 128),    # boat: navy
        5: (128, 0, 128),  # bottle: purple
        6: (0, 128, 128),  # bus: teal
        7: (128, 128, 128),# car: gray
        8: (64, 0, 0),     # cat: dark red
        9: (192, 0, 0),    # chair: red
        10: (64, 128, 0),  # cow: dark green
        11: (192, 128, 0), # diningtable: orange
        12: (64, 0, 128),  # dog: dark purple
        13: (192, 0, 128), # horse: pink
        14: (64, 128, 128),# motorbike: dark teal
        15: (192, 128, 128), # person: light pink
        16: (0, 64, 0),    # pottedplant: dark green
        17: (128, 64, 0),  # sheep: brown
        18: (0, 192, 0),   # sofa: bright green
        19: (128, 192, 0), # train: yellow
        20: (0, 64, 128),  # tvmonitor: dark blue
    }
    return voc_colormap
def colorize_mask(mask_array):
    # Define a colormap to assign colors to different classes
    colormap = get_voc_colormap()
    height, width = mask_array.shape
    colored_mask = np.zeros((height, width, 3), dtype=np.uint8)

    for class_id, color in colormap.items():
        colored_mask[mask_array == class_id] = color

    return colored_mask
class_name_to_id = {
        'background': 0,
        'aeroplane': 1,
        'bicycle': 2,
        'bird': 3,
        'boat': 4,
        'bottle': 5,
        'bus': 6,
        'car': 7,
        'cat': 8,
        'chair': 9,
        'cow': 10,
        'table': 11,
        'dog': 12,
        'horse': 13,
        'motorbike': 14,
        'person': 15,
        'plant': 16,
        'sheep': 17,
        'sofa': 18,
        'train': 19,
        'monitor': 20,
}
annotation_path = '/misc/lmbraid21/sharmaa/VOCdevkit/VOC2012/SegmentationClass'
    
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
    
    vis_seg(cfg, data_loader, model)

def find_channels(anno, arr1 ):
    # Step 1: Find all unique values from anno
    arr1 = arr1.cpu().numpy()
    unique_classes = np.unique(anno)
    #print("unique_values_arr2", unique_classes)

    # Step 2: Find positions of unique values in arr1
    result_array = np.zeros(anno.shape, dtype=np.uint8)
    var = {}
    
    for value in unique_classes:
        if value == 0 or value == 255:
            continue
        #print("value", value)
        position = np.where(anno == value)
        # print("positions", np.array(positions[0]).shape)
        #print("len positions", len(position[0]))
        # #define an array to store the result
        # # Step 3: Find channels at corresponding positions in arr1  
        channels = [arr1[:, row, col] for row,col in zip(position[0], position[1])] 
        #print("len channels", len(channels))
        channels = np.array(channels)
        #print("channels", channels.shape)
        mean_channels = np.mean(channels, axis=0)
        #find argmax index
        argmax_index = np.argmax(mean_channels)
        #print("argmax_index", argmax_index)
        variance = np.var(channels, axis=0)
        var[value] = variance[value]
        #print("mean_channels", mean_channels.shape)
        #print("var shape",variance.shape)
        #print("var", variance)
        #fill the positions in the result_array with the value from argmax_index
        for row,col in zip(position[0], position[1]):
            result_array[row, col] = np.uint8(argmax_index)
    return result_array, var
def get_metric(val_seg_map):
    cfg = mmcv.Config.fromfile(filename='/misc/student/sharmaa/groupvit/GroupViT/segmentation/configs/_base_/datasets/pascal_voc12.py')
    dataset = build_dataset(cfg.data.test)
    metrics = [dataset.evaluate(val_seg_map, metric='mIoU')]
    print("metrics", metrics)
    return metrics
@torch.no_grad()
def vis_seg(config, data_loader, model):
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

    if dist.get_rank() == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    loader_indices = data_loader.batch_sampler
    final_seg = []
    var_all = {}
    for batch_indices, data in zip(loader_indices, data_loader):
        assert len(batch_indices)==1
        # if batch_indices[0] >5:
        #      break
        img_metas = data['img_metas'][0].data[0]
        #print("img_metas", img_metas)
        filename = img_metas[0]['ori_filename']
        #print("filename", filename)
        out_file = osp.join(config.output, 'results', f'{batch_indices[0]:04d}')
        model.set_output_dir(out_file)
        img_tensor = data['img'][0]
        img_metas = data['img_metas'][0].data[0]
        #print("img_metas", img_metas)
        imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
        h, w, _ = img_metas[0]['img_shape']
        img_show = imgs[0][:h, :w, :]
        ori_h, ori_w = img_metas[0]['ori_shape'][:-1]
        img_show = mmcv.imresize(img_show, (ori_w, ori_h))

        with torch.no_grad():
            result = model.get_class_distribution(img_show, img_tensor.half().to(device))
        #print("result after pass", result[0].shape)
        #load annotation
        ann_file = osp.join(annotation_path, filename.replace('jpg', 'png'))
        #print("ann_file", ann_file)
        ann =  np.array(Image.open(ann_file))
        #print("ann", ann.shape)
        result, var = find_channels(ann, result[0])
        result = torch.from_numpy(result)
        result = np.array(result, dtype=np.uint8)
        final_seg.append(result)


        colored_mask = colorize_mask(result)

        # Save the colored mask as an image
        colored_mask_image = Image.fromarray(colored_mask)
        colored_mask_image.save(config.output+f'/{batch_indices[0]}.png')

        for key, value in var.items():
            if key not in var_all:
                var_all[key] = []
            var_all[key].append(value)

        #save result in the output dir with batchidx as suffix:
    print("final_seg", len(final_seg))
    mIoU = get_metric(final_seg)
    print("mIoU", mIoU)
    #print("var_all", var_all)
    for key, value in var_all.items():
        mean_var = np.mean(value, axis=0)
        # print("mean_var", mean_var.shape)
        # print("mean_var", mean_var)
        # print("key", key)
        var_all[key] = mean_var
    print("var_all", var_all)



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
