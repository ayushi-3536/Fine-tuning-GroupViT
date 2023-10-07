# -------------------------------------------------------------------------
# # Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License.
# To view a copy of this license, visit
# https://github.com/NVlabs/GroupViT/blob/main/LICENSE
#
# Written by Jiarui Xu
# -------------------------------------------------------------------------

import torch
import argparse
import os.path as osp
import sys
from omegaconf import OmegaConf
parentdir = osp.dirname(osp.dirname(__file__))
sys.path.insert(0, parentdir)
import mmcv
from datasets import build_text_transform
from mmcv.cnn.utils import revert_sync_batchnorm
from mmcv.image import tensor2imgs
from mmcv.parallel import collate, scatter
from models import build_model
from omegaconf import read_write
from segmentation.datasets import COCOObjectDataset, PascalContextDataset, PascalVOCDataset
from segmentation.evaluation import build_seg_demo_pipeline, build_train_assessment_pipeline
from utils import get_config, load_checkpoint
import numpy as np
from segmentation.evaluation import GroupViTTrainAssessment
from datasets import build_loader_sync, build_train_assessment_dataset
# import debugpy

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
    # parser.add_argument(
    #     '--dataset', default='voc', choices=['voc', 'coco', 'context'], help='dataset classes for visualization')

    parser.add_argument('--output_dir', type=str, help='output dir')
    parser.add_argument('--allow_shape_change', default=True, type=bool,  help='path to config file')
    
    args = parser.parse_args()
    args.local_rank = 0  # compatible with config

    return args

def train_assessment(args, config):
    model = build_model(config.model)
    model = revert_sync_batchnorm(model)
    model.to(args.device)
    model.eval()

    load_checkpoint(config, model, None, None, args.allow_shape_change)

    text_transform = build_text_transform(True, config.data.text_aug, with_dc=False,  train_assessment=True)
    
    #Choosing one of the classes to get the palette
    dataset = COCOObjectDataset
    data = build_train_assessment_dataset(is_train=True, config=config.data)
    model = GroupViTTrainAssessment(model)
    model.PALETTE = dataset.PALETTE
    
    test_pipeline = build_seg_demo_pipeline()
    from itertools import islice   
    for i, sample in enumerate(islice(data, 0, 200)):
        for key, value in sample.items():
            print(key, repr(value))
            # # # prepare data
            if key is 'image':
                img = np.array(value)
                print("image shape", img.shape)
                print("type image", type(img))
                data = dict(img=img)
                data = test_pipeline(data)
                data = collate([data], samples_per_gpu=1)
                #value.save(f'img_{i}.jpg')
            elif key is 'text':
                labels, texts = text_transform(value)
                print('label:', labels)
                #print('text', texts)
        model.CLASSES =  labels 
        #model.PALETTE = dataset.PALETTE
        #print("model palette", model.PALETTE)
        vis_seg(model, data, texts, args.output_dir + '/' + str(i), args.vis)
        #break
    # cfg = mmcv.Config.fromfile(config.cfg)
    # if len(config.opts):
    #     cfg.merge_from_dict(OmegaConf.to_container(OmegaConf.from_dotlist(OmegaConf.to_container(config.opts))))

    # for idx, samples in enumerate(data_loader_train):
    #     #assess only for single batch
    #     if idx > 0:
    #         break;
    #     images = samples['image'].data
    #     texts = samples['text'].data
        
    #     for image, text in zip(images,texts):
    #         vis_seg(model, image, text, args.output_dir, args.vis)


def vis_seg(seg_model, data, text, output_dir, vis_modes):
    device = next(seg_model.parameters()).device
    # test_pipeline = build_train_assessment_pipeline()
    # # # # prepare data
    # data = dict(img=[input_img])
    
    # kwargs = dict(text=[text])
    # data = test_pipeline(data)
    # data = collate([data], samples_per_gpu=1)
    if next(seg_model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        data['img_metas'] = [i.data[0] for i in data['img_metas']]
    print("data", data)
    with torch.no_grad():
        #text_transform = build_text_transform(is_train=True, config=config.text_aug, with_dc=False)
        seg_model.set_text_tokens(text)
        result = seg_model(return_loss=False, rescale=True, **data)
    print("results", result[0].shape)
    print("results", type(result))
    img_tensor = data['img'][0]
    img_metas = data['img_metas'][0]
    imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
    assert len(imgs) == len(img_metas)

    #np.savetxt(output_dir + '/output.txt', np.array(result[0]), delimiter=',', fmt='%d')
    for img, img_meta in zip(imgs, img_metas):
        h, w, _ = img_meta['img_shape']
        img_show = img[:h, :w, :]

        ori_h, ori_w = img_meta['ori_shape'][:-1]
        img_show = mmcv.imresize(img_show, (ori_w, ori_h))

        for vis_mode in vis_modes:
            out_file = osp.join(output_dir, 'vis_imgs', vis_mode, f'{vis_mode}.jpg')
            seg_model.show_result(img_show, img_tensor.to(device), result, out_file, vis_mode)


def main():
    args = parse_args()
    cfg = get_config(args)

    with read_write(cfg):
        cfg.evaluate.eval_only = True

    train_assessment(args, cfg)


if __name__ == '__main__':
    main()
