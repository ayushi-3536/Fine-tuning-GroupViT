'''
This is temp file that just loads the model to debug checkpoint loading
'''
import argparse
from collections import defaultdict
import torch
from models import build_model
from utils import get_config, load_checkpoint
from utils import (get_config, get_logger, load_checkpoint)

def parse_args():
    parser = argparse.ArgumentParser('GroupViT training and evaluation script')
    parser.add_argument('--cfg', default='configs/gs3_group_vit_gcc_yfcc_30e.yml', type=str, required=True, help='path to config file')
    parser.add_argument('--opts', help="Modify config options by adding 'KEY=VALUE' list. ", default=None, nargs='+')
    parser.add_argument('--device', default='cpu', help='Device used for inference')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--allow_shape_change', default=True, type=bool,  help='path to config file')
    args = parser.parse_args()
    args.local_rank = 0 #To make it compatibile with config file
    return args

def main():
    args = parse_args()
    cfg = get_config(args) 
    print("cfg", cfg)
    logger = get_logger()
    logger.info(f'Creating model:{cfg.model.type}/{cfg.model_name}')
    model = build_model(cfg.model)
    model.to(args.device)
    model.eval()
    load_checkpoint(cfg, model, None, None, args.allow_shape_change)

if __name__ == '__main__':
    main()