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

import os
from collections import defaultdict

import torch
import torch.distributed as dist
from mmcv.runner import CheckpointLoader
from omegaconf import read_write

from .logger import get_logger
from .config import load_config

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None
from copy import deepcopy

def find_mismatch_index(tensor1, tensor2):
    '''
    This function finds the first dimension of the mismatched dimensions between two tensor
    For our limited usecase, there is always only one mismatched dimension
    '''
    if tensor1.shape != tensor2.shape:
        mismatch_index = []
        for i, (dim1, dim2) in enumerate(zip(tensor1.shape, tensor2.shape)):
            if dim1 != dim2:
                mismatch_index.append(i)
        return mismatch_index
    return []

def shift_shape(param, weight):
    # If the shapes do not match, try to crop or pad the tensor
    mismatched_dimension = find_mismatch_index(param, weight)
    # find the index of the first non-zero value in shape_diff
    first_non_zero_idx =  mismatched_dimension[0]
    # extract the matched portion of the bigger tensor, this method is used in the 
    # weight transfer for 2 grouping layer to 3 grouping layer architecture with 
    # reduced number of token in the final layer. Therefore, the implementation is 
    # the limited case of param being the bigger tensor
    indices = torch.arange(param.shape[first_non_zero_idx])
    
    matched_portion = weight.index_select(first_non_zero_idx, indices)
    return matched_portion 

# def rev_shift_shape(param, weight):
#     # If the shapes do not match, try to crop or pad the tensor
#     mismatched_dimension = find_mismatch_index(param, weight)
#     # find the index of the first non-zero value in shape_diff
#     first_non_zero_idx =  mismatched_dimension[0]
#     # extract the matched portion of the bigger tensor
#     indices = torch.arange(weight.shape[first_non_zero_idx])
    
#     matched_portion = param.index_select(first_non_zero_idx, indices)
#     return matched_portion  

def load_state_dict(model, model_dict):
    logger = get_logger()
    missing_keys = []
    shape_mismatch=[]
    for name, param in model.named_parameters():
        if name not in model_dict:
             missing_keys.append(name)
             continue
        weight = deepcopy(model_dict[name])
        logger.info(f'name:{name},model param shape:{param.shape},new weights shape:{weight.shape}')
        if param.data.shape != weight.data.shape:
            shape_mismatch.append(name)
            data = shift_shape(param.data, weight.data)
            weight.data = data
            model_dict[name]=weight
    return  model_dict, missing_keys, shape_mismatch

def create_model_dict(config, checkpoint):
    model_dict = {}
    key_mapper = load_config(config.checkpoint.key_mapper, merge_base=False)
    for key, value in checkpoint['model'].items():
        modified = False
        if 'modified' in key_mapper:
            for mapperk, mapperv in  key_mapper.modified.items():
                if mapperk in key:
                    print("inside key", key, "key_mapper", mapperk)
                    new_key = key.replace(mapperk, mapperv)
                    modified = True
                    model_dict[new_key] = value
                    break
        if 'addas' in key_mapper:
            for mapperk, mapperv in  key_mapper.addas.items():
                if mapperv in key:
                    print("inside key", key, "addas_mapper", mapperv)
                    new_key = key.replace(mapperv, mapperk)
                    model_dict[new_key] = value
                    break
        if not modified:
            model_dict[key] = value
    return model_dict

#Taken from ViT codebase
def interpolate_posembed(posemb, num_tokens: int):
  """Interpolate given positional embedding parameters into a new shape.

  Args:
    posemb: positional embedding parameters.
    num_tokens: desired number of tokens.
    has_class_token: True if the positional embedding parameters contain a
      class token.

  Returns:
    Positional embedding parameters interpolated into the new shape.
  """
  import scipy.ndimage
  import numpy as np
  assert posemb.shape[0] == 1
  posemb_tok, posemb_grid = posemb[:, :0], posemb[0, 0:]
  logger = get_logger()
  gs_old = int(np.sqrt(len(posemb_grid)))
  gs_new = int(np.sqrt(num_tokens))
  logger.info('interpolate_posembed: grid-size from %s to %s', gs_old, gs_new)
  assert gs_old ** 2 == len(posemb_grid), f'{gs_old ** 2} != {len(posemb_grid)}'
  assert gs_new ** 2 == num_tokens, f'{gs_new ** 2} != {num_tokens}'
  posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)

  zoom = (gs_new / gs_old, gs_new / gs_old, 1)
  posemb_grid = scipy.ndimage.zoom(posemb_grid, zoom, order=1)
  posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
  
  logger.info(f'new grid size: {posemb_grid.shape[1]}')
  new_posembed = np.concatenate([posemb_tok, posemb_grid], axis=1)
  logger.info(f'new positional embedding shape: {new_posembed.shape}')
  #convert new pos embed to torch tensor
  return torch.from_numpy(new_posembed)

def interpolate_pos_embed(pos_embed, pos_embed_new):
    '''
    This function is used to interpolate the positional embedding from the previous checkpoint to the new checkpoint
    '''
    logger = get_logger()
    logger.info(f'Interpolating positional embedding from {pos_embed.shape} to {pos_embed_new.shape}')
    # interpolate the positional embedding from the previous checkpoint to the new checkpoint
    pos_embed_new[:, :pos_embed.shape[1]] = pos_embed
    if pos_embed.shape[1] > pos_embed_new.shape[1]:
        pos_embed_new[:, pos_embed_new.shape[1]:] = pos_embed[:, :pos_embed_new.shape[1]]
    return pos_embed_new

def load_checkpoint(config, model, optimizer, lr_scheduler, allow_shape_change: bool=False):
    logger = get_logger()
    logger.info(f'==============> Resuming form {config.checkpoint.resume}....................')
    checkpoint = CheckpointLoader.load_checkpoint(config.checkpoint.resume, map_location='cpu')
    model_dict = {}
    '''
    Key mapper is used to transfer weights optimally while training hierarchies other than standard 2 layer hierarchy
    '''
    print("config", config.checkpoint)
    if config.checkpoint.key_mapper:
        model_dict = create_model_dict(config, checkpoint)
        print("model_dict", model_dict.keys())
    if config.checkpoint.key_mapper and model_dict:
        if not allow_shape_change:
            msg = model.load_state_dict(model_dict, strict=False)   
            logger.info("msg", msg)
            logger.info("msg len missing - incompatible keys", len(msg.missing_keys))
        else:
            '''
            Iterate over all the layers in the model and load weight manually
            This is done to accomodate weight initialization for architecture with 3 grouping layer 
            with fewer final group tokens with weight initialization from pretrained model's 2 grouping
            layer architecture. We use cropping and padding strategy to find the matched dimensions and
            use them as initialization
            '''
            logger.info("Manually loading state dict")
            #missing_keys = load_state_dict(model, model_dict)
            model_dict, missing_keys, _ = load_state_dict(model, model_dict)
            logger.info(f'Manually loading state dict with {len(missing_keys)} missing keys')
            logger.info(f'missing keys: {missing_keys} ')

            msg = model.load_state_dict(model_dict, strict=False)   
            logger.info("msg", msg)
            logger.info("msg len missing - incompatible keys", len(msg.missing_keys))
    elif config.checkpoint.interpolate_pos_embedding:
        logger.info("Interpolating position embedding")
        model_dict = checkpoint['model']
        
        pos_embed = model_dict['img_encoder.pos_embed']
        #check if pos_embed is in img_encoder
        pos_embed_new = model.img_encoder.pos_embed
        logger.info(f'pos_embed shape: {pos_embed.shape}')
        logger.info(f'pos_embed_new shape: {pos_embed_new.shape}')
        if pos_embed.shape != pos_embed_new.shape:
            #pos_embed_new = interpolate_pos_embed(pos_embed, pos_embed_new)
            pos_embed_new = interpolate_posembed(pos_embed, pos_embed_new.shape[1])
            model_dict['img_encoder.pos_embed'] = pos_embed_new
            msg = model.load_state_dict(model_dict, strict=False)
            logger.info("msg", msg)
            logger.info("msg len missing - incompatible keys", len(msg.missing_keys))
        else:
            msg = model.load_state_dict(model_dict, strict=False)
            logger.info("msg", msg)
            logger.info("msg len missing - incompatible keys", len(msg.missing_keys))
    else:
        msg = model.load_state_dict(checkpoint['model'], strict=False)
        logger.info("msg", msg)
        logger.info("msg len missing - incompatible keys",len(msg.missing_keys))

    metrics = defaultdict(float)
    if (not config.evaluate.eval_only and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint
            and 'epoch' in checkpoint):
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        with read_write(config):
            config.train.start_epoch = checkpoint['epoch'] + 1
        if 'amp' in checkpoint and config.train.amp_opt_level != 'O0' and checkpoint[
                'config'].train.amp_opt_level != 'O0':
            amp.load_state_dict(checkpoint['amp'])
        logger.info(f"=> loaded successfully '{config.checkpoint.resume}' (epoch {checkpoint['epoch']})")
        metrics = checkpoint['metrics']

    del checkpoint
    torch.cuda.empty_cache()
    return metrics

# def load_state_dict(model, model_dict):
#     logger = get_logger()
#     missing_keys = []
#     for name, param in model.named_parameters():
#         if name not in model_dict:
#              missing_keys.append(name)
#              continue
#         weight = deepcopy(model_dict[name])
#         logger.info(f'name:{name},model param shape:{param.shape},new weights shape:{weight.shape}')
#         if param.data.shape != weight.data.shape:
#             data = shift_shape(param.data, weight.data)
#             model_dict[name].data = weight.data
#             weight.data = data
#             model_dict[name]=weight
        
#         # Set the weight of the current layer to the loaded weight
#     return  model_dict, missing_keys 

def save_checkpoint(config, epoch, model, metrics, optimizer, lr_scheduler, suffix=''):
    save_state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'metrics': metrics,
        'epoch': epoch,
        'config': config
    }
    logger = get_logger()

    for k, v in metrics.items():
        save_state[k] = v

    if config.train.amp_opt_level != 'O0':
        save_state['amp'] = amp.state_dict()

    if len(suffix) > 0 and not suffix.startswith('_'):
        suffix = '_' + suffix
    filename = f'ckpt_epoch_{epoch}{suffix}.pth'

    save_path = os.path.join(config.output, filename)
    logger.info(f'{save_path} saving......')
    torch.save(save_state, save_path)
    torch.save(save_state, os.path.join(config.output, 'checkpoint.pth'))
    logger.info(f'{save_path} saved !!!')

    if config.checkpoint.max_kept > 0:
        if epoch >= config.checkpoint.max_kept:
            logger.info(f'Epoch: {epoch}, greater than config.checkpoint.max_kept: {config.checkpoint.max_kept}')
            end_clean_epoch = epoch - config.checkpoint.max_kept
            old_path_list = []
            for cur_clean_epoch in range(end_clean_epoch + 1):
                old_path = os.path.join(config.output, f'ckpt_epoch_{cur_clean_epoch}{suffix}.pth')
                if os.path.exists(old_path):
                    logger.info(f'old checkpoint path {old_path} exits')
                    old_path_list.append(old_path)
            for old_path in old_path_list[:-config.checkpoint.max_kept]:
                os.remove(old_path)
                logger.info(f'old checkpoint path {old_path} removed!!!')

def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item()**norm_type
    total_norm = total_norm**(1. / norm_type)
    return total_norm

def auto_resume_helper(output_dir):
    if os.path.exists(os.path.join(output_dir, 'checkpoint.pth')):
        return os.path.join(output_dir, 'checkpoint.pth')
    checkpoints = os.listdir(output_dir)
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith('pth')]
    print(f'All checkpoints founded in {output_dir}: {checkpoints}')
    if len(checkpoints) > 0:
        latest_checkpoint = max([os.path.join(output_dir, d) for d in checkpoints], key=os.path.getmtime)
        print(f'The latest checkpoint founded: {latest_checkpoint}')
        resume_file = latest_checkpoint
    else:
        resume_file = None
    return resume_file

def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt
