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
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
from datasets import build_loader, build_text_transform, imagenet_classes
from mmcv.parallel import MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, set_random_seed
from mmcv.utils import collect_env, get_git_hash
from mmseg.apis import multi_gpu_test
from models import build_model
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
        '--output', type=str, help='root of output folder, '
        'the full path is <output>/<model_name>/<tag>')
    parser.add_argument('--tag', type=str, help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--wandb', action='store_true', help='Use W&B to log experiments')
    parser.add_argument('--keep', type=int, help='Maximum checkpoint to keep')

    # distributed training
    parser.add_argument('--local_rank', type=int, required=True, help='local rank for DistributedDataParallel')

    args = parser.parse_args()

    return args

def log_tensorboard_graph(data_loader_train, model, cfg):
    import copy
    from einops import rearrange
    from torch.utils.tensorboard import SummaryWriter
    
    logger = get_logger()

    # default `log_dir` is "runs" - we'll be more specific here
    writer = SummaryWriter(cfg.output)

    data_loader = copy.deepcopy(data_loader_train)
    dataiter = iter(data_loader)
    samples = next(dataiter)
        
    img_model = copy.deepcopy(model.module.img_encoder).cuda()
    
    text_model = copy.deepcopy(model.module.text_encoder).cuda()
    # single_sample = torch.unsqueeze(torch.unsqueeze(samples['image'].data[0][0][0], dim=0), dim=0)
    # logger.info('single sample', single_sample)
    
    # logger.info('shape single sample', single_sample.shape)
    #img_sample_data = single_sample.cuda()
    img_sample_data = samples['image'].data[0].cuda()
    img_sample_data.requires_grad = False
    text_sample_data = samples['text'].data[0].cuda()
    assert text_sample_data.ndim in [2, 3], text_sample_data.ndim
    squeeze_dim = False
    num_text = 1
    if text_sample_data.ndim == 3:
        num_text = text_sample_data.shape[1]
        text_sample_data = rearrange(text_sample_data, 'b n l -> (b n) l', n=num_text)
        squeeze_dim = True
    logger.info('text sample data', text_sample_data.shape)
    text_sample_data.requires_grad = False
    for para in img_model.parameters():
        para.required_grad = False
    for para in text_model.parameters():
        para.required_grad = False
    img_model.eval()
    text_model.eval()
    x = torch.rand((1,3,16,16)).cuda()

    text_input = torch.rand((1, 4, 77)).to(torch.int64).cuda()
    with amp.disable_casts():
        writer.add_graph(img_model, x)#img_sample_data.detach())
        writer.add_graph(text_model, text_sample_data)#text_input)
    writer.close()

def write_model_iteration(cfg, data_loader_train, model):
    #To track tensorboard model graph
    #log_tensorboard_graph()
    import copy
    from einops import rearrange
    from torch.utils.tensorboard import SummaryWriter
    
    logger = get_logger()
    writer = SummaryWriter(cfg.output)
    data_loader = copy.deepcopy(data_loader_train)
    dataiter = iter(data_loader)
    samples = next(dataiter)
        
    img_model = copy.deepcopy(model.module.img_encoder).cuda()
    
    text_model = copy.deepcopy(model.module.text_encoder).cuda()
    # single_sample = torch.unsqueeze(torch.unsqueeze(samples['image'].data[0][0][0], dim=0), dim=0)
    # logger.info('single sample', single_sample)
    
    # logger.info('shape single sample', single_sample.shape)
    #img_sample_data = single_sample.cuda()
    img_sample_data = samples['image'].data[0].cuda()
    img_sample_data.requires_grad = False
    text_sample_data = torch.rand((1, 4, 77)).to(torch.int64).cuda()# samples['text'].data[0].cuda()
    assert text_sample_data.ndim in [2, 3], text_sample_data.ndim
    squeeze_dim = False
    num_text = 1
    if text_sample_data.ndim == 3:
        num_text = text_sample_data.shape[1]
        text_sample_data = rearrange(text_sample_data, 'b n l -> (b n) l', n=num_text)
        squeeze_dim = True
    text_sample_data.requires_grad = False
    for para in img_model.parameters():
        para.required_grad = False
    for para in text_model.parameters():
        para.required_grad = False
    img_model.eval()
    text_model.eval()
    x = torch.rand((1,3,16,16)).cuda()


    text_input = torch.rand((1, 4, 77)).to(torch.int64).cuda()
    assert text_input.ndim in [2, 3], text_input.ndim
    squeeze_dim = False
    num_text = 1
    if text_input.ndim == 3:
        num_text = text_input.shape[1]
        text_input = rearrange(text_input, 'b n l -> (b n) l', n=num_text)
        squeeze_dim = True
    
    
    #writer.add_graph(text_model, text_input)#text_input)
    with amp.disable_casts():
        writer.add_graph(img_model, x)#img_sample_data.detach())
    writer.close()



def train(cfg):
    logger = get_logger()
    if cfg.wandb and dist.get_rank() == 0:
        import wandb
        if cfg.wandb_id:
            wandb.init(
                id=cfg.wandb_id,
                project='group_vit',
                sync_tensorboard=True,
                name=osp.join(cfg.model_name, cfg.tag),
                dir=cfg.output,#wandb_output,
                config=OmegaConf.to_container(cfg, resolve=True),
                resume='must'
            )
        else:
            wandb.init(
                project='group_vit',
                sync_tensorboard=True,
                name=osp.join(cfg.model_name, cfg.tag),
                dir=cfg.output,#wandb_output,
                config=OmegaConf.to_container(cfg, resolve=True),
            )
    else:
        wandb = None
    # waiting wandb init
    dist.barrier()
    dataset_train, dataset_val, \
        data_loader_train, data_loader_val = build_loader(cfg.data)
    data_loader_seg = build_seg_dataloader(build_seg_dataset(cfg.evaluate.seg))

    logger.info(f'Creating model:{cfg.model.type}/{cfg.model_name}')
    model = build_model(cfg.model)
    model.cuda()

    if cfg.wandb and dist.get_rank() == 0:
        wandb.watch(model, log="all")
        
    logger.info(str(model))

    optimizer = build_optimizer(cfg.train, model)
    if cfg.train.amp_opt_level != 'O0':
        model, optimizer = amp.initialize(model, optimizer, opt_level=cfg.train.amp_opt_level)
    model = MMDistributedDataParallel(model, device_ids=[torch.cuda.current_device()], broadcast_buffers=False)
    model_without_ddp = model.module

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
    text_transform = build_text_transform(False, cfg.data.text_aug, with_dc=False)
    if cfg.data.precompute_pad_mask:
        padword_tokenized = text_transform(cfg.data.pad_word)
        model_without_ddp.build_padtoken_embedding(padword_tokenized)

    if cfg.checkpoint.resume:
        max_metrics = load_checkpoint(cfg, model_without_ddp, optimizer, lr_scheduler)
        max_accuracy, max_miou = max_metrics['max_accuracy'], max_metrics['max_miou']
        # if 'cls' in cfg.evaluate.task:
        #     acc1, acc5, loss = validate_cls(cfg, data_loader_val, model)
        #     logger.info(f'Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%')
        # if 'seg' in cfg.evaluate.task:
        #     miou = validate_seg(cfg, data_loader_seg, model)
        #     logger.info(f'mIoU of the network on the {len(data_loader_seg.dataset)} test images: {miou:.2f}%')
        #     if wandb is not None:
        #         log_stat = {}
        #         log_stat.update({
        #             'epoch/val_miou': miou,
        #             'epoch/epoch': -1,
        #             'epoch/n_parameters': n_parameters
        #         })
        #         wandb.log(log_stat)

        if cfg.evaluate.eval_only:
            return


    #write_model_iteration(cfg, data_loader_train, model)

    
    logger.info('Start training')
    start_time = time.time()

    #default `log_dir` is "runs" - we'll be more specific here
    
    for epoch in range(cfg.train.start_epoch, cfg.train.epochs):
        # import ipdb
        # ipdb.set_trace()
        torch.cuda.empty_cache()
        logger.debug(f'epoch:{epoch}')
        loss_train_dict = train_one_epoch(cfg, model, data_loader_train, optimizer, epoch, lr_scheduler)
        print('loss_train_dict', loss_train_dict)
        if dist.get_rank() == 0 and (epoch % cfg.checkpoint.save_freq == 0 or epoch == (cfg.train.epochs - 1)):
            save_checkpoint(cfg, epoch, model_without_ddp, {
                'max_accuracy': max_accuracy,
                'max_miou': max_miou
            }, optimizer, lr_scheduler)
        dist.barrier()
        loss_train = loss_train_dict['total_loss']
        logger.info(f'Avg loss of the network on the {len(dataset_train)} train images: {loss_train:.2f}')

        # evaluate
        if (epoch % cfg.evaluate.eval_freq == 0 or epoch == (cfg.train.epochs - 1)):
            if 'cls' in cfg.evaluate.task:
                acc1, acc5, loss = validate_cls(cfg, data_loader_val, model)
                logger.info(f'Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%')
                max_metrics['max_accuracy'] = max(max_metrics['max_accuracy'], acc1)
                if cfg.evaluate.cls.save_best and dist.get_rank() == 0 and acc1 > max_accuracy:
                    save_checkpoint(
                        cfg, epoch, model_without_ddp, max_metrics, optimizer, lr_scheduler, suffix='best_acc1')
                dist.barrier()
                max_accuracy = max_metrics['max_accuracy']
                logger.info(f'Max accuracy: {max_accuracy:.2f}%')
            if 'seg' in cfg.evaluate.task:
                miou = validate_seg(cfg, data_loader_seg, model)
                logger.info(f'mIoU of the network on the {len(data_loader_seg.dataset)} test images: {miou:.2f}%')
                max_metrics['max_miou'] = max(max_metrics['max_miou'], miou)
                if cfg.evaluate.seg.save_best and dist.get_rank() == 0 and miou > max_miou:
                    save_checkpoint(
                        cfg, epoch, model_without_ddp, max_metrics, optimizer, lr_scheduler, suffix='best_miou')
                dist.barrier()
                max_miou = max_metrics['max_miou']
                logger.info(f'Max mIoU: {max_miou:.2f}%')

        if wandb is not None:
            log_stat = {f'epoch/train_{k}': v for k, v in loss_train_dict.items()}
            log_stat.update({
                'epoch/val_miou': miou,
                'epoch/epoch': epoch,
                'epoch/n_parameters': n_parameters
            })
            wandb.log(log_stat)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))
    dist.barrier()


def train_one_epoch(config, model, data_loader, optimizer, epoch, lr_scheduler):
    logger = get_logger()
    dist.barrier()
    model.train()
    optimizer.zero_grad()
    if config.wandb and dist.get_rank() == 0:
        import wandb
    else:
        wandb = None

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    log_vars_meters = defaultdict(AverageMeter)

    start = time.time()
    end = time.time()
    #getting some random training images for tensorboard
    

    for idx, samples in enumerate(data_loader):

        logger.info(f'idx:{idx}')

        batch_size = config.data.batch_size
        # print("batch size",batch_size)
        # print("samples",len(samples['image']))
        # images = samples['image'].data
        # print("batch tensor shape", len(images))
        # print("image1", images[0].shape)
        # imagebatch = images[0]

        # print("first image",imagebatch[0])
        
        # print("first image shape",imagebatch[0].shape)
        # import matplotlib.pyplot as plt
        # import numpy as np
        # fi=imagebatch[0]
        # print("fi",fi.shape)
        # print("type fi", type)
        #plt.imshow(np.transpose(fi.cpu().numpy(), (1,2,0)))
        #plt.imsave('test.png', np.transpose(fi.cpu().numpy()))
        #plt.imshow(fi.numpy()[0], cmap='gray')
        # print("second image",imagebatch[1])
        # #for img in imagebatch:
        # for img1 in imagebatch:
        #     if torch.equal(img1, imagebatch[0]):
        #         print("torch equal or not?", torch.equal(img1, imagebatch[0]))


        losses = model(**samples)
        loss, log_vars = parse_losses(losses)
        if config.train.accumulation_steps > 1:
            loss = loss / config.train.accumulation_steps
            if config.train.amp_opt_level != 'O0':
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    
                    # check whether loss requires grad or not
                    #set requires grad to true
                    scaled_loss.requires_grad = True
                    scaled_loss.backward()
                if config.train.clip_grad:
                    grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), config.train.clip_grad)
                else:
                    grad_norm = get_grad_norm(amp.master_params(optimizer))
            else:
                loss.backward()
                if config.train.clip_grad:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.clip_grad)
                else:
                    grad_norm = get_grad_norm(model.parameters())
            if (idx + 1) % config.train.accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step_update(epoch * num_steps + idx)
        else:
            optimizer.zero_grad()
            if config.train.amp_opt_level != 'O0':
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    # check whether loss requires grad or not
                    
                    #scaled_loss.requires_grad = True

                    scaled_loss.backward()
                if config.train.clip_grad:
                    grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), config.train.clip_grad)
                else:
                    grad_norm = get_grad_norm(amp.master_params(optimizer))
            else:
                loss.backward()
                if config.train.clip_grad:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.clip_grad)
                else:
                    grad_norm = get_grad_norm(model.parameters())
            optimizer.step()
            lr_scheduler.step_update(epoch * num_steps + idx)

        torch.cuda.synchronize()

        loss_meter.update(loss.item(), batch_size)
        for loss_name in log_vars:
            log_vars_meters[loss_name].update(log_vars[loss_name], batch_size)
        norm_meter.update(grad_norm)
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.print_freq == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            log_vars_str = '\t'.join(f'{n} {m.val:.4f} ({m.avg:.4f})' for n, m in log_vars_meters.items())
            logger.info(f'Train: [{epoch}/{config.train.epochs}][{idx}/{num_steps}]\t'
                        f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                        f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                        f'total_loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                        f'{log_vars_str}\t'
                        f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                        f'mem {memory_used:.0f}MB')
            if wandb is not None:
                log_stat = {f'iter/train_{n}': m.avg for n, m in log_vars_meters.items()}
                log_stat['iter/train_total_loss'] = loss_meter.avg
                log_stat['iter/learning_rate'] = lr
                wandb.log(log_stat)

    epoch_time = time.time() - start
    logger.info(f'EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}')
    result_dict = dict(total_loss=loss_meter.avg)
    for n, m in log_vars_meters.items():
        result_dict[n] = m.avg
    dist.barrier()
    return result_dict


@torch.no_grad()
def validate_cls(config, data_loader, model):
    logger = get_logger()
    dist.barrier()
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    text_transform = build_text_transform(False, config.data.text_aug, with_dc=False)

    end = time.time()
    logger.info('Building zero shot classifier')
    text_embedding = data2cuda(
        model.module.build_text_embedding(
            build_dataset_class_tokens(text_transform, config.evaluate.cls.template, imagenet_classes)))
    logger.info('Zero shot classifier built')
    for idx, samples in enumerate(data_loader):
        logger.debug(f'idx:{idx}')
        target = samples.pop('target').data[0].cuda()
        target = data2cuda(target)

        # compute output
        output = model(**samples, text=text_embedding)

        # measure accuracy and record loss
        loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))    
        logger.info(f'acc1:{acc1}, acc5:{acc5}, loss:{loss}')

        acc1 = reduce_tensor(acc1)
        acc5 = reduce_tensor(acc5)
        loss = reduce_tensor(loss)
        
        loss_meter.update(loss.item(),target.size(0))
        
        
        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))
        acc5_meter.update(acc5.item(), target.size(0))
        #measure elapsed time
        logger.debug("all metrics updated")
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.print_freq == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(f'Test: [{idx}/{len(data_loader)}]\t'
                        f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                        f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                        f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t'
                        f'Mem {memory_used:.0f}MB')
    logger.info('Clearing zero shot classifier')
    torch.cuda.empty_cache()
    logger.info(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}')
    dist.barrier()
    return acc1_meter.avg, acc5_meter.avg, loss_meter.avg


@torch.no_grad()
def validate_seg(config, data_loader, model):
    logger = get_logger()
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
    results = multi_gpu_test(
        model=mmddp_model,
        data_loader=data_loader,
        tmpdir=None,
        gpu_collect=True,
        efficient_test=False,
        pre_eval=True,
        format_only=False)

    if dist.get_rank() == 0:

        metrics = [data_loader.dataset.evaluate(results, metric='mIoU')]
        # import csv
        # import numpy as np
        # file_name = config.output+'/'+ str(data_loader.dataset.__class__.__name__) + '_per_class_iou.csv'
        # ret_metrics = metrics[0]
        # miou_result = ret_metrics['mIoU'] * 100
        # print("miou_result", miou_result)
        # print("metric", ret_metrics)
        # ret_metrics_class = OrderedDict({
        #     ret_metric: np.round(ret_metric_value * 100, 2)
        #     for ret_metric, ret_metric_value in ret_metrics.items()
        # })

        # Save the per-class IoU scores into a CSV file
        # with open(file_name, 'w', newline='') as f:
        #     writer = csv.writer(f)
        #     writer.writerow(['Class Name', 'IoU Score'])
        #     for key, val in ret_metrics_class.items():
        #       if 'IoU.' in key:
        #         writer.writerow([key.split('.')[1], val])
    else:
        metrics = [None]
    dist.broadcast_object_list(metrics)
    miou_result = metrics[0]['mIoU'] * 100
    torch.cuda.empty_cache()
    logger.info(f'Eval Seg mIoU {miou_result:.2f}')
    dist.barrier()
    return miou_result


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