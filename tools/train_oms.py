#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
import random
import warnings
from loguru import logger
import json

import torch
import torch.backends.cudnn as cudnn

from yolox.core import launch, TrainerOMS as Trainer 
from yolox.exp import get_exp
from yolox.utils import configure_nccl, configure_omp, get_num_devices



def make_parser():
    parser = argparse.ArgumentParser("YOLOX train parser")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")
    parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")
    parser.add_argument("--dist-url", default=None, type=str, help="url used to set up distributed training")
    parser.add_argument("-b", "--batch-size", type=int, default=64, help="batch size")
    parser.add_argument("-d", "--devices", default=None, type=int, help="device for training")
    parser.add_argument("--device_type", default='cuda', type=str, help="device type for training. cpu for cpu based training, otherwise gpu based training")
    parser.add_argument("-w", "--workers", default=None, type=int, help="number of workers per gpu")
    parser.add_argument("-f", "--exp_file", default=None, type=str, help="plz input your experiment description file")
    parser.add_argument("--resume", default=False, action="store_true", help="resume training")
    parser.add_argument("--dataset", default=None, type=str, help="dataset for training")
    parser.add_argument("--task", default="2dod", type=str, help="type of task for model eval")
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="checkpoint file")
    parser.add_argument("-odw", "--od-weights", default=None, type=str, help="weights for trained 2DOD network")
    parser.add_argument("-e", "--start_epoch", default=None, type=int, help="resume training start epoch")
    parser.add_argument("--num_machines", default=1, type=int, help="num of node for training")
    parser.add_argument("--machine_rank", default=0, type=int, help="node rank for multi-node training")
    parser.add_argument("--fp16", dest="fp16", default=False, action="store_true", help="Adopting mix precision training.")
    parser.add_argument("--cache", dest="cache", default=False, action="store_true", help="Caching imgs to RAM for fast training.")
    parser.add_argument("-o", "--occupy", dest="occupy", default=False, action="store_true", help="occupy GPU memory first for training.")
    parser.add_argument("--visualize", dest="visualize", default=False, action="store_true", help="Enable drawing of bounding cuboid")
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)
    parser.add_argument("--train_ann", help="train annotation file name", default=None)
    parser.add_argument("--val_ann", help="val annotation file name", default=None, nargs=argparse.REMAINDER)
    return parser


@logger.catch
def main(exp, args):
    if exp.seed is not None:
        random.seed(exp.seed)
        torch.manual_seed(exp.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed training. This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! You may see unexpected behavior "
            "when restarting from checkpoints."
        )

    # set environment variables for distributed training
    configure_nccl()
    configure_omp()
    cudnn.benchmark = True
    exp.max_epoch = 100

    trainer = Trainer(exp, args)
    trainer.train()


def run(**kwargs):
    '''
    This run function is not called from inside this repository
    This is for use from external programs such as the edgeai-modemaker
    '''
    
    args = make_parser().parse_args()
    for k, v in kwargs.items():
        setattr(args, k, v)
    
    exp = get_exp(args.exp_file, args.name)
    exp.max_epoch = args.max_epoch
    
    exp.warmup_epochs = max(min(exp.warmup_epochs, exp.max_epoch//4), 1)
    exp.no_aug_epochs = max(min(exp.no_aug_epochs, exp.max_epoch//4), 1)
    exp.output_dir = args.output_dir
    exp.visualize = args.visualize
    exp.data_dir = args.data_dir
    exp.train_ann = args.train_ann
    exp.val_ann = args.val_ann
    exp.img_folder_names = args.img_folder_names
    exp.flip_prob = 0.0
    exp.default_sigmas = False
    exp.device_type = args.device_type

    if args.ckpt is not None:
        exp.od_weights = args.ckpt

 

    if not args.experiment_name:
        args.experiment_name = ''

    if args.workers is not None:
        exp.data_num_workers = args.workers

    num_gpu = get_num_devices() if args.devices is None else args.devices
    assert num_gpu <= get_num_devices()

    dist_url = "auto" if args.dist_url is None else args.dist_url

    if num_gpu > 1:
        launch(
            main,
            num_gpu,
            args.num_machines,
            args.machine_rank,
            backend=args.dist_backend,
            dist_url=dist_url,
            args=(exp, args),
        )
    else:
        main(exp, args)


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.opts)

    # additional merges
    for key in ['device_type', 'train_ann', 'val_ann']:
        val = getattr(args, key)
        if val is not None:
            setattr(exp, key, val)

    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    num_gpu = get_num_devices() if args.devices is None else args.devices
    assert num_gpu <= get_num_devices()

    if args.workers is not None:
        exp.data_num_workers = args.workers

    dist_url = "auto" if args.dist_url is None else args.dist_url
    launch(
        main,
        num_gpu,
        args.num_machines,
        args.machine_rank,
        backend=args.dist_backend,
        dist_url=dist_url,
        args=(exp, args),
    )
