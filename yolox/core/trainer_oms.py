#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import datetime
import os
import time
from loguru import logger
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from yolox.data import DataPrefetcher, DataPrefetcherCPU
from yolox.utils import (
    MeterBuffer,
    ModelEMA,
    all_reduce_norm,
    get_local_rank,
    get_model_info,
    get_rank,
    get_world_size,
    gpu_mem_usage,
    is_parallel,
    load_ckpt,
    occupy_mem,
    save_checkpoint,
    setup_logger,
    synchronize,
    plots
)

def move_targets_to_device(targets, dtype=None, device=None):
    """
    Move all tensors inside a dictionary to a specified device and/or dtype.

    Args:
        targets (dict): Dictionary where each value is a tensor or None.
        dtype (torch.dtype, optional): Target data type (e.g., torch.float32). Defaults to None (keep same).
        device (torch.device, optional): Target device (e.g., 'cuda', 'cpu'). Defaults to None (keep same).

    Returns:
        dict: New dictionary with tensors moved to specified device/dtype.
    """
    
    if type(targets) is not dict:
        targets = targets.to(dtype=dtype, device=device)
        return targets
        
    else:
        new_targets = {}
        for k, v in targets.items():
            if isinstance(v, torch.Tensor):
                v = v.to(dtype=dtype, device=device)  # Move to device and/or dtype
            new_targets[k] = v  # Leave None or other types as-is
        return new_targets


class Trainer:
    def __init__(self, exp, args):
        self.exp = exp
        self.args = args
        # training related attr
        self.max_epoch = exp.max_epoch
        self.amp_training = args.fp16
        self.scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)
        self.is_distributed = get_world_size() > 1
        self.rank = get_rank()
        self.local_rank = get_local_rank()
        self.device = (exp.device_type if exp.device_type == "cpu" else "{}:{}".format(exp.device_type, self.local_rank))
        self.use_model_ema = exp.ema
        self.state_dict_object_detect = None

        # data/dataloader related attr
        self.data_type = torch.float16 if args.fp16 else torch.float32
        self.input_size = exp.input_size
        self.best_ap = 0

        # metric record
        self.meter = MeterBuffer(window_size=exp.print_interval)
        self.file_name = os.path.join(exp.output_dir, args.experiment_name)

        if self.rank == 0:
            os.makedirs(self.file_name, exist_ok=True)
        
        setup_logger(
            self.file_name,
            distributed_rank=self.rank,
            filename="train_log.txt",
            mode="a",
        )

    def train(self):
        self.before_train()
        try:
            self.train_in_epoch()
        except Exception:
            raise
        finally:
            self.after_train()

    def train_in_epoch(self):
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.train_in_iter()
            self.after_epoch()

    def train_in_iter(self):
        for self.iter in range(self.max_iter):
            self.before_iter()
            self.train_one_iter()
            self.after_iter()
           
            ## Debug 
            # if self.iter >= 40:
            #     break

    def _update_metrics(self, info_dict, iter_start_time, prefix='face'):
        metrics = {
            f"{prefix}_iter_time": time.time() - iter_start_time,
            f"{prefix}_lr": self.optimizer.param_groups[0]["lr"],
        }


        # Add each metric with the prefix in the key
        for key in info_dict.keys():
            if type(info_dict[key]) == torch.Tensor:
                metrics[f"{prefix}_{key}"] = info_dict[key].cpu().item()
            else:
                metrics[f"{prefix}_{key}"] = info_dict[key]
        self.meter.update(metrics)
        
    
    def train_one_iter(self):
        iter_start_time = time.time()
    
        # Different behavior based on training strategy
        if self.train_strategy == "alternating":
            # Process one batch from each dataset type
            if self.iter % 3 == 0:
                # Human batch
                inps, targets, _ = self.prefetcher_human.next()
                dataset_type = "human"
            elif self.iter % 3 == 1:
                # Object batch
                inps, targets, _ = self.prefetcher_face.next()
                dataset_type = "face"
            else:
                # Object batch
                inps, targets, _ = self.prefetcher_object.next()
                dataset_type = "object"
   
        
        # Skip invalid batches
        if inps is None:
            self.logger.warning("Empty batch encountered!")
            return
        
        inps = inps.to(self.device)
        
        #Todo targets 
        targets = move_targets_to_device(targets, device=self.device)
        
        # Preprocess data
        if dataset_type == "human":
            inps, targets = self.exp.preprocess_human(inps, targets, self.input_size)
        elif dataset_type == "face":
            inps, targets = self.exp.preprocess_face(inps, targets, self.input_size)
        elif dataset_type == "object":
            inps, targets = self.exp.preprocess_object(inps, targets, self.input_size)
                
        outputs = self.model(inps, targets, train_heads=[dataset_type])

        # Calculate loss based on dataset type
        if dataset_type == "human":
            loss = outputs["human"]["total_loss"]
            iou_loss = outputs["human"]["iou_loss"] 
            conf_loss = outputs["human"]["conf_loss"]
            cls_loss = outputs["human"]["cls_loss"]
            l1_loss = outputs["human"]["l1_loss"]
            kpts_loss = outputs["human"]["kpts_loss"]
            kpts_vis_loss = outputs["human"]["kpts_vis_loss"]
            l1_loss_kpts = outputs["human"]["l1_loss_kpts"]
            num_fg = outputs["human"]["num_fg"]
            # print(f'Yoooooooooooooooooooooooooooooooo: {outputs["human"]}')
            self._update_metrics(outputs["human"], iter_start_time=iter_start_time, prefix='human')
            # print(f'### loss human :{loss.item()}')
        
        elif dataset_type == "face":
            loss  = outputs["face"]["total_loss"]
            iou_loss = outputs["face"]["iou_loss"]
            conf_loss = outputs["face"]["conf_loss"]
            cls_loss = outputs["face"]["cls_loss"]
            l1_loss = outputs["face"]["l1_loss"]
            kpts_loss = outputs["face"]["kpts_loss"]
            kpts_vis_loss = outputs["face"]["kpts_vis_loss"]
            l1_loss_kpts = outputs["face"]["l1_loss_kpts"]
            num_fg = outputs["face"]["num_fg"]
            # print(f'Yoooooooooooooooooooooooooooooooo: {outputs["face"]}')
            
            self._update_metrics(outputs["face"], iter_start_time, prefix='face')
            
            # print(f'### loss face :{loss.item()}')
        
        elif dataset_type == "object":
            loss = outputs["object"]["total_loss"]
            iou_loss = outputs["object"]["iou_loss"] 
            conf_loss = outputs["object"]["conf_loss"]
            cls_loss = outputs["object"]["cls_loss"]
            l1_loss = outputs["object"]["l1_loss"]
            num_fg = outputs["object"]["num_fg"]
            
            # Set human-specific losses to 0 for object batches
            kpts_loss = 0
            kpts_vis_loss = 0
            l1_loss_kpts = 0
            # print(f'Yoooooooooooooooooooooooooooooooo: {outputs["object"]}')
            
            self._update_metrics(outputs["object"], iter_start_time, prefix='object')
            
            # print(f'### loss object :{loss.item()}')
            
            # Forward pass
        
        
        
        
           
        # elif dataset_type == "mixed":
        #     # For mixed batches, combine losses from both heads
        #     human_mask = batch["dataset_type"] == "human"
        #     object_mask = batch["dataset_type"] == "object"
            
        #     # Initialize all losses to 0
        #     loss = 0
        #     iou_loss = 0
        #     conf_loss = 0
        #     cls_loss = 0
        #     l1_loss = 0
        #     kpts_loss = 0
        #     kpts_vis_loss = 0
        #     l1_loss_kpts = 0
        #     num_fg = 0
            
        #     # Process human samples if present
        #     if human_mask.any():
        #         human_loss = outputs["human"]["total_loss"]
        #         loss += human_loss
        #         iou_loss += outputs["human"]["iou_loss"]
        #         conf_loss += outputs["human"]["conf_loss"]
        #         cls_loss += outputs["human"]["cls_loss"]
        #         l1_loss += outputs["human"]["l1_loss"]
        #         kpts_loss += outputs["human"]["kpts_loss"]
        #         kpts_vis_loss += outputs["human"]["kpts_vis_loss"]
        #         l1_loss_kpts += outputs["human"]["l1_loss_kpts"]
        #         num_fg += outputs["human"]["num_fg"]
                
        #     # Process object samples if present
        #     if object_mask.any():
        #         object_loss = outputs["object"]["total_loss"]
        #         loss += object_loss
        #         iou_loss += outputs["object"]["iou_loss"]
        #         conf_loss += outputs["object"]["conf_loss"]
        #         cls_loss += outputs["object"]["cls_loss"]
        #         l1_loss += outputs["object"]["l1_loss"]
        #         num_fg += outputs["object"]["num_fg"]
        
        # Backward pass
        self.optimizer.zero_grad()
        # Dododevu:
        # if dataset_type == "face":
        loss.backward()
        self.optimizer.step()
        
        # Update EMA model if using
        if self.use_model_ema:
            self.ema_model.update(self.model)
            
            
        lr = self.lr_scheduler.update_lr(self.progress_in_iter + 1)
        # lr = 0.0001
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

            
        # Track and log losses
        # self.meter.update(
        #     iter_time=time.time() - iter_start_time,
        #     lr=self.optimizer.param_groups[0]["lr"],
        #     dataset_type=dataset_type,
        #     total_loss=loss,
        #     iou_loss=iou_loss,
        #     conf_loss=conf_loss,
        #     cls_loss=cls_loss,
        #     l1_loss=l1_loss,
        #     kpts_loss=kpts_loss,
        #     kpts_vis_loss=kpts_vis_loss,
        #     l1_loss_kpts=l1_loss_kpts,
        #     num_fg=num_fg,
        # )
        
        # Update learning rate scheduler
        # print(f'WARNINGGGGGGGGGGGGGGGGGGGGGg. Fix update schedulaer')
        # self.lr_scheduler.step()

    def before_train(self):
        # logger.info("args: {}".format(self.args))
        logger.info("exp value:\n{}".format(self.exp))
        
        # Setup model
        model = self.exp.get_model()
        logger.info(
            "Model Summary: {}".format(get_model_info(model, self.exp.test_size))
        )
        model.to(self.device)
        
        # Load checkpoint if resuming training
        model = self.resume_train(model)
        
        self.no_aug = self.start_epoch >= self.max_epoch - self.exp.no_aug_epochs
        
        # Setup optimizer and loss
        self.optimizer = self.exp.get_optimizer(self.args.batch_size)
      
        
        # Setup multi-GPU training
        if self.is_distributed:
            model = DDP(model, device_ids=[self.local_rank], broadcast_buffers=False)
        
   
        
        
        # Setup dataloaders based on training strategy
        self.train_strategy = getattr(self.exp, "train_strategy", "alternating")
        logger.info(f"Using training strategy: {self.train_strategy}")
        
        # Get appropriate dataloaders
        data_loaders = self.exp.get_data_loader(
            batch_size=self.args.batch_size,
            is_distributed=self.is_distributed,
            no_aug=self.no_aug,
            cache_img=self.args.cache
        )
        
        # Setup data prefetchers based on strategy
        if self.train_strategy == "alternating":
            # Create separate prefetchers for human and object datasets
            self.prefetcher_human = DataPrefetcher(data_loaders["human"])
            self.prefetcher_face = DataPrefetcher(data_loaders["face"])
            self.prefetcher_object = DataPrefetcher(data_loaders["object"])
            self.max_iter = len(data_loaders["human"]) + len(data_loaders['face']) + len(data_loaders["object"])
            print(f'###### Max iter: {self.max_iter}')
        else:
            # Single prefetcher for combined or proportional strategy
            dataloader = data_loaders.get("combined", data_loaders.get("proportional"))
            self.prefetcher = DataPrefetcher(dataloader)
            self.max_iter = len(dataloader)
        
        # Setup learning rate scheduler
        self.lr_scheduler = self.exp.get_lr_scheduler(
            self.exp.basic_lr_per_img * self.args.batch_size, self.max_iter
        )
        
 
        
          
        # Initialize EMA model if enabled
        if self.exp.ema:
            self.ema_model = ModelEMA(model, 0.9998)
            self.ema_model.updates = self.max_epoch * self.max_iter
        else:
            self.ema_model = None
        
        # Setup logging tools
        self.model = model
        self.model.train()
        
        self.evaluator = self.exp.get_evaluator(
            batch_size=self.args.batch_size,
            is_distributed=self.is_distributed
        )
        
        # Setup metrics logging
        self.meter = MeterBuffer(window_size=self.exp.print_interval)
        self.file_name = os.path.join(
            self.exp.output_dir, self.args.experiment_name
        )
        
        # Setup TensorBoard logging if enabled
        if self.rank == 0:
            os.makedirs(self.file_name, exist_ok=True)
            self.tblogger = SummaryWriter(
                os.path.join(self.exp.output_dir, "tensorboard", self.args.experiment_name)
            )
            
            # Log training config to file
            # self.exp.update_config(self.file_name)

    def after_train(self):
        logger.info(
            "Training of experiment is done and the best AP is {:.2f}".format(self.best_ap * 100)
        )

    def before_epoch(self):
        """
        Reset dataloaders and prefetchers before each epoch
        """
        logger.info("---> Start training epoch {}".format(self.epoch + 1))
        
        # Reset prefetchers based on training strategy
        if self.train_strategy == "alternating":
            self.prefetcher_human.preload()
            self.prefetcher_face.preload()
            self.prefetcher_object.preload()
        else:
            self.prefetcher.preload()

    def after_epoch(self):
        self.save_ckpt(ckpt_name="latest")

        if (self.epoch + 1) % self.exp.eval_interval == 0:
            all_reduce_norm(self.model)
            self.evaluate_and_save_model() # Todo need to remove it

    def before_iter(self):
        pass

    def after_iter(self):
        """
        Log training metrics after each iteration
        """
        # Only log on certain iterations
        if (self.iter + 1) % self.exp.print_interval == 0:
            # Convert to average metrics
            avg_metrics = self.meter.get_filtered_meter(filter_key='loss')
            
            # Get current iteration and learning rate
            current_iter = self.epoch * self.max_iter + self.iter + 1
            # lr = self.meter.get_meter("lr")
            # print(f'warninnnnn')
            # fkae_lr = 0
            lr = self.optimizer.param_groups[0]["lr"]
            # Log to console
            if self.rank == 0:
                # Build metrics message based on available metrics
                msg = "Epoch[{}], Iter[{}], lr:{:.6f}, ".format(
                    self.epoch + 1, current_iter, lr
                )
                
                # Add common metrics
                # import pdb; pdb.set_trace();
                msg += "human_total_loss:{:.3f}, ".format(avg_metrics.get("human_total_loss", 0).avg)
                msg += "face_total_loss:{:.3f}, ".format(avg_metrics.get("face_total_loss", 0).avg)
                msg += "object_total_loss:{:.3f}, ".format(avg_metrics.get("object_total_loss", 0).avg)
                
                logger.info(msg)
                
                # Log to TensorBoard if enabled
                for loss_key in avg_metrics.keys():
                    task_name = loss_key.split("_")[0] # human, object, face
                    self.tblogger.add_scalar(f"train/{task_name}/{loss_key}", avg_metrics.get(loss_key, 0).avg, current_iter)
                    
        
        # # Save and evaluate model at specified intervals
        # if (self.iter + 1) % self.exp.eval_interval == 0:
        #     self.evaluate_and_save_model()

    @property
    def progress_in_iter(self):
        return self.epoch * self.max_iter + self.iter

    def resume_train(self, model):
        if self.args.resume:
            logger.info("resume training")
            if self.args.ckpt is None:
                ckpt_file = os.path.join(self.file_name, "latest" + "_ckpt.pth")
            else:
                ckpt_file = self.args.ckpt

            ckpt = torch.load(ckpt_file, map_location=self.device)
            # resume the model/optimizer state dict
            model.load_state_dict(ckpt["model"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            # resume the training states variables
            start_epoch = (
                self.args.start_epoch - 1
                if self.args.start_epoch is not None
                else ckpt["start_epoch"]
            )
            self.start_epoch = start_epoch
            logger.info(
                "loaded checkpoint '{}' (epoch {})".format(
                    self.args.resume, self.start_epoch
                )
            )  # noqa
        else:
            if self.args.ckpt is not None:
                logger.info("loading checkpoint for fine tuning")
                ckpt_file = self.args.ckpt
                ckpt = torch.load(ckpt_file, map_location=self.device)["model"]
                model = load_ckpt(model, ckpt)
            self.start_epoch = 0
        return model

    def evaluate_and_save_model(self):
        # Tododevu
        # Do evaluation here
        
        # Save model checkpoint
        synchronize()
        
        self.save_ckpt(ckpt_name="latest")

    def save_ckpt(self, ckpt_name, update_best_ckpt=False):
        if self.rank == 0:
            save_model = self.ema_model.ema if self.use_model_ema else self.model
            logger.info("Save weights to {}".format(self.file_name))
            ckpt_state = {
                "start_epoch": self.epoch + 1,
                "model": save_model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }
            save_checkpoint(
                ckpt_state,
                update_best_ckpt,
                self.file_name,
                ckpt_name,
            )