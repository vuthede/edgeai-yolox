#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os
from yolox.exp import BaseExp
import torch
import torch.distributed as dist
import torch.nn as nn
import numpy as np

class CombinedDataset(torch.utils.data.Dataset):
    """
    Dataset that combines human and object datasets with metadata to track source.
    """
    
    def __init__(self, human_dataset, object_dataset):
        self.human_dataset = human_dataset
        self.object_dataset = object_dataset
        self.human_size = len(human_dataset)
        self.object_size = len(object_dataset)
        self.total_size = self.human_size + self.object_size
    
    def __len__(self):
        return self.total_size
    
    def __getitem__(self, idx):
        if idx < self.human_size:
            # Get sample from human dataset
            data = self.human_dataset[idx]
            data["dataset_type"] = "human"
        else:
            # Get sample from object dataset
            data = self.object_dataset[idx - self.human_size]
            data["dataset_type"] = "object"
        
        return data

class ProportionalSampler(torch.utils.data.Sampler):
    """
    Sampler that samples from human and object datasets proportionally.
    """
    
    def __init__(self, human_dataset_size, object_dataset_size, human_ratio=0.5, seed=0):
        self.human_dataset_size = human_dataset_size
        self.object_dataset_size = object_dataset_size
        self.human_ratio = human_ratio
        self.object_ratio = 1.0 - human_ratio
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        
        # Calculate max epochs to balance datasets
        self.total_samples = max(
            int(human_dataset_size / human_ratio) if human_ratio > 0 else 0,
            int(object_dataset_size / self.object_ratio) if self.object_ratio > 0 else 0
        )
        
        # Calculate samples per epoch
        self.human_samples_per_epoch = int(self.total_samples * human_ratio)
        self.object_samples_per_epoch = int(self.total_samples * self.object_ratio)
    
    def __iter__(self):
        # Generate human indices
        human_indices = self.rng.randint(0, self.human_dataset_size, size=self.human_samples_per_epoch)
        
        # Generate object indices (shifted to account for human dataset size)
        object_indices = self.rng.randint(0, self.object_dataset_size, size=self.object_samples_per_epoch)
        object_indices = object_indices + self.human_dataset_size
        
        # Combine and shuffle
        all_indices = np.concatenate([human_indices, object_indices])
        self.rng.shuffle(all_indices)
        
        return iter(all_indices.tolist())
    
    def __len__(self):
        return self.total_samples

class ProportionalDataset(torch.utils.data.Dataset):
    """
    Dataset that works with ProportionalSampler to provide combined access to both datasets.
    """
    
    def __init__(self, human_dataset, object_dataset):
        self.human_dataset = human_dataset
        self.object_dataset = object_dataset
        self.human_size = len(human_dataset)
        self.object_size = len(object_dataset)
        self.total_size = self.human_size + self.object_size
    
    def __len__(self):
        return self.total_size
    
    def __getitem__(self, idx):
        if idx < self.human_size:
            # Get sample from human dataset
            data = self.human_dataset[idx]
            data["dataset_type"] = "human"
        else:
            # Get sample from object dataset
            data = self.object_dataset[idx - self.human_size]
            data["dataset_type"] = "object"
        
        return data
    
class Exp(BaseExp):
    def __init__(self):
        super(BaseExp, self).__init__()
        self.seed = None
        self.depth = 0.66
        self.width = 0.25
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.act = "relu"
        self.input_size = (640, 640)
        self.data_num_workers = 4
        
        
        self.warmup_epochs = 5
        self.max_epoch = 300
        self.warmup_lr = 0
        self.basic_lr_per_img = 0.01 / 64.0
        self.scheduler = "yoloxwarmcos"
        self.no_aug_epochs = 15
        self.min_lr_ratio = 0.05
        self.ema = False
        self.weight_decay = 5e-4
        self.momentum = 0.9
        self.print_interval = 10
        self.eval_interval = 10
        self.test_size = (640, 640)
        self.test_conf = 0.01
        self.nmsthre = 0.65
        self.output_dir = "./YOLOX_outputs"

        # Human branch
        self.human_params = {
            "num_classes": 1,
            "num_kpts": 17,
            "default_sigmas": False,
            "data_dir": "/home/vuthede/fiftyone/coco-2017/validation",
            "train_ann": "person_keypoints_val2017_with_random_bimometry.json",  
            "name": "val2017",
            "flip_prob": 0.5,
            "hsv_prob": 0.5,
            "degrees": 10.0,
            "translate": 0.1,
            "mosaic_scale": [0.8, 1.2],
            "mixup_scale": [0.5, 1.5],
            "shear": 2.0,
            "enable_mixup": True,
            "mosaic_prob": 1.0,
            "mixup_prob": 1.0,
        }      
        
        # face branch
        self.face_params = {
            "num_classes": 1,
            "num_kpts": 5,
            "default_sigmas": False,
            "data_dir": "/media/vuthede/Lexar/data/face_detection/widerface/train",
            "train_ann": "annotations_coco.json",  
            "name": "images",
            "flip_prob": 0.5,
            "hsv_prob": 0.5,
            "degrees": 10.0,
            "translate": 0.1,
            "mosaic_scale": [0.8, 1.2],
            "mixup_scale": [0.5, 1.5],
            "shear": 2.0,
            "enable_mixup": True,
            "mosaic_prob": 1.0,
            "mixup_prob": 1.0,
        }    
    
        # Object branch
        # Object branch
        self.object_params = {
            "num_classes": 4,  # phone, ciga, food, beverage
            "data_dir": "/media/vuthede/Lexar/phone_food_smoking_sampling_full",
            "train_ann": "sampling_20k.json",
            "name": "images",
            "flip_prob": 0.5,
            "hsv_prob": 0.5,
            "degrees": 10.0,
            "translate": 0.1,
            "mosaic_scale": [0.8, 1.2],
            "mixup_scale": [0.5, 1.5],
            "shear": 2.0,
            "enable_mixup": True,
            "mosaic_prob": 1.0,
            "mixup_prob": 1.0,
        }
        
    def get_optimizer(self, batch_size):
        if "optimizer" not in self.__dict__:
            if self.warmup_epochs > 0:
                lr = self.warmup_lr
            else:
                lr = self.basic_lr_per_img * batch_size

            pg0, pg1, pg2 = [], [], []  # optimizer parameter groups

            for k, v in self.model.named_modules():
                if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                    pg2.append(v.bias)  # biases
                if isinstance(v, nn.BatchNorm2d) or "bn" in k:
                    pg0.append(v.weight)  # no decay
                elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                    pg1.append(v.weight)  # apply decay

            optimizer = torch.optim.SGD(
                pg0, lr=lr, momentum=self.momentum, nesterov=True
            )
            optimizer.add_param_group(
                {"params": pg1, "weight_decay": self.weight_decay}
            )  # add pg1 with weight_decay
            optimizer.add_param_group({"params": pg2})
            self.optimizer = optimizer

        return self.optimizer

    def get_lr_scheduler(self, lr, iters_per_epoch):
        from yolox.utils import LRScheduler

        scheduler = LRScheduler(
            self.scheduler,
            lr,
            iters_per_epoch,
            self.max_epoch,
            warmup_epochs=self.warmup_epochs,
            warmup_lr_start=self.warmup_lr,
            no_aug_epochs=self.no_aug_epochs,
            min_lr_ratio=self.min_lr_ratio,
        )
        return scheduler
    
    def eval(self, model, evaluator, is_distributed, half=False):
        pass

        

        
    # override the default dataset
    def preprocess_human(self, inputs, targets, tsize):
        scale_y = tsize[0] / self.input_size[0]
        scale_x = tsize[1] / self.input_size[1]
        if scale_x != 1 or scale_y != 1:
            inputs = nn.functional.interpolate(
                inputs, size=tsize, mode="bilinear", align_corners=False
            )
            targets['target'][..., 1::2] = targets['target'][..., 1::2] * scale_x
            targets['target'][..., 2::2] = targets['target'][..., 2::2] * scale_y
        return inputs, targets
    
    def preprocess_face(self, inputs, targets, tsize):
        scale_y = tsize[0] / self.input_size[0]
        scale_x = tsize[1] / self.input_size[1]
        if scale_x != 1 or scale_y != 1:
            inputs = nn.functional.interpolate(
                inputs, size=tsize, mode="bilinear", align_corners=False
            )
            targets['target'][..., 1::2] = targets['target'][..., 1::2] * scale_x
            targets['target'][..., 2::2] = targets['target'][..., 2::2] * scale_y
        return inputs, targets
    
    # Preprocess objects
    def preprocess_object(self, inputs, targets, tsize):
        # assert type(targets) == dict, f'''targets should be dict, got {type(targets)}'''
        
        scale_y = tsize[0] / self.input_size[0]
        scale_x = tsize[1] / self.input_size[1]
        if scale_x != 1 or scale_y != 1:
            inputs = nn.functional.interpolate(
                inputs, size=tsize, mode="bilinear", align_corners=False
            )
            targets[..., 1::2] = targets[..., 1::2] * scale_x
            targets[..., 2::2] = targets[..., 2::2] * scale_y
        return inputs, targets
        
    def get_model(self):
        from yolox.models import YOLOXOMS, YOLOPAFPN, YOLOXHeadKPTS, YOLOFaceKPTSHead,YOLOXHead as YOLOXObjectHead
        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03
        
        
        if getattr(self, "model", None) is None:
            in_channels = [256, 512, 1024]
            backbone = YOLOPAFPN(self.depth, self.width, act=self.act, in_channels=in_channels)
            head_human = YOLOXHeadKPTS(self.human_params["num_classes"], self.width, in_channels=in_channels, act=self.act, num_kpts=self.human_params["num_kpts"], default_sigmas=self.human_params["default_sigmas"])
            head_face = YOLOXHeadKPTS(self.face_params["num_classes"], self.width, in_channels=in_channels, act=self.act, num_kpts=self.face_params["num_kpts"], default_sigmas=self.face_params["default_sigmas"])
            head_object = YOLOXObjectHead(self.object_params["num_classes"], width=self.width, in_channels=in_channels)
            self.model = YOLOXOMS(backbone,  head_dict={"human": head_human, "face":head_face,"object": head_object})
        
        self.model.apply(init_yolo)
        for head_key in self.model.head_dict.keys():
            self.model.head_dict[head_key].initialize_biases(1e-2)

        return self.model
    
    def _init_human_dataset(self, batch_size, is_distributed, no_aug=False, cache_img=False):
        from yolox.data import  COCOKPTSDataset, COCOFaceKPTSDataset, TrainTransformKpts, MosaicDetectionKpts
       
        dataset = COCOKPTSDataset(
            data_dir=self.human_params["data_dir"],
            json_file=self.human_params["train_ann"],
            num_kpts=self.human_params["num_kpts"],
            name=self.human_params["name"],
            img_size=self.input_size,
            preproc=TrainTransformKpts(
                max_labels=50,
                flip_prob=self.human_params["flip_prob"],
                hsv_prob=self.human_params["hsv_prob"],
                num_kpts=self.human_params["num_kpts"]),
            cache=cache_img,
        )

        dataset = MosaicDetectionKpts(
            dataset,
            mosaic=not no_aug,
            img_size=self.input_size,
            preproc=TrainTransformKpts(
                max_labels=120,
                flip_prob=self.human_params["flip_prob"],
                hsv_prob=self.human_params["hsv_prob"],
                object_pose=False,
                human_pose=True,
                flip_index=dataset.flip_index,
                num_kpts=self.human_params["num_kpts"],
            ),
            num_kpts=self.human_params["num_kpts"],
            degrees=self.human_params["degrees"],
            translate=self.human_params["translate"],
            mosaic_scale=self.human_params["mosaic_scale"],
            mixup_scale=self.human_params["mixup_scale"],
            shear=self.human_params["shear"],
            enable_mixup=self.human_params["enable_mixup"],
            mosaic_prob=self.human_params["mosaic_prob"],
            mixup_prob=self.human_params["mixup_prob"],
        )
        
        return dataset
    
    def _init_face_dataset(self, batch_size, is_distributed, no_aug=False, cache_img=False):
        from yolox.data import  COCOFaceKPTSDataset, TrainTransformKpts, MosaicDetectionKpts
       
        dataset = COCOFaceKPTSDataset(
            data_dir=self.face_params["data_dir"],
            json_file=self.face_params["train_ann"],
            num_kpts=self.face_params["num_kpts"],
            name=self.face_params["name"],
            img_size=self.input_size,
            preproc=TrainTransformKpts(
                max_labels=50,
                flip_prob=self.face_params["flip_prob"],
                hsv_prob=self.face_params["hsv_prob"],
                num_kpts=self.face_params["num_kpts"]),
            cache=cache_img,
        )

        dataset = MosaicDetectionKpts(
            dataset,
            mosaic=not no_aug,
            img_size=self.input_size,
            preproc=TrainTransformKpts(
                max_labels=120,
                flip_prob=self.face_params["flip_prob"],
                hsv_prob=self.face_params["hsv_prob"],
                object_pose=False,
                human_pose=True,
                flip_index=dataset.flip_index,
                num_kpts=self.face_params["num_kpts"],
            ),
            num_kpts=self.face_params["num_kpts"],
            degrees=self.face_params["degrees"],
            translate=self.face_params["translate"],
            mosaic_scale=self.face_params["mosaic_scale"],
            mixup_scale=self.face_params["mixup_scale"],
            shear=self.face_params["shear"],
            enable_mixup=self.face_params["enable_mixup"],
            mosaic_prob=self.face_params["mosaic_prob"],
            mixup_prob=self.face_params["mixup_prob"],
        )
        
        return dataset
    
    
    def _init_object_dataset(self, batch_size, is_distributed, no_aug=False, cache_img=False):
        from yolox.data import  COCODataset, TrainTransform, MosaicDetection
        dataset = COCODataset(
            data_dir=self.object_params["data_dir"],
            json_file=self.object_params["train_ann"],
            name=self.object_params["name"],
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=50, 
                flip_prob=self.object_params["flip_prob"], 
                hsv_prob=self.object_params["hsv_prob"]
            ),
            cache=cache_img,
        )

        dataset = MosaicDetection(
            dataset,
            mosaic=not no_aug,
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=120, 
                flip_prob=self.object_params["flip_prob"], 
                hsv_prob=self.object_params["hsv_prob"]
            ),
            degrees=self.object_params["degrees"],
            translate=self.object_params["translate"],
            mosaic_scale=self.object_params["mosaic_scale"],
            mixup_scale=self.object_params["mixup_scale"],
            shear=self.object_params["shear"],
            enable_mixup=self.object_params["enable_mixup"],
            mosaic_prob=self.object_params["mosaic_prob"],
            mixup_prob=self.object_params["mixup_prob"],
        )

        self.dataset = dataset
        return dataset

    def _create_alternating_dataloader(self, human_dataset, face_dataset, object_dataset, batch_size, is_distributed, no_aug):
        """
        Strategy 1: Create dataloaders that alternate between human and face and object batches.
        Returns a dict with separate dataloaders.
        """
        from yolox.data import (
            YoloBatchSampler,
            DataLoader,
            InfiniteSampler,
            worker_init_reset_seed,
        )
        
        sampler_h = InfiniteSampler(len(human_dataset), seed=self.seed if self.seed else 0)
        sampler_f = InfiniteSampler(len(face_dataset), seed=self.seed if self.seed else 0)
        sampler_o = InfiniteSampler(len(object_dataset), seed=self.seed if self.seed else 0)
        
        batch_sampler_h = YoloBatchSampler(
            sampler=sampler_h,
            batch_size=batch_size,
            drop_last=False,
            mosaic=not no_aug,
        )
        
        batch_sampler_f = YoloBatchSampler(
            sampler=sampler_f,
            batch_size=batch_size,
            drop_last=False,
            mosaic=not no_aug
        )
        
        batch_sampler_o = YoloBatchSampler(
            sampler=sampler_o,
            batch_size=batch_size,
            drop_last=False,
            mosaic=not no_aug,
        )
        
        dataloader_human = DataLoader(
            human_dataset,
            batch_sampler=batch_sampler_h,
            num_workers=self.data_num_workers,
            pin_memory=True,
            worker_init_fn=worker_init_reset_seed,
        )
        
        dataloader_face = DataLoader(
            face_dataset,
            batch_sampler=batch_sampler_f,
            num_workers=self.data_num_workers,
            pin_memory=True,
            worker_init_fn=worker_init_reset_seed,
        )
        
        dataloader_object = DataLoader(
            object_dataset,
            batch_sampler=batch_sampler_o,
            num_workers=self.data_num_workers,
            pin_memory=True,
            worker_init_fn=worker_init_reset_seed,
        )
    
        return {
            "human": dataloader_human,
            "face": dataloader_face,
            "object": dataloader_object,
            "strategy": "alternating"
        }

    def _create_combined_dataloader(self, human_dataset, object_dataset, batch_size, is_distributed):
        """
        Strategy 2: Create a dataloader using a combined dataset.
        Data from both sources will be mixed within batches.
        """
        from yolox.data import (
            DataLoader,
            InfiniteSampler,
            YoloBatchSampler,
            worker_init_reset_seed,
        )
        
        # Create a dataset that combines both - will need a custom CombinedDataset class
        combined_dataset = CombinedDataset(human_dataset, object_dataset)
        
        sampler = InfiniteSampler(len(combined_dataset), seed=self.seed if self.seed else 0)
        
        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            mosaic=not self.no_aug,
        )
        
        dataloader = DataLoader(
            combined_dataset,
            batch_sampler=batch_sampler,
            num_workers=self.data_num_workers,
            pin_memory=True,
            worker_init_fn=worker_init_reset_seed,
        )
        
        return {
            "combined": dataloader,
            "strategy": "combined"
        }

    def _create_proportional_dataloader(self, human_dataset, object_dataset, batch_size, is_distributed, human_ratio=0.5):
        """
        Strategy 3: Create a dataloader that samples from each dataset proportionally
        based on the human_ratio parameter.
        """
        from yolox.data import (
            DataLoader,
            worker_init_reset_seed,
        )
        
        # Create a custom sampler that handles proportional sampling
        sampler = ProportionalSampler(
            human_dataset_size=len(human_dataset),
            object_dataset_size=len(object_dataset),
            human_ratio=human_ratio,
            seed=self.seed if self.seed else 0
        )
        
        # Create a custom dataset that uses the sampler info to return data
        proportional_dataset = ProportionalDataset(human_dataset, object_dataset)
        
        dataloader = DataLoader(
            proportional_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=self.data_num_workers,
            pin_memory=True,
            worker_init_fn=worker_init_reset_seed,
            drop_last=False,
        )
        
        return {
            "proportional": dataloader,
            "strategy": "proportional"
        }
        
    def get_data_loader(self, batch_size, is_distributed, no_aug=False, cache_img=False):
        from yolox.data import (
            COCOKPTSDataset,
            TrainTransform,
            YoloBatchSampler,
            DataLoader,
            InfiniteSampler,
            MosaicDetection,
            worker_init_reset_seed,
        )
        from yolox.utils import (
            wait_for_the_master,
            get_local_rank,
        )
        
        # Configuration for training strategy
        self.train_strategy = "alternating"  # Options: "alternating", "combined", "proportional"
        self.human_ratio = 0.5  # For proportional strategy - ratio of human samples
        
        local_rank = get_local_rank()
        with wait_for_the_master(local_rank):
            human_dataset = self._init_human_dataset(batch_size, is_distributed, no_aug, cache_img)
            face_dataset = self._init_face_dataset(batch_size, is_distributed, no_aug, cache_img)
            object_dataset = self._init_object_dataset(batch_size, is_distributed, no_aug, cache_img)
        
            # Strategy 1: Alternating batches (1 human batch, 1 object batch)
            if self.train_strategy == "alternating":
                return self._create_alternating_dataloader(
                    human_dataset, face_dataset, object_dataset, batch_size, is_distributed, no_aug
                )
            
            # Strategy 2: Combined dataset (mix samples from both datasets)
            elif self.train_strategy == "combined":
                raise ValueError(f"Unsupported training strategy: {self.train_strategy}")
                
                # return self._create_combined_dataloader(
                #     human_dataset, object_dataset, batch_size, is_distributed
                # )
            
            # Strategy 3: Proportional sampling (control ratio between datasets)
            elif self.train_strategy == "proportional":
                raise ValueError(f"Unsupported training strategy: {self.train_strategy}")
                
                # return self._create_proportional_dataloader(
                #     human_dataset, object_dataset, batch_size, is_distributed, self.human_ratio
                # )
            
            else:
                raise ValueError(f"Unsupported training strategy: {self.train_strategy}")
            

    def get_eval_loader(self, batch_size, is_distributed, testdev=False, legacy=False):
        pass 

    
    def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False):
        pass