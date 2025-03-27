#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os
from yolox.exp import BaseExp
import torch
import torch.distributed as dist
import torch.nn as nn
import numpy as np

    
class Exp(BaseExp):
    def __init__(self):
        super(BaseExp, self).__init__()
        self.seed = None
        ## Config of yolox Dat
        # self.depth = 0.66
        # self.width = 0.25
        
        ## Default config of yolox-s version
        self.depth = 0.33
        self.width = 0.5
        
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.act = "relu"
        self.input_size = (640, 640)
        self.data_num_workers = 4
        
        
        # dododevu
        self.warmup_epochs = 0
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
        self.eval_interval = 1
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
    
    def using_backbone_pretrained_coco(self, backbone, ckpt="pretrained_models/yolox-s-ti-lite_39p1_57p9_checkpoint.pth"):
        """
        Load the pretrained model from YOLOX with COCO dataset.
        """
        ckpt = torch.load(ckpt, map_location="cpu")
        state_dict = ckpt["model"]
        new_state_dict = {}
        # Get only the backbone state dict
        # Because the checkpoint there is backbone.backbone . So will remove the first backbone.
        for k, v in state_dict.items():
            if "backbone" in k:
                new_k = k[9:] # remove "backbone."
                new_state_dict[new_k] = v
        
        
        backbone.load_state_dict(new_state_dict)
        
        return backbone
    

    
    def get_model(self):
        from yolox.models import YOLOXOMS, YOLOPAFPN, YOLOXHeadKPTS, YOLOFaceKPTSHead,YOLOXHead as YOLOXObjectHead
        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03
        
        
        if getattr(self, "model", None) is None:
            in_channels = [256, 512, 1024]
            backbone = YOLOPAFPN(self.depth, self.width, act=self.act, in_channels=in_channels, conv_focus=True)
            try:
                backbone = self.using_backbone_pretrained_coco(backbone)
                print("########################### Using pretrained model backbone from COCO")
            except Exception as e:
                print(f"Error when loading pretrained model from coco: {e}")
            
            head_human = YOLOXHeadKPTS(self.human_params["num_classes"], self.width, in_channels=in_channels, act=self.act, num_kpts=self.human_params["num_kpts"], default_sigmas=self.human_params["default_sigmas"])
            head_face = YOLOFaceKPTSHead(self.face_params["num_classes"], self.width, in_channels=in_channels, act=self.act, num_kpts=self.face_params["num_kpts"], default_sigmas=self.face_params["default_sigmas"])
            head_object = YOLOXObjectHead(self.object_params["num_classes"], act=self.act,width=self.width, in_channels=in_channels)
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
            else:
                raise ValueError(f"Unsupported training strategy: {self.train_strategy}")
            

    def get_eval_loader(self, batch_size, is_distributed, testdev=False, legacy=False):
        pass 

    
    def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False):
        pass



if __name__ == "__main__":
    # Play with load pretrained model for only the backbone
    in_channels = [256, 512, 1024]
    from yolox.models import YOLOPAFPN
    depth = 0.33
    width = 0.50
    backbone = YOLOPAFPN(depth, width, in_channels=in_channels, conv_focus=True)
    
    
    ckpt = torch.load("pretrained_models/yolox-s-ti-lite_39p1_57p9_checkpoint.pth", map_location="cpu")
    state_dict = ckpt["model"]
    new_state_dict = {}
    # Get only the backbone state dict
    # Because the checkpoint there is backbone.backbone . So will remove the first backbone.
    for k, v in state_dict.items():
        if "backbone" in k:
            new_k = k[9:] # remove "backbone."
            new_state_dict[new_k] = v
    
        
    
    # import ipdb; ipdb.set_trace();
    
    backbone.load_state_dict(new_state_dict)
    
    
    
    print("Done")
    
    
    