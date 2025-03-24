#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
import torch.nn as nn

from .yolo_head import YOLOXHead as YOLOXObjectHead
from .yolo_kpts_head import YOLOXHeadKPTS
from .yolo_face_kpts_head import  YOLOXHeadKPTS as YOLOFaceKPTSHead
from .yolo_pafpn import YOLOPAFPN


class YOLOX(nn.Module):
    """
    YOLOX model module with multiple heads support.
    The module can handle multiple detection heads and selectively train them.
    """

    def __init__(self, backbone=None, head_dict=None):
        super().__init__()
        # Default backbone
        if backbone is None:
            backbone = YOLOPAFPN()
        
        # Default head_dict with both human and object heads
        if head_dict is None:
            head_dict = {
                "human": YOLOXHeadKPTS(1, default_sigmas=False),  # Human head with keypoints
                "face" : YOLOFaceKPTSHead(1, default_sigmas=False),  # Face head with keypoints
                "object": YOLOXObjectHead(80)  # Object detection head
            }
        
        
        self.backbone = backbone
        self.head_dict = nn.ModuleDict(head_dict)

    def forward(self, x, targets=None, train_heads=["human", "face", "object"]):
        # fpn output content features of [dark3, dark4, dark5]
        fpn_outs = self.backbone(x)
        
        outputs = {}
        if self.training:
            assert targets is not None

            if "human" in train_heads:
                # Only train human head
                output_head_human = self.head_dict["human"](fpn_outs, targets, x)
                loss, iou_loss, conf_loss, cls_loss, l1_loss, kpts_loss, kpts_vis_loss, loss_l1_kpts, num_fg = output_head_human
                outputs["human"] = {"total_loss": loss, "iou_loss": iou_loss, "l1_loss": l1_loss, "conf_loss": conf_loss, "cls_loss": cls_loss, "kpts_loss": kpts_loss, "kpts_vis_loss": kpts_vis_loss, "l1_loss_kpts": loss_l1_kpts, "num_fg": num_fg}
            
            if "face" in train_heads:
                # Only train face head
                output_head_face = self.head_dict["face"](fpn_outs, targets, x)
                loss, iou_loss, conf_loss, cls_loss, l1_loss, kpts_loss, kpts_vis_loss, loss_l1_kpts, num_fg = output_head_face
                outputs["face"] = {"total_loss": loss, "iou_loss": iou_loss, "l1_loss": l1_loss, "conf_loss": conf_loss, "cls_loss": cls_loss, "kpts_loss": kpts_loss, "kpts_vis_loss": kpts_vis_loss, "l1_loss_kpts": loss_l1_kpts, "num_fg": num_fg}
            
            if "object" in train_heads:
                # Only train object head
                output_head_object = self.head_dict["object"](fpn_outs, targets, x)
                loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = output_head_object
                outputs["object"] = {"total_loss": loss, "iou_loss": iou_loss, "l1_loss": l1_loss, "conf_loss": conf_loss, "cls_loss": cls_loss, "num_fg": num_fg}
            
            
        
        else:
            # During inference, process all heads
            for head_name, head in self.head_dict.items():
                outputs[head_name] = head(fpn_outs)
            
            
            
            
        return outputs    
    
if __name__ == "__main__":
    import torch
    # Test YOLOX model
    model = YOLOX()
    model.eval()
    x = torch.rand(1, 3, 640, 640)
    with torch.no_grad():
        out = model(x)

    # import pdb; pdb.set_trace();
    print(f'Shape human output: {out["human"].shape}') # 1x8400x60. 60=3*17(n_keypoints) + 4(bounding box) + 1(objectness) + 1(class) + 3 (biometry) 
    print(f'Shape object output: {out["object"].shape}') # 1x8400x85. 85=4(bounding box) + 1(objectness) + 80(class)
    print("Finished testing YOLOX model!")    
    
    
            
