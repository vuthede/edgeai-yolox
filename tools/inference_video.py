import cv2
import torch
import numpy as np

def init_model(ckpt_path):
    from yolox.models import YOLOXOMS, YOLOPAFPN, YOLOXHeadKPTS, YOLOXHead as YOLOXObjectHead
    human_params = {
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
    face_params = {
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
    object_params = {
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
    depth = 0.33
    width = 0.50
    in_channels = [256, 512, 1024]
    act = "relu"
    backbone = YOLOPAFPN(depth, width, in_channels=in_channels, act=act,conv_focus=True)
    head_human = YOLOXHeadKPTS(human_params["num_classes"], width, in_channels=in_channels, act=act, num_kpts=human_params["num_kpts"], default_sigmas=human_params["default_sigmas"])
    head_face = YOLOXHeadKPTS(face_params["num_classes"], width, in_channels=in_channels, act=act, num_kpts=face_params["num_kpts"], default_sigmas=face_params["default_sigmas"])
    head_object = YOLOXObjectHead(object_params["num_classes"], width=width, in_channels=in_channels)
    model = YOLOXOMS(backbone, head_dict={"human": head_human, "face": head_face, "object": head_object})
    print("Init model!!!")
    
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    print(f"Load checkpoint from {ckpt_path}")
    
    return model

def preprocess_frame(frame, target_size=(640, 640)):
    """Preprocess the frame: resize with padding, normalize, and convert to tensor."""
    # Convert BGR (OpenCV) to RGB
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Get original dimensions
    h, w = img.shape[:2]
    target_h, target_w = target_size
    
    # Calculate scaling factor to maintain aspect ratio
    scale = min(target_h / h, target_w / w)
    new_h, new_w = int(h * scale), int(w * scale)
    
    # Resize image
    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Create a blank canvas (black padding)
    padded_img = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    
    # Place resized image in the center
    top = (target_h - new_h) // 2
    left = (target_w - new_w) // 2
    padded_img[top:top + new_h, left:left + new_w] = resized_img
    
    # Normalize to [0, 1] and apply YOLOX standard mean/std
    img_tensor = torch.from_numpy(padded_img).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3)
    img_tensor = (img_tensor - mean) / std
    
    # Permute to (C, H, W) and add batch dimension
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
    
    return img_tensor, (scale, top, left)

def postprocess_outputs(outputs, conf_thres=0.1, img_size=(640, 640), scale_info=None):
    """Post-process YOLOX outputs: filter by confidence, adjust coordinates."""
    human_out, face_out, object_out = outputs["human"], outputs["face"], outputs["object"]
    orig_scale, top_pad, left_pad = scale_info
    
    def decode_boxes(pred, num_classes, conf_thres, num_kpts=0):
        # pred: [B, 8400, N] where N = 4 (box) + 1 (obj) + num_classes + num_kpts*3
        boxes = pred[..., :4]  # [B, 8400, 4]
        obj_score = pred[..., 4:5]  # [B, 8400, 1]
        cls_score = pred[..., 5:5 + num_classes]  # [B, 8400, num_classes]
        kpts = pred[..., 5 + num_classes:5 + num_classes+num_kpts*3] if num_kpts > 0 else None  # [B, 8400, num_kpts*3]
        
        # import pdb; pdb.set_trace()
        
        # Compute confidence: objectness * max class score
        # scores = obj_score * cls_score.max(dim=-1, keepdim=True)[0]
        scores = cls_score.max(dim=-1, keepdim=True)[0]
        
        mask = scores.squeeze(-1) > conf_thres
        
        # Filter predictions
        filtered_boxes = boxes[mask]
        filtered_scores = scores[mask]
        filtered_cls = cls_score[mask].argmax(dim=-1)
        filtered_kpts = kpts[mask] if kpts is not None else None
        
        print(f'Box result shape :{filtered_boxes.shape}')
        
        # Adjust box coordinates: YOLOX outputs center_x, center_y, w, h
        box_result = filtered_boxes.clone()
        box_result[:, 0] -= filtered_boxes[:, 2] / 2  # x1 = center_x - w/2
        box_result[:, 1] -= filtered_boxes[:, 3] / 2  # y1 = center_y - h/2
        box_result[:, 2] = filtered_boxes[:, 0] + filtered_boxes[:, 2]/2      # x2 = x1 + w
        box_result[:, 3] = filtered_boxes[:, 1] + filtered_boxes[:, 3]/2    # y2 = y1 + h
        
        # Scale back to original image coordinates
        box_result -= torch.tensor([left_pad, top_pad, left_pad, top_pad], device=box_result.device)
        box_result /= orig_scale
        if filtered_kpts is not None:
            filtered_kpts[:, 0::3] -= left_pad  # x coordinates
            filtered_kpts[:, 1::3] -= top_pad   # y coordinates
            filtered_kpts[:, 0::3] /= orig_scale
            filtered_kpts[:, 1::3] /= orig_scale
        
        return box_result, filtered_scores, filtered_cls, filtered_kpts
    
    # Process each head
    # import pdb; pdb.set_trace();
    human_boxes, human_scores, human_cls, human_kpts = decode_boxes(human_out, 1, conf_thres,17)
    face_boxes, face_scores, face_cls, face_kpts = decode_boxes(face_out, 1, 0.1, 5)
    object_boxes, object_scores, object_cls, _ = decode_boxes(object_out, 4 , 0.1,0)
    
    return {
        "human": (human_boxes, human_scores, human_cls, human_kpts),
        "face": (face_boxes, face_scores, face_cls, face_kpts),
        "object": (object_boxes, object_scores, object_cls, None)
    }

def visualize_frame(frame, detections):
    """Draw detections on the frame."""
    colors = {
        "human": (0, 255, 0),    # Green
        "face": (255, 0, 0),     # Blue
        "object": (0, 0, 255)    # Red
    }
    object_labels = ["phone", "ciga", "food", "beverage"]
    
    for head, (boxes, scores, cls, kpts) in detections.items():
        for i in range(len(boxes)):
            x1, y1, x2, y2 = map(int, boxes[i])
            score = scores[i].item()
            label = "human" if head == "human" else "face" if head == "face" else object_labels[cls[i].item()]
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), colors[head], 2)
            cv2.putText(frame, f"{label} {score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[head], 2)
            
            # Draw keypoints if available
            if kpts is not None:
                for j in range(0, len(kpts[i]), 3):
                    x, y, conf = kpts[i][j:j+3]
                    # if conf > 0.5:
                    cv2.circle(frame, (int(x), int(y)), 3, colors[head], -1)
    
    return frame

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = init_model(ckpt_path="/home/vuthede/OMS_AI/.github/edgeai-yolox/YOLOX_outputs/train_s_oms/latest_ckpt.pth")
    model.to(device)
    model.eval()
    
    video_path = "/media/vuthede/Lexar/Ficosa_videos/CMC_P1_People.mp4"  # Replace with your video pat
    video_path = "/media/vuthede/Lexar/Ficosa_videos/DFC_P1_Smoking.mp4"
    # video_path = "/home/vuthede/Videos/example_phone_data.webm"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()
    
    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
            # frame = cv2.resize(frame, (640, 640))
            # frame = cv2.imread("/home/vuthede/fiftyone/coco-2017/validation/val2017/000000000785.jpg")
            # frame = cv2.flip(frame, 1)
            
            # frame = cv2.resize(frame, (640, 640))
            if not ret:
                print("End of video.")
                break
            
            # Preprocess frame
            input_tensor, scale_info = preprocess_frame(frame, target_size=(640, 640))
            input_tensor = input_tensor.to(device)
            
            # Model inference
            outputs = model(input_tensor)
            # Outputs: {"human": [B, 8400, 60], "face": [B, 8400, 24], "object": [B, 8400, 9]}
            
            # Post-process outputs
            detections = postprocess_outputs(outputs, conf_thres=0.5, img_size=(640, 640), scale_info=scale_info)
            
            # Visualize detections
            frame_with_dets = visualize_frame(frame, detections)
            
            # Display frame
            cv2.imshow("YOLOX Detections", frame_with_dets)
            if cv2.waitKey(0) & 0xFF == ord("q"):
                break
    
    cap.release()
    cv2.destroyAllWindows()