import cv2
import torch

def init_model(ckpt_path):
    from yolox.models import YOLOXOMS, YOLOPAFPN, YOLOXHeadKPTS, YOLOFaceKPTSHead,YOLOXHead as YOLOXObjectHead
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
    backbone = YOLOPAFPN(depth, width, in_channels=in_channels, conv_focus=True)
    head_human = YOLOXHeadKPTS(human_params["num_classes"], width, in_channels=in_channels, act=act, num_kpts=human_params["num_kpts"], default_sigmas=human_params["default_sigmas"])
    head_face = YOLOXHeadKPTS(face_params["num_classes"], width, in_channels=in_channels, act=act, num_kpts=face_params["num_kpts"], default_sigmas=face_params["default_sigmas"])
    head_object = YOLOXObjectHead(object_params["num_classes"], width=width, in_channels=in_channels)
    model = YOLOXOMS(backbone,  head_dict={"human": head_human, "face":head_face,"object": head_object})
    print(f'Init model!!!')
    
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    
    
    print(f'Load checkpoint from {ckpt_path}')
    
    
    return model


if __name__ == "__main__":
    device = 'cuda'
    model = init_model(ckpt_path="/home/vuthede/OMS_AI/.github/edgeai-yolox/YOLOX_outputs/train_s_oms/latest_ckpt.pth")
    model.to(device)
    model.eval()
    
    # x= torch.rand(1,3,640, 640)
    # x = x.to(device)
    # result = model(x)
    # import pdb; pdb.set_trace()
    
    video_path = ""
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        ## Preprocess frame here. The input into model is 640x640 so need to padding
        # Todo
        
        ## Model inference. Note that result has 3 keys represent for 3 heads: human, face, object
        # human has shape Bx8400x60, (body box has 4, 1 for objectness_score, 1 for class_human_score, kpts has 51 including 17 kpts x3 (x,y,score), 3 for body's biometry)
        # face has shape Bx8400x24, (face box has 4, 1 for objectness_score, 1 for class_face_score, kpts has 25 including 5 kpts x3 (x,y,score)), 3 for face's biometry)
        # object has shape Bx8400x9 (facebox has 4, 1 for objectness_score, 4 for classes_score (phone, ciga, food, beverage))
        # Todo
        
        
        ## Post processing
        # Todo
        
        
        ## Visualization
        
        
        
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()