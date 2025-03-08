python -m yolox.tools.train \
    -n yolox_s_human_pose_ti_lite \
    --train_ann "person_keypoints_val2017.json"\
    --task human_pose \
    --dataset coco_kpts \
    -d 1 \
    -b 8 \
    --fp16 \