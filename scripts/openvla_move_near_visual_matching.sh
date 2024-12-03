#!/bin/bash
checkpoint_path=$1

env_name=MoveNearGoogleBakedTexInScene-v0
scene_name=google_pick_coke_can_1_v4
rgb_overlay_path=./ManiSkill2_real2sim/data/real_inpainting/google_move_near_real_eval_1.png


python simpler_env/move_near_visual_matching.py --policy-model openvla --ckpt-path ${checkpoint_path} \
    --robot google_robot_static \
    --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
    --env-name ${env_name} --scene-name ${scene_name} \
    --rgb-overlay-path ${rgb_overlay_path} \
    --robot-init-x 0.35 0.35 1 --robot-init-y 0.21 0.21 1 --obj-variation-mode episode --obj-episode-range 0 60 \
    --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 -0.09 -0.09 1 \
    --additional-env-save-tags baked_except_bpb_orange; # google_move_near_real_eval_1.png
