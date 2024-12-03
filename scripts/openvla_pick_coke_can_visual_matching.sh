#!/bin/bash
checkpoint_path=$1


python simpler_env/pick_coke_can_visual_matching.py --policy-model openvla --ckpt-path ${checkpoint_path} \
    --robot google_robot_static \
    --env-name GraspSingleOpenedCokeCanInScene-v0 \
    --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
    --scene-name google_pick_coke_can_1_v4 \
    --rgb-overlay-path ./ManiSkill2_real2sim/data/real_inpainting/google_coke_can_real_eval_1.png \
    --robot-init-x 0.35 0.35 1 --robot-init-y 0.20 0.20 1 --obj-init-x -0.35 -0.12 5 --obj-init-y -0.02 0.42 5 \
    --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1
