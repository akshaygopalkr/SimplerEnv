#!/bin/bash
checkpoint_path=$1


python simpler_env/move_near_variant_agg.py --policy-model openvla --ckpt-path ${checkpoint_path} \
    --robot google_robot_static \
    --env-name MoveNearGoogleInScene-v0 \
    --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
    --scene-name google_pick_coke_can_1_v4 \
    --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 -0.09 -0.09 1
