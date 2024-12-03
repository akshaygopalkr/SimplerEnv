#!/bin/bash
checkpoint_path=$1

python simpler_env/put_in_drawer_variant_agg.py --policy-model openvla --ckpt-path ${checkpoint_path} \
    --robot google_robot_static \
    --env-name PlaceIntoClosedTopDrawerCustomInScene-v0 \
    --control-freq 3 --sim-freq 513 --max-episode-steps 200 \
    --scene-name frl_apartment_stage_simple \
    --robot-init-x 0.65 0.65 1 --robot-init-y -0.2 0.2 3 --obj-init-x -0.35 -0.12 5 --obj-init-y -0.02 0.42 5 \
    --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0.0 0.0 1 \
    --obj-init-x-range -0.08 -0.02 3 --obj-init-y-range -0.02 0.08 3 
