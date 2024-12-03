#!/bin/bash
checkpoint_path=$1


python simpler_env/drawer_variant_agg.py --policy-model openvla --ckpt-path ${checkpoint_path} \
    --robot google_robot_static \
    --env-name OpenTopDrawerCustomInScene-v0 \
    --control-freq 3 --sim-freq 513 --max-episode-steps 113 \
    --scene-name frl_apartment_stage_simple \
    --robot-init-x 0.65 0.85 3 --robot-init-y -0.2 0.2 3 \
    --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0.0 0.0 1 \
    --obj-init-x-range 0 0 1 --obj-init-y-range 0 0 1
