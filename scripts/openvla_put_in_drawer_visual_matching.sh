#!/bin/bash
checkpoint_path=$1
EXTRA_ARGS="--enable-raytracing --additional-env-build-kwargs station_name=mk_station_recolor light_mode=simple disable_bad_material=True model_ids=baked_apple_v2"

python simpler_env/put_in_drawer_visual_matching.py openvla --ckpt-path ${checkpoint_path} \
    --robot google_robot_static \
    --env-name OpenTopDrawerCustomInScene-v0 \
    --control-freq 3 --sim-freq 513 --max-episode-steps 200 \
    --scene-name dummy_drawer \
    --robot-init-rot-quat-center 0 0 0 1 \
    --obj-init-x-range -0.08 -0.02 3 --obj-init-y-range -0.02 0.08 3 \
