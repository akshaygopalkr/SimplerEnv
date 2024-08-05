#!/bin/bash
EXTRA_ARGS="--enable-raytracing --additional-env-build-kwargs station_name=mk_station_recolor light_mode=simple disable_bad_material=True"
model_type=$1
checkpoint_path=$2


if [ "${model_type}" == "opendvla" ]; then
  echo "Running opendvla evaluation"
  python simpler_env/multi_inference.py --policy-model ${model_type} --ckpt-path ${checkpoint_path} \
    --robot google_robot_static \
    --env-name OpenTopDrawerCustomInScene-v0 \
    --use-depth-anything \
    --control-freq 3 --sim-freq 513 --max-episode-steps 113 \
    --scene-name dummy_drawer \
    --robot-init-rot-quat-center 0 0 0 1 \
    --obj-init-x-range 0 0 1 --obj-init-y-range 0 0 1 \
    ${EXTRA_ARGS}
else
  python simpler_env/multi_inference.py --policy-model ${model_type} --ckpt-path None \
    --robot google_robot_static \
    --env-name OpenTopDrawerCustomInScene-v0 \
    --use-depth-anything \
    --control-freq 3 --sim-freq 513 --max-episode-steps 113 \
    --scene-name dummy_drawer \
    --robot-init-rot-quat-center 0 0 0 1 \
    --obj-init-x-range 0 0 1 --obj-init-y-range 0 0 1 \
    ${EXTRA_ARGS}
fi
