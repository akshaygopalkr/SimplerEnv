import os

import numpy as np
import tensorflow as tf
from simpler_env.evaluation.argparse import get_args
from simpler_env.evaluation.maniskill2_evaluator import maniskill2_evaluator
from simpler_env.policies.openvla.openvla_model import OPENVLAInference
import torch
from sapien.core import Pose
from transforms3d.euler import euler2quat

def parse_range_tuple(t):
    return np.linspace(t[0], t[1], int(t[2]))

os.environ["TOKENIZERS_PARALLELISM"] = "false"

env_scene_names = [
    ("MoveNearGoogleInScene-v0", "google_pick_coke_can_1_v4"),
    ("MoveNearGoogleInScene-v0", "google_pick_coke_can_1_v4", "no_distractor", True),
    ("MoveNearGoogleInScene-v0", "google_pick_coke_can_1_v4_alt_background"),
    ("MoveNearGoogleInScene-v0", "google_pick_coke_can_1_v4_alt_background_2"),
    ("MoveNearGoogleInScene-v0", "google_pick_coke_can_1_v4", "slightly_darker_lighting", True),
    ("MoveNearGoogleInScene-v0", "google_pick_coke_can_1_v4", "slightly_brighter_lighting", True),
    ("MoveNearGoogleInScene-v0", "Baked_sc1_staging_objaverse_cabinet1_h870"),
    ("MoveNearGoogleInScene-v0", "Baked_sc1_staging_objaverse_cabinet2_h870"),
    ("MoveNearAltGoogleCameraInScene-v0", "google_pick_coke_can_1_v4"),
    ("MoveNearAltGoogleCamera2InScene-v0", "google_pick_coke_can_1_v4"),
]

def run_eval_loop(args, model, depth_model=None):
    
    success_arr = []
    for env_tuple in env_scene_names:
        
        env_args = args
        
        env_args.env_name = env_tuple[0]
        env_args.scene_name = env_tuple[1]
        
        if len(env_tuple) > 2:
            args.additional_env_build_kwargs[env_tuple[2]] = env_tuple[3]
                
        # env args: robot pose
        env_args.robot_init_xs = parse_range_tuple(env_args.robot_init_x_range)
        env_args.robot_init_ys = parse_range_tuple(env_args.robot_init_y_range)
        env_args.robot_init_quats = []
        for r in parse_range_tuple(env_args.robot_init_rot_rpy_range[:3]):
            for p in parse_range_tuple(env_args.robot_init_rot_rpy_range[3:6]):
                for y in parse_range_tuple(env_args.robot_init_rot_rpy_range[6:]):
                    env_args.robot_init_quats.append((Pose(q=euler2quat(r, p, y)) * Pose(q=args.robot_init_rot_quat_center)).q)
        # env args: object position
        if env_args.obj_variation_mode == "xy":
            env_args.obj_init_xs = parse_range_tuple(env_args.obj_init_x_range)
            env_args.obj_init_ys = parse_range_tuple(env_args.obj_init_y_range)
            
        # update logging info (args.additional_env_save_tags) if using a different camera from default
        if env_args.obs_camera_name is not None:
            if env_args.additional_env_save_tags is None:
                env_args.additional_env_save_tags = f"obs_camera_{env_args.obs_camera_name}"
            else:
                env_args.additional_env_save_tags = env_args.additional_env_save_tags + f"_obs_camera_{env_args.obs_camera_name}"
        
        success = maniskill2_evaluator(model, env_args)
        success_arr.append(success)


if __name__ == "__main__":
    args = get_args()
    os.environ["DISPLAY"] = ""
    # prevent a single jax process from taking up all the GPU memory
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) > 0:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        # prevent a single tf process from taking up all the GPU memory
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=args.tf_memory_limit)],
        )
    if "openvla" in args.policy_model:
        model = OPENVLAInference(model_id_or_path=args.ckpt_path,
                                 policy_setup=args.policy_setup)
        
    # run real-to-sim evaluation
    run_eval_loop(args, model)
