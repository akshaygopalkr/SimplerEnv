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

urdf_versions = ["recolor_cabinet_visual_matching_1", "recolor_tabletop_visual_matching_1", "recolor_tabletop_visual_matching_2", "None"]
def run_eval_loop(args, model):
    
    success_arr = []
    args.additional_env_build_kwargs = {}
    for urdf_version in urdf_versions:
        
        env_args = args
        env_args.additional_env_build_kwargs['urdf_version'] = urdf_version
                
        # env args: robot pose
        env_args.robot_init_xs = parse_range_tuple(env_args.robot_init_x_range)
        env_args.robot_init_ys = parse_range_tuple(env_args.robot_init_y_range)
        env_args.robot_init_quats = []
        for r in parse_range_tuple(env_args.robot_init_rot_rpy_range[:3]):
            for p in parse_range_tuple(env_args.robot_init_rot_rpy_range[3:6]):
                for y in parse_range_tuple(env_args.robot_init_rot_rpy_range[6:]):
                    env_args.robot_init_quats.append((Pose(q=euler2quat(r, p, y)) * Pose(q=env_args.robot_init_rot_quat_center)).q)
        # env args: object position
        if env_args.obj_variation_mode == "xy":
            env_args.obj_init_xs = parse_range_tuple(args.obj_init_x_range)
            env_args.obj_init_ys = parse_range_tuple(args.obj_init_y_range)
            
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
        print(args.ckpt_path)
        model = OPENVLAInference(model_id_or_path=args.ckpt_path,
                                 policy_setup=args.policy_setup)
        print('Done loading model...')
    else:
        raise NotImplementedError()

    # run real-to-sim evaluation
    run_eval_loop(args, model)
    # print(args)
    # print(" " * 10, "Average success", np.mean(success_arr))
