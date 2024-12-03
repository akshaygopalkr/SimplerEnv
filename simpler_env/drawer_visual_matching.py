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

env_names = [
    "OpenTopDrawerCustomInScene-v0",
    "OpenMiddleDrawerCustomInScene-v0",
    "OpenBottomDrawerCustomInScene-v0",
    "CloseTopDrawerCustomInScene-v0",
    "CloseMiddleDrawerCustomInScene-v0",
    "CloseBottomDrawerCustomInScene-v0",
]
urdf_versions = [
    "recolor_cabinet_visual_matching_1",
    "recolor_tabletop_visual_matching_1",
    "recolor_tabletop_visual_matching_2",
    "None"
]


robot_init_x_list = [
    [0.644, 0.644, 1.0], [0.765, 0.765, 1.0], [0.889, 0.889, 1.0],
    [0.652, 0.652, 1.0], [0.752, 0.752, 1.0], [0.851, 0.851, 1.0],
    [0.665, 0.665, 1.0], [0.765, 0.765, 1.0], [0.865, 0.865, 1.0]
]
robot_init_y_list = [
    [-0.179, -0.179, 1.0], [-0.182, -0.182, 1.0], [-0.203, -0.203, 1.0],
    [0.009, 0.009, 1.0], [0.009, 0.009, 1.0], [0.035, 0.035, 1.0],
    [0.224, 0.224, 1.0], [0.222, 0.222, 1.0], [0.222, 0.222, 1.0]
]
robot_rpy_range_list = [
    [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, -0.03, -0.03, 1.0],
    [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, -0.02, -0.02, 1.0],
    [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, -0.06, -0.06, 1.0],
    [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
    [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
    [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
    [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
    [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, -0.025, -0.025, 1.0],
    [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, -0.025, -0.025, 1.0]  
]
rgb_overlay_list = [
    'a0', 'a1', 'a2',
    'b0', 'b1', 'b2',
    'c0', 'c1', 'c2'
]


def run_eval_loop(args, model):
    success_arr = []
    
    for urdf_version in urdf_versions:
        for env_name in env_names:
            for robot_init_x, robot_init_y, robot_rpy_range, rgb_overlay in zip(robot_init_x_list, robot_init_y_list, robot_rpy_range_list, rgb_overlay_list):
                args.env_name = env_name
                args.robot_init_x_range = robot_init_x
                args.robot_init_y_range = robot_init_y
                args.robot_init_rot_rpy_range = robot_rpy_range
                args.rgb_overlay_path = f"./ManiSkill2_real2sim/data/real_inpainting/open_drawer_{rgb_overlay}.png"
                args.additional_env_build_kwargs['urdf_version'] = urdf_version
                
                # env args: robot pose
                args.robot_init_xs = parse_range_tuple(args.robot_init_x_range)
                args.robot_init_ys = parse_range_tuple(args.robot_init_y_range)
                args.robot_init_quats = []
                for r in parse_range_tuple(args.robot_init_rot_rpy_range[:3]):
                    for p in parse_range_tuple(args.robot_init_rot_rpy_range[3:6]):
                        for y in parse_range_tuple(args.robot_init_rot_rpy_range[6:]):
                            args.robot_init_quats.append((Pose(q=euler2quat(r, p, y)) * Pose(q=args.robot_init_rot_quat_center)).q)
                # env args: object position
                if args.obj_variation_mode == "xy":
                    args.obj_init_xs = parse_range_tuple(args.obj_init_x_range)
                    args.obj_init_ys = parse_range_tuple(args.obj_init_y_range)
                    
                # update logging info (args.additional_env_save_tags) if using a different camera from default
                if args.obs_camera_name is not None:
                    if args.additional_env_save_tags is None:
                        args.additional_env_save_tags = f"obs_camera_{args.obs_camera_name}"
                    else:
                        args.additional_env_save_tags = args.additional_env_save_tags + f"_obs_camera_{args.obs_camera_name}"
                
                success = maniskill2_evaluator(model, args)
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
    else:
        raise NotImplementedError()

    # run real-to-sim evaluation
    run_eval_loop(args, model)
    # print(args)
    # print(" " * 10, "Average success", np.mean(success_arr))
