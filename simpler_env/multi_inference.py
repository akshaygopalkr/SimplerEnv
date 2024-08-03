import os

import numpy as np
import tensorflow as tf

from simpler_env.evaluation.argparse import get_args
from simpler_env.evaluation.maniskill2_evaluator import maniskill2_evaluator
from simpler_env.policies.openvla.openvla_model import OPENVLAInference
from simpler_env.policies.openvla.opendvla_model import OPENDVLAInference
from Depth_Anything_V2.metric_depth.depth_anything_v2.dpt import DepthAnythingV2
import torch
os.environ["TOKENIZERS_PARALLELISM"] = "false"

bridge_robot_init_x = 0.0147
bridge_robot_init_y = 0.028
urdf_version_arr = ["recolor_cabinet_visual_matching_1",
    "recolor_tabletop_visual_matching_1",
    "recolor_tabletop_visual_matching_2",
    "None"
]

env_names = [
    "OpenTopDrawerCustomInScene-v0",
    "OpenMiddleDrawerCustomInScene-v0",
    "OpenBottomDrawerCustomInScene-v0",
    "CloseTopDrawerCustomInScene-v0",
    "CloseMiddleDrawerCustomInScene-v0",
    "CloseBottomDrawerCustomInScene-v0",
]


robot_init_x_list = [
    [0.644, 0.644, 1], [0.765, 0.765, 1], [0.889, 0.889, 1],
    [0.652, 0.652, 1], [0.752, 0.752, 1], [0.851, 0.851, 1],
    [0.665, 0.665, 1], [0.765, 0.765, 1], [0.865, 0.865, 1]
]
robot_init_y_list = [
    [-0.179, -0.179, 1], [-0.182, -0.182, 1], [-0.203, -0.203, 1],
    [0.009, 0.009, 1], [0.009, 0.009, 1], [0.035, 0.035, 1],
    [0.224, 0.224, 1], [0.222, 0.222, 1], [0.222, 0.222, 1]
]
robot_rpy_range_list = [
    [0, 0, 1, 0, 0, 1, -0.03, -0.03, 1],
    [0, 0, 1, 0, 0, 1, -0.02, -0.02, 1],
    [0, 0, 1, 0, 0, 1, -0.06, -0.06, 1],
    [0, 0, 1, 0, 0, 1, 0, 0, 1],
    [0, 0, 1, 0, 0, 1, 0, 0, 1],
    [0, 0, 1, 0, 0, 1, 0, 0, 1],
    [0, 0, 1, 0, 0, 1, 0, 0, 1],
    [0, 0, 1, 0, 0, 1, -0.025, -0.025, 1],
    [0, 0, 1, 0, 0, 1, -0.025, -0.025, 1]  
]
rgb_overlay_list = [
    'a0', 'a1', 'a2',
    'b0', 'b1', 'b2',
    'c0', 'c1', 'c2'
]


def run_eval_loop(args, model, depth_model=None):
    success_arr = []
    for urdf_version in urdf_version_arr:
        for env_name in env_names:
            for robot_init_x, robot_init_y, robot_rpy_range, rgb_overlay in zip(robot_init_x_list, robot_init_y_list, robot_rpy_range_list, rgb_overlay_list):
                args.env_name = env_name
                args.robot_init_x = robot_init_x
                args.robot_init_y = robot_init_y
                args.robot_rpy_range = robot_rpy_range
                args.rgb_overlay = f"./ManiSkill2_real2sim/data/real_inpainting/open_drawer_{rgb_overlay}.png"
                args.urdf_version = urdf_version
                success = maniskill2_evaluator(model, args, depth_model=depth_model)
                success_arr.append(success)
    print(" " * 10, "Average success", np.mean(success_arr))


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
        model = OPENVLAInference(policy_setup=args.policy_setup)
    elif "opendvla" in args.policy_model:
        model = OPENDVLAInference(model_id_or_path=args.ckpt_path,
                                  policy_setup=args.policy_setup)
        print('Done loading model...')
    else:
        raise NotImplementedError()
    
    if "opendvla" in args.policy_model and args.use_depth_anything:
        
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        }
        encoder = 'vitl'
        dataset = 'hypersim'
        max_depth = 20
        print('Loading DepthAnythingV2 model...')
        da_model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})
        da_model.load_state_dict(torch.load(os.path.join(args.da_ckpt, "depth_anything_v2_metric_hypersim_vitl.pth"), map_location='cpu'))
        da_model.to('cuda')
        da_model.eval()
    else:
        da_model = None

    # run real-to-sim evaluation
    success_arr = maniskill2_evaluator(model, args, depth_model=da_model)
    print(args)
    print(" " * 10, "Average success", np.mean(success_arr))
