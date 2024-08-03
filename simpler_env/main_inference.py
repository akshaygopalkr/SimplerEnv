import os

import numpy as np
import tensorflow as tf

from simpler_env.evaluation.argparse import get_args
from simpler_env.evaluation.maniskill2_evaluator import maniskill2_evaluator
from simpler_env.policies.octo.octo_server_model import OctoServerInference
from simpler_env.policies.openvla.openvla_model import OPENVLAInference
from simpler_env.policies.openvla.opendvla_model import OPENDVLAInference
from simpler_env.policies.rt1.rt1_model import RT1Inference
from Depth_Anything_V2.metric_depth.depth_anything_v2.dpt import DepthAnythingV2
import torch
os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    from simpler_env.policies.octo.octo_model import OctoInference
except ImportError as e:
    print("Octo is not correctly imported.")
    print(e)


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

    # policy model creation; update this if you are using a new policy model
    if args.policy_model == "rt1":
        assert args.ckpt_path is not None
        model = RT1Inference(
            saved_model_path=args.ckpt_path,
            policy_setup=args.policy_setup,
            action_scale=args.action_scale,
        )
    elif "octo" in args.policy_model:
        if args.ckpt_path is None or args.ckpt_path == "None":
            args.ckpt_path = args.policy_model
        if "server" in args.policy_model:
            model = OctoServerInference(
                model_type=args.ckpt_path,
                policy_setup=args.policy_setup,
                action_scale=args.action_scale,
            )
        else:
            model = OctoInference(
                model_type=args.ckpt_path,
                policy_setup=args.policy_setup,
                init_rng=args.octo_init_rng,
                action_scale=args.action_scale,
            )
    elif "openvla" in args.policy_model:
        model = OPENVLAInference(policy_setup=args.policy_setup)
    elif "opendvla" in args.policy_model:
        model = OPENDVLAInference(model_id_or_path=args.ckpt_path,
                                  policy_setup=args.policy_setup)
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
        da_model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})
        da_model.load_state_dict(torch.load(os.path.join(args.da_ckpt, "depth_anything_v2_metric_hypersim_vitl.pth"), map_location='cpu'))
        da_model.to('cuda:0')
        da_model.eval()
    else:
        da_model = None

    # run real-to-sim evaluation
    success_arr = maniskill2_evaluator(model, args, depth_model=da_model)
    print(args)
    print(" " * 10, "Average success", np.mean(success_arr))
