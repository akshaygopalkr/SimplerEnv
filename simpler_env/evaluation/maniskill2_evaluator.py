"""
Evaluate a model on ManiSkill2 environment.
"""

import os
import numpy as np
from transforms3d.euler import quat2euler

from simpler_env.utils.env.env_builder import build_maniskill2_env, get_robot_control_mode
from simpler_env.utils.env.observation_utils import *
from simpler_env.utils.visualization import write_video
from simpler_env.evaluation.task_augmentations import TASK_AUGMENTATIONS
move_near_pattern = r"move (\w+) near (\w+)"
pick_pattern = r"pick (\w+)"
place_into_pattern = r"place (\w+) into (\w+) drawer"



def run_multitext_maniskill2_eval_single_episode(
    model,
    ckpt_path,
    robot_name,
    env_name,
    scene_name,
    robot_init_x,
    robot_init_y,
    robot_init_quat,
    control_mode,
    obj_init_x=None,
    obj_init_y=None,
    obj_episode_id=None,
    additional_env_build_kwargs=None,
    rgb_overlay_path=None,
    obs_camera_name=None,
    control_freq=3,
    sim_freq=513,
    max_episode_steps=80,
    instruction=None,
    enable_raytracing=False,
    additional_env_save_tags=None,
    logging_dir="./results",
):
    
    if additional_env_build_kwargs is None:
        additional_env_build_kwargs = {}

    # Create environment
    kwargs = dict(
        obs_mode="rgbd",
        robot=robot_name,
        sim_freq=sim_freq,
        control_mode=control_mode,
        control_freq=control_freq,
        max_episode_steps=max_episode_steps,
        scene_name=scene_name,
        camera_cfgs={"add_segmentation": True},
        rgb_overlay_path=rgb_overlay_path,
    )
    if enable_raytracing:
        ray_tracing_dict = {"shader_dir": "rt"}
        ray_tracing_dict.update(additional_env_build_kwargs)
        # put raytracing dict keys before other keys for compatibility with existing result naming and metric calculation
        additional_env_build_kwargs = ray_tracing_dict
    env = build_maniskill2_env(
        env_name,
        **additional_env_build_kwargs,
        **kwargs,
    )

    # initialize environment
    env_reset_options = {
        "robot_init_options": {
            "init_xy": np.array([robot_init_x, robot_init_y]),
            "init_rot_quat": robot_init_quat,
        }
    }
    if obj_init_x is not None:
        assert obj_init_y is not None
        obj_variation_mode = "xy"
        env_reset_options["obj_init_options"] = {
            "init_xy": np.array([obj_init_x, obj_init_y]),
        }
    else:
        assert obj_episode_id is not None
        obj_variation_mode = "episode"
        env_reset_options["obj_init_options"] = {
            "episode_id": obj_episode_id,
        }

    # Obtain language instruction
    if instruction is not None:
        task_description = instruction
    else:
        # get default language instruction
        task_description = env.get_language_instruction()
    
    if env_name.startswith("Place"):
        obj_1 = env.model_id
        
        if obj_1 == "baked_apple_v2":
            obj_1 = "apple"
            
        drawer = env.drawer_id
        args = {"object": obj_1, "drawer": drawer}
        task_descriptions = TASK_AUGMENTATIONS["place into drawer"]
    elif env_name.startswith("Open"):
        drawer = env.drawer_id
        args = {"drawer": drawer}
        task_descriptions = TASK_AUGMENTATIONS["open drawer"]
    elif env_name.startswith("Close"):
        drawer = env.drawer_id
        args = {"drawer": drawer}
        task_descriptions = TASK_AUGMENTATIONS["close drawer"]
    elif env_name.startswith("MoveNear"):
        obj_1 = re.search(move_near_pattern, task_description).group(1)
        obj_2 = re.search(move_near_pattern, task_description).group(2)
        args = {"object_1": obj_1, "object_2": obj_2}
        task_descriptions = TASK_AUGMENTATIONS["move near"]
    elif env_name.startswith("Grasp"):
        obj_1 = re.search(pick_pattern, task_description).group(1)
        args = {"object": obj_1}
        task_descriptions = TASK_AUGMENTATIONS["pick"]
    else:
        raise NotImplementedError(f"Task {env_name} not implemented!")

    if "Place" in env_name:
        task_descriptions = itertools.product(TASK_AUGMENTATIONS["open drawer"],
                                              TASK_AUGMENTATIONS["place into drawer"])
    
    success = "failure"
    
    # Iterate through all task descriptions
    for orig_task_description in task_descriptions:
        
        env = build_maniskill2_env(
            env_name,
            **additional_env_build_kwargs,
            **kwargs,
        )
        
        # Set task description to 1st subtask if the environment is a "Place" environment
        if "Place" in env_name:
            task_description = orig_task_description[0].format(**{"drawer": args["drawer"]})
        else:
            task_description = orig_task_description.format(**args)

        obs, _ = env.reset(options=env_reset_options)
        # for long-horizon environments, we check if the current subtask is the final subtask
        is_final_subtask = env.is_final_subtask()

        # Initialize logging
        image = get_image_from_maniskill2_obs_dict(env, obs, camera_name=obs_camera_name)
        images = [image]
        predicted_actions = []
        predicted_terminated, done, truncated = False, False, False

        # Initialize model
        model.reset(task_description)

        timestep = 0

        # Step the environment
        while not (predicted_terminated or truncated):
            # step the model; "raw_action" is raw model action output; "action" is the processed action to be sent into maniskill env
            raw_action, action = model.step(image, task_description)
            predicted_actions.append(raw_action)
            predicted_terminated = bool(action["terminate_episode"][0] > 0)
            if predicted_terminated:
                if not is_final_subtask:
                    # advance the environment to the next subtask
                    predicted_terminated = False
                    env.advance_to_next_subtask()

            # step the environment
            obs, reward, done, truncated, info = env.step(
                np.concatenate([action["world_vector"], action["rot_axangle"], action["gripper"]]),
            )
            
            success = "success" if done else "failure"
            new_task_description = env.get_language_instruction()
            
            # This would only be for Place environments
            if "place" in new_task_description:
                print(f"Completed subtask: {task_description}")
                task_description = orig_task_description[1].format(**args)
                print(f"New subtask: {task_description}")
            
            print(f"Task: {task_description}")
                
            is_final_subtask = env.is_final_subtask()

            image = get_image_from_maniskill2_obs_dict(env, obs, camera_name=obs_camera_name)
            images.append(image)
            timestep += 1

        episode_stats = info.get("episode_stats", {})

        # save video
        env_save_name = env_name
        for k, v in additional_env_build_kwargs.items():
            env_save_name = env_save_name + f"_{k}_{v}"
        if additional_env_save_tags is not None:
            env_save_name = env_save_name + f"_{additional_env_save_tags}"
        ckpt_path_basename = ckpt_path if ckpt_path[-1] != "/" else ckpt_path[:-1]
        ckpt_path_basename = ckpt_path_basename.split("/")[-1]
        if obj_variation_mode == "xy":
            video_name = f"{success}_obj_{obj_init_x}_{obj_init_y}"
        elif obj_variation_mode == "episode":
            video_name = f"{success}_obj_episode_{obj_episode_id}"
        for k, v in episode_stats.items():
            video_name = video_name + f"_{k}_{v}"
        video_name = video_name + ".mp4"
        if rgb_overlay_path is not None:
            rgb_overlay_path_str = os.path.splitext(os.path.basename(rgb_overlay_path))[0]
        else:
            rgb_overlay_path_str = "None"
        r, p, y = quat2euler(robot_init_quat)
        
        if "Place" in env_name:
            task_description = orig_task_description[1].format(**args)
        
        vid_dir = f"{ckpt_path_basename}/{scene_name}/{control_mode}/{env_save_name}/rob_{robot_init_x}_{robot_init_y}_rot_{r:.3f}_{p:.3f}_{y:.3f}_rgb_overlay_{rgb_overlay_path_str}"
        
        if os.path.exists(vid_dir) and video_name in os.listdir(vid_dir):
            num_videos = len(vid_dir)
            video_name = f"{success}_obj_{obj_init_x}_{obj_init_y}_{num_videos}.mp4"
            
        video_path = f"{ckpt_path_basename}/{scene_name}/{control_mode}/{env_save_name}/{task_description}/rob_{robot_init_x}_{robot_init_y}_rot_{r:.3f}_{p:.3f}_{y:.3f}_rgb_overlay_{rgb_overlay_path_str}/{video_name}"
        video_path = os.path.join(logging_dir, video_path)
        print(f"Writing video to {video_path}")
        write_video(video_path, images, fps=5)

        # save action trajectory
        action_path = video_path.replace(".mp4", ".png")
        action_root = os.path.dirname(action_path) + "/actions/"
        os.makedirs(action_root, exist_ok=True)
        action_path = action_root + os.path.basename(action_path)
        model.visualize_epoch(predicted_actions, images, save_path=action_path)

    return success == "success"

def run_maniskill2_eval_single_episode(
    model,
    ckpt_path,
    robot_name,
    env_name,
    scene_name,
    robot_init_x,
    robot_init_y,
    robot_init_quat,
    control_mode,
    obj_init_x=None,
    obj_init_y=None,
    obj_episode_id=None,
    additional_env_build_kwargs=None,
    rgb_overlay_path=None,
    obs_camera_name=None,
    control_freq=3,
    sim_freq=513,
    max_episode_steps=80,
    instruction=None,
    enable_raytracing=False,
    additional_env_save_tags=None,
    logging_dir="./results",
):

    if additional_env_build_kwargs is None:
        additional_env_build_kwargs = {}

    # Create environment
    kwargs = dict(
        obs_mode="rgbd",
        robot=robot_name,
        sim_freq=sim_freq,
        control_mode=control_mode,
        control_freq=control_freq,
        max_episode_steps=max_episode_steps,
        scene_name=scene_name,
        camera_cfgs={"add_segmentation": True},
        rgb_overlay_path=rgb_overlay_path,
    )
    if enable_raytracing:
        ray_tracing_dict = {"shader_dir": "rt"}
        ray_tracing_dict.update(additional_env_build_kwargs)
        # put raytracing dict keys before other keys for compatibility with existing result naming and metric calculation
        additional_env_build_kwargs = ray_tracing_dict
    env = build_maniskill2_env(
        env_name,
        **additional_env_build_kwargs,
        **kwargs,
    )

    # initialize environment
    env_reset_options = {
        "robot_init_options": {
            "init_xy": np.array([robot_init_x, robot_init_y]),
            "init_rot_quat": robot_init_quat,
        }
    }
    if obj_init_x is not None:
        assert obj_init_y is not None
        obj_variation_mode = "xy"
        env_reset_options["obj_init_options"] = {
            "init_xy": np.array([obj_init_x, obj_init_y]),
        }
    else:
        assert obj_episode_id is not None
        obj_variation_mode = "episode"
        env_reset_options["obj_init_options"] = {
            "episode_id": obj_episode_id,
        }
    obs, _ = env.reset(options=env_reset_options)
    # for long-horizon environments, we check if the current subtask is the final subtask
    is_final_subtask = env.is_final_subtask() 

    # Obtain language instruction
    if instruction is not None:
        task_description = instruction
    else:
        # get default language instruction
        task_description = env.get_language_instruction()
    print(task_description)

    # Initialize logging
    image = get_image_from_maniskill2_obs_dict(env, obs, camera_name=obs_camera_name)
    images = [image]
    predicted_actions = []
    predicted_terminated, done, truncated = False, False, False

    # Initialize model
    model.reset(task_description)

    timestep = 0
    success = "failure"

    # Step the environment
    while not (predicted_terminated or truncated):
        # step the model; "raw_action" is raw model action output; "action" is the processed action to be sent into maniskill env
        raw_action, action = model.step(image, task_description)
        predicted_actions.append(raw_action)
        predicted_terminated = bool(action["terminate_episode"][0] > 0)
        if predicted_terminated:
            if not is_final_subtask:
                # advance the environment to the next subtask
                predicted_terminated = False
                env.advance_to_next_subtask()

        # step the environment
        obs, reward, done, truncated, info = env.step(
            np.concatenate([action["world_vector"], action["rot_axangle"], action["gripper"]]),
        )
        
        success = "success" if done else "failure"
        new_task_description = env.get_language_instruction()
        if new_task_description != task_description:
            task_description = new_task_description
            print(task_description)
        is_final_subtask = env.is_final_subtask()

        print(timestep, info)

        image = get_image_from_maniskill2_obs_dict(env, obs, camera_name=obs_camera_name)
        images.append(image)
        timestep += 1

    episode_stats = info.get("episode_stats", {})

    # save video
    env_save_name = env_name
    for k, v in additional_env_build_kwargs.items():
        env_save_name = env_save_name + f"_{k}_{v}"
    if additional_env_save_tags is not None:
        env_save_name = env_save_name + f"_{additional_env_save_tags}"
    ckpt_path_basename = ckpt_path if ckpt_path[-1] != "/" else ckpt_path[:-1]
    ckpt_path_basename = ckpt_path_basename.split("/")[-1]
    if obj_variation_mode == "xy":
        video_name = f"{success}_obj_{obj_init_x}_{obj_init_y}"
    elif obj_variation_mode == "episode":
        video_name = f"{success}_obj_episode_{obj_episode_id}"
    for k, v in episode_stats.items():
        video_name = video_name + f"_{k}_{v}"
    video_name = video_name + ".mp4"
    if rgb_overlay_path is not None:
        rgb_overlay_path_str = os.path.splitext(os.path.basename(rgb_overlay_path))[0]
    else:
        rgb_overlay_path_str = "None"
    r, p, y = quat2euler(robot_init_quat)
    video_path = f"{ckpt_path_basename}/{scene_name}/{control_mode}/{env_save_name}/rob_{robot_init_x}_{robot_init_y}_rot_{r:.3f}_{p:.3f}_{y:.3f}_rgb_overlay_{rgb_overlay_path_str}/{video_name}"
    video_path = os.path.join(logging_dir, video_path)
    write_video(video_path, images, fps=5)

    # save action trajectory
    action_path = video_path.replace(".mp4", ".png")
    action_root = os.path.dirname(action_path) + "/actions/"
    os.makedirs(action_root, exist_ok=True)
    action_path = action_root + os.path.basename(action_path)
    model.visualize_epoch(predicted_actions, images, save_path=action_path)

    return success == "success"


def maniskill2_evaluator(model, args):
    control_mode = get_robot_control_mode(args.robot, args.policy_model)
    success_arr = []

    # run inference
    for robot_init_x in args.robot_init_xs:
        for robot_init_y in args.robot_init_ys:
            for robot_init_quat in args.robot_init_quats:
                kwargs = dict(
                    model=model,
                    ckpt_path=args.ckpt_path,
                    robot_name=args.robot,
                    env_name=args.env_name,
                    scene_name=args.scene_name,
                    robot_init_x=robot_init_x,
                    robot_init_y=robot_init_y,
                    robot_init_quat=robot_init_quat,
                    control_mode=control_mode,
                    additional_env_build_kwargs=args.additional_env_build_kwargs,
                    rgb_overlay_path=args.rgb_overlay_path,
                    control_freq=args.control_freq,
                    sim_freq=args.sim_freq,
                    max_episode_steps=args.max_episode_steps,
                    enable_raytracing=args.enable_raytracing,
                    additional_env_save_tags=args.additional_env_save_tags,
                    obs_camera_name=args.obs_camera_name,
                    logging_dir=args.logging_dir,
                )
                if args.obj_variation_mode == "xy":
                    for obj_init_x in args.obj_init_xs:
                        for obj_init_y in args.obj_init_ys:
                            success_arr.append(
                                run_maniskill2_eval_single_episode(
                                    obj_init_x=obj_init_x,
                                    obj_init_y=obj_init_y,
                                    **kwargs,
                                )
                            )
                elif args.obj_variation_mode == "episode":
                    for obj_episode_id in range(args.obj_episode_range[0], args.obj_episode_range[1]):
                        success_arr.append(run_maniskill2_eval_single_episode(obj_episode_id=obj_episode_id, **kwargs))
                else:
                    raise NotImplementedError()

    return success_arr
