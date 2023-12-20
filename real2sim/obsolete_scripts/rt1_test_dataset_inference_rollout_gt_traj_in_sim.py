import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf
from PIL import Image
from IPython import display
import tqdm, os
import mani_skill2.envs, gymnasium as gym
from transforms3d.euler import euler2axangle, euler2quat
from transforms3d.quaternions import quat2axangle, axangle2quat
from sapien.core import Pose
from copy import deepcopy
import cv2

from real2sim.utils.visualization import write_video

DATASETS = ['fractal20220817_data', 'bridge']

def dataset2path(dataset_name):
    if dataset_name == 'robo_net':
        version = '1.0.0'
    elif dataset_name == 'language_table':
        version = '0.0.1'
    else:
        version = '0.1.0'
    return f'gs://gresearch/robotics/{dataset_name}/{version}'


def main(dset_iter, iter_num, episode_id, set_actual_reached=False, 
         control_mode='arm_pd_ee_delta_pose_align_interpolate_gripper_pd_joint_pos'):
    for _ in range(iter_num):
        episode = next(dset_iter)
    print("episode tfds id", episode['tfds_id'])
    episode_steps = list(episode['steps'])
    pred_actions, gt_actions, images = [], [], []
    
    language_instruction = episode_steps[0]['observation']['natural_language_instruction']
    print(language_instruction)
    
    sim_freq, control_freq, action_repeat = 510, 3, 5
    action_scale = 1.0
    env = gym.make('PickCube-v0',
                        control_mode=control_mode,
                        obs_mode='rgbd',
                        robot='google_robot_static',
                        sim_freq=sim_freq,
                        control_freq=control_freq * action_repeat,
                        max_episode_steps=50 * action_repeat,
                        camera_cfgs={"add_segmentation": True},
                        rgb_overlay_path=f'/home/xuanlin/Downloads/{episode_id}_0_cleanup.png',
                        rgb_overlay_cameras=['overhead_camera'],
            )
    
    images = []
    ee_poses_at_base = []
    gt_images = []
    obs, _ = env.reset()
    images.append(obs['image']['overhead_camera']['rgb'])
    ee_poses_at_base.append(env.agent.robot.pose.inv() * env.tcp.pose)
    
    for i in range(len(episode_steps) - 1):
        episode_step = episode_steps[i] # episode_step['observation']['base_pose_tool_reached'] = [xyz, quat xyzw]
        gt_images.append(episode_step['observation']['image'])
        next_episode_step = episode_steps[i + 1]
        current_pose_at_robot_base = env.agent.robot.pose.inv() * env.tcp.pose
        if not set_actual_reached:
            gt_action_world_vector = episode_step['action']['world_vector']
            gt_action_rotation_delta = np.asarray(episode_step['action']['rotation_delta'], dtype=np.float64)
            # gt_action_rotation_ax, gt_action_rotation_angle = euler2axangle(*gt_action_rotation_delta)
            # gt_action_rotation_axangle = gt_action_rotation_ax * gt_action_rotation_angle
            gt_action_rotation_angle = np.linalg.norm(gt_action_rotation_delta)
            gt_action_rotation_ax = gt_action_rotation_delta / gt_action_rotation_angle if gt_action_rotation_angle > 1e-6 else np.array([0., 1., 0.])
            gt_action_rotation_axangle = gt_action_rotation_ax * gt_action_rotation_angle
            gt_action_gripper_closedness_action = env.agent.get_gripper_closedness() + episode_step['action']['gripper_closedness_action']
            target_gripper_closedness_action = gt_action_gripper_closedness_action
            # print(i, "gripper", env.agent.get_gripper_closedness(), episode_step['action']['gripper_closedness_action'])
            target_tcp_pose_at_base = Pose(p=current_pose_at_robot_base.p + gt_action_world_vector * action_scale,
                                           q=(Pose(q=axangle2quat(gt_action_rotation_ax, gt_action_rotation_angle)) 
                                              * Pose(q=current_pose_at_robot_base.q)).q
            )
        else:
            assert control_mode == 'arm_pd_ee_target_delta_pose_base_gripper_pd_joint_pos'
            next_xyz = next_episode_step['observation']['base_pose_tool_reached'][:3]
            next_xyzw = next_episode_step['observation']['base_pose_tool_reached'][3:]
            next_pose_at_robot_base = Pose(p=np.array(next_xyz), q=np.concatenate([next_xyzw[-1:], next_xyzw[:-1]]))
            target_delta_pose_at_robot_base = next_pose_at_robot_base * current_pose_at_robot_base.inv()
                        
            gt_action_world_vector = target_delta_pose_at_robot_base.p
            gt_action_rotation_ax, gt_action_rotation_angle = quat2axangle(np.array(target_delta_pose_at_robot_base.q, dtype=np.float64))
            gt_action_rotation_axangle = gt_action_rotation_ax * gt_action_rotation_angle
            gt_action_gripper_closedness_action = env.agent.get_gripper_closedness() + episode_step['action']['gripper_closedness_action']
            target_gripper_closedness_action = gt_action_gripper_closedness_action
            
            target_tcp_pose_at_base = Pose(p=gt_action_world_vector * action_scale, 
                                           q=axangle2quat(gt_action_rotation_ax, gt_action_rotation_angle)) * current_pose_at_robot_base
            
        # print(i, "gripper", env.agent.get_gripper_closedness(), episode_step['action']['gripper_closedness_action'])
        action = np.concatenate(
                            [gt_action_world_vector * action_scale, 
                            gt_action_rotation_axangle * action_scale,
                            gt_action_gripper_closedness_action,
                            ],
                        ).astype(np.float64)
        
        obs, reward, terminated, truncated, info = env.step(action) 
        images.append(obs['image']['overhead_camera']['rgb'])
        ee_poses_at_base.append(env.agent.robot.pose.inv() * env.tcp.pose)
        arm_ctrl_mode, gripper_ctrl_mode = control_mode.split('gripper')
        for _ in range(action_repeat - 1):
            interp_action = action.copy()
            if 'target' in arm_ctrl_mode:
                interp_action[:6] *= 0
            else:
                cur_tcp_pose_at_base = env.agent.robot.pose.inv() * env.tcp.pose
                delta_tcp_pose_at_base = target_tcp_pose_at_base * cur_tcp_pose_at_base.inv()
                interp_action[:3] = target_tcp_pose_at_base.p - cur_tcp_pose_at_base.p
                interp_rot_ax, interp_rot_angle = quat2axangle(np.array(delta_tcp_pose_at_base.q, dtype=np.float64))
                interp_action[3:6] = interp_rot_ax * interp_rot_angle
                
            if 'target' in gripper_ctrl_mode:
                interp_action[6:] *= 0
            obs, reward, terminated, truncated, info = env.step(interp_action)
            images.append(obs['image']['overhead_camera']['rgb'])
            ee_poses_at_base.append(env.agent.robot.pose.inv() * env.tcp.pose)
            
    # for i, ee_pose in enumerate(ee_poses_at_base):
    #     print(i, "ee pose wrt robot base", ee_pose)
    gt_images = [gt_images[np.clip((i - 1) // action_repeat + 1, 0, len(gt_images) - 1)] for i in range(len(images))]
    for i in range(len(images)):
        images[i] = np.concatenate([images[i], cv2.resize(np.asarray(gt_images[i]), (images[i].shape[1], images[i].shape[0]))], axis=1)
    if not set_actual_reached:
        write_video(f'/home/xuanlin/Downloads/tmp_pick_coke_can/{episode_id}_0_cleanup_overlay_arm.mp4', images, fps=5)
    else:
        write_video(f'/home/xuanlin/Downloads/tmp_pick_coke_can/{episode_id}_0_cleanup_overlay_arm_actual_reached.mp4', images, fps=5)
    write_video(f'/home/xuanlin/Downloads/tmp_pick_coke_can/{episode_id}_gt.mp4', gt_images, fps=5)
    

if __name__ == '__main__':
    dataset_name = DATASETS[0]
    dset = tfds.builder_from_directory(builder_dir=dataset2path(dataset_name))
    
    dset = dset.as_dataset(split='train', read_config=tfds.ReadConfig(add_tfds_id=True))
    dset_iter = iter(dset)
    last_episode_id = 0
    for ep_idx in [805, 1257, 1495, 1539, 1991, 2398, 3289]:
        if last_episode_id == 0:
            main(dset_iter, ep_idx + 1 - last_episode_id, ep_idx, False)
        else:
            main(dset_iter, ep_idx - last_episode_id, ep_idx, False)
        last_episode_id = ep_idx