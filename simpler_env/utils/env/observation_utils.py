import numpy as np
import matplotlib
cmap = matplotlib.colormaps.get_cmap('Spectral')

def get_image_from_maniskill2_obs_dict(env, obs, camera_name=None):
    # obtain image from observation dictionary returned by ManiSkill2 environment
    if camera_name is None:
        if "google_robot" in env.robot_uid:
            camera_name = "overhead_camera"
        elif "widowx" in env.robot_uid:
            camera_name = "3rd_view_camera"
        else:
            raise NotImplementedError()
    return obs["image"][camera_name]["rgb"]


def get_image_and_depth_from_maniskill2_obs_dict(env, obs, camera_name=None, depth_model=None):
    # obtain image and depth from observation dictionary returned by ManiSkill2 environment
    if camera_name is None:
        if "google_robot" in env.robot_uid:
            camera_name = "overhead_camera"
        elif "widowx" in env.robot_uid:
            camera_name = "3rd_view_camera"
        else:
            raise NotImplementedError()
    
    if depth_model is not None:
        depth = depth_model.infer_image(obs["image"][camera_name]["rgb"])
    else:
        depth = obs["image"][camera_name]["depth"]
    
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth = depth.astype(np.uint8)
    depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
        
    return obs["image"][camera_name]["rgb"], depth