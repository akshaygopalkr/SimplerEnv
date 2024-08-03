from collections import defaultdict
from typing import Optional, Sequence
from typing import Any, Callable, ClassVar, Dict, List, Optional, Tuple, Union

from prismatic import load_dvla
import matplotlib.pyplot as plt
import numpy as np
import torch
from transforms3d.euler import euler2axangle
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
from prismatic.models.backbones.llm.prompting import LLaMa2ChatPromptBuilder
import transformers
import torch
from transformers import AutoTokenizer
from typing import Union
from pathlib import Path
text_model = "meta-llama/Llama-2-7b-chat-hf"

t_tokenizer = AutoTokenizer.from_pretrained(text_model)
class OPENDVLAInference:
    def __init__(
        self,
        model_id_or_path: Union[str, Path],
        image_width: int = 224,
        image_height: int = 224,
        action_scale: float = 1.0,
        policy_setup: str = "google_robot",
    ) -> None:
        self.input_processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)   
        hf_token = "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"
        device = torch.device("cuda")

        # Load a pretrained D-VLM (either local path, or ID to auto-download from the HF Hub)
        self.policy = load_dvla(model_id_or_path, hf_token=hf_token).to(device, dtype=torch.bfloat16)
        self.vlm_features = []
        self.feats = []
        self.image_width = image_width
        self.image_height = image_height
        self.action_scale = action_scale

        self.observation = None
        self.tfa_time_step = None
        self.policy_state = None
        self.task_description = None
        self.task_description_embedding = None
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        # self.gripper_is_closed = False
        self.previous_gripper_action = None
        self.gripper_action_repeat = 0
        self.sticky_gripper_num_repeat = 15

        self.policy_setup = policy_setup
        if self.policy_setup == "google_robot":
            self.unnormalize_action = False
            self.unnormalize_action_fxn = None
            self.invert_gripper_action = False
            self.action_rotation_mode = "axis_angle"
        elif self.policy_setup == "widowx_bridge":
            self.unnormalize_action = True
            self.unnormalize_action_fxn = self._unnormalize_action_widowx_bridge
            self.invert_gripper_action = True
            self.action_rotation_mode = "rpy"
        else:
            raise NotImplementedError()

    def init_model(self):
        self.observation = {}
        self.observation["image"] = torch.zeros((224, 224, 3))
        self.observation["depth"] = torch.zeros((224, 224, 3))
        self.observation["natural_language_embedding"] = torch.zeros((512,), dtype=torch.float32)


    @staticmethod
    def _rescale_action_with_bound(
        actions: np.ndarray | torch.Tensor,
        low: float,
        high: float,
        safety_margin: float = 0.0,
        post_scaling_max: float = 1.0,
        post_scaling_min: float = -1.0,
    ) -> np.ndarray:
        """Formula taken from https://stats.stackexchange.com/questions/281162/scale-a-number-between-a-range."""
        resc_actions = (actions - low) / (high - low) * (post_scaling_max - post_scaling_min) + post_scaling_min
        return np.clip(
            resc_actions,
            post_scaling_min + safety_margin,
            post_scaling_max - safety_margin,
        )

    def _unnormalize_action_widowx_bridge(self, action: dict[str, np.ndarray | torch.Tensor]) -> dict[str, np.ndarray]:
        action["world_vector"] = self._rescale_action_with_bound(
            action["world_vector"],
            low=-1.75,
            high=1.75,
            post_scaling_max=0.05,
            post_scaling_min=-0.05,
        )
        action["rotation_delta"] = self._rescale_action_with_bound(
            action["rotation_delta"],
            low=-1.4,
            high=1.4,
            post_scaling_max=0.25,
            post_scaling_min=-0.25,
        )
        return action

 
    def _resize_image(self, image: np.ndarray | torch.Tensor) -> torch.Tensor:
        #image = tf.image.resize_with_pad(image, target_width=self.image_width, target_height=self.image_height)
        #image = tf.cast(image, tf.uint8)
        return image

    def _initialize_task_description(self, task_description: Optional[str] = None) -> None:
        if task_description is not None:
            self.task_description = task_description
            self.task_description_embedding = torch.zeros((512,), dtype=torch.float32)
        else:
            self.task_description = ""
            self.task_description_embedding = torch.zeros((512,), dtype=torch.float32)

    def reset(self, task_description: str) -> None:
        # self._initialize_model()i
        self.num_image_history = 0

        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        # self.gripper_is_closed = False
        self.previous_gripper_action = None
        self._initialize_task_description(task_description)

    @staticmethod
    def _small_action__filter_google_robot(raw_action: dict[str, np.ndarray | torch.Tensor], arm_movement: bool = False, gripper: bool = True) -> dict[str, np.ndarray | torch.Tensor]:
        # small action filtering for google robot
        if arm_movement:
            raw_action["world_vector"] = torch.where(
                torch.abs(raw_action["world_vector"]) < 5e-3,
                torch.zeros_like(raw_action["world_vector"]),
                raw_action["world_vector"],
            )
            raw_action["rotation_delta"] = torch.where(
                torch.abs(raw_action["rotation_delta"]) < 5e-3,
                torch.zeros_like(raw_action["rotation_delta"]),
                raw_action["rotation_delta"],
            )
            raw_action["base_displacement_vector"] = torch.where(
                raw_action["base_displacement_vector"] < 5e-3,
                torch.zeros_like(raw_action["base_displacement_vector"]),
                raw_action["base_displacement_vector"],
            )
            raw_action["base_displacement_vertical_rotation"] = torch.where(
                raw_action["base_displacement_vertical_rotation"] < 1e-2,
                torch.zeros_like(raw_action["base_displacement_vertical_rotation"]),
                raw_action["base_displacement_vertical_rotation"],
            )
        if gripper:
            raw_action["gripper_closedness_action"] = torch.where(
                torch.abs(raw_action["gripper_closedness_action"]) < 1e-2,
                torch.zeros_like(raw_action["gripper_closedness_action"]),
                raw_action["gripper_closedness_action"],
            )
        return raw_action


    def step(self, image: np.ndarray, depth: np.ndarray, task_description: Optional[str] = None) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        Input:
            image: np.ndarray of shape (H, W, 3), uint8
            task_description: Optional[str], task description; if different from previous task description, policy state is reset
        Output:
            raw_action: dict; raw policy action output
            action: dict; processed action to be sent to the maniskill2 environment, with the following keys:
                - 'world_vector': np.ndarray of shape (3,), xyz translation of robot end-effector
                - 'rot_axangle': np.ndarray of shape (3,), axis-angle representation of end-effector rotation
                - 'gripper': np.ndarray of shape (1,), gripper action
                - 'terminate_episode': np.ndarray of shape (1,), 1 if episode should be terminated, 0 otherwise
        """
        if task_description is not None:
            if task_description != self.task_description:
                # task description has changed; update language embedding
                # self._initialize_task_description(task_description)
                self.reset(task_description)
        
        assert image.dtype == np.uint8
        self.init_model()
        
        prompt = "In: What action should the robot take to {}?\nOut:".format(task_description)
        self.observation["image"] = image
        self.observation["depth"] = depth
        self.observation["natural_language_embedding"] = self.task_description_embedding
        
        _actions = self.policy.predict_action(image=Image.fromarray(image), instruction=task_description,
                                              depth=Image.fromarray(depth), unnorm_key="fractal20220817_depth_data")
        
        raw_action = {
            "world_vector": np.array(_actions[:3]),
            "rotation_delta": np.array(_actions[3:6]),
            "open_gripper": np.array(_actions[6:7]),  # range [0, 1]; 1 = open; 0 = close
        }
        action = {}
        action["world_vector"] = raw_action["world_vector"]
        action_rotation_delta = np.asarray(raw_action["rotation_delta"], dtype=np.float64)
        roll, pitch, yaw = action_rotation_delta
        action_rotation_ax, action_rotation_angle = euler2axangle(roll, pitch, yaw)
        action_rotation_axangle = action_rotation_ax * action_rotation_angle
        action["rot_axangle"] = action_rotation_axangle
        if self.policy_setup == "google_robot":
            current_gripper_action = raw_action["open_gripper"]

            # alternative implementation
            if self.previous_gripper_action is None:
                relative_gripper_action = np.array([0])
            else:
                relative_gripper_action = (
                    self.previous_gripper_action - current_gripper_action
                )  # google robot 1 = close; -1 = open
            print("relative_", relative_gripper_action, current_gripper_action,
                  relative_gripper_action)
            self.previous_gripper_action = current_gripper_action

            if np.abs(relative_gripper_action) > 0.5 and self.sticky_action_is_on is False:
                self.sticky_action_is_on = True
                self.sticky_gripper_action = relative_gripper_action
                print("second last", relative_gripper_action)
            if self.sticky_action_is_on:
                self.gripper_action_repeat += 1
                relative_gripper_action = self.sticky_gripper_action

                print("last", relative_gripper_action)
            if self.gripper_action_repeat == self.sticky_gripper_num_repeat:
                self.sticky_action_is_on = False
                self.gripper_action_repeat = 0
                self.sticky_gripper_action = 0.0

            print(relative_gripper_action)
            action["gripper"] = relative_gripper_action

        elif self.policy_setup == "widowx_bridge":
            action["gripper"] = (
                2.0 * (raw_action["open_gripper"] > 0.5) - 1.0
            )  # binarize gripper action to 1 (open) and -1 (close)
            # self.gripper_is_closed = (action['gripper'] < 0.0)

        action["terminate_episode"] = np.array([0.0])

        return raw_action, action
    
    def visualize_epoch(self, predicted_raw_actions: Sequence[np.ndarray], images: Sequence[np.ndarray], save_path: str) -> None:
        images = [self._resize_image(image) for image in images]
        ACTION_DIM_LABELS = ["x", "y", "z", "roll", "pitch", "yaw", "grasp"]

        img_strip = np.concatenate(np.array(images[::3]), axis=1)

        # set up plt figure
        figure_layout = [["image"] * len(ACTION_DIM_LABELS), ACTION_DIM_LABELS]
        plt.rcParams.update({"font.size": 12})
        fig, axs = plt.subplot_mosaic(figure_layout)
        fig.set_size_inches([45, 10])

        # plot actions
        pred_actions = np.array(
            [
                np.concatenate([a["world_vector"], a["rotation_delta"], a["open_gripper"]], axis=-1)
                for a in predicted_raw_actions
            ]
        )
        for action_dim, action_label in enumerate(ACTION_DIM_LABELS):
            # actions have batch, horizon, dim, in this example we just take the first action for simplicity
            axs[action_label].plot(pred_actions[:, action_dim], label="predicted action")
            axs[action_label].set_title(action_label)
            axs[action_label].set_xlabel("Time in one episode")

        axs["image"].imshow(img_strip)
        axs["image"].set_xlabel("Time in one episode (subsampled)")
        plt.legend()
        plt.savefig(save_path)
