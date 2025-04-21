import dataclasses
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms


def make_xarm_example() -> dict:
    """Creates a random input example for the xArm policy."""
    return {
        "state": np.ones((16,)),
        "images": {
            "base_0_rgb": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
            "left_wrist_0_rgb": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
            "right_wrist_0_rgb": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        },
        "prompt": "pick the ripe strawberry",
    }


@dataclasses.dataclass(frozen=True)
class XArmInputs(transforms.DataTransformFn):
    """Inputs for the xArm policy.

    Expected inputs:
    - images: dict[name, img] where img is [channel, height, width]. name must be in EXPECTED_CAMERAS.
    - state: [16]
    - actions: [action_horizon, 16]
    """

    # The action dimension of the model. Will be used to pad state and actions.
    action_dim: int

    # The expected cameras names. All input cameras must be in this set. Missing cameras will be
    # replaced with black images and the corresponding `image_mask` will be set to False.
    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = (
        "base_0_rgb",
        "left_wrist_0_rgb",
        "right_wrist_0_rgb",
    )

    def __call__(self, data: dict) -> dict:
        data = _decode_xarm(data)

        # Get the state. We are padding from 14 to the model action dim.
        state = transforms.pad_to_dim(data["state"], self.action_dim)

        in_images = data["images"]
        if set(in_images) - set(self.EXPECTED_CAMERAS):
            raise ValueError(f"Expected images to contain {self.EXPECTED_CAMERAS}, got {tuple(in_images)}")

        # Initialize image dictionary with required keys
        images = {}
        image_masks = {}

        # Use top view as base image
        for name in self.EXPECTED_CAMERAS:
            images[name] = in_images[name]
            image_masks[name] = np.True_

        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": state,
        }

        # Actions are only available during training.
        if "actions" in data:
            actions = np.asarray(data["actions"])
            inputs["actions"] = transforms.pad_to_dim(actions, self.action_dim)

        if "prompt" in data:
            inputs["prompt"] = str(data["prompt"])
        else:
            inputs["prompt"] = "fold the towel"

        return inputs


@dataclasses.dataclass(frozen=True)
class XArmOutputs(transforms.DataTransformFn):
    """Outputs for the xArm policy."""

    def __call__(self, data: dict) -> dict:
        # Only return the first 16 dims.
        actions = np.asarray(data["actions"][:, :16])
        actions = actions * _joint_flip_mask()
        return {"actions": actions}


def _decode_xarm(data: dict) -> dict:
    # State is [left_arm_joint_angles, left_arm_gripper, right_arm_joint_angles, right_arm_gripper]
    # dimension sizes: [7, 1, 7, 1]
    state = np.asarray(data["state"])
    state = state * _joint_flip_mask()
    data["state"] = state

    if "actions" in data:
        actions = np.asarray(data["actions"])
        actions = actions * _joint_flip_mask()
        data["actions"] = actions

    data["images"] = {name: _parse_image(img) for name, img in data["images"].items()}

    return data


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def _joint_flip_mask() -> np.ndarray:
    return np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
