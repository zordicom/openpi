"""
Configurable PI0 Action Server supporting multiple model types
Copyright 2025 Zordi, Inc. All rights reserved.
"""

import os
import sys
import types
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import jax
import numpy as np
from zordi_policy_rpc.image_transforms import ImageEncoder, ImageId, ResizeAndEncodeV1
from zordi_policy_rpc.policy.server.interface import ActionServer
from zordi_policy_rpc.transport import (
    GetPolicyActionRequest,
    GetPolicyActionResponse,
    GetPolicyMetadataRequest,
    GetPolicyMetadataResponse,
    PolicyAction,
    ServerError,
)
from zordi_policy_rpc.vectors import VectorAsFieldsFactory

# Add the src directory to Python path so we can import from openpi.utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from openpi.utils.image_utils import resize_with_pad
from openpi.utils.model_serializer import restore_exported
from openpi.utils.processor_utils import load_processor, spec_select, make_batch
from openpi.utils.exported_model import ExportedModel
from openpi.utils.processor import (
    DiscretizeStates,
    ConvertStateToText,
    PredictDCTActions,
    ToInterleaved,
    PaligemmaFormatter,
    TokenizeForPaligemmaEncoder,
)


# Create dummy modules for processor loading
def create_dummy_module(module_path, class_name, class_obj):
    parts = module_path.split(".")
    current_module = sys.modules.get(parts[0])
    if current_module is None:
        current_module = types.ModuleType(parts[0])
        sys.modules[parts[0]] = current_module

    for i in range(1, len(parts)):
        sub_module_name = ".".join(parts[: i + 1])
        if not hasattr(current_module, parts[i]):
            setattr(current_module, parts[i], types.ModuleType(sub_module_name))
        current_module = getattr(current_module, parts[i])

    setattr(current_module, class_name, class_obj)
    sys.modules[module_path] = current_module


# Register the processor classes
create_dummy_module("monopi.model.processors.text", "DiscretizeStates", DiscretizeStates)
create_dummy_module("monopi.model.processors.text", "ConvertStateToText", ConvertStateToText)
create_dummy_module("monopi.model.processors.text", "TokenizeForPaligemmaEncoder", TokenizeForPaligemmaEncoder)
create_dummy_module("monopi.model.processors.interleaved_examples", "PredictDCTActions", PredictDCTActions)
create_dummy_module("monopi.model.processors.interleaved", "ToInterleaved", ToInterleaved)
create_dummy_module("monopi.lib.py.ml.tokenizer", "PaligemmaFormatter", PaligemmaFormatter)

# Constants
RAW_TEXT = "raw_text"
ROBOT_TASK_STRING = "robot_task_string"


class ConfigurablePI0ActionServer(ActionServer):
    """Configurable PI0 action server supporting multiple model types."""

    def __init__(self, config_path: str = "model_configs.yaml", model_name: Optional[str] = None):
        """
        Initialize the server with a specific model configuration.

        Args:
            config_path: Path to the YAML configuration file
            model_name: Name of the model to load (if None, uses default from config)
        """
        print("=" * 60)
        print("Initializing Configurable PI0 Action Server")

        # Load configuration
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # Select model
        if model_name is None:
            model_name = self.config["default_model"]

        if model_name not in self.config["models"]:
            raise ValueError(
                f"Model '{model_name}' not found in configuration. Available: {list(self.config['models'].keys())}"
            )

        self.model_config = self.config["models"][model_name]
        self.control_config = self.config["control"]
        self.image_config = self.config["images"]

        print(f"Loading model: {self.model_config['name']}")
        print(f"Description: {self.model_config['description']}")
        print(f"Model type: {self.model_config['model_type']}")
        print("=" * 60)

        # Initialize JAX
        print("Initializing JAX...")
        try:
            devices = jax.devices("gpu")
            print(f"✓ Using GPU for inference: {devices}")
        except Exception as e:
            print(f"Warning: No GPU available: {e}")
            devices = jax.devices("cpu")
            print(f"✓ Using CPU for inference: {devices}")

        # Load model based on type
        self._load_model()

        # Load processor
        self._load_processor()

        # Initialize field mappings
        self._initialize_field_mappings()

        # Initialize RNG key
        self.rng = jax.random.PRNGKey(0)
        print("✓ RNG key initialized")

        print("✓ Server initialization complete")

    def _load_model(self):
        """Load the model based on configuration."""
        print("-" * 40)
        print("Loading model...")

        model_path = Path(self.model_config["paths"]["model"])
        if not model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

        loading_method = self.model_config["loading_method"]
        print(f"Using loading method: {loading_method}")

        if loading_method == "restore_exported":
            # PI05 style loading
            inference_function = self.model_config["inference"]["function"]
            print(f"Loading with restore_exported, function: {inference_function}")

            self.state_dict, self.exported = restore_exported(str(model_path), names=inference_function)
            self.model_method = self.exported[inference_function]

            # For PI05, we need to handle the model call differently
            self.model_type = "pi05"

        elif loading_method == "exported_model":
            # PI0 style loading
            print("Loading with ExportedModel")
            self.model = ExportedModel.load(str(model_path))
            self.model_type = "pi0"

            # Get the input fields for PI0 models
            self.pi0_input_fields = self.model.get_sample_actions_fields()[0]
            print(f"PI0 input fields: {self.pi0_input_fields}")

        else:
            raise ValueError(f"Unknown loading method: {loading_method}")

        # Set sampling kwargs
        self.sampling_kwargs = self.model_config["inference"]["sampling_kwargs"]
        print(f"Sampling kwargs: {self.sampling_kwargs}")
        print("✓ Model loading complete")

    def _load_processor(self):
        """Load the processor based on configuration."""
        print("-" * 40)
        print("Loading processor...")

        # Update tokenizer path in PaligemmaFormatter
        tokenizer_path = self.model_config["paths"].get("tokenizer")
        if tokenizer_path:
            PaligemmaFormatter.tokenizer_path = tokenizer_path
            print(f"Set tokenizer path: {tokenizer_path}")

        # Build processor path
        model_path = Path(self.model_config["paths"]["model"])
        processor_name = self.model_config["paths"]["processor_name"]
        processor_path = model_path / "processors" / processor_name

        print(f"Processor path: {processor_path}")

        try:
            self.processor = load_processor(str(processor_path), processor_name)
            print(f"✓ Processor loaded with {len(self.processor.transformations)} transformations")
        except Exception as e:
            print(f"ERROR: Failed to load processor: {e}")
            raise

    def _initialize_field_mappings(self):
        """Initialize mappings for state and action fields."""
        print("-" * 40)
        print("Initializing field mappings...")

        # Get dimensions from config
        self.state_dim = self.model_config["state_space"]["dimensions"]
        self.action_dim = self.model_config["action_space"]["dimensions"]

        # Initialize fields dictionaries
        self.state_fields = {}
        self.action_fields = {}

        # Store slicing configuration
        self.state_slicing = self.model_config["state_space"].get("slicing")
        self.action_slicing = self.model_config["action_space"].get("slicing")

        # Create state fields from slicing configuration
        if self.state_slicing:
            for component_name, (start, end) in self.state_slicing.items():
                self.state_fields[component_name] = (start, end)

        # Create action fields from slicing configuration
        if self.action_slicing:
            for component_name, (start, end) in self.action_slicing.items():
                self.action_fields[component_name] = (start, end)

        print(f"State dimensions: {self.state_dim}, fields: {self.state_fields}")
        print(f"Action dimensions: {self.action_dim}, fields: {self.action_fields}")

    def _slice_array(self, array: np.ndarray, slicing_config: Optional[Dict[str, List[int]]]) -> np.ndarray:
        """Apply slicing configuration to an array.
        
        Args:
            array: Input array to slice
            slicing_config: Dictionary mapping component names to [start, end] indices
            
        Returns:
            Concatenated array of sliced components
        """
        if slicing_config is None:
            return array

        sliced_parts = []
        for component_name, (start, end) in slicing_config.items():
            sliced_parts.append(array[..., start:end])

        return np.concatenate(sliced_parts, axis=-1)

    def _unslice_array(
        self, sliced_array: np.ndarray, slicing_config: Optional[Dict[str, List[int]]], full_size: int = 33
    ) -> np.ndarray:
        """Reconstruct full array from sliced array.
        
        Args:
            sliced_array: Input array to unslice
            slicing_config: Dictionary mapping component names to [start, end] indices
            full_size: Size of the full output array
            
        Returns:
            Full array with sliced components placed in their original positions
        """
        if slicing_config is None:
            return sliced_array

        # Initialize full array with zeros
        full_array = np.zeros((*sliced_array.shape[:-1], full_size), dtype=sliced_array.dtype)

        # Fill in the sliced parts
        slice_idx = 0
        for component_name, (start, end) in slicing_config.items():
            slice_len = end - start
            full_array[..., start:end] = sliced_array[..., slice_idx : slice_idx + slice_len]
            slice_idx += slice_len

        return full_array

    def get_policy_metadata(self, request: GetPolicyMetadataRequest) -> GetPolicyMetadataResponse:
        """Get the policy metadata."""
        print("=" * 60)
        print("Processing metadata request")

        # Image descriptions based on config
        image_descriptions: dict[ImageId, ImageEncoder] = {}

        camera_mapping = {
            "static_center_rgb": ImageId.STATIC_CENTER_RGB,
            "eoat_left_bottom_rgb": ImageId.EOAT_LEFT_BOTTOM_RGB,
            "eoat_right_bottom_rgb": ImageId.EOAT_RIGHT_BOTTOM_RGB,
        }

        for camera in self.image_config["cameras"]:
            if camera in camera_mapping:
                image_descriptions[camera_mapping[camera]] = ResizeAndEncodeV1(
                    width=self.image_config["width"],
                    height=self.image_config["height"],
                    pad_ratio=0.0,
                )

        print(f"Image descriptions created for {len(image_descriptions)} cameras")

        state_ff = VectorAsFieldsFactory(fields=self.state_fields)
        action_ff = VectorAsFieldsFactory(fields=self.action_fields)

        response = GetPolicyMetadataResponse(
            image_descriptions=image_descriptions,
            state_fields_factory=state_ff,
            action_fields_factory=action_ff,
            observation_length=self.control_config["observation_length"],
            timedelta_sec=self.control_config["timedelta_sec"],
        )

        print(f"✓ Metadata response created")
        print(f"  - Model: {self.model_config['name']}")
        print(f"  - Observation length: {response.observation_length}")
        print(f"  - Time delta: {response.timedelta_sec}s")

        return response

    def process_request(self, request: GetPolicyActionRequest) -> Union[GetPolicyActionResponse, ServerError]:
        """Process the request and generate actions."""
        print("=" * 60)
        print(f"Processing action request for {self.model_config['name']}")

        try:
            # Update RNG key
            self.rng, action_rng = jax.random.split(self.rng)

            # Prepare observation dictionary
            outgoing_step = {}

            # Process state if provided
            if request.states and request.states[-1]:  # Use latest state
                state_vector = np.array(request.states[-1], dtype=np.float32)

                # Apply slicing if needed
                sliced_state = self._slice_array(state_vector, self.state_slicing)
                outgoing_step["observation/observation.state"] = sliced_state

                print(f"✓ State processed: original={state_vector.shape}, sliced={sliced_state.shape}")
            else:
                print("Warning: No state provided in request")

            # Map ImageId enum values to observation keys
            image_id_to_obs_key = {
                ImageId.STATIC_CENTER_RGB: "observation.images.static_center_rgb",
                ImageId.EOAT_LEFT_BOTTOM_RGB: "observation.images.eoat_left_bottom_rgb",
                ImageId.EOAT_RIGHT_BOTTOM_RGB: "observation.images.eoat_right_bottom_rgb",
            }

            print(f"Processing {len(request.images)} images...")
            images_processed = 0

            for img_id, byte_list in request.images.items():
                if not byte_list:
                    continue

                if img_id not in image_id_to_obs_key:
                    print(f"Warning: Unknown image ID: {img_id}")
                    continue

                obs_key = image_id_to_obs_key[img_id]
                img_bytes = byte_list[-1]  # Use latest image

                arr = np.frombuffer(img_bytes, dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
                if img is None:
                    print(f"ERROR: Failed to decode image for {obs_key}")
                    continue

                # Convert BGR to RGB
                if img.ndim == 3 and img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Resize and pad image
                img = resize_with_pad(img, self.image_config["height"], self.image_config["width"])

                # Add to outgoing step
                key_suffix = f"{self.image_config['height']}_{self.image_config['width']}"
                outgoing_step[f"observation/images/{obs_key}_{key_suffix}"] = img
                outgoing_step[f"observation/images/{obs_key}_{key_suffix}_mask"] = np.array(1, dtype=np.bool_)
                images_processed += 1

            print(f"✓ Processed {images_processed} images")

            # Add text instruction
            # raw_text = "pick the ripe strawberry"  # Could be made configurable
            raw_text = "be a good robot"
            outgoing_step[RAW_TEXT] = raw_text
            outgoing_step[ROBOT_TASK_STRING] = raw_text
            print(f"✓ Added instruction: '{raw_text}'")

            # Process the step through the processor
            print("Processing through transformations...")
            processed_step, _ = self.processor.process(outgoing_step, {})

            # Generate actions based on model type
            print(f"Generating actions with {self.model_type} model...")
            import time

            start_time = time.time()

            if self.model_type == "pi05":
                # PI05 style inference
                processed_step = make_batch(processed_step)
                predicted_action = self.model_method.call(
                    self.state_dict, action_rng, processed_step, **self.sampling_kwargs
                )
                # Unprocess the predictions
                unprocessed = self.processor.unprocess(processed_step, predicted_action, has_batch_dim=True)
                raw_predicted_action = unprocessed[1]["actions"][0]  # remove batch dimension

            else:  # pi0
                # PI0 style inference
                # Only pass the fields the model expects
                model_inputs = spec_select(processed_step, self.pi0_input_fields)
                predicted_action = self.model.sample_actions(action_rng, model_inputs, sample_args=self.sampling_kwargs)
                # Add loss_multiplier for unprocessing
                predicted_action["loss_multiplier"] = np.ones_like(predicted_action["actions"])
                unprocessed_action = self.processor.unprocess(processed_step, predicted_action, has_batch_dim=False)[1][
                    "actions"
                ]
                raw_predicted_action = unprocessed_action

            elapsed = time.time() - start_time
            print(f"✓ Model inference completed in {elapsed:.3f}s")
            print(f"✓ Raw actions shape: {raw_predicted_action.shape}")

            # Convert to RPC response format
            rpc_policy_actions: List[PolicyAction] = []
            time_delta = self.control_config["timedelta_sec"]
            num_actions_to_use = min(self.control_config["actions_per_inference"], len(raw_predicted_action))

            print(f"Converting {num_actions_to_use} actions to RPC format...")

            for i in range(num_actions_to_use):
                # Unslice actions if needed (to full 33-dim)
                action = self._unslice_array(raw_predicted_action[i], self.action_slicing, self.action_dim)

                rpc_policy_actions.append(
                    PolicyAction(
                        action=action.tolist(),
                        time_from_start_sec=(i + 1) * time_delta,
                    )
                )

            response = GetPolicyActionResponse(action=rpc_policy_actions)
            print(f"✓ Response created with {len(rpc_policy_actions)} actions")
            print("✓ Request processing complete")

            return response

        except Exception as e:
            print(f"ERROR: Exception in process_request:")
            import traceback

            traceback.print_exc()
            error_msg = f"Error in process_request: {str(e)}"
            return ServerError(message=error_msg)


def main():
    """Example usage of the configurable server."""
    import argparse

    parser = argparse.ArgumentParser(description="Configurable PI0 Action Server")
    parser.add_argument("--config", default="model_configs.yaml", help="Path to configuration file")
    parser.add_argument("--model", help="Model name to load (if not specified, uses default from config)")
    args = parser.parse_args()

    # Create server
    server = ConfigurablePI0ActionServer(config_path=args.config, model_name=args.model)

    # Example: get metadata
    metadata_request = GetPolicyMetadataRequest()
    metadata_response = server.get_policy_metadata(metadata_request)
    print(f"\nMetadata response: {metadata_response}")

    # The server is now ready to process action requests
    print("\n✓ Server is ready to process action requests")


if __name__ == "__main__":
    main()
