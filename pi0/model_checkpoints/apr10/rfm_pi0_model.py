"""
Copyright 2025 Zordi, Inc. All rights reserved.
"""

import abc
import contextlib
import dataclasses
import functools
import os
import pathlib
import random
import string
import time
from typing import Any

import cloudpathlib
import equinox
from etils import epath
import flax.struct as struct
import jax
from jax import export
import jax.numpy as jnp
import jaxtyping
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
import matplotlib.pyplot as plt
import numpy as np
import orbax.checkpoint as ocp
import sentencepiece
import tqdm

# Download from s3://zordi/paligemma_tokenizer.model
_TOKENIZER_PATH = "../../paligemma_tokenizer.model"

PROCESSORS = "processors"
PROCESS = "process"
UNPROCESS = "unprocess"
FN = "fn"
RAW_TEXT = "raw_text"
SERIALIZATION_META = "transformation.yaml"


# Helper functions
def resize_with_pad(
    images,
    height: int,
    width: int,
    method: jax.image.ResizeMethod = jax.image.ResizeMethod.LINEAR,
):
    """Replicates tf.image.resize_with_pad. Resizes an image to a target height and width without distortion
    by padding with zeros.
    """
    has_batch_dim = images.ndim == 4
    if not has_batch_dim:
        images = images[None]  # type: ignore
    cur_height, cur_width = images.shape[1:3]
    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_images = jax.image.resize(
        images,
        (images.shape[0],) + (resized_height, resized_width) + (images.shape[3],),
        method=method,
    )
    # round from float back to uint8
    resized_images = jnp.round(resized_images).clip(0, 255).astype(jnp.uint8)

    pad_h0, remainder_h = divmod(height - resized_height, 2)
    pad_h1 = pad_h0 + remainder_h
    pad_w0, remainder_w = divmod(width - resized_width, 2)
    pad_w1 = pad_w0 + remainder_w
    padded_images = jnp.pad(resized_images, ((0, 0), (pad_h0, pad_h1), (pad_w0, pad_w1), (0, 0)))

    if not has_batch_dim:
        padded_images = padded_images[0]
    return padded_images


def resize_with_pad_collection(
    images,
    height: int,
    width: int,
    method: jax.image.ResizeMethod = jax.image.ResizeMethod.LINEAR,
):
    """Resizes a collection of batched images coming out of the data source, skipping the masks."""

    # Need to make a new dict, rather than just returning the resized images.
    # Otherwise the jaxtyping will complain.
    out = {}

    for k, v in images.items():
        if k.endswith("_mask") or v.shape[1:3] == (height, width):
            out[k] = v
        else:
            out[k] = resize_with_pad(v, height, width, method=method)
    return out


def batch_select(batch: dict[str, Any], fields) -> dict[str, Any]:
    """Helper function to pick out desired fields in the batch."""
    output = {}
    for field in fields:
        if field in batch:
            output[field] = batch[field]
    return output


def spec_select(batch: dict[str, Any], spec: set) -> dict[str, Any]:
    """Helper function to pick out desired fields based on spec."""
    for k in spec:
        if k not in batch:
            raise ValueError(f"Field {k} not found in batch {batch.keys()=}")
    return {k: batch[k] for k in spec}


def make_batch(batch: dict | np.ndarray | str, name: str = "batch"):
    if isinstance(batch, dict):
        return {k: make_batch(v, k) for k, v in batch.items()}

    if isinstance(batch, str):
        # This field should be made into a list.
        # Note: important for this to come first, since numpy string arrays have shape,
        # so would get caught by the hasattr(batch, "shape") check below & throw error.
        return [batch]

    if hasattr(batch, "shape"):
        # This field is an array, insert a dimension.
        return batch[None]

    raise ValueError(f"Unknown batch field {name}: {batch}")


def unmake_batch(batch: dict | np.ndarray | str, name: str = "batch", strict_check: bool = True):
    if isinstance(batch, dict):
        return {k: unmake_batch(v, k, strict_check=strict_check) for k, v in batch.items()}

    if hasattr(batch, "shape"):
        # This field is an array, return first value.
        if batch.shape[0] != 1 and strict_check:
            raise ValueError(f"Removing batch dimension for field {name} with shape {batch.shape}: {batch}")
        return batch[0]

    # This is a list.
    if not hasattr(batch, "__len__"):
        raise ValueError(f"Removing batch dimension for field {name} without: {batch}")
    if len(batch) != 1 and strict_check:
        raise ValueError(f"Removing batch dimension for field {name} with length {len(batch)}: {batch}")
    return batch[0]


# Helper for loading model weights
@contextlib.contextmanager
def disable_typechecking():
    initial = jaxtyping.config.jaxtyping_disable
    jaxtyping.config.update("jaxtyping_disable", True)
    yield
    jaxtyping.config.update("jaxtyping_disable", initial)


@dataclasses.dataclass
class FlaxRestore(ocp.args.CheckpointArgs):
    item: struct.PyTreeNode | None


@dataclasses.dataclass
class InferenceModel(abc.ABC):
    """A model that supports inference via `sample_actions`."""

    # The only required key is "actions", which is used to infer the action horizon and action dim of the model.
    # However, subclasses can (and should) use the example batch to validate inputs.
    example_batch: dict[str, Any]

    @property
    def action_horizon(self) -> int:
        assert self.example_batch["actions"].ndim == 3, self.example_batch["actions"].shape
        return self.example_batch["actions"].shape[1]

    @property
    def action_dim(self) -> int:
        assert self.example_batch["actions"].ndim == 3, self.example_batch["actions"].shape
        return self.example_batch["actions"].shape[2]

    @abc.abstractmethod
    def sample_actions(self, rng, batch: jaxtyping.PyTree, sample_args: dict[str, Any]) -> jaxtyping.PyTree:
        """Sample actions from the model, with or without a batch dimension."""

    def __str__(self) -> str:
        s_example_batch = equinox.tree_pformat(self.example_batch, struct_as_array=True).replace("\n", "\n  ")
        s = f"{self.__class__.__name__}(example_batch={s_example_batch}\n)"
        return s


class FlaxCheckpointHandler(ocp.AsyncCheckpointHandler):
    """A CheckpointHandler for saving `flax.struct.dataclass` objects."""

    def __init__(self):
        self._pytree_handler = ocp.PyTreeCheckpointHandler(use_ocdbt=True, use_zarr3=True)

    def save(self):
        pass

    def async_save(self):
        pass

    def restore(self, directory: str | epath.Path):
        directory = epath.Path(directory)

        # then, restore the non-metadata as a PyTree adhering to the structure
        with disable_typechecking():
            # if structure is None, we still need to get the "raw" (nested dict) structure to construct the restore args
            raw_structure = self._pytree_handler.metadata(directory)
            restored = self._pytree_handler.restore(
                directory,
                args=ocp.args.PyTreeRestore(
                    item=None,
                    # restore as NumPy arrays, ignoring sharding
                    restore_args=jax.tree.map(
                        lambda _: ocp.RestoreArgs(restore_type=np.ndarray),
                        raw_structure,
                    ),
                ),
            )

        return restored

    def finalize(self, directory: epath.Path) -> None:
        return self._pytree_handler.finalize(directory)

    def close(self):
        return self._pytree_handler.close()


@functools.partial(jax.jit, static_argnums=(0,))
def sample_actions_fn(exported, params, rng, inputs, sample_args):
    return exported.call(params, rng, inputs, sample_args)


@dataclasses.dataclass
class ExportedModel(InferenceModel):
    """A model deserialized from an exported computational graph. Does not support batched sampling."""

    params: Any
    exported: export.Exported
    _positional_output: bool
    _input_fields: set[str]
    _output_fields: set[str]
    _sample_args_fields: set[str]

    @property
    def sample_args_fields(self) -> set[str]:
        return self._sample_args_fields

    def sample_actions(
        self,
        rng: jaxtyping.PRNGKeyArray,
        inputs: jaxtyping.PyTree,
        sample_args: dict[str, Any],
    ) -> jaxtyping.PyTree:
        if str(rng.dtype) == "key<fry>":
            rng = jax.random.key_data(rng)

        outputs = sample_actions_fn(self.exported, self.params, rng, inputs, sample_args)
        return outputs

    def get_sample_actions_fields(self) -> tuple[set, set]:
        """This function should return two sets corresponding to the names of inputs and outputs of sample_actions."""
        return self._input_fields, self._output_fields

    @classmethod
    def load(cls, path: str) -> "ExportedModel":
        """Loads an exported model. Expects a directory containing a `graph` file (created by `Model.export`) as
        well as serialized model parameters (created by `orbax.FlaxSave(model)`).
        """
        path = epath.Path(path).expanduser().resolve()

        # item=None avoids loading the metadata, returning a bare nested dict instead
        params = FlaxCheckpointHandler().restore(path)["params"]

        with (path / "graph").open("rb") as f:
            exported = export.deserialize(f.read())

        input_spec = jax.tree.unflatten(exported.in_tree, exported.in_avals)
        output_spec = jax.tree.unflatten(exported.out_tree, exported.out_avals)

        param_dtype = jax.tree.leaves(input_spec[0][0])[0].dtype
        params = jax.tree.map(lambda x: x.astype(param_dtype), params)
        params = jax.device_put(params)  # put params on the default device, which will be a GPU if available
        print("Loaded exported model")

        input_fields = set(input_spec[0][2].keys())
        example_input = {**input_spec[0][2], **output_spec}
        output_fields = set(output_spec.keys())

        # Get sample_args fields.
        sample_args_fields = set(input_spec[0][3].keys())

        example_batch = jax.tree.map(lambda x: jax.ShapeDtypeStruct((1,) + x.shape, x.dtype), example_input)

        return cls(
            example_batch=example_batch,
            params=params,
            exported=exported,
            _positional_output=(not isinstance(output_spec, dict)),
            _input_fields=input_fields,
            _output_fields=output_fields,
            _sample_args_fields=sample_args_fields,
        )


@dataclasses.dataclass(frozen=False)
class GraphTransformation:
    ops: dict

    def process(self, inputs: dict[str, Any], outputs: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        spec = self.ops[PROCESS]["input_spec"][0]
        if len(spec) == 1:
            new_inputs, new_outputs = self.ops[PROCESS][FN](spec_select(inputs, spec[0].keys()))
        else:
            new_inputs, new_outputs = self.ops[PROCESS][FN](
                spec_select(inputs, spec[0].keys()),
                spec_select(outputs, spec[1].keys()),
            )
        inputs.update(new_inputs)
        outputs.update(new_outputs)
        return inputs, outputs

    def unprocess(self, inputs: dict[str, Any], outputs: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        spec = self.ops[UNPROCESS]["input_spec"][0]
        new_inputs, new_outputs = self.ops[UNPROCESS][FN](
            spec_select(inputs, spec[0].keys()), spec_select(outputs, spec[1].keys())
        )
        inputs.update(new_inputs)
        outputs.update(new_outputs)
        return inputs, outputs


@dataclasses.dataclass(frozen=False)
class Processor:
    """
    WARNING: This class mutates inputs and outputs in place. This is necessary to avoid copying large arrays.
    Do not make assumptions about the contents of the original inputs and outputs dictionaries after calling process or unprocess.
    """

    name: str
    transformations: list[GraphTransformation]

    def process(
        self, inputs: dict[str, Any], outputs: dict[str, Any] = {}, has_batch_dim: bool = False
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Process batch (dictionary mapping fields to values) to transform inputs before passing to the model.
        If the batch contains actions, these are transformed to make the suitable for use in training.
        The arguments inputs and outputs are unprocessed batches.
        """
        if not has_batch_dim:
            inputs = make_batch(inputs)
            outputs = make_batch(outputs)

        for transformation in self.transformations:
            try:
                inputs, outputs = transformation.process(inputs, outputs)
            except Exception as e:
                print(f"Error in transformation {transformation}: {e}")
                raise e

        if not has_batch_dim:
            inputs = unmake_batch(inputs)
            outputs = unmake_batch(outputs)
        return inputs, outputs

    def unprocess(
        self, inputs: dict[str, Any], outputs: dict[str, Any], has_batch_dim: bool = False
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Process model outputs to convert them to raw actions. The outputs batch should contain whatever the
        model outputs, and the inputs batch corresponds to the inputs thta were provided to the model. This
        assumes that the inputs were returned by the process_inputs method above.
        The arguments inputs and outputs are processed batches.
        """
        if not has_batch_dim:
            inputs = make_batch(inputs)
            outputs = make_batch(outputs)

        for transformation in reversed(self.transformations):
            inputs, outputs = transformation.unprocess(inputs, outputs)

        if not has_batch_dim:
            inputs = unmake_batch(inputs)
            outputs = unmake_batch(outputs)
        return inputs, outputs


@dataclasses.dataclass(frozen=False)
class TokenizeForPaligemmaEncoder:
    text_len: int = 48
    _tokenizer: sentencepiece.SentencePieceProcessor | None = None

    def _load_tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = sentencepiece.SentencePieceProcessor(_TOKENIZER_PATH)

    def process(self, inputs: dict[str, Any], outputs: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        if RAW_TEXT not in inputs:
            raise ValueError(f"Raw text not found in inputs. Current input keys: {inputs.keys()}")

        self._load_tokenizer()

        ret_tokens = []
        ret_mask_input = []
        for text in inputs[RAW_TEXT]:
            cleaned_text = text.lower().strip().replace("_", " ").replace("\n", " ")
            # tokenize "\n" separately as the "start of answer" token
            tokens = self._tokenizer.encode(cleaned_text, add_bos=True) + self._tokenizer.encode("\n")
            input_mask = [1] * len(tokens)
            padding = [0] * max(0, self.text_len - len(tokens))
            tokens = tokens[: self.text_len] + padding
            input_mask = input_mask[: self.text_len] + padding

            ret_tokens.append(tokens)
            ret_mask_input.append(input_mask)

        inputs["prompt_tokens"] = np.array(ret_tokens)
        inputs["mask_input"] = np.array(ret_mask_input)
        assert inputs["prompt_tokens"].shape == (len(inputs[RAW_TEXT]), self.text_len)
        assert inputs["mask_input"].shape == (len(inputs[RAW_TEXT]), self.text_len)
        return inputs, outputs

    def unprocess(self, inputs: dict[str, Any], outputs: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        return inputs, outputs

    def update_params(self, inputs: dict[str, Any], outputs: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        # Skip to make param initialization faster
        # assumes that no downstream processor needs outputs of this processor
        return inputs, outputs


def load_transformation(path: str) -> GraphTransformation:
    if os.path.exists(os.path.join(path, PROCESS)):
        operations = [PROCESS, UNPROCESS]
        out = {}
        for operation_name in operations:
            with open(os.path.join(path, operation_name), "rb") as f:
                exported = export.deserialize(f.read())
            input_spec = jax.tree.unflatten(exported.in_tree, exported.in_avals)
            output_spec = jax.tree.unflatten(exported.out_tree, exported.out_avals)
            out[operation_name] = {
                FN: exported.call,
                "input_spec": input_spec,
                "output_spec": output_spec,
            }

        return GraphTransformation(out)

    with open(os.path.join(path, SERIALIZATION_META)) as f:
        transformation = f.read()

    transformation_dict = {}

    for line in transformation.split("\n"):
        if ":" not in line:
            continue
        key, value = line.split(":")
        transformation_dict[key] = value.strip()

    assert transformation_dict["__name__"] == "TokenizeForPaligemmaEncoder", transformation_dict

    return TokenizeForPaligemmaEncoder(text_len=int(transformation_dict["text_len"]))


class Pi0Model:
    """Class representing the Pi0 robotics model for predicting actions based on observations."""

    def __init__(self, model_path=None, image_height=224, image_width=224):
        """Initialize the Pi0Model.

        Args:
            model_path: Path to the checkpoint directory containing the model
            image_height: Height to resize input images to
            image_width: Width to resize input images to
        """
        # Check if GPU is available
        jax.devices("gpu")

        self.model_path = model_path
        self.model = None
        self.processor = None
        self.image_height = image_height
        self.image_width = image_width
        self.masking = np.array(0, dtype=np.bool_)
        self.non_masking = np.array(1, dtype=np.bool_)

        if model_path:
            self.init_model(model_path)

    def init_model(self, model_path):
        """Initialize the model from a checkpoint path.

        Args:
            model_path: Path to the checkpoint directory containing the model
        """
        self.model_path = model_path
        ckpt_path = pathlib.Path(model_path)
        processor_path = ckpt_path / "processors" / "zordi_pick_strawberry"

        # Load the model
        self.model = ExportedModel.load(str(ckpt_path))
        print(f"Model loaded from {ckpt_path}")

        # Load the processor
        self.init_processor(processor_path)

    def init_processor(self, processor_path):
        """Initialize the processor for the model.

        Args:
            processor_path: Path to the processor directory
        """
        transformations = []
        paths = os.listdir(processor_path)
        paths = [path for path in paths if path.isdigit()]
        indices = [int(path) for path in paths]
        order = np.argsort(indices)

        for i in order:
            transformations.append(load_transformation(os.path.join(processor_path, paths[i])))

        self.processor = Processor("zordi_processor", transformations)
        print(f"Processor initialized with {len(transformations)} transformations")

    def predict(self, obs_dict: dict[str, np.ndarray], raw_text="pick a ripe strawberry"):
        """Run prediction on a single observation step.

        Args:
            obs_dict: Dictionary containing observation data
            raw_text: Text instruction for the model

        Returns:
            Predicted actions
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call init_model first.")

        image_keys = [
            "observation.images.static_top_down",
            "observation.images.eoat_top",
            "observation.images.eoat_bottom",
        ]
        state_key = "observation.state"
        sample_args = {"num_steps": 10}

        # Process the observation data
        outgoing_step = {}

        for image_key in image_keys:
            outgoing_step[f"observation/images/{image_key}_{self.image_height}_{self.image_width}"] = resize_with_pad(
                obs_dict[image_key], self.image_height, self.image_width
            )
            outgoing_step[f"observation/images/{image_key}_{self.image_height}_{self.image_width}_mask"] = (
                self.non_masking
            )

        outgoing_step[f"observation/{state_key}"] = obs_dict[state_key]
        outgoing_step[RAW_TEXT] = raw_text

        # Process the step for model input
        processed_step, _ = self.processor.process(outgoing_step, {})

        batch = batch_select(processed_step, self.model.get_sample_actions_fields()[0])
        # Run model prediction
        predicted_action = self.model.sample_actions(jax.random.PRNGKey(0), batch, sample_args=sample_args)

        # Add dummy loss multiplier for the processor
        predicted_action["loss_multiplier"] = np.ones_like(predicted_action["actions"])

        # Unprocess the predictions to get raw actions
        raw_predicted_action = self.processor.unprocess(processed_step, predicted_action, has_batch_dim=False)[1][
            "actions"
        ]

        return raw_predicted_action


def _get_tmp_download_dir(root: str) -> str:
    random_string = "".join(random.choices(string.ascii_letters + string.digits, k=8))
    download_dir = root.replace("s3://", "/tmp/")
    download_dir += random_string
    pathlib.Path(download_dir).mkdir(parents=True, exist_ok=True)
    return download_dir


def _files_for_episode(dataset_meta: LeRobotDatasetMetadata, episode_index: int) -> list[str]:
    files = [str(dataset_meta.get_data_file_path(episode_index))]
    if len(dataset_meta.video_keys) > 0:
        video_files = [
            str(dataset_meta.get_video_file_path(episode_index, vid_key)) for vid_key in dataset_meta.video_keys
        ]
        files += video_files

    return files


def _download_files(remote_path: cloudpathlib.CloudPath, download_dir: str, files: list[str]) -> list[str]:
    paths = []
    for file in files:
        file_path = os.path.join(download_dir, file)
        pathlib.Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        (remote_path / file).download_to(file_path)
        paths.append(file_path)
    return paths


def visualize_predictions(
    ground_truth_actions, predicted_actions, save_path=None, title="Ground Truth vs Predicted Actions"
):
    """Visualize the ground truth actions vs predicted actions.

    Args:
        ground_truth_actions: Array of ground truth actions
        predicted_actions: Array of predicted actions
        save_path: Path to save the figure to (if None, will display the figure)
        title: Title for the plot
    """
    # Process the predicted actions if they have an extra dimension
    if predicted_actions.ndim > 2:
        # For each predicted action chunk, we execute the first half of the chunk
        timestep_between_inference = predicted_actions.shape[1]

        actions_to_execute = []

        for i in range(0, predicted_actions.shape[0], timestep_between_inference):
            actions_to_execute.append(predicted_actions[i, :timestep_between_inference])

        actions_to_execute = np.concatenate(actions_to_execute, axis=0)
    else:
        actions_to_execute = predicted_actions

    # Get the number of timesteps and action dimensions
    n_timesteps, n_dims = ground_truth_actions.shape

    # Create a figure with subplots for each action dimension
    fig, axes = plt.subplots(n_dims, 1, figsize=(12, 4 * n_dims), sharex=True)
    fig.suptitle(title)

    # Plot each dimension
    for i in range(min(n_dims, actions_to_execute.shape[1])):
        ax = axes[i] if n_dims > 1 else axes

        ax.plot(ground_truth_actions[:, i], label="Ground Truth", color="blue")
        ax.plot(actions_to_execute[:, i], label="Predicted", color="red", linestyle="--")
        ax.set_ylabel(f"Dim {i + 1}")
        ax.legend()

    # Set common x-label
    axes[-1].set_xlabel("Timestep")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()  # Close the figure to free up memory
        print(f"Saved: {save_path}")
    else:
        plt.show()


def main():
    """Main function to load model, process data and visualize results."""
    # Initialize the model
    model_path = "./nov4_zordi_pick_strawberry/0ke1t39y/20000/model"
    pi0_model = Pi0Model(model_path=model_path)

    # Define data loading parameters
    root_path = "s3://zordi/h-003-lerobot-converted/dataset"
    episode_index = 1

    # Download and load dataset
    download_dir = _get_tmp_download_dir(root_path)
    print("Temporary download directory:", download_dir)

    custom_client = cloudpathlib.s3.S3Client(profile_name="pi")
    remote_path = cloudpathlib.S3Path(root_path, client=custom_client)

    # Download meta/ only because we download videos later
    to_clean_up = []
    s = time.time()
    to_clean_up.extend(_download_files(remote_path, download_dir, ["meta"]))
    print(f"Featurizer downloaded meta in {time.time() - s} seconds")

    dataset_meta = LeRobotDatasetMetadata(
        repo_id="dataset",
        root=download_dir,
        local_files_only=True,
    )

    s = time.time()
    files = _files_for_episode(dataset_meta, episode_index)
    to_clean_up.extend(_download_files(remote_path, download_dir, files))
    print(f"Featurizer downloaded files in {time.time() - s} seconds")

    dataset = LeRobotDataset(
        repo_id="dataset",
        root=download_dir,
        episodes=[episode_index],
        local_files_only=True,
    )

    # Process data and collect predictions
    from_idx = dataset.episode_data_index["from"][0].item()
    to_idx = dataset.episode_data_index["to"][0].item()

    image_keys = [
        "observation.images.static_top_down",
        "observation.images.eoat_top",
        "observation.images.eoat_bottom",
    ]
    state_key = "observation.state"
    raw_text = "pick a ripe strawberry"
    num_step_to_visualize = 20000

    predicted_actions = []
    gt_actions = []

    for step_idx in tqdm.tqdm(range(from_idx, to_idx)):
        step = dataset[step_idx]

        obs_dict = {}

        # Preprocess images
        for image_key in image_keys:
            image = step[image_key].numpy()
            image = image.transpose(1, 2, 0)
            obs_dict[image_key] = (image * 255).astype(np.uint8)

        obs_dict[state_key] = step[state_key].numpy()

        # Get ground truth action
        gt_actions.append(step["action"].numpy())

        # Get model prediction
        predicted_action = pi0_model.predict(obs_dict, raw_text=raw_text)

        predicted_actions.append(predicted_action)

        if step_idx - from_idx == num_step_to_visualize:
            break

    # Convert to arrays for visualization
    ground_truth_actions = np.array(gt_actions)
    predicted_actions = np.array(predicted_actions)

    print(f"{ground_truth_actions.shape = }")
    print(f"{predicted_actions.shape = }")

    # Visualize results
    visualize_predictions(
        ground_truth_actions,
        predicted_actions,
        save_path="ground_truth_vs_predicted_actions_pi0.png",
        title="Ground Truth vs Predicted Actions pi_0",
    )


if __name__ == "__main__":
    main()
