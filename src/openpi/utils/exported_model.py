# ExportedModel class for loading PI0 models
import dataclasses
import abc
import functools
from typing import Any
from pathlib import Path

import jax
import numpy as np
import jaxtyping
import equinox
from etils import epath
import orbax.checkpoint as ocp
import flax.struct as struct
import contextlib
from jax import export


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
    example_batch: dict

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
        self, rng: jaxtyping.PRNGKeyArray, inputs: jaxtyping.PyTree, sample_args: dict[str, Any]
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
        print(f"Loaded exported model with {param_dtype} parameters")

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
