# Model serialization
import dataclasses
import functools
import jax
import jax.numpy as jnp
import numpy as np
import pprint
import json
from etils import epath
from typing import Any
import jax.export
from jax._src.export.shape_poly import _DimExpr
import jaxtyping
import jax._src.tree_util as private_tree_util
import orbax.checkpoint as ocp


PROCESSORS = "processors"
PROCESS = "process"
UNPROCESS = "unprocess"
FN = "fn"
ROBOT_TASK_STRING = "robot_task_string"
RAW_TEXT = "raw_text"
SERIALIZATION_META = "transformation.yaml"

Batch = dict[str, Any]
DimSize = _DimExpr | int


@dataclasses.dataclass(slots=True, frozen=True)
class _PolymorphicInteger:
    value: DimSize

    def __repr__(self):
        return f"PolymorphicInteger({self.value})"


def _represents_polymorphic_integer(shape_dtype) -> bool:
    if hasattr(shape_dtype, "shape"):
        if shape_dtype.dtype == jnp.float32 and len(shape_dtype.shape) == 2 and shape_dtype.shape[0] == 0:
            return True
        if any(s == 0 for s in shape_dtype.shape):
            raise ValueError(
                f"Shape contains zeros, but does not match expected format for a polymorphic integer (0, n): {shape_dtype}"
            )
    return False


def _symbolic_shapes_to_polymorphic_integers(pytree, *, wrap: bool):
    def f(shape_dtype):
        if _represents_polymorphic_integer(shape_dtype):
            if wrap:
                return _PolymorphicInteger(shape_dtype.shape[1])
            return shape_dtype.shape[1]
        return shape_dtype

    return jax.tree.map(f, pytree)


def _polymorphic_integers_to_arrays(pytree, spec_pytree):
    def f(x, s):
        if isinstance(s, _PolymorphicInteger):
            return np.zeros((0, x), dtype=jnp.float32)
        return x

    return jax.tree.map(f, pytree, spec_pytree)


def unwrap_prng_keys(pytree):
    """Utility function to unwrap new-style key arrays, since they do not currently play well with Flax serialization.

    See https://jax.readthedocs.io/en/latest/jep/9263-typed-keys.html.
    """

    def _convert(x):
        if hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jax.dtypes.prng_key):
            return jax.random.key_data(x)
        return x

    return jax.tree.map(_convert, pytree)


def check_pytree_equality(
    *, expected: jaxtyping.PyTree, got: jaxtyping.PyTree, check_shapes: bool = False, check_dtypes: bool = False
):
    """Checks that two PyTrees have the same structure and optionally checks shapes and dtypes. Creates a much nicer
    error message than if `jax.tree.map` is naively used on PyTrees with different structures.
    """

    if errors := list(private_tree_util.equality_errors(expected, got)):
        msg = ["PyTrees have different structure:"]
        msg.extend(
            f"   - at keypath '{jax.tree_util.keystr(path)}': expected {thing1}, got {thing2}, so {explanation}."
            for path, thing1, thing2, explanation in errors
        )
        msg.extend(["Expected structure:", str(jax.tree.structure(expected))])
        msg.extend(["Got structure:", str(jax.tree.structure(got))])
        raise ValueError("\n".join(msg))

    if check_shapes or check_dtypes:

        def check(kp, x, y):
            if check_shapes or check_dtypes:
                if not hasattr(x, "shape"):
                    x = np.asarray(x)
                if not hasattr(y, "shape"):
                    y = np.asarray(y)

            if check_shapes and x.shape != y.shape:
                raise ValueError(f"Shape mismatch at {jax.tree_util.keystr(kp)}: expected {x.shape}, got {y.shape}")

            if check_dtypes and x.dtype != y.dtype:
                raise ValueError(f"Dtype mismatch at {jax.tree_util.keystr(kp)}: expected {x.dtype}, got {y.dtype}")

        jax.tree_util.tree_map_with_path(check, expected, got)


@dataclasses.dataclass(frozen=True, eq=False)
class ExportedModuleMethod:
    exported: jax.export.Exported
    defaults: dict[str, float | int | bool]

    @property
    def state_dict_spec(self):
        return jax.tree.unflatten(self.exported.in_tree, self.exported.in_avals)[0][0]

    @property
    def args_spec(self):
        spec = jax.tree.unflatten(self.exported.in_tree, self.exported.in_avals)[0][1:]
        return _symbolic_shapes_to_polymorphic_integers(spec, wrap=True)

    @property
    def kwargs_spec(self):
        spec = jax.tree.unflatten(self.exported.in_tree, self.exported.in_avals)[1]
        return _symbolic_shapes_to_polymorphic_integers(spec, wrap=True)

    @property
    def out_spec(self):
        return jax.tree.unflatten(self.exported.out_tree, self.exported.out_avals)

    def __repr__(self):
        return (
            f"ExportedModuleMethod(\n"
            f"  name={self.exported.fun_name}\n"
            f"  args_spec={pprint.pformat(self.args_spec)}\n"
            f"  kwargs_spec={pprint.pformat(self.kwargs_spec)}\n"
            f"  out_spec={pprint.pformat(self.out_spec)}\n"
            f"  defaults={pprint.pformat(self.defaults)}\n"
            f")"
        )

    @functools.partial(jax.jit, static_argnums=0)
    def _jitted_call(self, state_dict, *args, **kwargs):
        args, kwargs = unwrap_prng_keys((args, kwargs))
        # JITing this dtype cast reduces memory usage, since JAX can modify the state_dict buffers in place.
        state_dict = jax.tree.map(lambda x, y: x.astype(y.dtype), state_dict, self.state_dict_spec)
        return self.exported.call(state_dict, *args, **kwargs)

    def call(self, state_dict, *args, ignore_extra_keys: bool = True, **kwargs):
        """Call the exported method with the given state dict and arguments.

        If `ignore_extra_keys` is True, any extra keys in `args` or `kwargs` that are not present in the exported
        method's input signature are silently ignored. Note that "keys" refer specifically to dictionary keys at or below
        one level of nesting -- i.e., extra args and kwargs cannot be ignored, but if any args or kwargs are
        dictionaries, elements of those dictionaries can be ignored.
        """
        kwargs = {**self.defaults, **kwargs}
        if len(args) != len(self.args_spec):
            raise ValueError(f"Expected {len(self.args_spec)} arguments, got {len(args)}")
        if kwargs.keys() != self.kwargs_spec.keys():
            raise ValueError(f"Expected kwargs {self.kwargs_spec.keys()}, got {kwargs.keys()}")
        if ignore_extra_keys:

            def intersect(d1, d2):
                if not isinstance(d1, dict) or not isinstance(d2, dict):
                    return d2
                return {
                    k: jax.tree.map(intersect, d1[k], d2[k], is_leaf=lambda x: isinstance(x, dict))
                    for k in set(d1.keys()) & set(d2.keys())
                }

            args = tuple(intersect(a, b) for a, b in zip(self.args_spec, args, strict=True))
            kwargs = {k: intersect(self.kwargs_spec[k], v) for k, v in kwargs.items()}
        check_pytree_equality(expected=(self.args_spec, self.kwargs_spec), got=(args, kwargs))
        args, kwargs = _polymorphic_integers_to_arrays((args, kwargs), (self.args_spec, self.kwargs_spec))
        return self._jitted_call(state_dict, *args, **kwargs)


def restore_state_dict(path, *, restore_type=jax.Array, dtype=None, sharding=None, meta_only=False):
    """Restores unstructured state PyTree from a checkpoint"""
    path = epath.Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Model state not found at: {path}")

    if not meta_only and restore_type is jax.Array and sharding is None:
        mesh = jax.sharding.Mesh(jax.devices(), ("x",))
        sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    with ocp.PyTreeCheckpointer() as ckptr:
        metadata = ckptr.metadata(path)
        if meta_only:
            return metadata.tree
        if dtype is None:
            restore_args = jax.tree.map(
                lambda _: ocp.ArrayRestoreArgs(sharding=sharding, restore_type=restore_type), metadata
            )
        else:
            try:
                dtype = jnp.dtype(dtype)
                dtype = jax.tree.map(lambda _: dtype, metadata)
            except TypeError:
                pass
            except ValueError:
                pass
            restore_args = jax.tree.map(
                lambda d: ocp.ArrayRestoreArgs(sharding=sharding, restore_type=restore_type, dtype=d), dtype
            )

        return ckptr.restore(path, ocp.args.PyTreeRestore(item=metadata, restore_args=restore_args))


def load_exported_collection(path):
    path = epath.Path(path).resolve()
    collection = {}
    for exported_file in path.glob("*.exported"):
        defaults_file = exported_file.with_suffix(".json")
        if defaults_file.exists():
            with defaults_file.open("r") as f:
                defaults = json.load(f)
        else:
            defaults = {}
        with exported_file.open("rb") as f:
            exported = ExportedModuleMethod(jax.export.deserialize(f.read()), defaults)  # type: ignore
            collection[exported_file.stem] = exported
    if len(collection) == 0:
        raise ValueError(f"No exported methods found in {path}")
    return collection


def restore_exported(path, names=None, dtype=None, sharding=None, meta_only=False):
    """Restore a model from a directory containing exported methods and PyTree state."""
    path = epath.Path(path).resolve()
    exported = load_exported_collection(path / "exported")
    if names is None:
        names = exported.keys()
    elif isinstance(names, str):
        names = [names]
    exported = {name: exported[name] for name in names}

    if meta_only:
        state_dict = restore_state_dict(path / "state", meta_only=True)
        return state_dict, exported

    if dtype is not None:
        state_dict = restore_state_dict(path / "state", dtype=dtype, sharding=sharding)
    else:
        # for convenience: if all exported methods have the same state_dict_spec, and all dtypes match, we can directly
        # restore as the correct dtype.
        state_dict_specs = tuple(e.state_dict_spec for e in exported.values())
        if all(
            jax.tree.structure(s) == jax.tree.structure(state_dict_specs[0]) for s in state_dict_specs
        ) and jax.tree.all(jax.tree.map(lambda *ss: all(s.dtype == ss[0].dtype for s in ss), *state_dict_specs)):
            dtype_tree = jax.tree.map(lambda s: s.dtype, state_dict_specs[0])
            state_dict = restore_state_dict(path / "state", dtype=dtype_tree, sharding=sharding)
        else:
            state_dict = restore_state_dict(path / "state")
    return state_dict, exported
