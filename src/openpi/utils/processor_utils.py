# Processor-related

import abc
import dataclasses
import importlib
import os
import yaml
import jax
import numpy as np
import logging
from typing import Any
import frozendict
import sentencepiece
import sentencepiece.sentencepiece_model_pb2

# Type aliases and constants
SPProcessor = sentencepiece.SentencePieceProcessor
SPModelProto = sentencepiece.sentencepiece_model_pb2.ModelProto
kw_dataclass = dataclasses.dataclass(kw_only=True)

PROCESSORS = "processors"
PROCESS = "process"
UNPROCESS = "unprocess"
FN = "fn"
ROBOT_TASK_STRING = "robot_task_string"
RAW_TEXT = "raw_text"
SERIALIZATION_META = "transformation.yaml"

Batch = dict[str, Any]


def spec_select(batch: Batch, spec: set) -> Batch:
    """Helper function to pick out desired fields based on spec."""
    for k in spec:
        if k not in batch:
            raise ValueError(f"Field {k} not found in batch {spec=}")
    return {k: batch[k] for k in spec}


def make_batch(batch: dict | np.ndarray | str, name: str = "batch"):
    if isinstance(batch, dict):
        return {k: make_batch(v, k) for k, v in batch.items()}
    else:
        if isinstance(batch, str):
            # This field should be made into a list.
            # Note: important for this to come first, since numpy string arrays have shape,
            # so would get caught by the hasattr(batch, "shape") check below & throw error.
            return [batch]
        elif hasattr(batch, "shape"):
            # This field is an array, insert a dimension.
            return batch[None]
        else:
            raise ValueError(f"Unknown batch field {name}: {batch}")


def unmake_batch(batch: dict | np.ndarray | str, name: str = "batch", strict_check: bool = True):
    if isinstance(batch, dict):
        return {k: unmake_batch(v, k, strict_check=strict_check) for k, v in batch.items()}
    else:
        if hasattr(batch, "shape"):
            # This field is an array, return first value.
            if batch.shape[0] != 1 and strict_check:
                raise ValueError(f"Removing batch dimension for field {name} with shape {batch.shape}: {batch}")
            return batch[0]
        else:
            # This is a list.
            if not hasattr(batch, "__len__"):
                raise ValueError(f"Removing batch dimension for field {name} without: {batch}")
            if len(batch) != 1 and strict_check:
                raise ValueError(f"Removing batch dimension for field {name} with length {len(batch)}: {batch}")
            return batch[0]


@dataclasses.dataclass(frozen=False)
class GraphTransformation:
    ops: dict

    def process(self, inputs: Batch, outputs: Batch) -> tuple[Batch, Batch]:
        spec = self.ops[PROCESS]["input_spec"][0]
        if len(spec) == 1:
            new_inputs, new_outputs = self.ops[PROCESS][FN](spec_select(inputs, spec[0].keys()))
        else:
            new_inputs, new_outputs = self.ops[PROCESS][FN](
                spec_select(inputs, spec[0].keys()), spec_select(outputs, spec[1].keys())
            )
        inputs.update(new_inputs)
        outputs.update(new_outputs)
        return inputs, outputs

    def unprocess(self, inputs: Batch, outputs: Batch) -> tuple[Batch, Batch]:
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

    def process(self, inputs: Batch, outputs: Batch = {}, has_batch_dim: bool = False) -> tuple[Batch, Batch]:
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

    def unprocess(self, inputs: Batch, outputs: Batch, has_batch_dim: bool = False) -> tuple[Batch, Batch]:
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


class DataclassLoader(yaml.SafeLoader):
    ignore_unknown_fields: bool

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_constructor("!array", self._construct_array)
        self.add_constructor("!dataclass", self._construct_dataclass)
        self.add_constructor("!enum", self._construct_enum)
        self.add_constructor("!tuple", self._construct_tuple)
        self.add_constructor("!frozendict", self._construct_frozendict)

    def _construct_array(self, loader: yaml.Loader, node: yaml.MappingNode):
        data = loader.construct_mapping(node, deep=True)
        if "data" in data:
            return np.array(data["data"], dtype=np.dtype(data["dtype"])).reshape(data["shape"])
        return jax.ShapeDtypeStruct(data["shape"], np.dtype(data["dtype"]))

    def _construct_dataclass(self, loader: yaml.Loader, node: yaml.MappingNode):
        # NOTE: We need deep=True set so that contents of dataclasses are not constructed lazily
        # Without this, __post_init__ methods may not function as expected since objects will not be fully constructed
        data = loader.construct_mapping(node, deep=True)
        module = importlib.import_module(data["__module__"])  # Import the module
        cls = getattr(module, data["__name__"])  # Get the class from the module
        del data["__module__"], data["__name__"]
        if self.ignore_unknown_fields:
            known_fields = {field.name for field in dataclasses.fields(cls)}
            data = {k: v for k, v in data.items() if k in known_fields}
        return cls(**data)  # Instantiate the dataclass

    def _construct_enum(self, loader: yaml.Loader, node: yaml.MappingNode):
        data = loader.construct_mapping(node)
        module = importlib.import_module(data["__module__"])
        cls = getattr(module, data["__name__"])
        return cls[data["name"]]

    def _construct_tuple(self, loader: yaml.Loader, node: yaml.SequenceNode):
        return tuple(loader.construct_sequence(node))

    def _construct_frozendict(self, loader: yaml.Loader, node: yaml.MappingNode):
        return frozendict(loader.construct_mapping(node, deep=True))


def from_yaml(data: str, ignore_unknown_fields: bool = False):
    """Deserializes a structure from YAML, loading shape and dtype information into ShapeDtypeStructs.

    Args:
        data: YAML string.
        ignore_unknown_fields: If True, unknown fields in dataclasses will be ignored during instantiation.

    Returns:
        Structure deserialized from YAML.
    """
    loader_cls = type("DataclassLoader", (DataclassLoader,), {"ignore_unknown_fields": ignore_unknown_fields})
    return yaml.load(data, Loader=loader_cls)


def load_transformation(path: str, live_only: bool = False):
    if os.path.exists(os.path.join(path, PROCESS)) and not live_only:
        # Load transformation from exported graph.
        live_only_override = False
        operations = [PROCESS, UNPROCESS]
        out = {}
        for operation_name in operations:
            with open(os.path.join(path, operation_name), "rb") as f:
                exported = jax.export.deserialize(f.read())
            input_spec = jax.tree.unflatten(exported.in_tree, exported.in_avals)
            output_spec = jax.tree.unflatten(exported.out_tree, exported.out_avals)
            if _check_dali_image_override(input_spec):
                live_only_override = True
                break
            out[operation_name] = {FN: exported.call, "input_spec": input_spec, "output_spec": output_spec}

        if not live_only_override:
            return GraphTransformation(out)

    # Load live transformation from YAML.
    with open(os.path.join(path, SERIALIZATION_META)) as f:
        yaml_content = f.read()

    # Check if this is a TokenizeForPaligemmaEncoder by looking at the YAML content
    if "TokenizeForPaligemmaEncoder" in yaml_content:
        # Parse it manually for TokenizeForPaligemmaEncoder
        transformation_dict = {}
        for line in yaml_content.split("\n"):
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            transformation_dict[key.strip()] = value.strip()

        if transformation_dict.get("__name__") == "TokenizeForPaligemmaEncoder":
            # Import the class
            from openpi.utils.processor import TokenizeForPaligemmaEncoder

            text_len = int(transformation_dict.get("text_len", 48))
            return TokenizeForPaligemmaEncoder(text_len=text_len)

    # Otherwise, use the standard YAML loader
    transformation = from_yaml(yaml_content)
    transformation.deserialize(path)
    return transformation


def _check_dali_image_override(input_spec) -> bool:
    process_input_spec = input_spec[0][0]
    for key in process_input_spec:
        # This is an image.
        if "rgb" in key and len(process_input_spec[key].shape) == 2:
            # This is a DALI compressed image.
            logging.info(
                f"Found a processor with a graph that expects DALI compressed images, with input spec: {process_input_spec}"
            )
            return True
    return False


def load_processor(path: str, name: str):  # load live
    transformations = []
    paths = os.listdir(path)
    paths = [path for path in paths if path.isdigit()]
    indices = [int(path) for path in paths]
    order = np.argsort(indices)
    for i in order:
        print("Processor path:", os.path.join(path, paths[i]))
        transformations.append(load_transformation(os.path.join(path, paths[i])))
    processor = Processor(name, transformations)
    return processor


# originally from pi's code but had to copy here since some processors were not serialized
@kw_dataclass
class Modality(abc.ABC):
    mask: bool | None = None
    bidirectional_attention: bool | None = None
    output_group: int | None = None


@kw_dataclass
class Image(Modality):
    image_key: str | None = None
    bidirectional_attention: bool = True
    mask: bool = True
    output_group: int = 0


@kw_dataclass
class ContinuousAction(Modality):
    output_group: int
    bidirectional_attention: bool = True
    mask: bool = True

    def __post_init__(self):
        assert self.output_group > 0, "Output group for a continuous action must be greater than 0"


@kw_dataclass
class Language(Modality):
    text: str | None = None
    tokens: list[int] | None = None
    piece: str | None = None
    loss: bool | None = None
    mask: bool = True
    bidirectional_attention: bool = False
    output_group: int = 0
    logging_loss_name: str | None = None

    def __post_init__(self):
        if len([x for x in [self.text, self.tokens, self.piece] if x is not None]) != 1:
            raise ValueError("Only one can be active")

    def to_tokens(self, tokenizer) -> list[int]:
        if self.text is not None:
            return tokenizer.tokenize(self.text)
        if self.tokens is not None:
            return self.tokens
        return [tokenizer.piece_to_id(self.piece)]


@kw_dataclass
class BOS(Language):
    bidirectional_attention: bool
    piece: str = "<bos>"
    loss: bool = False
    mask: bool = True


@kw_dataclass
class EOS(Language):
    piece: str = "<eos>"
    loss: bool = True
    mask: bool = False


@dataclasses.dataclass
class ModalityGroup:
    modalities: list[Modality]

    def __iter__(self):
        for modality in self.modalities:
            if isinstance(modality, ModalityGroup):
                yield from modality
            else:
                yield modality


@kw_dataclass
class OutputGroup(ModalityGroup):
    output_group: int

    def __iter__(self):
        for modality in super().__iter__():
            yield dataclasses.replace(modality, output_group=self.output_group)


class BidirectionalPrefix(ModalityGroup):
    def __iter__(self):
        for modality in super().__iter__():
            updates = {"bidirectional_attention": True}
            if isinstance(modality, Language):
                updates["loss"] = False
            yield dataclasses.replace(modality, **updates)


class AutoregressivePrefix(ModalityGroup):
    def __iter__(self):
        for modality in super().__iter__():
            updates = {"bidirectional_attention": False}
            if isinstance(modality, Language):
                updates["loss"] = False
            yield dataclasses.replace(modality, **updates)


@kw_dataclass
class Prediction(ModalityGroup):
    logging_loss_name: str | None = None

    def __iter__(self):
        updates = {"bidirectional_attention": False, "loss": True}
        if self.logging_loss_name is not None:
            updates["logging_loss_name"] = self.logging_loss_name
        for modality in super().__iter__():
            if isinstance(modality, Language):
                yield dataclasses.replace(modality, **updates)
            elif isinstance(modality, ModalityGroup):
                yield from (dataclasses.replace(m, **updates) for m in modality)
            elif isinstance(modality, ContinuousAction):
                yield modality
            else:
                raise ValueError(f"Cannot predict modality of type {type(modality)}")


class TokenAccumulator:
    def __init__(self, logging_loss_groups: list[str] | None = None):
        self.tokens: list[int] = []
        self.idcs: list[int] = []
        self.mask_loss: list[bool] = []

        # Capture self in the lambda to use the current token count
        self.loss_groups: dict[str, list[bool]] | None = None
        if logging_loss_groups is not None:
            self.loss_groups = {k: [] for k in logging_loss_groups}

    def add(self, tokens: list[int], loss: list[bool] | bool, idcs: list[int], loss_group: str | None = None):
        if isinstance(loss, bool):
            loss = [loss] * len(tokens)

        self.tokens.extend(tokens)
        self.idcs.extend(idcs)
        self.mask_loss.extend(loss)

        if self.loss_groups is not None:
            if loss_group is not None:
                self.loss_groups[loss_group].extend(loss)

            for other_loss_group in self.loss_groups:
                if other_loss_group != loss_group:
                    self.loss_groups[other_loss_group].extend([0] * len(tokens))


PALIGEMMA_NUM_ACTION_TOKENS = 2048
PALIGEMMA_DEFAULT_REPLACEMENTS = {
    250_000: "<start_of_image>",
    250_001: "<end_of_image>",
    250_002: "<end_of_cot>",
    255_999: "<_duplicate_start_of_image>",
} | {256_000 - 33 - i: f"<act{i}>" for i in range(PALIGEMMA_NUM_ACTION_TOKENS)}


def _replace_pieces(model_bytes, replacements):
    tokenizer = SPProcessor()
    tokenizer.LoadFromSerializedProto(model_bytes)

    def get_id(x):
        if isinstance(x, str):
            return tokenizer.PieceToId(x)
        return x

    replacements = {get_id(old_piece): new_piece.replace(" ", "_") for old_piece, new_piece in replacements.items()}
    proto = SPModelProto()
    proto.ParseFromString(model_bytes)
    for indx, new_piece in replacements.items():
        proto.pieces[indx].piece = new_piece
        proto.pieces[indx].score = 0.0
        proto.pieces[indx].type = proto.SentencePiece().Type.USER_DEFINED

    return proto.SerializeToString()


def load_tokenizer(fname, replacements):
    with open(fname, "rb") as f:
        model_bytes = f.read()
    model_bytes = _replace_pieces(model_bytes, replacements)
    processor = SPProcessor()
    processor.LoadFromSerializedProto(model_bytes)
    return processor


def load_paligemma_tokenizer(fname, replacements):
    return load_tokenizer(fname, replacements)


def load_tokenizer_for_model(fname, tokenizer_type, replacements):
    return load_paligemma_tokenizer(fname, replacements)


@dataclasses.dataclass(frozen=False)
class SentencepieceFormatter:
    tokenizer_name: str | None = None
    replacements: Any | None = None
    _processor: Any | None = None
    tokenizer_path: str | None = "/home/zordi/zordi_ws/openpi/pi0/paligemma_tokenizer.model"

    def __repr__(self):
        return f"{self.__class__.__name__}(tokenizer_name={self.tokenizer_name})"

    @property
    def processor(self):
        assert self.tokenizer_name is not None
        if self._processor is None:
            if self.tokenizer_path is None:
                raise ValueError("tokenizer_path must be set before using the processor")
            self._processor = load_tokenizer_for_model(self.tokenizer_path, self.tokenizer_name, self.replacements)
        return self._processor

    def tokenize(self, text: str) -> list[int]:
        return self.processor.Encode(text)

    def detokenize(self, tokens: list[int]) -> str:
        return self.processor.Decode(tokens)

    def piece_to_id(self, piece: str) -> int:
        return self.processor.PieceToId(piece)

    def id_to_piece(self, id: int) -> str:
        return self.processor.IdToPiece(id)

    @property
    def pad_id(self) -> int:
        return self.processor.pad_id()

    @property
    def bos_id(self) -> int:
        return self.processor.bos_id()

    @property
    def eos_id(self) -> int:
        return self.processor.eos_id()

    @property
    def vocab_size(self) -> int:
        return self.processor.vocab_size()


@dataclasses.dataclass(frozen=False)
class PaligemmaFormatter(SentencepieceFormatter):
    """skip encode_geoms and decode_geoms"""

    tokenizer_name: str = "paligemma"
    replacements: Any | None = dataclasses.field(default_factory=lambda: PALIGEMMA_DEFAULT_REPLACEMENTS)
    bbox_separator: str = ","

    def __repr__(self):
        return f"{self.__class__.__name__}(tokenizer_name={self.tokenizer_name})"

    def image_start_delimiter(self, image_name: str = "") -> str:
        return "<start_of_image>"

    def image_end_delimiter(self, image_name: str = "") -> str:
        return "<end_of_image>"

    @property
    def action_token_range(self) -> tuple[int, int]:
        return (256_000 - 33 - PALIGEMMA_NUM_ACTION_TOKENS, 256_000 - 33)

    def map_action_tokens(self, tokens: list[int]) -> list[int]:
        return [256_000 - 33 - i for i in tokens]

    def unmap_action_tokens(self, tokens: list[int] | str) -> list[int]:
        if isinstance(tokens, str):
            tokens = self.tokenize(tokens)
        return [256_000 - 33 - i for i in tokens]
