# processors
import dataclasses
import numpy as np
import jax
from typing import Any
from collections import defaultdict
from openpi.utils.processor_utils import (
    PaligemmaFormatter,
    ModalityGroup,
    BidirectionalPrefix,
    Image,
    BOS,
    Language,
    ContinuousAction,
    TokenAccumulator,
)
from openpi.utils.image_utils import dummy_image_like, encode_str, decode_str
import sentencepiece
from openpi.utils.processor_utils import kw_dataclass
from openpi.utils.processor_utils import Batch, RAW_TEXT


class LiveTransformation:
    def deserialize(self, path):
        return


@dataclasses.dataclass(frozen=True)
class TransformationSpec:
    """This class specifies the input (argument) and output (return) fields for a processor transformation.
    Note that "inputs" and "outputs" refers to inputs and outputs batches (i.e., inputs contain state, outputs
    contain actions), while _in and _out represents the input (arguments) and output (return) for the
    transformation, such that "inputs_in" means the fields that "inputs" should have when passed as an
    argument, and "inputs_out" means the fields that the transformation will return."""

    inputs_in: set[str]
    outputs_in: set[str]
    inputs_out: set[str]
    outputs_out: set[str]


@dataclasses.dataclass(frozen=False)
class DiscretizeStates(LiveTransformation):
    num_bins: int = 256

    def get_process_fields(self):
        return TransformationSpec({"state"}, set(), {"discretized_state"}, set())

    def get_unprocess_fields(self):
        return TransformationSpec(set(), set(), set(), set())

    def process(self, inputs, outputs):
        inputs["discretized_state"] = np.digitize(inputs["state"], bins=np.linspace(-1, 1, self.num_bins + 1)[:-1]) - 1
        return inputs, outputs

    def unprocess(self, inputs, outputs):
        return inputs, outputs

    def update_params(self, inputs, outputs):
        # Skip to make param initialization faster
        # assumes that no downstream processor needs outputs of this processor
        return inputs, outputs

    def requires_param_computation(self):
        return False


@dataclasses.dataclass(frozen=False)
class ConvertStateToText(LiveTransformation):
    eor_token: str = ";"
    mask: list[bool] | None = None

    def _array_to_text(self, array):
        if array.ndim == 1:
            if self.mask is not None:
                array = array[self.mask]
            return " ".join(map(str, array)) + self.eor_token
        if array.ndim == 2:
            if self.mask is not None:
                raise NotImplementedError
                # array = [row[mask_row] for row, mask_row in zip(array, mask, strict=True)]
            rows_as_text = [" ".join(map(str, row)) + self.eor_token for row in array]
            return " ".join(rows_as_text)
        raise ValueError(f"Unsupported number of dimensions for discretization: {array.ndim}")

    def get_process_fields(self):
        return TransformationSpec({"discretized_state"}, set(), {"text_state"}, set())

    def get_unprocess_fields(self) -> TransformationSpec:
        return TransformationSpec(set(), set(), set(), set())

    def process(self, inputs, outputs):
        assert "discretized_state" in inputs, f"Nothing to discretize... Current input keys: {inputs.keys()}"
        if "discretized_state" in inputs:
            inputs["text_state"] = [self._array_to_text(state) for state in inputs["discretized_state"]]
        return inputs, outputs

    def unprocess(self, inputs, outputs):
        return inputs, outputs

    def update_params(self, inputs, outputs):
        # Skip to make param initialization faster
        # assumes that no downstream processor needs outputs of this processor
        return inputs, outputs

    def requires_param_computation(self):
        return False


@dataclasses.dataclass
class PredictDCTActions(LiveTransformation):
    """skip tokenizer"""

    action_horizon: int
    action_dim: int
    include_state: bool = True
    include_diffusion: bool = True
    pad_output_token: int = 300

    add_future_image: bool = False
    ar_predict_future_image: bool = False
    condition_on_future_image: bool = False
    drop_task_prob: float = 0.0
    include_critic_value: bool = False
    load_only_future_base_image: bool = False
    mask_dct_token_loss: bool = False
    tokenizer: Any = dataclasses.field(default_factory=PaligemmaFormatter)
    with_ar_prefix: bool = False

    def process(self, inputs, outputs):
        bs = len(inputs["raw_text"])
        inputs["modalities"] = []
        num_images = len([k for k in inputs["image"] if not k.endswith("mask")])
        for i in range(bs):
            modalities = []
            prompt = f"Task: {inputs['raw_text'][i].lower().strip().replace('_', ' ')}"
            prompt += f", State: {inputs['text_state'][i]}" if self.include_state else ""
            prompt += "\nAction: "
            modalities.append(
                BidirectionalPrefix(
                    [
                        *[Image() for _ in range(num_images)],
                        BOS(),
                        Language(text=prompt),
                    ]
                )
            )
            inputs["modalities"].append(modalities)
        return inputs, outputs

    def unprocess(self, inputs, outputs):
        return inputs, outputs

    def get_process_fields(self):
        return TransformationSpec({"raw_text", "image"}, {"actions"}, {"modalities"}, {"output_tokens"})

    def get_unprocess_fields(self):
        return TransformationSpec(
            set(),
            {"output_tokens", "actions"},
            set(),
            {"uncompressed_action_reconstruction", "action_is_valid", "actions"},
        )

    def update_params(self, inputs, outputs):
        # Skip to make param initialization faster
        # assumes that no downstream processor needs outputs of this processor
        return inputs, outputs

    def requires_param_computation(self):
        return False


@kw_dataclass
class ToInterleaved(LiveTransformation):
    text_sequence_length: int
    max_num_images: int
    max_num_actions: int
    tokens_per_image: int
    tokens_per_action: int
    tokenizer: Any = dataclasses.field(default_factory=PaligemmaFormatter)

    _tokenizer: Any
    add_future_image: bool = False
    condition_on_future_image: bool = False
    include_image_delimiters: bool = False
    latent_queries_per_image_chunk: Any = None
    log_modalities_to_html_str: bool = True
    logging_loss_groups: Any = None
    max_language_inference_injection_length: Any = None
    max_num_future_images: int = 4
    max_num_image_chunks: Any = None

    def process_example(self, modalities, image_dict, inference=False):
        image_names = [k for k in image_dict if not k.endswith("mask")]
        new_image_dict = {}
        image_order = []
        new_data = defaultdict(list)
        length = 0
        any_loss = False
        language_length = 0

        if not isinstance(modalities, ModalityGroup):
            modalities = ModalityGroup(modalities)
        modalities = list(modalities)

        # Make sure all modalities have been initialized
        assert all(modality.mask is not None for modality in modalities)
        assert all(modality.bidirectional_attention is not None for modality in modalities)
        assert all(modality.output_group is not None for modality in modalities)
        assert all(modality.loss is not None for modality in modalities if isinstance(modality, Language))

        # Language padding
        modalities.append(Language(tokens=[0] * self.text_sequence_length, loss=False, mask=False))

        # Image padding
        num_images = len(list(filter(lambda x: isinstance(x, Image), modalities)))
        assert (
            num_images <= self.max_num_images
        ), f"Found {num_images} Images(), but max_num_images is {self.max_num_images}"
        if num_images < self.max_num_images:
            dummy_image, _ = dummy_image_like(image_dict[image_names[0]][None])
            image_dict["dummy"] = dummy_image[0]
            image_dict["dummy_mask"] = False
            modalities.extend([Image(image_key="dummy")] * (self.max_num_images - num_images))
        # Action padding
        num_actions = len(list(filter(lambda x: isinstance(x, ContinuousAction), modalities)))
        if not inference:
            assert (
                num_actions <= self.max_num_actions
            ), f"Found {num_actions} ContinuousActions(), but max_num_actions is {self.max_num_actions}"
            modalities.extend([ContinuousAction(output_group=1, mask=False)] * (self.max_num_actions - num_actions))
        else:
            assert num_actions == 0, "There should be no ContinuousAction() in inference mode"

        token_accumulator = TokenAccumulator(logging_loss_groups=None)

        for modality in modalities:
            if isinstance(modality, Language):  # not dealing with injection
                tokens = modality.to_tokens(self.tokenizer)
                tokens = tokens[: self.text_sequence_length - language_length]
                language_length += len(tokens)

                token_accumulator.add(
                    tokens,
                    [modality.loss] * len(tokens),
                    list(range(length, length + len(tokens))),
                    modality.logging_loss_name,
                )
                any_loss = any_loss or modality.loss
                modality_length = len(tokens)

            elif isinstance(modality, Image):
                image_key = modality.image_key or image_names[len(new_image_dict)]
                image_order.append(image_key)
                new_image_dict[f"image_{len(new_image_dict)}"] = image_dict[image_key]
                modality.mask = modality.mask and image_dict[f"{image_key}_mask"]
                new_data["image_indices"].append(range(length, length + self.tokens_per_image))
                modality_length = self.tokens_per_image

            elif isinstance(modality, ContinuousAction):
                new_data["action_indices"].append(range(length, length + self.tokens_per_action))
                modality_length = self.tokens_per_action

            new_data["mask_input"].extend([modality.mask] * modality_length)
            new_data["mask_ar"].extend([int(not modality.bidirectional_attention)] * modality_length)
            new_data["output_group"].extend([modality.output_group] * modality_length)
            new_data["example_id"].extend([0] * modality_length)
            length += modality_length

        new_data["language_indices"] = token_accumulator.idcs
        new_data["language"] = token_accumulator.tokens
        new_data["mask_loss"] = token_accumulator.mask_loss

        new_data = {k: np.array(v).astype(np.int32) for k, v in new_data.items() if len(v) > 0}
        new_data["image_order"] = encode_str("\n".join(image_order), pad_length=1024)
        new_data["image"] = new_image_dict
        return new_data

    def process(self, inputs, outputs):
        batched_new_data = []
        bs = len(inputs["modalities"])
        inference = "actions" not in outputs
        for i in range(bs):
            new_data = self.process_example(
                inputs["modalities"][i], {k: inputs["image"][k][i] for k in inputs["image"]}, inference
            )
            batched_new_data.append(new_data)
        new_inputs = jax.tree.map(lambda *x: np.stack(x), *batched_new_data)
        inputs.update(new_inputs)
        del inputs["modalities"]
        return inputs, outputs

    def unprocess(self, inputs, outputs):
        bs = len(inputs["image_order"])
        new_image_dict = defaultdict(list)
        for batch_idx in range(bs):
            image_order = str(decode_str(np.asarray(inputs["image_order"][batch_idx]))).split("\n")
            for i, image_key in enumerate(image_order):
                if image_key != "dummy":
                    new_image_dict[image_key].append(inputs["image"][f"image_{i}"][batch_idx])
                    new_image_dict[f"{image_key}_mask"].append(True)
        inputs["image"] = {k: np.stack(v) for k, v in new_image_dict.items()}
        return inputs, outputs

    def get_process_fields(self):
        interleaved_keys = {
            "mask_input",
            "mask_loss",
            "mask_ar",
            "output_group",
            "example_id",
            "language",
            "language_indices",
            "image_indices",
            "action_indices",
        }
        input_out_spec = interleaved_keys | {"image", "image_order", "modalities"}
        return TransformationSpec(
            {"modalities", "image"},
            {"actions"},
            input_out_spec,
            set(),
        )

    def get_unprocess_fields(self):
        return TransformationSpec(
            {"image", "image_order"},
            set(),
            {"image", "output_tokens"},
            set(),
        )

    def update_params(self, inputs, outputs):
        # Skip to make param initialization faster
        # assumes that no downstream processor needs outputs of this processor
        return inputs, outputs

    def requires_param_computation(self):
        return False


@dataclasses.dataclass(frozen=False)
class TokenizeForPaligemmaEncoder:
    """Tokenizer for PI0 models that use PaliGemma encoder."""

    text_len: int = 48
    _tokenizer: sentencepiece.SentencePieceProcessor | None = None

    def _load_tokenizer(self):
        if self._tokenizer is None:
            # Get the tokenizer path from the model config
            tokenizer_path = getattr(PaligemmaFormatter, "tokenizer_path", None)
            if tokenizer_path is None:
                raise ValueError("tokenizer_path must be set in PaligemmaFormatter before using the tokenizer")
            self._tokenizer = sentencepiece.SentencePieceProcessor(tokenizer_path)

    def process(self, inputs: Batch, outputs: Batch) -> tuple[Batch, Batch]:
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

    def unprocess(self, inputs: Batch, outputs: Batch) -> tuple[Batch, Batch]:
        return inputs, outputs

    def update_params(self, inputs: Batch, outputs: Batch) -> tuple[Batch, Batch]:
        # Skip to make param initialization faster
        # assumes that no downstream processor needs outputs of this processor
        return inputs, outputs
