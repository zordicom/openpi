# Resizing, string, etc
import jax
import jax.numpy as jnp
import numpy as np
import simplejpeg
import functools
import logging


def resize_with_pad(images, height, width, method: jax.image.ResizeMethod = jax.image.ResizeMethod.LINEAR):
    """Resizes an image to a target height and width without distortion by padding with zeros"""
    has_batch_dim = images.ndim == 4
    if not has_batch_dim:
        images = images[None]  # type: ignore
    cur_height, cur_width = images.shape[1:3]
    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_images = jax.image.resize(
        images, (images.shape[0],) + (resized_height, resized_width) + (images.shape[3],), method=method
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


def decode_str(encoded: np.ndarray) -> np.ndarray:
    """Decodes a padded array of uint8s as a UTF-8 string."""
    if encoded.dtype != np.uint8:
        raise ValueError("Expected encoded array to have dtype np.uint8")

    bytestrings = np.ascontiguousarray(encoded).view(f"|S{encoded.shape[-1]}").squeeze(-1)
    return np.char.decode(bytestrings, "utf-8")


def get_lib(array: np.ndarray | jnp.ndarray):
    # TODO: switch to jax-jumpy ops to do this automatically
    if isinstance(array, np.ndarray):
        return np
    if isinstance(array, jnp.ndarray):
        return jnp
    raise ValueError(f"Unknown image type {type(array)}")


@functools.cache
def _black_image_np(shape: tuple[int, ...]) -> np.ndarray:
    return np.zeros(shape, dtype=np.uint8)


def _black_image(shape, lib: str = "np"):
    if lib == "np":
        return _black_image_np(shape)
    if lib == "jnp":
        return jnp.zeros(shape, dtype=jnp.uint8)
    raise ValueError(f"Unknown library {lib}")


@functools.cache
def get_black_compressed_jpeg(h: int, w: int, quality: int = 10):
    # 10 arbitrary quality value
    image_np = np.zeros((h, w, 3), dtype=np.uint8)
    dummy_image_str = simplejpeg.encode_jpeg(image_np, quality=quality)
    return np.frombuffer(dummy_image_str, dtype=np.uint8)


def dummy_image_like(img, allow_compressed: bool = True):
    lib = get_lib(img)
    if img.ndim in [4, 5]:
        if lib == np:
            lib_str = "np"
        elif lib == jnp:
            lib_str = "jnp"
        else:
            raise ValueError(f"Unknown image library {lib}")

        dummy_image = _black_image(img.shape, lib=lib_str)
        dummy_mask = lib.zeros(img.shape[:-3], dtype=lib.bool_)
    elif allow_compressed:
        # [0] to remove batch dimension
        h, w, *_ = simplejpeg.decode_jpeg_header(img[0])
        dummy_image = get_black_compressed_jpeg(h, w)[None]
        dummy_mask = lib.zeros(*img.shape[:-1], dtype=lib.bool_)
    else:
        raise ValueError(f"Expected to point to image, but shape is {img.shape}.")
    return dummy_image, dummy_mask


def encode_str(strings, pad_length: int = 128, str_warning: str | None = None) -> np.ndarray:
    """Encodes string as a padded array of uint8s using UTF-8."""

    strings = np.asarray(strings)
    if strings.dtype.kind not in {"U", "S"}:
        raise ValueError(f"Expected string, got {strings.dtype} {strings=}")

    batch_shape = strings.shape
    strings = strings.reshape(-1)

    # Encode all strings and get their lengths
    # Use utf-8 instead of ascii to handle non-ascii characters
    encoded = np.array([s.encode("utf-8") for s in strings], dtype=object)
    lengths = np.array([len(e) for e in encoded], dtype=np.int32)
    max_length = lengths.max()

    if max_length > pad_length:
        max_length_string = strings[lengths.argmax()]
        logging.warning(f"String too long: {max_length} > {pad_length} {max_length_string=} {str_warning=}")
        max_length = pad_length

    # Preallocate the padded array with zeros
    padded = np.zeros((strings.shape[0], pad_length), dtype=np.uint8)

    # Fill the padded array with the encoded bytes
    for i, e in enumerate(encoded):
        padded[i, : lengths[i]] = np.frombuffer(e[:max_length], dtype=np.uint8)

    return padded.reshape(*batch_shape, pad_length)
