"""
Copyright 2025 Zordi, Inc. All rights reserved.

WebSocket client for Pi0 prediction service.
"""

import logging
import time

import cv2
import numpy as np
from rfm_websocket_manager import WebSocketClient

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_HOST = "localhost"
DEFAULT_PORT = 10012


def process_image(
    image: np.ndarray,
    image_size: tuple[int, int] = (224, 224),
    pad_ratio: float = 0.0,
) -> np.ndarray:
    """Perform center crop with padding based on the shorter dimension.

    Args:
        image: Input image as numpy array
        image_size: Tuple of (height, width)
        pad_ratio: Ratio to determine padding size (e.g., 0.2 for 20% padding)

    Returns:
        Cropped image as numpy array
    """
    height, width = image.shape[:2]
    shorter_dim = min(width, height)
    crop_size = int(shorter_dim * (1.0 - pad_ratio))

    start_x = (width - crop_size) // 2
    start_y = (height - crop_size) // 2

    cropped = image[start_y : start_y + crop_size, start_x : start_x + crop_size]
    resized: np.ndarray = cv2.resize(cropped, (image_size[1], image_size[0]))
    return resized


def encode_image_binary(image: np.ndarray) -> bytes:
    """Encode image to binary format.

    Args:
        image: Image array of shape (H, W, C)

    Returns:
        bytes: Binary encoded image
    """
    if image.ndim != 3:
        raise ValueError("Color image must have 3 dimensions")

    # Ensure image is in BGR format for OpenCV
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if image.shape[2] == 3 else image

    # Encode to JPEG
    _, buffer = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return buffer.tobytes()


class Pi0Client:
    """WebSocket client for Pi0 prediction service."""

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        auto_reconnect: bool = True,
        reconnect_attempts: int = 3,
        reconnect_delay: float = 0.5,
    ):
        """Initialize client.

        Args:
            host: Server hostname
            port: Server port
            auto_reconnect: Whether to automatically reconnect
            reconnect_attempts: Number of reconnection attempts
            reconnect_delay: Delay between reconnection attempts
        """
        self.client = WebSocketClient(
            host=host,
            port=port,
            auto_reconnect=auto_reconnect,
            reconnect_attempts=reconnect_attempts,
            reconnect_delay=reconnect_delay,
        )

    def connect(self) -> None:
        """Connect to WebSocket server."""
        self.client.connect()

    def disconnect(self) -> None:
        """Disconnect from WebSocket server."""
        self.client.disconnect()

    def predict(self, images: dict[str, np.ndarray], state: np.ndarray, raw_text: str) -> np.ndarray:
        """Get prediction from server.

        Args:
            images: Dictionary mapping camera names to images as numpy arrays (H,W,C)
            state: State array
            raw_text: Text prompt describing the task

        Returns:
            np.ndarray: Predicted action
        """
        # Process and encode images
        encoded_images = {}
        for key, image in images.items():
            # Directly encode the raw image. Resizing/padding is handled by the model server-side.
            encoded_images[key] = encode_image_binary(image)

        # Prepare binary data
        binary_data = {
            "images": encoded_images,
            "state": state.tolist(),
            "raw_text": raw_text,
        }

        # Send request and get response
        result = self.client.send_receive(binary_data)

        # Parse response
        try:
            return np.array(result["action"])
        except Exception as e:
            raise ValueError(f"Failed to parse server response: {e}") from e


def main() -> None:
    """Example usage of Pi0Client."""
    # Create client
    client = Pi0Client()

    # Generate random data for testing
    rng = np.random.default_rng(42)

    # Create dummy images - random uint8 images with values 0-255
    # Use simplified camera names that match what the server expects
    image_keys = ["static_top_down", "eoat_top", "eoat_bottom"]

    images = {key: rng.integers(0, 256, size=(720, 1280, 3), dtype=np.uint8) for key in image_keys}

    # Create dummy state
    state = rng.standard_normal((6,))

    # Sample task description
    raw_text = "pick a ripe strawberry"

    client.connect()

    # Get prediction
    try:
        start_time = time.time()
        actions = None  # Initialize actions variable

        # Make multiple predictions to demonstrate connection reuse
        num_predictions = 10
        for i in range(num_predictions):
            actions = client.predict(images, state, raw_text)
            print(f"Prediction {i + 1}/{num_predictions} complete")

        end_time = time.time()
        duration = end_time - start_time

        print(f"Completed {num_predictions} predictions in {duration:.2f} seconds")
        print(f"Average time per prediction: {(duration / num_predictions) * 1000:.2f}ms")
        if actions is not None:
            print(f"Last received actions shape: {actions.shape}")
            # print(f"Actions:\n{actions}")
    except Exception as e:
        print(f"Prediction failed: {e}")
    finally:
        client.disconnect()


if __name__ == "__main__":
    main()
