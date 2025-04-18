"""
Copyright 2025 Zordi, Inc. All rights reserved.

WebSocket server for Pi0 prediction model.
"""

import asyncio
import os
import pickle
import time
from typing import Any

import cv2
import numpy as np
from rfm_pi0_model import Pi0Model
from rfm_websocket_manager import WebSocketServer
from rfm_websocket_manager import send_error_response

# Default configuration
MODEL_CHECKPOINT = os.getenv("PI0_MODEL_CHECKPOINT", "./nov4_zordi_pick_strawberry/0ke1t39y/20000/model")
HOST = "0.0.0.0"
PORT = 10012


class Pi0Server:
    """WebSocket server for Pi0 model predictions."""

    def __init__(
        self,
        checkpoint_path: str = MODEL_CHECKPOINT,
        max_clients: int = 100,
        connection_timeout: float = 600.0,  # 10 minute timeout
        host: str = HOST,
        port: int = PORT,
    ):
        """Initialize the Pi0Server.

        Args:
            checkpoint_path: Path to the model checkpoint
            max_clients: Maximum number of simultaneous clients
            connection_timeout: Connection timeout in seconds
            host: Server host
            port: Server port
        """
        self.checkpoint_path = checkpoint_path
        self._model_lock = asyncio.Lock()
        self.model = None

        # Performance metrics
        self.request_count = 0
        self.total_processing_time = 0.0

        # Create WebSocket server
        self.server = WebSocketServer(
            message_handler=self.process_message,
            max_clients=max_clients,
            connection_timeout=connection_timeout,
            host=host,
            port=port,
        )

    async def initialize_model(self) -> None:
        """Initialize model asynchronously."""
        async with self._model_lock:
            if self.model is not None:
                return

            try:
                print(f"Loading Pi0 model from {self.checkpoint_path}")
                start_time = time.time()
                self.model = Pi0Model(model_path=self.checkpoint_path)
                load_time = time.time() - start_time
                print(f"Pi0 model loaded successfully in {load_time:.2f} seconds")
            except Exception as e:
                print(f"[ERROR] Failed to load Pi0 model: {e}")
                raise RuntimeError(f"Failed to initialize Pi0 model: {e}") from e

    @staticmethod
    def decode_binary_image(binary_data: bytes) -> np.ndarray:
        """Decode binary image data to numpy array.

        Args:
            binary_data: Binary image data

        Returns:
            np.ndarray: Image array of shape (H, W, C)

        Raises:
            ValueError: If image decoding fails
        """
        try:
            # Decode the binary data to an image
            nparr = np.frombuffer(binary_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # OpenCV reads as BGR, convert to RGB
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        except Exception as e:
            print(f"[ERROR] Binary image decoding failed: {e}")
            raise ValueError(f"Failed to decode binary image: {e}") from e

    async def process_message(self, websocket: Any, message: bytes, client_id: int) -> None:
        """Process client message and send prediction.

        Args:
            websocket: WebSocket connection
            message: Client message (binary pickle data)
            client_id: Client ID for logging
        """
        processing_start = time.time()

        # Initialize model if needed
        if self.model is None:
            try:
                await self.initialize_model()
            except Exception as e:
                await send_error_response(websocket, f"Model initialization failed: {e}")
                return

        try:
            # Parse binary pickle message
            data = pickle.loads(message)

            # Validate request
            if "images" not in data or "state" not in data or "raw_text" not in data:
                raise ValueError("Missing required fields: images, state, raw_text")

            # Convert images to numpy arrays
            images = {}
            for key, binary_image in data["images"].items():
                images[key] = self.decode_binary_image(binary_image)

            # Convert state to tensor
            state = np.array(data["state"], dtype=np.float32)

            # Get raw text
            raw_text = data["raw_text"]

            # Validate values
            if not np.all(np.isfinite(state)):
                raise ValueError("State contains invalid values")

            # Prepare observation dictionary for the model
            obs_dict = {
                "observation.images.static_top_down": images.get("static_top_down"),
                "observation.images.eoat_top": images.get("eoat_top"),
                "observation.images.eoat_bottom": images.get("eoat_bottom"),
                "observation.state": state,
            }

            # Remove any None entries (for cameras that might not be available)
            obs_dict = {k: v for k, v in obs_dict.items() if v is not None}

            # Get prediction (model is now guaranteed to be initialized)
            assert self.model is not None
            actions = self.model.predict(obs_dict, raw_text=raw_text)

            # Prepare and send response
            response_data = {"action": actions.tolist()}
            await websocket.send(pickle.dumps(response_data))

            # Update metrics
            processing_time = time.time() - processing_start
            self.request_count += 1
            self.total_processing_time += processing_time

            # Log metrics every 10 requests
            if self.request_count % 10 == 0:
                avg_time = self.total_processing_time / self.request_count
                print(f"Performance stats: Requests={self.request_count}, Avg processing time={avg_time * 1000:.2f}ms")

        except Exception as e:
            print(f"[ERROR] Client {client_id} prediction failed: {e}")
            await send_error_response(websocket, f"Prediction failed: {e}")

    async def start(self) -> None:
        """Start the Pi0 server."""
        await self.initialize_model()
        print(f"Pi0Server is ready and listening on {HOST}:{PORT}")
        await self.server.start()


async def main() -> None:
    """Start the WebSocket server."""
    server = Pi0Server()
    await server.start()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Server stopped by user")
    except Exception as e:
        print(f"[ERROR] Server error: {e}")
