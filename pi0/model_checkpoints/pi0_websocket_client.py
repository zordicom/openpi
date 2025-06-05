#!/usr/bin/env python3
"""
PI0 WebSocket Client - Client for testing the PI0 WebSocket server
Copyright 2025 Zordi, Inc. All rights reserved.
"""

import sys
import os
import asyncio
import argparse
import numpy as np
import cv2
from pathlib import Path
from websockets.sync.client import connect

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from zordi_policy_rpc.transport import (
    GetPolicyActionRequest,
    GetPolicyActionResponse,
    GetPolicyMetadataRequest,
    GetPolicyMetadataResponse,
    structure,
    unstructure,
)
from zordi_policy_rpc.image_transforms import ImageId


def encode_image(image: np.ndarray, target_height: int, target_width: int) -> bytes:
    """Encode image to JPEG bytes with resizing."""
    # Resize image
    h, w = image.shape[:2]
    scale = min(target_width / w, target_height / h)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(image, (new_w, new_h))

    # Pad to target size
    pad_h = (target_height - new_h) // 2
    pad_w = (target_width - new_w) // 2
    padded = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    padded[pad_h : pad_h + new_h, pad_w : pad_w + new_w] = resized

    # Encode to JPEG
    _, encoded = cv2.imencode(".jpg", cv2.cvtColor(padded, cv2.COLOR_RGB2BGR))
    return encoded.tobytes()


def main():
    """Main function to test the PI0 server with a WebSocket client."""
    parser = argparse.ArgumentParser(description="PI0 WebSocket Client")
    parser.add_argument("--host", default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=10012, help="Server port")
    parser.add_argument("--instruction", default="pick the ripe strawberry", help="Instruction to send")
    args = parser.parse_args()

    print("=" * 80)
    print("PI0 WebSocket Client")
    print("=" * 80)
    print(f"Connecting to: {args.host}:{args.port}")
    print(f"Instruction: {args.instruction}")
    print("=" * 80)

    try:
        # Connect to WebSocket server
        print("\nConnecting to server...")
        websocket = connect(f"ws://{args.host}:{args.port}")
        print("✓ Connected to server")

        # Get metadata
        print("\nRequesting metadata...")
        websocket.send(unstructure(GetPolicyMetadataRequest()))
        metadata_bytes = websocket.recv()
        metadata = structure(metadata_bytes, GetPolicyMetadataResponse)

        print("✓ Metadata received:")
        print(f"  - Observation length: {metadata.observation_length}")
        print(f"  - Time delta: {metadata.timedelta_sec}s")
        print(f"  - Image encoders: {len(metadata.image_descriptions)}")
        print(f"  - Metadata: {metadata}")

        # Get image dimensions from metadata
        img_desc = list(metadata.image_descriptions.values())[0]
        img_height = img_desc.height
        img_width = img_desc.width
        print(f"  - Image dimensions: {img_width}x{img_height}")

        # Create dummy test images
        print("\nPreparing test data...")
        raw_images = {
            ImageId.STATIC_CENTER_RGB: np.zeros((480, 640, 3), dtype=np.uint8),
            ImageId.EOAT_LEFT_BOTTOM_RGB: np.zeros((480, 640, 3), dtype=np.uint8),
            ImageId.EOAT_RIGHT_BOTTOM_RGB: np.zeros((480, 640, 3), dtype=np.uint8),
        }

        # Add some visual pattern to images for testing
        for i, (img_id, img) in enumerate(raw_images.items()):
            # Add a colored rectangle to each image
            color = [(255, 0, 0), (0, 255, 0), (0, 0, 255)][i]  # RGB colors
            cv2.rectangle(img, (100, 100), (200, 200), color, -1)
            raw_images[img_id] = img

        # Encode images to bytes
        encoded_images = {}
        for img_id, img in raw_images.items():
            if img_id in metadata.image_descriptions:
                encoded_images[img_id] = [encode_image(img, img_height, img_width)]

        # Example state vector (33D)
        state = [0.0] * 16

        # Request action
        print("\nRequesting action from server...")
        action_request = GetPolicyActionRequest(states=[state], images=encoded_images)

        websocket.send(unstructure(action_request))
        response_bytes = websocket.recv()
        response = structure(response_bytes, GetPolicyActionResponse)

        print("✓ Action response received:")
        print(f"  - Number of actions: {len(response.action)}")
        if response.action:
            print(f"  - First action dimension: {len(response.action[0].action)}")
            print(f"  - First action time: {response.action[0].time_from_start_sec}s")

            # Print first few actions
            print("\nFirst 3 actions:")
            for i, action in enumerate(response.action[:3]):
                print(f"  Action {i}: time={action.time_from_start_sec:.3f}s")
                print(f"    Values: {action.action[:5]}... (showing first 5 of {len(action.action)})")

        # Close connection
        websocket.close()
        print("\n✓ Disconnected from server")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
