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

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from third_party.zordi_policy_rpc.src.zordi_policy_rpc.client.websocket import WebSocketClient
from third_party.zordi_policy_rpc.src.zordi_policy_rpc.client.interface import (
    ActionRequest,
    BimanualPair,
    Pose,
    Position,
)
from third_party.zordi_policy_rpc.src.zordi_policy_rpc.image_transforms import ImageId


def encode_image(image: np.ndarray, target_height: int, target_width: int) -> np.ndarray:
    """Encode image to the format expected by the client."""
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

    return padded


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
        # Connect to WebSocket server using the proper client
        print("\nConnecting to server...")
        with WebSocketClient(args.host, args.port) as client:
            print("✓ Connected to server")

            # Get metadata
            print("\n✓ Metadata received:")
            print(f"  - Observation length: {client.metadata.observation_length}")
            print(f"  - Time delta: {client.metadata.timedelta_sec}s")
            print(f"  - Image encoders: {len(client.metadata.image_descriptions)}")

            # Get image dimensions from metadata
            img_desc = list(client.metadata.image_descriptions.values())[0]
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

            # Encode images to the expected format
            encoded_images = {}
            for img_id, img in raw_images.items():
                if img_id in client.metadata.image_descriptions:
                    encoded_images[img_id] = encode_image(img, img_height, img_width)

            # Create test state data
            print("\nCreating test state...")
            
            # Example joint positions (7D each arm)
            left_joints = [0.0] * 7
            right_joints = [0.0] * 7
            
            # Example gripper positions (1D each)
            left_gripper = [0.5]  # Half open
            right_gripper = [0.5]  # Half open
            
            # Example tool poses (7D each: x, y, z, qx, qy, qz, qw)
            left_tool_pose = Pose.from_list([0.5, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0])
            right_tool_pose = Pose.from_list([0.5, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0])
            
            # Example goal poses
            left_goal_pose = Pose.from_list([0.6, 0.1, 0.4, 0.0, 0.0, 0.0, 1.0])
            right_goal_pose = Pose.from_list([0.6, 0.1, 0.4, 0.0, 0.0, 0.0, 1.0])
            
            # Example tool to goal position (3D vector for right arm)
            right_tool_to_goal_position = Position(x=0.1, y=0.1, z=0.1)  # 3D offset vector
            left_tool_to_goal_position = Position(x=0.0, y=0.0, z=0.0)   # 3D offset vector for left arm

            # Get the expected fields from metadata
            expected_fields = set(client.metadata.state_fields_factory.fields.keys())
            print(f"Server expects fields: {expected_fields}")

            # Create action request with only the fields the server expects
            print("\nRequesting action from server...")
            action_request_kwargs = {
                'images': encoded_images,
            }
            
            # Add fields based on what the server expects
            if any('joint' in field for field in expected_fields):
                action_request_kwargs['joint'] = BimanualPair(left=left_joints, right=right_joints)
            
            if any('gripper' in field for field in expected_fields):
                action_request_kwargs['gripper'] = BimanualPair(left=left_gripper, right=right_gripper)
            
            if any('tool_pose' in field for field in expected_fields):
                action_request_kwargs['tool_pose'] = BimanualPair(left=left_tool_pose, right=right_tool_pose)
            
            if any('goal_tool_pose' in field for field in expected_fields):
                action_request_kwargs['tool_goal_pose'] = BimanualPair(left=left_goal_pose, right=right_goal_pose)
            
            if any('tool_to_goal_position' in field for field in expected_fields):
                action_request_kwargs['tool_to_goal_position'] = BimanualPair(left=left_tool_to_goal_position, right=right_tool_to_goal_position)
            
            action_request = ActionRequest(**action_request_kwargs)

            # Get prediction
            actions = client.predict(action_request)

            if actions is None:
                print("No actions received (need more observation history)")
                return

            print("✓ Action response received:")
            print(f"  - Number of actions: {len(actions)}")
            
            if actions:
                print(f"  - First action time: {actions[0].time_from_start_sec}s")

                # Print first few actions
                print("\nFirst 3 actions:")
                for i, action in enumerate(actions[:3]):
                    print(f"  Action {i}: time={action.time_from_start_sec:.3f}s")
                    
                    if action.joint:
                        print(f"    Joint left: {action.joint.left[:3]}... (showing first 3 of {len(action.joint.left)})")
                        print(f"    Joint right: {action.joint.right[:3]}... (showing first 3 of {len(action.joint.right)})")
                    
                    if action.gripper:
                        print(f"    Gripper left: {action.gripper.left}")
                        print(f"    Gripper right: {action.gripper.right}")
                    
                    if action.tool_pose:
                        print(f"    Tool pose left: {action.tool_pose.left.to_list()[:3]}... (showing first 3 of 7)")
                        print(f"    Tool pose right: {action.tool_pose.right.to_list()[:3]}... (showing first 3 of 7)")
                    
                    if hasattr(action, 'tool_to_goal_position') and action.tool_to_goal_position:
                        print(f"    Tool to goal position left: {action.tool_to_goal_position.left}")
                        print(f"    Tool to goal position right: {action.tool_to_goal_position.right}")

        print("\n✓ Disconnected from server")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
