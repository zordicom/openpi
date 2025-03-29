#!/usr/bin/env python3
"""Test script to check connection to the Galaxea policy server using the OpenPI client library."""

import argparse
import logging
import time

import numpy as np
from openpi_client.websocket_client_policy import WebsocketClientPolicy

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def generate_dummy_data(task_prompt=None):
    """Generate dummy robot state and camera images."""
    # Create dummy robot state (14 joints: 6 left arm, 1 left gripper, 6 right arm, 1 right gripper)
    state = np.zeros(14)

    # Create dummy camera images (224x224x3) in CHW format
    img_size = 224
    images = {
        "static_rs415_top": np.random.randint(0, 256, (3, img_size, img_size), dtype=np.uint8),
        "static_rs405_center": np.random.randint(0, 256, (3, img_size, img_size), dtype=np.uint8),
        "eoat_rs405_left_top": np.random.randint(0, 256, (3, img_size, img_size), dtype=np.uint8),
        "eoat_rs405_right_top": np.random.randint(0, 256, (3, img_size, img_size), dtype=np.uint8),
    }

    # Prepare observation
    observation = {
        "state": state,
        "images": images,
    }

    # Add task prompt if provided
    if task_prompt:
        observation["task"] = [task_prompt]

    return observation


def test_policy_server(host, port, task_prompt, num_requests=5):
    """Test the policy server using WebsocketClientPolicy."""
    # Create the client - automatically connects to the server
    server_url = f"ws://{host}:{port}"
    logger.info(f"Creating client for server at {server_url}")

    try:
        # WebsocketClientPolicy automatically connects and waits for the server
        client = WebsocketClientPolicy(host=host, port=port)
        logger.info(f"Connected to server. Metadata: {client.get_server_metadata()}")

        # Send multiple test requests
        for i in range(num_requests):
            start_time = time.time()

            # Generate and send dummy observation
            observation = generate_dummy_data(task_prompt)
            logger.info(f"Request {i+1}/{num_requests}: Sending observation to server")

            try:
                # Use the client's infer method
                response = client.infer(observation)
                elapsed = time.time() - start_time

                if "actions" in response:
                    actions = response["actions"]
                    logger.info(f"Received actions with shape {actions.shape}")
                    # Print first action (first timestep)
                    if len(actions) > 0:
                        logger.info(f"First action: {actions[0]}")
                    logger.info(f"Round trip time: {elapsed:.3f} seconds")
                else:
                    logger.error(f"Response missing 'actions': {list(response.keys())}")

            except Exception as e:
                logger.error(f"Error during inference: {e}")

            # Wait a bit between requests
            time.sleep(1.0)

        logger.info("Test completed successfully!")

    except Exception as e:
        logger.error(f"Error during test: {e}")


def main():
    """Parse arguments and run the test client."""
    parser = argparse.ArgumentParser(description="Galaxea Policy Server Test Client")
    parser.add_argument("--host", type=str, default="localhost", help="Hostname of the policy server")
    parser.add_argument("--port", type=int, default=8000, help="Port of the policy server")
    parser.add_argument("--prompt", type=str, default="fold the towel", help="Task prompt for the policy")
    parser.add_argument("--requests", type=int, default=5, help="Number of test requests to send")
    args = parser.parse_args()

    try:
        test_policy_server(args.host, args.port, args.prompt, args.requests)
    except KeyboardInterrupt:
        logger.info("Test terminated by user")


if __name__ == "__main__":
    main()
