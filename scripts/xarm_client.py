#!/usr/bin/env python3
"""Galaxea robot client for connecting to the OpenPI policy server."""

import argparse
import logging

import numpy as np
from openpi_client.websocket_client_policy import WebsocketClientPolicy

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# This is a placeholder. Replace with your actual robot controller implementation
class GalaxeaController:
    """Interface for controlling the Galaxea robot."""

    def __init__(self):
        """Initialize the robot controller."""
        logger.info("Initializing Galaxea robot controller")
        self.state = np.zeros(14)  # Default state with 14 joints

    def get_joint_positions(self) -> np.ndarray:
        """Get the current joint positions of the robot."""
        # In a real implementation, this would read from robot hardware
        return self.state

    def get_camera_image(self, camera_name: str) -> np.ndarray:
        """Get the current image from the specified camera."""
        # In a real implementation, this would read from camera hardware
        # Return dummy image with correct shape (C, H, W)
        logger.debug(f"Capturing image from camera: {camera_name}")
        return np.zeros((3, 224, 224), dtype=np.uint8)

    def get_all_camera_images(self) -> dict:
        """Get images from all cameras."""
        return {
            "static_rs415_top": self.get_camera_image("static_rs415_top"),
            "static_rs405_center": self.get_camera_image("static_rs405_center"),
            "eoat_rs405_left_top": self.get_camera_image("eoat_rs405_left_top"),
            "eoat_rs405_right_top": self.get_camera_image("eoat_rs405_right_top"),
        }

    def execute_action(self, action: np.ndarray) -> None:
        """Execute the given action on the robot."""
        # In a real implementation, this would send commands to robot hardware
        logger.info(f"Executing action: {action}")  # Log full action
        self.state = action  # Update internal state

    def reset(self) -> None:
        """Reset the robot to a safe position."""
        logger.info("Resetting robot to home position")
        self.state = np.zeros(14)


def run_galaxea_client(host: str, port: int, task_prompt: str | None = None):
    """Run the Galaxea robot client, connecting to the policy server."""
    # Initialize robot controller
    controller = GalaxeaController()
    controller.reset()

    # Connect to policy server
    logger.info(f"Connecting to policy server at ws://{host}:{port}")
    client = WebsocketClientPolicy(host=host, port=port)
    logger.info(f"Connected to server. Metadata: {client.get_server_metadata()}")

    try:
        while True:
            # Get current robot state and camera images
            state = controller.get_joint_positions()
            images = controller.get_all_camera_images()

            # Prepare observation
            observation = {
                "state": state,
                "images": images,
            }

            # Add task prompt if provided
            if task_prompt:
                observation["task"] = [task_prompt]

            # Send observation to server and get response
            response = client.infer(observation)
            logger.debug("Received response from server")

            if "actions" in response:
                actions = response["actions"]
                # Execute the first action from the sequence
                if len(actions) > 0:
                    controller.execute_action(actions[0])
                else:
                    logger.error("Received empty actions array")
            else:
                logger.error(f"Unexpected response format: {list(response.keys())}")

            # Sleep is handled by the time taken to execute the action in a real robot

    except KeyboardInterrupt:
        logger.info("Client terminated by user")
    except Exception as e:
        logger.error(f"Error during client operation: {e}")

    logger.info("Client connection closed")


def main():
    """Parse arguments and run the client."""
    parser = argparse.ArgumentParser(description="Galaxea Robot Policy Client")
    parser.add_argument("--host", type=str, default="localhost", help="Policy server hostname")
    parser.add_argument("--port", type=int, default=8000, help="Policy server port")
    parser.add_argument("--prompt", type=str, default="fold the towel", help="Task prompt for the policy")
    args = parser.parse_args()

    try:
        run_galaxea_client(args.host, args.port, args.prompt)
    except KeyboardInterrupt:
        logger.info("Client terminated by user")


if __name__ == "__main__":
    main()
