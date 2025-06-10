#!/usr/bin/env python3
"""
PI0 WebSocket Server - Runs the configurable PI0 action server as a WebSocket service
Copyright 2025 Zordi, Inc. All rights reserved.
"""

import sys
import os
import asyncio
import argparse
import logging
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pi0_action_server_configurable import ConfigurablePI0ActionServer
from third_party.zordi_policy_rpc.src.zordi_policy_rpc.server.websocket import WebSocketServer

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


async def main():
    """Main async function to run the WebSocket server."""
    parser = argparse.ArgumentParser(description="PI0 WebSocket Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=10012, help="Port to bind to")
    parser.add_argument("--config", default="model_configs.yaml", help="Path to model configuration file")
    parser.add_argument("--model", help="Model name to load (if not specified, uses default from config)")
    args = parser.parse_args()

    print("=" * 80)
    print("PI0 WebSocket Server")
    print("=" * 80)
    print(f"Configuration file: {args.config}")
    print(f"Model: {args.model or 'default from config'}")
    print(f"Server will listen on: {args.host}:{args.port}")
    print("=" * 80)

    try:
        # Create the PI0 action server
        print("\nInitializing PI0 Action Server...")
        action_server = ConfigurablePI0ActionServer(config_path=args.config, model_name=args.model)
        print("✓ Action server initialized successfully")

        # Create and run WebSocket server
        print(f"\nStarting WebSocket server on {args.host}:{args.port}...")
        print("Server is running. Press Ctrl+C to stop.")

        async with WebSocketServer(host=args.host, port=args.port, policy=action_server) as server:
            # Keep the server running
            await asyncio.Future()  # Run forever until interrupted

    except KeyboardInterrupt:
        print("\n\nShutting down server...")
    except Exception as e:
        logger.error(f"Server error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # Run the async main function
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nServer stopped by user")
