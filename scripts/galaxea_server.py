#!/usr/bin/env python3
"""Serve the Galaxea policy for the towel folding task."""

import logging

from serve_policy import Args
from serve_policy import EnvMode

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config

logging.basicConfig(level=logging.INFO, force=True)

CHECKPOINT_PATH = "../checkpoints/pi0_galaxea/galaxea_towel_folding/17500"


def main():
    """Create and serve the Galaxea policy."""
    args = Args(
        env=EnvMode.GALAXEA,
        port=8000,
        default_prompt="fold the towel",
        record=False,  # Set to True to record policy inputs/outputs for debugging
    )

    # Create policy
    policy_conf = _config.get_config("pi0_galaxea")
    policy = _policy_config.create_trained_policy(
        policy_conf,
        CHECKPOINT_PATH,
        default_prompt=args.default_prompt,
    )
    policy_metadata = policy.metadata

    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    # Start the server
    logging.info(f"Starting Galaxea policy server on port {args.port}")
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
