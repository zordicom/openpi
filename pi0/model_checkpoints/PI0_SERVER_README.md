# PI0 WebSocket Server and Client

This setup provides a WebSocket-based server for the PI0 action models that can handle requests from multiple clients.

## Files

- `pi0_websocket_server.py` - WebSocket server that runs continuously
- `pi0_websocket_client.py` - Example client for testing the server
- `pi0_action_server_configurable.py` - The configurable action server that supports multiple models
- `model_configs.yaml` - Configuration file defining all available models

## Available Models

1. **pi05_full** - PI05 model with full 33-dimensional action space
2. **pi0_ee** - PI0 model with 16-dimensional end-effector action space  
3. **pi0_joint** - PI0 model with 16-dimensional joint action space

## Running the Server

To start the server in one terminal:

```bash
# Run with default model (pi05_full)
python pi0_websocket_server.py

# Run with a specific model
python pi0_websocket_server.py --model pi0_ee

# Run on a different port
python pi0_websocket_server.py --port 8080

# See all options
python pi0_websocket_server.py --help
```

The server will:
- Load the specified model
- Listen for WebSocket connections on the specified port (default: 8765)
- Process action requests from clients
- Run continuously until stopped with Ctrl+C

## Running the Client

In another terminal, test the server with the client:

```bash

# Connect to different host/port
python pi0_websocket_client.py --host 0.0.0.0 --port 8080


```

The client will:
- Connect to the server
- Request metadata
- Send dummy images and state
- Receive and display action predictions

## Model Configuration

Edit `model_configs.yaml` to:
- Add new models
- Change model paths
- Adjust inference parameters
- Modify action/state dimensions

## Troubleshooting

### Tokenizer not found error
If you see an error about missing tokenizer files, update the tokenizer path in `model_configs.yaml` to point to an existing tokenizer file (usually in the may21 directory).

### GPU/CUDA errors
The server will automatically fall back to CPU if GPU is not available.

### Connection refused
Make sure the server is running before starting the client. 