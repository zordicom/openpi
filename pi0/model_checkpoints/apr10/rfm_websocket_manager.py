"""
Copyright 2025 Zordi, Inc. All rights reserved.

WebSocket client and server implementations
"""

import asyncio
from collections.abc import Awaitable, Callable
import logging
import pickle
import time
from typing import Any, Generic, TypeVar, cast

import websockets
from websockets.sync.client import connect as sync_connect

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")

WEBSOCKET_CLOSE_SERVICE_OVERLOAD = 1008

DEFAULT_SERVER_HOST = "0.0.0.0"
DEFAULT_CLIENT_HOST = "localhost"
DEFAULT_PORT = 8080

DEFAULT_AUTO_RECONNECT = True
DEFAULT_RECONNECT_ATTEMPTS = 3
DEFAULT_RECONNECT_DELAY = 0.5

DEFAULT_MAX_CLIENTS = 100
DEFAULT_CONNECTION_TIMEOUT = 600.0
DEFAULT_PING_TIMEOUT = 20.0
DEFAULT_MAX_QUEUE = 32


class WebSocketClient(Generic[T, R]):
    """Synchronous WebSocket client implementation."""

    def __init__(
        self,
        host: str = DEFAULT_CLIENT_HOST,
        port: int = DEFAULT_PORT,
        auto_reconnect: bool = DEFAULT_AUTO_RECONNECT,
        reconnect_attempts: int = DEFAULT_RECONNECT_ATTEMPTS,
        reconnect_delay: float = DEFAULT_RECONNECT_DELAY,
        serialize_func: Callable[[T], bytes] | None = None,
        deserialize_func: Callable[[bytes], R] | None = None,
    ):
        """Initialize client.

        Args:
            host: Server hostname
            port: Server port
            auto_reconnect: Whether to automatically reconnect
            reconnect_attempts: Number of reconnection attempts
            reconnect_delay: Delay between reconnection attempts
            serialize_func: Function to serialize data (default: pickle.dumps)
            deserialize_func: Function to deserialize data (default: pickle.loads)
        """
        self.uri = f"ws://{host}:{port}"
        self.websocket = None
        self.connected = False
        self.auto_reconnect = auto_reconnect
        self.reconnect_attempts = reconnect_attempts
        self.reconnect_delay = reconnect_delay
        self.serialize_func = serialize_func or cast(Callable[[T], bytes], pickle.dumps)
        self.deserialize_func = deserialize_func or cast(Callable[[bytes], R], pickle.loads)

    def connect(self) -> None:
        """Connect to WebSocket server."""
        if self.connected:
            return

        last_exception = None
        for attempt in range(1, self.reconnect_attempts + 1):
            try:
                self.websocket = sync_connect(self.uri)
                self.connected = True
                logger.info("Connected to WebSocket server at %s", self.uri)
                return
            except Exception as e:
                last_exception = e
                if attempt < self.reconnect_attempts:
                    logger.warning(
                        "Connection attempt %d/%d failed: %s. Retrying in %.1f " "seconds...",
                        attempt,
                        self.reconnect_attempts,
                        e,
                        self.reconnect_delay,
                    )
                    time.sleep(self.reconnect_delay)

        logger.error("Failed to connect to server: %s", last_exception)
        raise ConnectionError(f"Failed to connect to server: {last_exception}") from last_exception

    def disconnect(self) -> None:
        """Disconnect from WebSocket server."""
        if self.websocket and self.connected:
            self.websocket.close()
            self.connected = False
            logger.info("Disconnected from WebSocket server")

    def send_receive(self, data: T) -> R:
        """Send data to server and receive response.

        Args:
            data: Data to send

        Returns:
            Response data

        Raises:
            ConnectionError: If connection is lost or communication fails
            ValueError: If server returns an error message
        """
        # Ensure connection
        if not self.connected or not self.websocket:
            self.connect()

        # Ensure websocket is connected after connect attempt
        if not self.websocket:
            raise ConnectionError("Failed to establish WebSocket connection")

        result: R | None = None
        try:
            # Serialize and send
            binary_message = self.serialize_func(data)
            self.websocket.send(binary_message)

            # Receive response
            response = self.websocket.recv()

            # Ensure response is bytes before deserializing
            if not isinstance(response, bytes):
                raise TypeError(f"Expected bytes from server, received {type(response).__name__}")

            result = self.deserialize_func(response)

        except Exception as e:
            # Try to reconnect if connection was lost
            logger.warning("Connection lost: %s. Attempting to reconnect...", e)
            self.connected = False
            if self.auto_reconnect:
                self.connect()
                # Retry once after reconnection
                return self.send_receive(data)
            raise ConnectionError(f"WebSocket connection lost: {e}") from e

        # Check for error
        if isinstance(result, dict) and "error" in result:
            raise ValueError(f"Server error: {result['error']}")

        # We ensure result is not None before returning
        if result is None:
            # This case should theoretically not happen if deserialization works
            # and no exception was raised, but we handle it defensively.
            raise ConnectionError("Failed to receive a valid response from the server.")

        return result


class AsyncWebSocketClient(Generic[T, R]):
    """Generic WebSocket client implementation."""

    def __init__(
        self,
        host: str = DEFAULT_CLIENT_HOST,
        port: int = DEFAULT_PORT,
        auto_reconnect: bool = DEFAULT_AUTO_RECONNECT,
        reconnect_attempts: int = DEFAULT_RECONNECT_ATTEMPTS,
        reconnect_delay: float = DEFAULT_RECONNECT_DELAY,
        serialize_func: Callable[[T], bytes] = pickle.dumps,
        deserialize_func: Callable[[bytes], Any] = pickle.loads,
    ):
        """Initialize client.

        Args:
            host: Server hostname
            port: Server port
            auto_reconnect: Whether to automatically reconnect
            reconnect_attempts: Number of reconnection attempts
            reconnect_delay: Delay between reconnection attempts
            serialize_func: Function to serialize data (default: pickle.dumps)
            deserialize_func: Function to deserialize data (default: pickle.loads)
        """
        self.uri = f"ws://{host}:{port}"
        self.websocket: Any | None = None
        self.connected = False
        self.auto_reconnect = auto_reconnect
        self.reconnect_attempts = reconnect_attempts
        self.reconnect_delay = reconnect_delay
        self.serialize_func = serialize_func
        self.deserialize_func = deserialize_func
        self._lock = asyncio.Lock()

    async def _attempt_connect(self) -> tuple[bool, Exception | None]:
        """Attempt to connect once to the WebSocket server.

        Returns:
            Tuple[bool, Optional[Exception]]: (True, None) on success, or
            (False, exception) on failure.
        """
        try:
            self.websocket = await websockets.connect(self.uri)
            self.connected = True
            return True, None
        except Exception as e:
            return False, e

    async def connect(self) -> None:
        """Connect to WebSocket server."""
        async with self._lock:
            if self.connected:
                return

            last_exception = None
            for attempt in range(1, self.reconnect_attempts + 1):
                success, error = await self._attempt_connect()
                if success:
                    logger.info("Connected to WebSocket server at %s", self.uri)
                    return
                last_exception = error
                if attempt < self.reconnect_attempts:
                    logger.warning(
                        "Connection attempt %d/%d failed: %s. Retrying in %.1f " "seconds...",
                        attempt,
                        self.reconnect_attempts,
                        error,
                        self.reconnect_delay,
                    )
                    await asyncio.sleep(self.reconnect_delay)
            logger.error("Failed to connect to server: %s", last_exception)
            raise ConnectionError(f"Failed to connect to server: {last_exception}") from last_exception

    async def disconnect(self) -> None:
        """Disconnect from WebSocket server."""
        async with self._lock:
            if self.websocket and self.connected:
                await self.websocket.close()
                self.connected = False
                logger.info("Disconnected from WebSocket server")

    async def send_receive(self, data: T) -> R:
        """Send data to server and receive response.

        Args:
            data: Data to send

        Returns:
            Response data

        Raises:
            ConnectionError: If connection is lost or communication fails
            ValueError: If server returns an error message
        """
        # Ensure connection
        if not self.connected or not self.websocket:
            await self.connect()

        # Ensure websocket is connected after connect attempt
        if not self.websocket:
            raise ConnectionError("Failed to establish WebSocket connection")

        try:
            # Serialize and send
            binary_message = self.serialize_func(data)
            await self.websocket.send(binary_message)

            # Receive response
            response = await self.websocket.recv()

            # Ensure response is bytes before deserializing
            if not isinstance(response, bytes):
                raise TypeError(f"Expected bytes from server, received {type(response).__name__}")

            result = self.deserialize_func(response)

        except (websockets.exceptions.ConnectionClosed, ConnectionResetError) as e:
            # Try to reconnect if connection was lost
            logger.warning("Connection lost: %s. Attempting to reconnect...", e)
            self.connected = False
            if self.auto_reconnect:
                await self.connect()
                # Retry once after reconnection
                return await self.send_receive(data)
            raise ConnectionError(f"WebSocket connection lost: {e}") from e
        except Exception as e:
            logger.error("WebSocket communication error: %s", e)
            self.connected = False
            raise ConnectionError(f"WebSocket communication error: {e}") from e

        # Check for error
        if isinstance(result, dict) and "error" in result:
            raise ValueError(f"Server error: {result['error']}")

        return cast(R, result)


class WebSocketServer:
    """Generic WebSocket server implementation."""

    def __init__(
        self,
        message_handler: Callable[[Any, bytes, int], Awaitable[None]],
        max_clients: int = DEFAULT_MAX_CLIENTS,
        connection_timeout: float = DEFAULT_CONNECTION_TIMEOUT,
        host: str = DEFAULT_SERVER_HOST,
        port: int = DEFAULT_PORT,
    ):
        """Initialize the WebSocket server.

        Args:
            message_handler: Coroutine function that handles messages
                             (websocket, message, client_id) -> response
            max_clients: Maximum number of simultaneous clients
            connection_timeout: Connection timeout in seconds
            host: Server host
            port: Server port
        """
        self.message_handler = message_handler
        self.clients: set[Any] = set()
        self.max_clients = max_clients
        self.connection_timeout = connection_timeout
        self.host = host
        self.port = port

        # Performance metrics
        self.request_count = 0
        self.total_processing_time = 0.0
        self.connection_count = 0

    async def handle_client(self, websocket: Any) -> None:
        """Handle client connection.

        Args:
            websocket: WebSocket connection object
        """
        # Check max clients
        if len(self.clients) >= self.max_clients:
            logger.warning("Max clients (%d) reached, rejecting connection", self.max_clients)
            await websocket.close(WEBSOCKET_CLOSE_SERVICE_OVERLOAD, "Server at capacity")
            return

        # Add client and track connection
        self.clients.add(websocket)
        self.connection_count += 1
        client_id = self.connection_count
        logger.info(
            "Client %d connected. Total active clients: %d",
            client_id,
            len(self.clients),
        )

        try:
            # Set timeout
            websocket.ping_interval = self.connection_timeout / 2
            websocket.ping_timeout = DEFAULT_PING_TIMEOUT

            # Process messages
            async for message in websocket:
                await self.message_handler(websocket, message, client_id)
        except (websockets.exceptions.ConnectionClosed, ConnectionResetError) as e:
            logger.info("Client %d connection closed: %s", client_id, e)
        except Exception as e:
            logger.error("Error handling client %d: %s", client_id, e)
        finally:
            self.clients.remove(websocket)
            logger.info(
                "Client %d disconnected. Total active clients: %d",
                client_id,
                len(self.clients),
            )

    async def start(self) -> None:
        """Start the WebSocket server."""
        # Configure WebSocket server with optimal settings
        async with websockets.serve(
            self.handle_client,
            self.host,
            self.port,
            max_size=None,  # No limit on message size
            max_queue=DEFAULT_MAX_QUEUE,
            compression=None,  # Disable compression for speed
        ):
            logger.info("WebSocket server started at ws://%s:%s", self.host, self.port)
            logger.info(
                "Server configuration: Max clients=%d, Connection timeout=%.1fs",
                self.max_clients,
                self.connection_timeout,
            )

            try:
                await asyncio.Future()  # Run forever
            except asyncio.CancelledError:
                logger.info("Server shutting down...")


async def send_error_response(websocket: Any, error_message: str) -> None:
    """Send error response to client.

    Args:
        websocket: WebSocket connection
        error_message: Error message
    """
    try:
        error_response = {"error": error_message}
        await websocket.send(pickle.dumps(error_response))
    except Exception as e:
        logger.error("Failed to send error response: %s", e)
