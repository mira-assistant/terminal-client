import sys
import requests
import atexit
import asyncio
import signal
from whisper_live import (
    silent_observer,
    kill_observer,
    start_observer,
    is_observer_running,
)

base_url = "http://localhost:8000"
client_id = "Ankur's MacBook Air"
version = "2.0.0"


def connect_to_server():
    try:
        response = requests.post(url=f"{base_url}/register_client", params={"client_id": client_id})
        response.raise_for_status()

        if response.json().get("version") != version:
            print(
                f"ERROR: Version mismatch - expected {version}, got {response.json().get('version')}"
            )
            print("Please update your client to match the server version.")
            sys.exit(1)

        print("Connected to server")

    except Exception as e:
        print(f"Error connecting to server: {e}")
        sys.exit(1)


def disconnect_from_server():
    try:
        response = requests.post(
            url=f"{base_url}/deregister_client", params={"client_id": client_id}
        )
        response.raise_for_status()
        print("Disconnected from server.")
    except Exception as e:
        print(f"Error disconnecting from server: {e}")


def enable_mira():
    try:
        response = requests.patch(url=f"{base_url}/enable")
        response.raise_for_status()
        print("Mira enabled.")
    except Exception as e:
        print(f"Error enabling Mira: {e}")


def disable_mira():
    try:
        response = requests.patch(url=f"{base_url}/disable")
        response.raise_for_status()
        print("Mira disabled.")
    except Exception as e:
        print(f"Error disabling Mira: {e}")


async def process_interaction(sentence_buf: bytearray):
    try:
        response = requests.post(
            url=f"{base_url}/interactions/register",
            data=bytes(sentence_buf),
            headers={"Content-Type": "application/octet-stream"},
        )

        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error processing interaction: {e}")
        return None


def signal_handler(signum, frame):
    """Handle keyboard interrupt (Ctrl+C) gracefully."""
    print("\nMira interrupted.")
    sys.exit(0)


if __name__ == "__main__":
    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)

    sentence_buf = bytearray()

    atexit.register(disable_mira)
    atexit.register(disconnect_from_server)

    connect_to_server()
    enable_mira()

    try:
        while True:
            status = requests.get(url=f"{base_url}/").json()

            if status.get("enabled") is False:
                if is_observer_running():
                    kill_observer()
                    sentence_buf = bytearray()

                continue

            if not is_observer_running():
                start_observer()
                sentence_buf = bytearray()

            sentence_buf, is_sentence_complete = silent_observer(sentence_buf)

            if not is_sentence_complete:
                continue

            asyncio.run(process_interaction(sentence_buf))

    except KeyboardInterrupt:
        print("\nMira interrupted.")
        sys.exit(0)
    except Exception as e:
        print(f"Runtime error: {e}")
        sys.exit(1)
