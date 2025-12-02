import os
import signal
import subprocess
import sys
import time

backend_proc = None


def start_backend():
    """Start manga_translator in shared (API) mode on BACKEND_PORT."""
    global backend_proc

    backend_port = os.getenv("BACKEND_PORT", "8001")

    cmd = [
        sys.executable,
        "-m",
        "manga_translator",
        "shared",
        "--host",
        "127.0.0.1",
        "--port",
        backend_port,
    ]

    print(f"[orchestrator] Starting shared backend on 127.0.0.1:{backend_port} ...", flush=True)
    backend_proc = subprocess.Popen(cmd)

    # Give backend time to start up
    time.sleep(5)


def start_api():
    """Start FastAPI wrapper with uvicorn on $PORT (Render’s public port)."""
    port = os.getenv("PORT", "8000")

    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "server.main:app",
        "--host",
        "0.0.0.0",
        "--port",
        port,
    ]

    print(f"[orchestrator] Starting API server on 0.0.0.0:{port} ...", flush=True)
    # This blocks until uvicorn exits
    return subprocess.call(cmd)


def cleanup(*_):
    """Terminate backend process on shutdown."""
    global backend_proc
    print("[orchestrator] Cleaning up...", flush=True)

    if backend_proc and backend_proc.poll() is None:
        backend_proc.terminate()
        try:
            backend_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            backend_proc.kill()

    # Don't force sys.exit here so caller can control exit code


def main():
    # Handle SIGTERM / SIGINT so Render restarts cleanly
    signal.signal(signal.SIGTERM, cleanup)
    signal.signal(signal.SIGINT, cleanup)

    start_backend()
    exit_code = start_api()
    cleanup()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
