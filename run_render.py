import os
import signal
import subprocess
import sys
import time
import uvicorn

backend_proc = None

def start_backend():
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
    time.sleep(5)

def cleanup(*_):
    global backend_proc
    print("[orchestrator] Cleaning up...", flush=True)
    if backend_proc and backend_proc.poll() is None:
        backend_proc.terminate()
        try:
            backend_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            backend_proc.kill()

def main():
    signal.signal(signal.SIGTERM, cleanup)
    signal.signal(signal.SIGINT, cleanup)

    start_backend()

    port = int(os.getenv("PORT", "8000"))
    print(f"[orchestrator] Running uvicorn on 0.0.0.0:{port}", flush=True)

    # Run uvicorn in this same process so any errors show up in logs
    uvicorn.run("server.main:app", host="0.0.0.0", port=port)

    cleanup()

if __name__ == "__main__":
    main()