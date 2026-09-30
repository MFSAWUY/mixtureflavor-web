"""Serve the built frontend and API on one local port, without modifying the ML environment."""
from pathlib import Path
import argparse
import os
import secrets
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / ".python-deps"), str(ROOT / "backend")]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    env_file = ROOT / "runtime/local.env"
    if not env_file.exists():
        env_file.parent.mkdir(exist_ok=True)
        env_file.write_text("FLAVOR_ADMIN_USER=admin\nFLAVOR_ADMIN_PASSWORD=" + secrets.token_urlsafe(32) + "\n", encoding="utf-8")
        try:
            env_file.chmod(0o600)
        except OSError:
            pass
    for line in env_file.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            name, value = line.split("=", 1)
            os.environ.setdefault(name, value)
    print(f"Local administrator credentials: {env_file}", flush=True)
    print(f"Website: http://{args.host}:{args.port}", flush=True)
    if not (ROOT / "frontend/dist/index.html").is_file():
        print("Frontend is not built. Run npm run build --prefix frontend, then restart this service.", flush=True)
    import uvicorn
    uvicorn.run("mixtureflavor_web.api:app", host=args.host, port=args.port, access_log=False)


if __name__ == "__main__":
    main()
