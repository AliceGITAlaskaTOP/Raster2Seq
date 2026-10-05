"""HTTP service: floorplan image in, Floorplan2Walkthru plan out.

    POST /recognize?name=<plan name>&ppm=<pixels per metre, optional>
         body: PNG / JPEG / WEBP bytes
         → {"recognized": {width, height, rooms, doors, windows},   # pixels
            "pxPerMeter", "origin",                                  # pixels → metres
            "plan":   Floorplan2Walkthru Plan (metres),
            "svg":    CubiCasa model.svg (centimetres),
            "config": Floorplan2Walkthru config.json}
    POST /build?name=<plan name>
         body: Floorplan2Walkthru Plan drawn or edited by hand (JSON, metres)
         → {"plan", "svg", "config", "origin"} — same writer as /recognize
    GET  /health → {"ready": bool}

Meant to sit on a private network next to the app that calls it; with
RECOGNIZE_TOKEN set, every request needs ``Authorization: Bearer <token>``.
The checkpoint is fetched from Hugging Face on first start into HF_HOME —
mount a volume there and restarts / rebuilds never download it again.
"""

import hmac
import io
import json
import os
import sys
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from serve.cubicasa import from_plan, to_cubicasa  # noqa: E402

MAX_BYTES = 20 * 1024 * 1024
TOKEN = os.environ.get("RECOGNIZE_TOKEN", "")

state = {"recognizer": None, "error": None}


def load():
    try:
        import torch

        threads = int(os.environ.get("TORCH_THREADS", "0") or 0)
        if threads:
            torch.set_num_threads(threads)
        from serve.recognize import Recognizer

        state["recognizer"] = Recognizer()
        print("recognizer ready", flush=True)
    except Exception as e:  # noqa: BLE001
        state["error"] = f"{type(e).__name__}: {e}"
        traceback.print_exc()


class Handler(BaseHTTPRequestHandler):
    server_version = "raster2seq"

    def _json(self, code, body):
        data = json.dumps(body, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _allowed(self):
        if not TOKEN:
            return True
        got = self.headers.get("Authorization", "")
        return hmac.compare_digest(got.encode(), f"Bearer {TOKEN}".encode())

    def do_GET(self):
        if urlparse(self.path).path != "/health":
            return self._json(404, {"error": "not found"})
        self._json(200, {"ready": state["recognizer"] is not None, "error": state["error"]})

    def do_POST(self):
        url = urlparse(self.path)
        if url.path not in ("/recognize", "/build"):
            return self._json(404, {"error": "not found"})
        if not self._allowed():
            return self._json(401, {"error": "unauthorized"})
        if url.path == "/build":
            return self._build(url)
        rec = state["recognizer"]
        if rec is None:
            return self._json(503, {"error": state["error"] or "model is loading"})
        size = int(self.headers.get("Content-Length") or 0)
        if size <= 0 or size > MAX_BYTES:
            return self._json(413, {"error": "image size"})
        data = self.rfile.read(size)
        q = parse_qs(url.query)
        name = (q.get("name") or ["План"])[0][:200]
        try:
            ppm = float((q.get("ppm") or [0])[0]) or None
        except ValueError:
            ppm = None
        try:
            gray = np.array(Image.open(io.BytesIO(data)).convert("L"))
            recognized = rec(data)
            out = to_cubicasa(recognized, name=name, gray=gray, scale=ppm)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            return self._json(422, {"error": f"{type(e).__name__}: {e}"})
        out["recognized"] = recognized
        self._json(200, out)

    def _build(self, url):
        """A drawn plan needs no model: it works while the model is still loading."""
        size = int(self.headers.get("Content-Length") or 0)
        if size <= 0 or size > MAX_BYTES:
            return self._json(413, {"error": "plan size"})
        try:
            plan = json.loads(self.rfile.read(size))
            name = (parse_qs(url.query).get("name") or ["План"])[0][:200]
            out = from_plan(plan if isinstance(plan, dict) else {}, name=name)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            return self._json(422, {"error": f"{type(e).__name__}: {e}"})
        self._json(200, out)

    def log_message(self, fmt, *args):
        print("%s %s" % (self.address_string(), fmt % args), flush=True)


def main():
    threading.Thread(target=load, daemon=True).start()
    port = int(os.environ.get("PORT", "8000"))
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
