"""Raster2Seq inference for one floorplan image (CPU or GPU).

Loads the CubiCasa5K checkpoint once and turns an image into labelled
polygons in the image's own pixel coordinates:

    {"width", "height",
     "rooms":   [{"type": "Kitchen", "polygon": [[x, y], ...]}],
     "doors":   [[[x1, y1], [x2, y2]]],
     "windows": [[[x1, y1], [x2, y2]]]}
"""

import copy
import io
import os
import sys
import threading

import numpy as np
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from datasets.discrete_tokenizer import DiscreteTokenizer  # noqa: E402
from datasets.transforms import ResizeAndPad  # noqa: E402
from detectron2.data import transforms as T  # noqa: E402
from engine import generate  # noqa: E402
from models import build_model  # noqa: E402
from predict import get_args_parser  # noqa: E402
from raster2seq_hub import resolve_checkpoint_path  # noqa: E402
from util.plot_utils import CC5K_LABEL  # noqa: E402

# Inference flags of the CubiCasa5K checkpoint (its config.json on the Hub)
CC5K_ARGS = [
    "--dataset_name=cubicasa",
    "--semantic_classes=12",
    "--input_channels=3",
    "--poly2seq",
    "--seq_len=512",
    "--num_bins=32",
    "--disable_poly_refine",
    "--dec_attn_concat_src",
    "--per_token_sem_loss",
    "--use_anchor",
    "--ema4eval",
]
WINDOW, DOOR = 9, 10
IMAGE_SIZE = 256
# Long side the plan is brought to before the model sees it: tiny scans
# upscale, huge sheets downscale — the model was trained on 256 px.
MAX_SIDE = 2048


class Recognizer:
    def __init__(self, checkpoint=None, device=None):
        checkpoint = checkpoint or os.environ.get("RASTER2SEQ_CHECKPOINT", "hf:cubicasa5k")
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        args = get_args_parser().parse_args(CC5K_ARGS + ["--device", device])
        self.args = args
        self.device = torch.device(device)
        tokenizer = DiscreteTokenizer(args.num_bins, args.seq_len, add_cls=args.add_cls_token)
        args.vocab_size = len(tokenizer)
        args.pretrained_backbone = False
        args.with_poly_refine = not args.disable_poly_refine
        model = build_model(args, train=False, tokenizer=tokenizer)

        ckpt = torch.load(resolve_checkpoint_path(checkpoint), map_location="cpu")
        state = copy.deepcopy(ckpt["ema"] if "ema" in ckpt else ckpt["model"])
        state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
        model.load_state_dict(state, strict=False)
        del ckpt
        for p in model.parameters():
            p.requires_grad = False
        self.model = model.to(self.device).eval()
        self.transform = T.AugmentationList([ResizeAndPad((IMAGE_SIZE, IMAGE_SIZE), pad_value=255)])
        # The model is not re-entrant: one image at a time
        self.lock = threading.Lock()

    def __call__(self, data: bytes) -> dict:
        img = Image.open(io.BytesIO(data))
        if img.mode in ("RGBA", "LA", "P"):
            img = img.convert("RGBA")
            bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(bg, img)
        img = np.array(img.convert("RGB"))
        h, w = img.shape[:2]

        aug = T.AugInput(img)
        self.transform(aug)
        x = torch.as_tensor(aug.image.transpose(2, 0, 1).copy()).float().div(255)[None].to(self.device)

        with self.lock, torch.no_grad():
            out = generate(
                self.model,
                x,
                semantic_rich=True,
                use_cache=True,
                per_token_sem_loss=self.args.per_token_sem_loss,
                drop_wd=False,
                poly2seq=True,
            )

        # Model space (256 px, letterboxed) → original image pixels
        s = min(IMAGE_SIZE / h, IMAGE_SIZE / w)
        nh, nw = int(h * s), int(w * s)
        top, left = (IMAGE_SIZE - nh) // 2, (IMAGE_SIZE - nw) // 2
        sx, sy = w / nw, h / nh

        def back(poly):
            pts = np.asarray(poly, dtype=float).reshape(-1, 2)
            pts = np.stack([(pts[:, 0] - left) * sx, (pts[:, 1] - top) * sy], 1)
            pts[:, 0] = pts[:, 0].clip(0, w)
            pts[:, 1] = pts[:, 1].clip(0, h)
            return [[round(float(a), 1), round(float(b), 1)] for a, b in pts]

        rooms, doors, windows = [], [], []
        for poly, cls in zip(out["room"][0], out["labels"][0]):
            cls = int(cls)
            pts = back(poly)
            if cls == DOOR:
                doors.append(pts[:2])
            elif cls == WINDOW:
                windows.append(pts[:2])
            elif len(pts) >= 3:
                rooms.append({"type": CC5K_LABEL.get(cls, "Undefined"), "polygon": pts})
        return {"width": w, "height": h, "rooms": rooms, "doors": doors, "windows": windows}
