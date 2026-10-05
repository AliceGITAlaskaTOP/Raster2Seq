# Raster2Seq recognition service

A small HTTP service that turns a floorplan image into a 3D-ready plan in the
[Floorplan2Walkthru](https://github.com/AliceGITAlaskaTOP/Floorplan2Walkthru)
format (a CubiCasa5K `model.svg` and its parsed `Plan`). It runs on CPU and is
used by the «Лист» app (`smeta3`, `/api/raspoznat`) as a private neighbour in
docker compose.

```
POST /recognize?name=<plan name>&ppm=<pixels per metre, optional>
     body: PNG / JPEG / WEBP bytes
GET  /health   → {"ready": true|false, "error": null|"…"}
```

`/recognize` answers:

| key          | what                                                                 |
|--------------|----------------------------------------------------------------------|
| `recognized` | raw model output in image pixels: rooms (labelled polygons), doors and windows (segments) |
| `pxPerMeter`, `origin` | scale and shift used to go from pixels to metres          |
| `plan`       | Floorplan2Walkthru `Plan` (metres): walls, doors, windows, rooms with areas, outer perimeter |
| `svg`        | CubiCasa `model.svg` (centimetres) — Floorplan2Walkthru «Import SVG» opens it as is |
| `config`     | Floorplan2Walkthru `config.json` (name, scale, start room, perimeter) |

## How it works

1. `recognize.py` — the CubiCasa5K checkpoint (`hf:cubicasa5k`, room F1 88.7)
   predicts room polygons with labels plus door and window segments. On CPU the
   multi-scale deformable attention falls back to the pure PyTorch kernel.
2. `cubicasa.py` — the model has no walls, so they are derived:
   - near-axis edges are straightened and the same line in two rooms gets one
     coordinate; overlaps go to the smaller room; spikes are cut;
   - a hole enclosed by rooms (an unlabelled hallway) becomes a room;
   - two facing room edges across a gap → an interior wall in the gap;
   - an edge with nothing behind it → an exterior wall, its thickness measured
     on the image (dark ink along the normal);
   - doors and windows are attached to the nearest parallel wall and nested in
     it, as CubiCasa does;
   - scale: median door width = 0.85 m unless `ppm` is given.

## Run

```bash
docker build -f serve/Dockerfile -t raster2seq-serve .
docker run -p 8000:8000 -v raster2seq-models:/models raster2seq-serve
```

The checkpoint (~1.5 GB) is downloaded from Hugging Face on first start into
`HF_HOME=/models`; keep that path on a named volume and rebuilds never download
it again. Env: `RECOGNIZE_TOKEN` (require `Authorization: Bearer <token>`),
`TORCH_THREADS`, `RASTER2SEQ_CHECKPOINT` (default `hf:cubicasa5k`), `PORT`.

One floor per image. About 2–3 s per image on 2 CPU cores, ~2 GB RAM.
