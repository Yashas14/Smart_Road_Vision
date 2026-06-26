# Model weights

Place downloaded model checkpoints here. This directory is git-ignored except
for this README.

## Automatic download

```bash
python scripts/download_models.py --all
```

This fetches:

| File                         | Model        | Source                              |
|------------------------------|--------------|-------------------------------------|
| (ultralytics cache) `yolo11n.pt` | YOLOv11 base | ultralytics (auto)              |
| `sam2_hiera_small.pt`        | SAM2         | Meta segment-anything-2 release     |
| (torch.hub cache)            | MiDaS v3.1   | `intel-isl/MiDaS` via `torch.hub`   |

## Fine-tuned weights

After training (`python training/train.py`), copy the best checkpoint here and
point `.env` at it:

```bash
cp runs/train/smartroad_yolov11/weights/best.pt models/yolov11-pothole.pt
```

```env
YOLO_WEIGHTS=models/yolov11-pothole.pt
```

If `YOLO_WEIGHTS` does not exist, the detector falls back to a base YOLOv11
checkpoint so the system still runs end-to-end.
