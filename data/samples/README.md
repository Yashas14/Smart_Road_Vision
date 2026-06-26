# Sample media

Place sample images and short videos here to try the demo and tests.

```bash
# Image demo
python scripts/run_demo.py --source data/samples/road.jpg --save out.jpg

# Video demo
python scripts/run_demo.py --source data/samples/drive.mp4 --save out.mp4
```

Suggested files (add your own):

- `road.jpg` — a single road image (ideally with EXIF GPS for the map view)
- `drive.mp4` — a short dashcam clip
- `drone.mp4` — an aerial road clip

This directory is git-ignored except for this README, so large media files are
not committed.
