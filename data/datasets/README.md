# Datasets

This directory holds training/validation datasets. It is git-ignored except for
this README.

## Roboflow Pothole Dataset v2

SmartRoadVision fine-tunes YOLOv11 on the **Roboflow Pothole Dataset v2**.

### Automatic download

1. Create a free account at [Roboflow](https://roboflow.com) and copy your API key.
2. Add it to your `.env`:

   ```env
   ROBOFLOW_API_KEY=your_key_here
   ```

3. Run the preparation script (downloads in YOLOv11 format):

   ```bash
   python training/dataset_prep.py
   ```

   This produces:

   ```text
   data/datasets/pothole-detection-v2/
   ├── data.yaml
   ├── train/ { images, labels }
   ├── valid/ { images, labels }
   └── test/  { images, labels }
   ```

### Manual download

Visit `universe.roboflow.com/road-detection`, export the dataset in
**YOLOv11** format, and unzip it into this directory so that `data.yaml`
lives at `data/datasets/pothole-detection-v2/data.yaml`.

## Class map

| id | class             |
|----|-------------------|
| 0  | pothole           |
| 1  | hump              |
| 2  | crack             |
| 3  | road_degradation  |
