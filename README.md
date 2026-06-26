<div align="center">

<img src="https://img.shields.io/badge/SmartRoadVision-v2.0-3da9fc?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAxMDAgMTAwIj48dGV4dCB5PSIuOWVtIiBmb250LXNpemU9IjkwIj7wn5ujPC90ZXh0Pjwvc3ZnPg==&logoColor=white" alt="SmartRoadVision v2.0" />

# 🛣️ SmartRoadVision

### *Research-Grade Road Surface Anomaly Detection Platform*

[![IEEE Published](https://img.shields.io/badge/IEEE-Published%20%7C%20DOI%3A10.1109%2F11280062-00629B?style=flat-square&logo=ieee&logoColor=white)](https://ieeexplore.ieee.org/abstract/document/11280062)
[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![YOLOv11](https://img.shields.io/badge/YOLOv11-Ultralytics-FF6B35?style=flat-square&logo=ultralytics&logoColor=white)](https://ultralytics.com)
[![OpenCV](https://img.shields.io/badge/OpenCV-Fallback%20Mode-5C3EE8?style=flat-square&logo=opencv&logoColor=white)](https://opencv.org)
[![Tests](https://img.shields.io/badge/Tests-133%20passing-2ecc71?style=flat-square&logo=pytest&logoColor=white)](tests/)
[![Coverage](https://img.shields.io/badge/Coverage-≥85%25-brightgreen?style=flat-square)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat-square&logo=docker&logoColor=white)](docker-compose.yml)

<br/>

> **IEEE-published, production-grade road anomaly intelligence system** — detecting potholes, cracks, road degradation and speed humps from images, video and live webcam streams. Features composite severity scoring, maintenance cost estimation, GPS mapping, interactive analytics, and PDF reporting — fully operational without any GPU or deep-learning framework via an intelligent OpenCV fallback.

<br/>

![SmartRoadVision Demo](https://img.shields.io/badge/Live%20Demo-http%3A%2F%2Flocalhost%3A8000%2Fapp%2F-3da9fc?style=for-the-badge)

</div>

---

## 📖 IEEE Publication

This system is the software implementation of the following peer-reviewed publication:

<div align="center">

> **"SmartRoadVision: An Intelligent Road Surface Anomaly Detection System"**
>
> *Published in IEEE Xplore*
> **DOI:** [`10.1109/11280062`](https://ieeexplore.ieee.org/abstract/document/11280062)

</div>

### Research Contributions

The paper introduces a novel **hybrid detection architecture** that bridges the gap between high-accuracy deep learning and lightweight edge deployment:

| Contribution | Description |
|---|---|
| **YOLOv11 fine-tuning** | Custom training on a curated multi-class road-surface dataset |
| **OpenCV heuristic detector** | Illumination-normalised thresholding enabling GPU-free deployment |
| **Composite severity model** | Multi-factor scoring (area + depth + confidence + class weight) |
| **SAM2 segmentation** | Pixel-accurate polygon masks per anomaly |
| **MiDaS depth estimation** | Monocular depth → pothole depth in millimetres |
| **Cost-aware prioritisation** | Maintenance cost tied to severity score and estimated real-world area |

---

## ✨ Feature Overview

<table>
<tr>
<td valign="top" width="50%">

**🔍 Detection**
- Potholes, cracks, road degradation, speed humps
- YOLOv11 (GPU) with automatic OpenCV fallback (CPU)
- Bounding boxes + SAM2 polygon masks
- Monocular depth estimation (MiDaS)
- Non-maximum suppression + confidence sorting

**📊 Scoring & Cost**
- Continuous severity score [0–1]
- 4-tier classification: LOW → CRITICAL
- 4-tier urgency: MONITOR → IMMEDIATE
- Maintenance cost per class × severity × area

**🎬 Input Modes**
- Single image upload
- Multi-image batch
- Video (inline + async Celery)
- Live webcam stream

</td>
<td valign="top" width="50%">

**🌐 Web Platform (8 tabs)**
- Detect · Batch · Video · Live Webcam
- Interactive Leaflet map (GPS pins)
- Analytics: Chart.js severity/class/urgency/timeline
- Detection history with thumbnails
- One-click PDF report download

**🗄️ Backend**
- FastAPI REST + WebSocket
- SQLite offline persistence (no setup needed)
- PostgreSQL/PostGIS for production
- Prometheus metrics
- Redis + Celery async video queue

**🧪 Quality**
- 133 automated tests (unit + integration)
- ≥85% code coverage
- Ruff linting + mypy type checking
- GitHub Actions CI pipeline

</td>
</tr>
</table>

---

## 🏗️ Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                         SmartRoadVision v2.0                           │
│                                                                        │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                  Web Frontend (Vanilla JS SPA)                   │  │
│  │  Detect │ Batch │ Video │ Webcam │ Map │ Analytics │ History │ Reports  │
│  └──────────────────────────────────┬───────────────────────────────┘  │
│                                     │ HTTP / WebSocket                  │
│  ┌──────────────────────────────────▼───────────────────────────────┐  │
│  │                     FastAPI Application                           │  │
│  │  /detect  /batch  /video  /stream  /analytics  /reports  /health │  │
│  └──────┬──────────┬────────────────────────────────────────────────┘  │
│         │          │                                                    │
│  ┌──────▼──────┐  ┌▼─────────────────────────────────────┐            │
│  │  Detection  │  │             Data Layer                │            │
│  │  Pipeline   │  │  SQLite (offline) ←→ PostgreSQL (prod)│            │
│  │             │  └──────────────────────────────────────┘            │
│  │  YOLO v11   │                                                       │
│  │     ↓       │  ┌───────────────────────────────────────┐           │
│  │  SAM2 Masks │  │          Analytics Engine             │           │
│  │     ↓       │  │  summary · geojson · timeline · cost  │           │
│  │  MiDaS Depth│  └───────────────────────────────────────┘           │
│  │     ↓       │                                                       │
│  │  OpenCV     │  ┌───────────────────────────────────────┐           │
│  │  Fallback   │  │         Reporting Engine              │           │
│  └─────────────┘  │  ReportLab PDF · Jinja2 templates     │           │
│                   └───────────────────────────────────────┘           │
└────────────────────────────────────────────────────────────────────────┘
```

### Directory Structure

```
smartroadvision/
├── frontend/                    # Vanilla JS 8-tab SPA (no build step)
│   ├── index.html               # Tab shell: Detect · Batch · Video · Live
│   ├── styles.css               #            Map · Analytics · History · Reports
│   └── app.js                   # All logic: fetch, Chart.js, Leaflet, webcam
│
├── src/
│   ├── api/
│   │   ├── main.py              # FastAPI factory — lifespan, CORS, static mount
│   │   ├── dependencies.py      # DI: pipeline · DB session · Redis · store
│   │   └── routers/
│   │       ├── detection.py     # POST /detect/image · /batch · /video/sync · /video
│   │       ├── analytics.py     # GET  /analytics/summary · /geojson · /timeline · /history
│   │       ├── health.py        # GET  /health · /metrics (Prometheus)
│   │       ├── reports.py       # POST /reports/offline/generate · /generate
│   │       └── stream.py        # WS   /detect/stream
│   │
│   ├── detection/
│   │   ├── detector.py          # AnomalyDetector: YOLO + graceful OpenCV fallback
│   │   ├── fallback_detector.py # ClassicalAnomalyDetector (no torch required)
│   │   ├── severity_scorer.py   # Composite multi-factor severity scoring
│   │   ├── depth_estimator.py   # MiDaS monocular depth → depth_mm
│   │   ├── segmentor.py         # SAM2 polygon mask generation
│   │   ├── postprocessor.py     # NMS · bounding-box annotation · draw_annotations
│   │   ├── preprocessor.py      # Resize · denoise · CLAHE normalisation
│   │   └── types.py             # AnomalyDetection · FrameResult · SeverityLevel · UrgencyTag
│   │
│   ├── pipeline/
│   │   ├── image_pipeline.py    # preprocess → detect → score → mask → annotate
│   │   ├── video_pipeline.py    # Frame sampling · IoU/ByteTrack tracking
│   │   ├── batch_pipeline.py    # Parallel multi-image processing
│   │   └── stream_pipeline.py   # Real-time WebSocket frame loop
│   │
│   ├── analytics/
│   │   └── cost_estimator.py    # cost = base × (1 + mult × score) × area_factor
│   │
│   ├── storage/
│   │   └── detection_store.py   # Thread-safe SQLite store (singleton, file-backed)
│   │
│   ├── reporting/
│   │   ├── report_generator.py  # ReportLab + Jinja2 PDF builder
│   │   └── statistics.py        # Aggregate stats helpers
│   │
│   ├── database/                # Production PostgreSQL/PostGIS layer
│   ├── utils/                   # image_utils · geo_utils · export · visualizer
│   └── core/                    # config (Pydantic Settings) · logging · exceptions
│
├── dashboard/                   # Streamlit dashboard (alternative UI)
│   └── pages/
│       ├── 1_live_detection.py
│       ├── 2_analytics.py
│       ├── 3_map_view.py
│       └── 4_reports.py
│
├── training/                    # YOLOv11 fine-tuning
│   ├── train.py                 # Training loop (Roboflow datasets + W&B logging)
│   ├── dataset_prep.py          # Roboflow download and YOLO formatting
│   └── evaluate.py              # mAP · precision · recall evaluation
│
├── notebooks/                   # Research Jupyter notebooks
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_model_benchmarking.ipynb
│   └── 03_depth_severity_analysis.ipynb
│
├── tests/                       # 133 tests — unit + integration
├── scripts/                     # Demo · benchmarking · model download
├── configs/                     # YAML config: app + model settings
├── docker-compose.yml           # API · PostgreSQL · Redis · Nginx · Celery
├── Dockerfile
└── pyproject.toml
```

---

## 🚀 Quick Start

### Option A — Instant Start (OpenCV only, zero GPU/download)

```bash
git clone https://github.com/Yashas14/Smart_Road_Vision.git
cd Smart_Road_Vision

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS

pip install fastapi "uvicorn[standard]" opencv-python numpy pillow piexif \
            pydantic pydantic-settings python-multipart reportlab jinja2 \
            structlog prometheus-client aiofiles python-dotenv pyyaml httpx

uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000/app/** — start detecting immediately.

> The system detects that `ultralytics`/`torch` are absent and silently activates the built-in **OpenCV heuristic detector**. All 8 UI tabs, analytics, history, and PDF reports remain fully functional.

---

### Option B — Full AI Stack (YOLOv11 + SAM2 + MiDaS)

```bash
git clone https://github.com/Yashas14/Smart_Road_Vision.git
cd Smart_Road_Vision

python -m venv .venv && source .venv/bin/activate

# Core + dev dependencies
pip install -e ".[dev,tracking]"

# PyTorch with GPU (CUDA 12.x)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
# Or CPU-only:
# pip install torch torchvision

# Download pretrained weights
python scripts/download_models.py

# Configure environment
cp .env.example .env      # edit YOLO_WEIGHTS, DB URL, etc.

# Launch
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

---

### Option C — Docker (Full Stack)

```bash
git clone https://github.com/Yashas14/Smart_Road_Vision.git
cd Smart_Road_Vision
cp .env.example .env

docker compose up -d --build
```

| Service | URL |
|---|---|
| Web Application | http://localhost:8000/app/ |
| API Docs (Swagger) | http://localhost:8000/docs |
| Streamlit Dashboard | http://localhost:8501 |
| Prometheus Metrics | http://localhost:8000/api/v1/metrics |

---

## 🌐 Web Application — All 8 Tabs

### 🔍 Detect
Drag-and-drop (or browse) a road image. Optionally enter GPS coordinates or click **Use my GPS** to auto-fill from the browser. Results: annotated image overlay, metric cards (anomaly count, road score, cost, latency, engine mode), severity-coded table with export to CSV.

### 🗂️ Batch
Upload tens of images at once. Each is analysed independently; results appear in a thumbnail grid with per-image road-score bars and cost estimates. All are persisted to history automatically.

### 🎬 Video
Upload an MP4/AVI road video. Frames are sampled evenly (configurable max), tracking is applied across frames for unique anomaly counts, and sample annotated frames are returned immediately — no Celery worker required.

### 📷 Live Webcam
Accesses your browser camera (`getUserMedia`). Captures a frame every 2.5 seconds, POSTs to `/detect/image`, and updates the live annotated preview. Works with USB dashcam capture cards.

### 🗺️ Map
All geotagged detections rendered as colour-coded circles on Leaflet / OpenStreetMap. Circle colour: green (score ≥ 80) → amber → red (score < 40). Circle size scales with anomaly count. Click any pin for detail popup.

### 📊 Analytics
Auto-refreshed from the live store:
- **Severity Distribution** — doughnut chart (LOW/MEDIUM/HIGH/CRITICAL)
- **Anomaly Types** — bar chart (pothole/crack/degradation/hump)
- **Urgency Breakdown** — polar area chart
- **Road Score Timeline** — line chart, chronological

Plus 6 summary metric cards: total detections, anomalies, critical count, avg road score, avg confidence, total repair cost.

### 🕓 History
Scrollable thumbnail grid of all stored detections. Filter by source (image/batch/video/webcam). Road-score progress bars. One-click **Clear All** with confirmation.

### 📄 Reports
Enter a title and click **Generate & Download PDF**. Produces a full maintenance report from all stored detections — offline, no database connection needed.

---

## 🔬 Detection Pipeline (Deep Dive)

```
Raw Image (bytes / np.ndarray)
        │
        ▼
  ┌─────────────┐
  │ Preprocessor│  resize(max 1280px) → optional denoise → CLAHE histogram eq.
  └──────┬──────┘
         │
    ┌────┴─────────────────────────────────────────┐
    │                                              │
    ▼  (torch available)             ▼  (fallback) │
┌──────────┐                  ┌───────────────────┐│
│ YOLOv11  │                  │ ClassicalDetector ││
│ detector │                  │ ─────────────────  ││
│ conf=0.35│                  │ grayscale          ││
│ iou=0.45 │                  │ GaussianBlur(3×3)  ││
│ imgsz=640│                  │ large blur(71×71)  ││
└────┬─────┘                  │ subtract(bg, gray) ││
     │                        │ threshold @Δ=18    ││
     │                        │ morphology opening ││
     │                        │ findContours       ││
     │                        │ classify by aspect ││
     │                        └──────────┬─────────┘│
     └──────────────────┬─────────────────┘          │
                        │                            │
                        ▼
              ┌──────────────────┐
              │  SAM2 Segmentor  │  polygon_mask per detection (optional)
              └────────┬─────────┘
                       │
                       ▼
              ┌──────────────────┐
              │  MiDaS Depth     │  relative depth map → depth_mm (optional)
              └────────┬─────────┘
                       │
                       ▼
              ┌──────────────────────────────────┐
              │       Severity Scorer            │
              │  score = Σ(wᵢ × featureᵢ)       │
              │  LOW < 0.30 ≤ MEDIUM < 0.55      │
              │  ≤ HIGH < 0.75 ≤ CRITICAL        │
              └────────┬─────────────────────────┘
                       │
                       ▼
              ┌──────────────────┐
              │  Postprocessor   │  NMS (iou=0.45) → sort confidence → annotate
              └────────┬─────────┘
                       │
                       ▼
              ┌──────────────────┐
              │ Cost Estimator   │  base × (1 + mult × score) × √(area/ref)
              └────────┬─────────┘
                       │
                       ▼
              ┌──────────────────┐
              │ Detection Store  │  SQLite: detection + anomaly rows + thumbnail
              └────────┬─────────┘
                       │
                       ▼
                  API Response
          FrameResult → DetectionResponse JSON
```

---

## 🎯 Anomaly Classes & Severity Model

### Classes

| Class | Detection Criteria | Base Cost (USD) |
|---|---|---|
| `pothole` | Compact blob, high extent (≥0.45), area ≥0.25% frame | $120 |
| `crack` | Elongated (aspect >3.5 or <0.28 with low extent) | $45 |
| `road_degradation` | Diffuse surface damage (fallback class) | $90 |
| `hump` | Speed hump / raised feature | $60 |

### Severity Formula

$$\text{score} = w_{\text{area}} \cdot \hat{a} + w_{\text{depth}} \cdot \hat{d} + w_{\text{conf}} \cdot c + w_{\text{class}} \cdot k$$

Default weights: `area=0.30 · depth=0.35 · confidence=0.15 · class=0.20`

### Severity Thresholds

| Score | Level | Urgency | Action |
|---|---|---|---|
| 0.00 – 0.29 | 🟢 LOW | MONITOR | Schedule next inspection |
| 0.30 – 0.54 | 🟡 MEDIUM | SCHEDULE_REPAIR | Plan within 30 days |
| 0.55 – 0.74 | 🟠 HIGH | URGENT | Repair within 7 days |
| 0.75 – 1.00 | 🔴 CRITICAL | IMMEDIATE | Emergency repair |

### Cost Formula

$$\text{cost} = \text{base} \times \left(1 + \text{sev\_mult} \times \text{score}\right) \times \sqrt{\frac{\text{area\_px}}{\text{area\_ref}}}$$

---

## 📡 REST API Reference

Base path: `/api/v1` | Interactive docs: http://localhost:8000/docs

### Detection Endpoints

```
POST   /detect/image           Single image analysis
POST   /detect/batch           Multi-image batch analysis
POST   /detect/video/sync      Inline video analysis (no queue)
POST   /detect/video           Async video (Celery)
GET    /detect/video/{task_id} Poll async task
WS     /detect/stream          Real-time WebSocket stream
```

### Analytics Endpoints

```
GET    /analytics/summary      Aggregate stats (totals, by-class, by-severity)
GET    /analytics/geojson      GeoJSON FeatureCollection (geotagged only)
GET    /analytics/timeline     Road-score time series (chronological)
GET    /analytics/history      Paginated detection history
GET    /analytics/history/{id} Single detection + anomaly list
DELETE /analytics/history      Clear all records
```

### Report & System Endpoints

```
POST   /reports/offline/generate  PDF from SQLite store (no DB needed)
POST   /reports/generate          PDF from PostgreSQL (production)
GET    /reports/{id}/download     Download generated PDF
GET    /health                    Liveness probe (includes engine/fallback info)
GET    /metrics                   Prometheus metrics
```

### Example Request & Response

```bash
curl -X POST http://localhost:8000/api/v1/detect/image \
  -F "file=@pothole_road.jpg" \
  -F "latitude=12.9716" \
  -F "longitude=77.5946" \
  -F "annotate=true" \
  -F "persist=true"
```

```json
{
  "count": 2,
  "road_condition_score": 58.7,
  "processing_time_ms": 38.4,
  "model_version": "yolov11-pothole-v1",
  "estimated_repair_cost": 312.0,
  "currency": "USD",
  "location": { "latitude": 12.9716, "longitude": 77.5946 },
  "detections": [
    {
      "class_name": "pothole",
      "confidence": 0.91,
      "severity_level": "HIGH",
      "severity_score": 0.68,
      "urgency": "URGENT",
      "depth_mm": 52.1,
      "area_px": 10842.0,
      "bbox": { "x1": 124, "y1": 88, "x2": 312, "y2": 201 },
      "polygon_mask": [[124,88],[180,85],[312,100],[308,201],[120,199]],
      "estimated_repair_cost": 248.0
    },
    {
      "class_name": "crack",
      "confidence": 0.76,
      "severity_level": "MEDIUM",
      "severity_score": 0.41,
      "urgency": "SCHEDULE_REPAIR",
      "depth_mm": null,
      "area_px": 2104.0,
      "estimated_repair_cost": 64.0
    }
  ],
  "annotated_image_base64": "/9j/4AAQ..."
}
```

---

## 🧪 Tests

```bash
# Full suite (133 tests)
python -m pytest -o addopts="" -q

# With coverage report
python -m pytest --cov=src --cov-report=html
open htmlcov/index.html

# Specific suites
python -m pytest tests/unit/ -o addopts="" -q
python -m pytest tests/integration/ -o addopts="" -q

# Single module
python -m pytest tests/unit/test_detection_store.py -o addopts="" -v
```

**Test breakdown:**

| Test Module | Count | What it covers |
|---|---|---|
| `test_detection_store` | 8 | SQLite save/list/summary/geojson/clear |
| `test_fallback_detector` | 12 | OpenCV heuristic detection + NMS + polygon masks |
| `test_severity_scorer` | 10 | Scoring formula, threshold mapping |
| `test_cost_estimator` | 9 | Per-class cost, CostReport, by_urgency |
| `test_export` | 8 | CSV columns, GeoJSON records |
| `test_postprocessor` | 10 | NMS IoU, annotation rendering |
| `test_types` | 8 | Dataclass serialisation, BoundingBox math |
| `test_api_detection` | 6 | Integration: fake pipeline, detect endpoint |
| `test_analytics_api` | 5 | Integration: persist+summary+geojson+clear |
| `test_report_generation` | 6 | PDF generator |
| `test_image_utils` | 7 | bytes→ndarray, base64, resize |
| Others | 44 | config, exceptions, geo, preprocessor, visualizer, video |

---

## ⚙️ Configuration

```dotenv
# .env.example — copy to .env

# --- Application ---
APP_ENV=development              # development | production
APP_SECRET_KEY=change-me         # CHANGE IN PRODUCTION
APP_HOST=0.0.0.0
APP_PORT=8000
CORS_ORIGINS=http://localhost:8501

# --- Detection Model ---
YOLO_WEIGHTS=models/yolov11-pothole.pt
YOLO_CONFIDENCE=0.35
YOLO_IOU=0.45
YOLO_IMGSZ=640
YOLO_DEVICE=auto                 # auto | cpu | cuda:0 | mps

# --- Severity Weights (must sum to 1.0) ---
SEVERITY_W_AREA=0.30
SEVERITY_W_DEPTH=0.35
SEVERITY_W_CONFIDENCE=0.15
SEVERITY_W_CLASS=0.20

# --- Database (production) ---
DATABASE_URL=postgresql+asyncpg://user:pass@host:5432/smartroadvision

# --- Redis / Celery (async video) ---
REDIS_URL=redis://localhost:6379/0
CELERY_BROKER_URL=redis://localhost:6379/1

# --- Optional Integrations ---
ROBOFLOW_API_KEY=               # for training dataset download
WANDB_API_KEY=                  # for experiment tracking
```

---

## 🧠 Model Training

```bash
# 1. Download dataset from Roboflow
ROBOFLOW_API_KEY=your_key python training/dataset_prep.py

# 2. Fine-tune YOLOv11
WANDB_API_KEY=your_key python training/train.py \
    --epochs 100 --imgsz 640 --batch 16 --device cuda:0

# 3. Evaluate
python training/evaluate.py --weights runs/train/weights/best.pt

# 4. Export to ONNX (for CPU deployment)
python -c "from ultralytics import YOLO; YOLO('best.pt').export(format='onnx')"
```

Augmentation config: `training/augmentation_config.yaml`

---

## 🔧 OpenCV Fallback — How It Works

When `torch` / `ultralytics` are unavailable, `ClassicalAnomalyDetector` activates automatically:

```
Frame → Grayscale → GaussianBlur(3×3)
     → Large-kernel background estimate (71×71 blur)
     → cv2.subtract(background, gray)           ← illumination normalisation
     → threshold at darkness_delta=18
     → morphological opening (3×3 ellipse)      ← noise removal
     → findContours
     → filter area: [0.06%, 25%] of frame
     → classify:
         aspect > 3.5 or < 0.28 (& extent < 0.55)  → crack
         extent ≥ 0.45 & area ≥ 0.25% frame          → pothole
         else                                         → road_degradation
     → polygon approximation (approxPolyDP)
     → Non-Maximum Suppression (IoU=0.4, class-agnostic)
     → sort by confidence, cap at 12
```

**No GPU. No internet. No model files. Works on any machine with Python + OpenCV.**

---

## 📊 Streamlit Dashboard

```bash
pip install -e ".[dashboard]"
streamlit run dashboard/app.py --server.port 8501
```

| Page | Description |
|---|---|
| Live Detection | Real-time video / webcam analysis via WebRTC |
| Analytics | Plotly interactive charts from detection history |
| Map View | Folium choropleth heatmap of road conditions |
| Reports | PDF generation and download |

---

## 🐳 Docker Reference

```bash
# Build and start all services
docker compose up -d --build

# View logs
docker compose logs -f api

# Scale API workers
docker compose up -d --scale api=4

# Stop everything
docker compose down

# Reset data volumes
docker compose down -v
```

**docker-compose.yml services:**
- `api` — FastAPI + Uvicorn
- `postgres` — PostgreSQL 16 + PostGIS
- `redis` — Redis 7
- `celery` — Async video worker
- `nginx` — Reverse proxy + static files

---

## 🗺️ Roadmap

- [ ] ONNX runtime export for optimised CPU inference
- [ ] YOLOv11-nano mobile variant
- [ ] Pothole volume estimation (depth × area integration)
- [ ] Multi-city comparative dashboard
- [ ] REST webhook alerts for CRITICAL detections
- [ ] Ground-truth annotation tool integration (CVAT/Label Studio)
- [ ] Federated edge deployment via Docker + Kubernetes
- [ ] 3D point cloud depth fusion (LiDAR integration)
- [ ] Road condition change detection (temporal diff)

---

## 📄 Citation

If you use SmartRoadVision in your research or project, please cite:

```bibtex
@inproceedings{smartroadvision2025,
  title     = {SmartRoadVision: An Intelligent Road Surface Anomaly Detection System},
  author    = {Yashas, D. and others},
  booktitle = {IEEE Xplore},
  year      = {2025},
  doi       = {10.1109/11280062},
  url       = {https://ieeexplore.ieee.org/abstract/document/11280062}
}
```

---

## 🤝 Contributing

1. **Fork** this repository
2. **Create** a feature branch: `git checkout -b feature/your-feature`
3. **Implement** — follow the existing patterns (Pydantic types, structlog, ruff)
4. **Test** — add tests for any new functionality: `python -m pytest -o addopts="" -q`
5. **Lint** — `ruff check src/ tests/`
6. **Submit** a Pull Request with a clear description

Code style: Python 3.12+, type-annotated, Ruff (100 char limit), mypy-strict.

---
## 📸 Snapshots


<img width="1897" height="909" alt="image" src="https://github.com/user-attachments/assets/ce1db101-6333-441d-94fd-bab9450f973e" />



--
<img width="1891" height="906" alt="image" src="https://github.com/user-attachments/assets/b03e275b-c643-4faf-add1-55be037f9cdc" />

---
<img width="1897" height="904" alt="image" src="https://github.com/user-attachments/assets/ed1e79cc-96d2-4b88-9475-5bcd8c7784ce" />


--
<img width="1919" height="906" alt="image" src="https://github.com/user-attachments/assets/c34e371f-24f9-489a-b5aa-c7ffa4175177" />

--
<img width="1919" height="904" alt="image" src="https://github.com/user-attachments/assets/7b8dcf37-2aa2-4edf-ab49-280c1f917b8e" />

--

<img width="1898" height="909" alt="image" src="https://github.com/user-attachments/assets/9ad963df-7a72-4f73-a278-36912c539f5f" />

---

## 📃 License

MIT License — see [LICENSE](LICENSE) for full terms. Free to use, modify, and distribute with attribution.

---

<div align="center">

**SmartRoadVision** — *From IEEE research to production-ready platform.*

[![IEEE Paper](https://img.shields.io/badge/Read%20the%20Paper-IEEE%20Xplore-00629B?style=for-the-badge&logo=ieee&logoColor=white)](https://ieeexplore.ieee.org/abstract/document/11280062)
[![GitHub Profile](https://img.shields.io/badge/Author-Yashas14-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Yashas14)
[![API Docs](https://img.shields.io/badge/API%20Docs-Swagger-85EA2D?style=for-the-badge&logo=swagger&logoColor=black)](http://localhost:8000/docs)

*Star ⭐ the repo if SmartRoadVision helped your research or project!*

</div>
