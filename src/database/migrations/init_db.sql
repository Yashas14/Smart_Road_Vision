-- ============================================================================
-- SmartRoadVision — PostGIS schema initialisation
-- Executed automatically by the postgres container on first start.
-- The SQLModel layer can also create these tables; this script guarantees the
-- PostGIS extension, geospatial indexes and helper triggers exist.
-- ============================================================================

CREATE EXTENSION IF NOT EXISTS postgis;

-- ---------------------------------------------------------------------------
-- locations
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS locations (
    id          SERIAL PRIMARY KEY,
    latitude    DOUBLE PRECISION NOT NULL,
    longitude   DOUBLE PRECISION NOT NULL,
    altitude    DOUBLE PRECISION,
    road_name   VARCHAR(255),
    city        VARCHAR(255),
    geom        geometry(Point, 4326),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_locations_geom ON locations USING GIST (geom);
CREATE INDEX IF NOT EXISTS ix_locations_road_name ON locations (road_name);

-- ---------------------------------------------------------------------------
-- detections
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS detections (
    id                    SERIAL PRIMARY KEY,
    source                VARCHAR(32) NOT NULL DEFAULT 'image',
    model_version         VARCHAR(64) NOT NULL DEFAULT 'yolov11-pothole-v2.0',
    image_width           INTEGER NOT NULL DEFAULT 0,
    image_height          INTEGER NOT NULL DEFAULT 0,
    anomaly_count         INTEGER NOT NULL DEFAULT 0,
    road_condition_score  DOUBLE PRECISION NOT NULL DEFAULT 100.0,
    processing_time_ms    DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    location_id           INTEGER REFERENCES locations (id) ON DELETE SET NULL,
    created_at            TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_detections_created_at ON detections (created_at DESC);
CREATE INDEX IF NOT EXISTS ix_detections_source ON detections (source);
CREATE INDEX IF NOT EXISTS ix_detections_anomaly_count ON detections (anomaly_count);

-- ---------------------------------------------------------------------------
-- anomalies
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS anomalies (
    id              SERIAL PRIMARY KEY,
    detection_id    INTEGER NOT NULL REFERENCES detections (id) ON DELETE CASCADE,
    class_name      VARCHAR(64) NOT NULL,
    confidence      DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    severity_score  DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    severity_level  VARCHAR(16) NOT NULL DEFAULT 'LOW',
    urgency         VARCHAR(32) NOT NULL DEFAULT 'MONITOR',
    depth_mm        DOUBLE PRECISION,
    area_px         DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    area_m2         DOUBLE PRECISION,
    bbox_x1         DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    bbox_y1         DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    bbox_x2         DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    bbox_y2         DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    track_id        INTEGER,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_anomalies_detection_id ON anomalies (detection_id);
CREATE INDEX IF NOT EXISTS ix_anomalies_severity_level ON anomalies (severity_level);
CREATE INDEX IF NOT EXISTS ix_anomalies_class_name ON anomalies (class_name);
CREATE INDEX IF NOT EXISTS ix_anomalies_severity_class
    ON anomalies (severity_level, class_name);

-- ---------------------------------------------------------------------------
-- reports
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS reports (
    id              SERIAL PRIMARY KEY,
    title           VARCHAR(255) NOT NULL DEFAULT 'Road Condition Report',
    file_path       VARCHAR(512),
    total_anomalies INTEGER NOT NULL DEFAULT 0,
    avg_road_score  DOUBLE PRECISION NOT NULL DEFAULT 100.0,
    date_from       TIMESTAMPTZ,
    date_to         TIMESTAMPTZ,
    status          VARCHAR(32) NOT NULL DEFAULT 'completed',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_reports_created_at ON reports (created_at DESC);

-- ---------------------------------------------------------------------------
-- Trigger: keep locations.geom in sync with lat/lon on insert/update
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION sync_location_geom()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.longitude IS NOT NULL AND NEW.latitude IS NOT NULL THEN
        NEW.geom := ST_SetSRID(ST_MakePoint(NEW.longitude, NEW.latitude), 4326);
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_sync_location_geom ON locations;
CREATE TRIGGER trg_sync_location_geom
    BEFORE INSERT OR UPDATE ON locations
    FOR EACH ROW EXECUTE FUNCTION sync_location_geom();
