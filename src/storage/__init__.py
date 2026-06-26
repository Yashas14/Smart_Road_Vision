"""Offline persistence layer (SQLite) for detection history and analytics.

The production system persists to PostgreSQL/PostGIS, but that requires a running
database. This lightweight, dependency-free SQLite store keeps the application
fully functional offline so that history, analytics, the map view and reports all
work out-of-the-box. It is thread-safe and used as the default local backend.
"""

from src.storage.detection_store import DetectionStore, get_store

__all__ = ["DetectionStore", "get_store"]
