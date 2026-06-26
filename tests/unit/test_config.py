"""Unit tests for application configuration (:mod:`src.core.config`)."""

from __future__ import annotations

from src.core.config import Settings, get_settings, load_yaml_config


def test_default_settings_values() -> None:
    settings = Settings()
    assert settings.app_name == "SmartRoadVision"
    assert settings.app_port == 8000
    assert 0.0 < settings.yolo_confidence < 1.0


def test_cors_origins_list_splits_and_strips() -> None:
    settings = Settings(cors_origins=" http://a.com , http://b.com ,")
    assert settings.cors_origins_list == ["http://a.com", "http://b.com"]


def test_is_production_flag() -> None:
    assert Settings(app_env="production").is_production is True
    assert Settings(app_env="prod").is_production is True
    assert Settings(app_env="staging").is_production is True
    assert Settings(app_env="development").is_production is False


def test_severity_weights_mapping() -> None:
    settings = Settings()
    weights = settings.severity_weights
    assert set(weights) == {"area", "depth", "confidence", "class"}
    # Default weights should sum to 1.0.
    assert round(sum(weights.values()), 4) == 1.0


def test_get_settings_is_cached_singleton() -> None:
    assert get_settings() is get_settings()


def test_load_yaml_config_reads_existing_file() -> None:
    cfg = load_yaml_config("model_config.yaml")
    assert isinstance(cfg, dict)
    assert cfg  # non-empty


def test_load_yaml_config_missing_returns_empty() -> None:
    assert load_yaml_config("does_not_exist_42.yaml") == {}


def test_env_override(monkeypatch) -> None:
    monkeypatch.setenv("APP_PORT", "9999")
    settings = Settings()
    assert settings.app_port == 9999
