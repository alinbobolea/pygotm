"""Driver-facing configuration helpers built on top of ``config.settings``."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

from pygotm.config.settings import GotmSettings, _normalize_settings_document

yaml: Any = import_module("yaml")

__all__ = [
    "ConfigLike",
    "GotmConfig",
    "coerce_config",
    "load_config",
    "save_config",
]


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        msg = f"top-level YAML document in {path} must be a mapping"
        raise TypeError(msg)
    normalized = _normalize_settings_document(raw)
    assert isinstance(normalized, dict)
    return normalized


def _deep_merge(target: dict[str, Any], update: dict[str, Any]) -> None:
    for key, value in update.items():
        if key in target and isinstance(target[key], dict) and isinstance(value, dict):
            _deep_merge(target[key], value)
        else:
            target[key] = deepcopy(value)


def _resolve_relative_file_paths(node: object, base_dir: Path) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "file" and isinstance(value, str):
                if value and not Path(value).is_absolute():
                    node[key] = str((base_dir / value).resolve())
            else:
                _resolve_relative_file_paths(value, base_dir)
        return

    if isinstance(node, list):
        for value in node:
            _resolve_relative_file_paths(value, base_dir)
        return

    if isinstance(node, tuple):
        for value in node:
            _resolve_relative_file_paths(value, base_dir)


@dataclass(frozen=True)
class GotmConfig:
    """A GOTM YAML document plus a typed view and source-location metadata."""

    settings: GotmSettings
    document: dict[str, Any]
    source_path: Path | None = None

    @classmethod
    def from_path(cls, path: str | Path) -> GotmConfig:
        """Load a GOTM YAML document from *path*."""

        source_path = Path(path).resolve()
        document = _load_yaml_mapping(source_path)
        settings = GotmSettings.model_validate(document)
        return cls(settings=settings, document=document, source_path=source_path)

    @classmethod
    def from_settings(
        cls,
        settings: GotmSettings,
        *,
        source_path: str | Path | None = None,
        document: dict[str, Any] | None = None,
    ) -> GotmConfig:
        """Construct a config wrapper from an in-memory settings object."""

        materialized_document = (
            deepcopy(document)
            if document is not None
            else settings.model_dump(by_alias=True, exclude_none=True)
        )
        resolved_path = None if source_path is None else Path(source_path).resolve()
        return cls(
            settings=settings.model_copy(deep=True),
            document=materialized_document,
            source_path=resolved_path,
        )

    @property
    def source_dir(self) -> Path | None:
        """Directory containing the source YAML document, if known."""

        if self.source_path is None:
            return None
        return self.source_path.parent

    def materialize_document(self) -> dict[str, Any]:
        """Return the saved YAML document with typed settings merged back in."""

        document = deepcopy(self.document)
        _deep_merge(
            document,
            self.settings.model_dump(by_alias=True, exclude_none=True),
        )
        return document

    def resolved_document(self) -> dict[str, Any]:
        """Return a document copy with relative ``file`` entries resolved."""

        document = self.materialize_document()
        if self.source_dir is not None:
            _resolve_relative_file_paths(document, self.source_dir)
        return document

    def resolved_settings(self) -> GotmSettings:
        """Return a settings model with relative input paths resolved."""

        return GotmSettings.model_validate(self.resolved_document())

    def save(self, path: str | Path) -> None:
        """Write the current configuration document to *path*."""

        save_config(self, path)


type ConfigLike = GotmConfig | GotmSettings | str | Path


def load_config(path: str | Path) -> GotmConfig:
    """Load a :class:`GotmConfig` from *path*."""

    return GotmConfig.from_path(path)


def save_config(config: GotmConfig | GotmSettings, path: str | Path) -> None:
    """Write a configuration document or settings model to *path*."""

    if isinstance(config, GotmSettings):
        document = config.model_dump(by_alias=True, exclude_none=True)
    else:
        document = config.materialize_document()
    Path(path).write_text(
        yaml.safe_dump(document, sort_keys=False),
        encoding="utf-8",
    )


def coerce_config(config: GotmConfig | GotmSettings | str | Path) -> GotmConfig:
    """Normalise config-like inputs into :class:`GotmConfig`."""

    if isinstance(config, GotmConfig):
        return config
    if isinstance(config, GotmSettings):
        return GotmConfig.from_settings(config)
    return load_config(config)
