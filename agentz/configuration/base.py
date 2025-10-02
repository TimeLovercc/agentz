"""Shared infrastructure for pipeline configuration objects."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

from agentz.utils import load_config


def _deep_merge_dicts(base: Mapping[str, Any], overrides: Mapping[str, Any]) -> Dict[str, Any]:
    """Recursively merge two mappings without mutating the originals."""

    merged: Dict[str, Any] = {key: deepcopy(value) for key, value in base.items()}
    for key, value in overrides.items():
        if (
            key in merged
            and isinstance(merged[key], MutableMapping)
            and isinstance(value, Mapping)
        ):
            merged[key] = _deep_merge_dicts(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _resolve_config_path(
    config_path: Union[str, Path],
    *,
    base_source: Optional[str] = None,
) -> Path:
    """Resolve ``config_path`` against an optional base configuration file."""

    candidate = Path(config_path)
    if not candidate.is_absolute() and base_source:
        try:
            base_dir = Path(base_source).expanduser().parent
        except Exception:
            base_dir = None
        if base_dir is not None:
            candidate = base_dir / candidate
    candidate = candidate.expanduser()
    return candidate


class BaseConfig(BaseModel):
    """Base class for strongly-typed pipeline configuration objects."""

    config_path: Optional[Path] = Field(default=None, exclude=True)
    source_path: Optional[str] = Field(default=None, exclude=True)

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        populate_by_name=True,
        extra="allow",
    )

    def _resolve_external_source(self) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
        if self.config_path is None:
            return None, self.source_path

        candidate = _resolve_config_path(self.config_path, base_source=self.source_path)

        data = load_mapping_from_path(candidate)
        resolved_path = str(candidate)
        self.config_path = candidate
        self.source_path = resolved_path
        return data, resolved_path

    def _merge_with_external_payload(
        self, payload: Mapping[str, Any]
    ) -> tuple[Dict[str, Any], Optional[str]]:
        inline_payload: Dict[str, Any] = {
            key: deepcopy(value) for key, value in payload.items()
        }
        external_data, external_source = self._resolve_external_source()
        if external_data is not None:
            merged_payload = _deep_merge_dicts(external_data, inline_payload)
        else:
            merged_payload = inline_payload

        source = external_source or self.source_path
        if source is not None:
            self.source_path = source
        return merged_payload, source

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain serialisable dictionary of the configuration."""
        return self.model_dump(exclude_none=True)

    def resolve(self) -> tuple[Dict[str, Any], Optional[str]]:
        """Produce the resolved configuration as (config_dict, source_path)."""
        return self._merge_with_external_payload(self.model_dump(exclude_none=True))

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "BaseConfig":
        """Instantiate the config object from a mapping."""
        return cls.model_validate(data)

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "BaseConfig":
        """Instantiate the config object from a YAML or JSON file."""
        data = load_mapping_from_path(path)
        config = cls.from_mapping(data)
        config.source_path = str(Path(path))
        return config


def load_mapping_from_path(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a mapping from YAML or JSON file, supporting env substitution."""

    data = load_config(Path(path))
    if not isinstance(data, MutableMapping):
        raise ValueError(f"Configuration file must define a mapping, got {type(data)!r}")
    return dict(data)


def resolve_config_source(
    source: Union[BaseConfig, Mapping[str, Any], str, Path],
    *,
    config_cls: Optional[type[BaseConfig]] = None,
) -> tuple[Dict[str, Any], Optional[str]]:
    """Normalise multiple configuration entry-points into (config_dict, source_path)."""

    if isinstance(source, BaseConfig):
        if config_cls and not isinstance(source, config_cls):
            source = config_cls.model_validate(source.model_dump())
        config_dict, source_path = source.resolve()
        if source_path is None and hasattr(source, "source_path"):
            source_path = getattr(source, "source_path")
        return config_dict, source_path

    if isinstance(source, Mapping):
        if config_cls is None:
            raw_mapping = dict(source)
            config_path_value = raw_mapping.pop("config_path", None)
            source_path_value = raw_mapping.pop("source_path", None)

            if config_path_value is not None:
                resolved_path = _resolve_config_path(
                    config_path_value, base_source=source_path_value
                )
                external_data = load_mapping_from_path(resolved_path)
                config_dict = _deep_merge_dicts(external_data, raw_mapping)
                source_path_value = source_path_value or str(resolved_path)
            else:
                config_dict = raw_mapping

            return config_dict, source_path_value

        config_obj = config_cls.from_mapping(source)
        config_dict, source_path = config_obj.resolve()
        if source_path is None:
            source_path = getattr(config_obj, "source_path", None)
        return config_dict, source_path

    if isinstance(source, (str, Path)):
        path = Path(source)
        if config_cls is None:
            data = load_mapping_from_path(path)
            return data, str(path)
        config_obj = config_cls.from_file(path)
        config_dict, _ = config_obj.resolve()
        return config_dict, str(path)

    raise TypeError(
        "Unsupported configuration source type. Expected BaseConfig, mapping, or path string."
    )


def normalize_pipeline_config(
    source: Optional[Union[BaseConfig, Mapping[str, Any], str, Path]],
    *,
    config_cls: Optional[type[BaseConfig]] = None,
    default_path: Optional[Union[str, Path]] = None,
) -> tuple[Dict[str, Any], Optional[str]]:
    """Normalise unstructured pipeline inputs into (config_dict, source_path).

    Args:
        source: User-supplied configuration input (BaseConfig, mapping, path, or None).
        config_cls: Optional Pydantic model used to coerce mappings into strongly typed objects.
        default_path: Optional fallback invoked when ``source`` is ``None``.

    Returns:
        A tuple of (config_dict, source_path).
    """

    if source is None:
        if default_path is None:
            raise ValueError(
                "A configuration source must be provided or a default_path must be set."
            )
        source = default_path

    config_dict, source_path = resolve_config_source(source, config_cls=config_cls)
    return deepcopy(config_dict), source_path


__all__ = [
    "BaseConfig",
    "load_mapping_from_path",
    "normalize_pipeline_config",
    "resolve_config_source",
]
