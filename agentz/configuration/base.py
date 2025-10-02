"""Shared infrastructure for pipeline configuration objects."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, MutableMapping, Optional, Union

from pydantic import BaseModel, ConfigDict

from agentz.utils import load_config

PipelineConfigSource = Union["PipelineConfigBase", Mapping[str, Any], str, Path]
PipelineConfigInput = Optional[Union[PipelineConfigSource, "ResolvedPipelineConfig"]]
ConfigResolver = Callable[[PipelineConfigSource], "ResolvedPipelineConfig"]


@dataclass
class ResolvedPipelineConfig:
    """Concrete configuration payload handed to pipelines."""

    config_dict: Dict[str, Any]
    config_object: Optional["PipelineConfigBase"] = None
    attachments: Dict[str, Any] = field(default_factory=dict)
    source_path: Optional[str] = None

    def copy(self) -> "ResolvedPipelineConfig":
        return ResolvedPipelineConfig(
            config_dict=deepcopy(self.config_dict),
            config_object=self.config_object,
            attachments=deepcopy(self.attachments),
            source_path=self.source_path,
        )


class PipelineConfigBase(BaseModel):
    """Base class for strongly-typed pipeline configuration objects."""

    model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain serialisable dictionary of the configuration."""

        return self.model_dump(exclude_none=True)

    def resolve(self) -> ResolvedPipelineConfig:
        """Produce the resolved configuration payload for pipeline consumption."""

        return ResolvedPipelineConfig(config_dict=self.to_dict(), config_object=self)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "PipelineConfigBase":
        """Instantiate the config object from a mapping."""

        return cls.model_validate(data)

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "PipelineConfigBase":
        """Instantiate the config object from a YAML or JSON file."""

        data = load_mapping_from_path(path)
        return cls.from_mapping(data)


def load_mapping_from_path(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a mapping from YAML or JSON file, supporting env substitution."""

    data = load_config(Path(path))
    if not isinstance(data, MutableMapping):
        raise ValueError(f"Configuration file must define a mapping, got {type(data)!r}")
    return dict(data)


def resolve_config_source(
    source: PipelineConfigSource,
    *,
    config_cls: Optional[type[PipelineConfigBase]] = None,
) -> ResolvedPipelineConfig:
    """Normalise multiple configuration entry-points into a single payload."""

    if isinstance(source, PipelineConfigBase):
        if config_cls and not isinstance(source, config_cls):
            source = config_cls.model_validate(source.model_dump())
        resolved = source.resolve()
        if resolved.source_path is None and hasattr(source, "source_path"):
            resolved.source_path = getattr(source, "source_path")
        return resolved

    if isinstance(source, Mapping):
        if config_cls is None:
            return ResolvedPipelineConfig(config_dict=dict(source))
        config_obj = config_cls.from_mapping(source)
        resolved = config_obj.resolve()
        resolved.source_path = getattr(config_obj, "source_path", None)
        return resolved

    if isinstance(source, (str, Path)):
        path = Path(source)
        if config_cls is None:
            data = load_mapping_from_path(path)
            return ResolvedPipelineConfig(config_dict=data, source_path=str(path))
        config_obj = config_cls.from_file(path)
        resolved = config_obj.resolve()
        resolved.source_path = str(path)
        return resolved

    raise TypeError(
        "Unsupported configuration source type. Expected PipelineConfigBase, mapping, or path string."
    )


def normalize_pipeline_config(
    source: PipelineConfigInput,
    *,
    config_cls: Optional[type[PipelineConfigBase]] = None,
    default_path: Optional[Union[str, Path]] = None,
) -> ResolvedPipelineConfig:
    """Normalise unstructured pipeline inputs into a resolved configuration.

    Args:
        source: User-supplied configuration input. Accepts the same variants as
            :data:`PipelineConfigSource`, an already resolved payload, or ``None``.
        config_cls: Optional Pydantic model used to coerce mappings into strongly
            typed objects.
        default_path: Optional fallback invoked when ``source`` is ``None``.

    Returns:
        A :class:`ResolvedPipelineConfig` instance that encapsulates the merged
        configuration dictionary, attachments, and provenance metadata.
    """

    if source is None:
        if default_path is None:
            raise ValueError(
                "A configuration source must be provided or a default_path must be set."
            )
        source = default_path

    if isinstance(source, ResolvedPipelineConfig):
        resolved = source.copy()
        if config_cls and resolved.config_object is not None:
            if not isinstance(resolved.config_object, config_cls):
                coerced = config_cls.from_mapping(resolved.config_dict)
                refreshed = coerced.resolve()
                refreshed.attachments.update(resolved.attachments)
                if resolved.source_path and refreshed.source_path is None:
                    refreshed.source_path = resolved.source_path
                return refreshed
        return resolved

    resolved = resolve_config_source(source, config_cls=config_cls)
    return resolved.copy()


__all__ = [
    "ConfigResolver",
    "PipelineConfigBase",
    "PipelineConfigSource",
    "PipelineConfigInput",
    "ResolvedPipelineConfig",
    "load_mapping_from_path",
    "normalize_pipeline_config",
    "resolve_config_source",
]
