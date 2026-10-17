"""Generation-parameter selection and usage accounting for gameplay inference."""

from __future__ import annotations

import dataclasses
import hashlib
import logging
import math
import random
from collections import defaultdict
from typing import Any, Optional

logger = logging.getLogger(__name__)

GENERATION_PARAM_NAMES = ("temperature", "top_p", "top_k")
_TOP_LEVEL_FIELDS = {
    "strategy",
    "seed",
    "seed_scope",
    "value_precision",
    "key_precision",
    "usage_summary_top_n",
    "ranges",
    "selection",
    "options",
}


@dataclasses.dataclass(frozen=True)
class GenerationParamRange:
    minimum: float | int
    maximum: float | int

    def sample(
        self,
        rng: random.Random,
        parameter: str,
        value_precision: Optional[int],
    ) -> float | int:
        if parameter == "top_k":
            return rng.randint(int(self.minimum), int(self.maximum))
        value = rng.uniform(float(self.minimum), float(self.maximum))
        if value_precision is not None:
            value = round(value, value_precision)
            if value == 0:
                value = 0.0
        return value


GenerationParamSpec = float | int | GenerationParamRange


@dataclasses.dataclass(frozen=True)
class GenerationParamOption:
    name: Optional[str]
    values: dict[str, GenerationParamSpec]

    @property
    def has_ranges(self) -> bool:
        return any(isinstance(value, GenerationParamRange) for value in self.values.values())

    def sample(
        self,
        rng: random.Random,
        value_precision: Optional[int],
    ) -> dict[str, float | int]:
        sampled: dict[str, float | int] = {}
        for parameter in GENERATION_PARAM_NAMES:
            if parameter not in self.values:
                continue
            value = self.values[parameter]
            if isinstance(value, GenerationParamRange):
                sampled[parameter] = value.sample(rng, parameter, value_precision)
            else:
                sampled[parameter] = value
        return sampled


@dataclasses.dataclass(frozen=True)
class GenerationParamsConfig:
    strategy: str = "off"
    seed: Optional[int] = None
    seed_scope: str = "per_run"
    value_precision: Optional[int] = None
    key_precision: int = 3
    usage_summary_top_n: int = 10
    ranges: dict[str, GenerationParamRange] = dataclasses.field(default_factory=dict)
    selection: str = "sequential"
    options: tuple[GenerationParamOption, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def enabled(self) -> bool:
        return self.strategy != "off"

    @property
    def configured_param_names(self) -> set[str]:
        if self.strategy == "random_ranges":
            return set(self.ranges)
        if self.strategy == "options":
            return {
                parameter
                for option in self.options
                for parameter in option.values
            }
        return set()

    @classmethod
    def from_config(cls, data: Any) -> "GenerationParamsConfig":
        if data is None:
            return cls()
        if not isinstance(data, dict):
            raise ValueError("gameplay.model_settings.generation_params must be a mapping")

        warnings: list[str] = []
        for field in data:
            if field not in _TOP_LEVEL_FIELDS:
                warnings.append(
                    "Unknown generation_params field "
                    f"{field!r}; it will be ignored"
                )

        strategy = str(data.get("strategy", "off")).strip().lower()
        if strategy not in {"off", "random_ranges", "options"}:
            raise ValueError(
                "generation_params.strategy must be one of: "
                "off, random_ranges, options"
            )

        seed = _optional_int(data.get("seed"), "generation_params.seed")
        seed_scope = str(data.get("seed_scope", "per_run")).strip().lower()
        if seed_scope not in {"per_run", "same_per_suite"}:
            raise ValueError(
                "generation_params.seed_scope must be per_run or same_per_suite"
            )

        value_precision = _optional_non_negative_int(
            data.get("value_precision"),
            "generation_params.value_precision",
        )
        key_precision = _non_negative_int(
            data.get("key_precision", 3),
            "generation_params.key_precision",
        )
        usage_summary_top_n = _non_negative_int(
            data.get("usage_summary_top_n", 10),
            "generation_params.usage_summary_top_n",
        )

        if strategy == "off":
            config = cls(
                strategy=strategy,
                seed=seed,
                seed_scope=seed_scope,
                value_precision=value_precision,
                key_precision=key_precision,
                usage_summary_top_n=usage_summary_top_n,
                warnings=tuple(warnings),
            )
            _log_warnings(config.warnings)
            return config

        if strategy == "random_ranges":
            ranges = _parse_ranges(
                data.get("ranges", {}),
                "generation_params.ranges",
                warnings,
            )
            config = cls(
                strategy=strategy,
                seed=seed,
                seed_scope=seed_scope,
                value_precision=value_precision,
                key_precision=key_precision,
                usage_summary_top_n=usage_summary_top_n,
                ranges=ranges,
                warnings=tuple(warnings),
            )
            _log_warnings(config.warnings)
            return config

        selection = str(data.get("selection", "sequential")).strip().lower()
        if selection not in {"sequential", "random"}:
            raise ValueError(
                "generation_params.selection must be sequential or random"
            )
        options_raw = data.get("options")
        if not isinstance(options_raw, list) or not options_raw:
            raise ValueError(
                "generation_params.options must be a non-empty list "
                "when strategy is options"
            )
        options = tuple(
            _parse_option(option, index, warnings)
            for index, option in enumerate(options_raw)
        )
        _validate_unique_options(options, key_precision)
        config = cls(
            strategy=strategy,
            seed=seed,
            seed_scope=seed_scope,
            value_precision=value_precision,
            key_precision=key_precision,
            usage_summary_top_n=usage_summary_top_n,
            selection=selection,
            options=options,
            warnings=tuple(warnings),
        )
        _log_warnings(config.warnings)
        return config


@dataclasses.dataclass(frozen=True)
class GenerationParamsSelection:
    source: str
    requested: dict[str, float | int]
    selection_index: Optional[int] = None
    option_name: Optional[str] = None
    option_has_ranges: bool = False


class GenerationParamsSampler:
    """Select per-inference generation parameter overrides."""

    def __init__(
        self,
        config: GenerationParamsConfig,
        *,
        effective_seed: Optional[int] = None,
    ) -> None:
        self.config = config
        self.effective_seed = (
            config.seed if effective_seed is None else effective_seed
        )
        self._rng = random.Random(self.effective_seed)
        self._sequential_index = 0

    def select(self) -> GenerationParamsSelection:
        if not self.config.enabled:
            raise RuntimeError("Generation-parameter sampler is disabled")

        if self.config.strategy == "random_ranges":
            requested = {
                parameter: value_range.sample(
                    self._rng,
                    parameter,
                    self.config.value_precision,
                )
                for parameter, value_range in self.config.ranges.items()
            }
            return GenerationParamsSelection(
                source="random_ranges",
                requested=_ordered_params(requested),
            )

        if self.config.selection == "sequential":
            index = self._sequential_index % len(self.config.options)
            self._sequential_index += 1
        else:
            index = self._rng.randrange(len(self.config.options))
        option = self.config.options[index]
        return GenerationParamsSelection(
            source="options",
            requested=option.sample(self._rng, self.config.value_precision),
            selection_index=index,
            option_name=option.name,
            option_has_ranges=option.has_ranges,
        )


@dataclasses.dataclass
class _UsageEntry:
    base_key: str
    display_key: str
    effective: dict[str, float | int]
    count: int = 0


class GenerationParamsUsageTracker:
    """Count attempted gameplay inferences by selected/effective parameters."""

    def __init__(
        self,
        config: GenerationParamsConfig,
        *,
        effective_seed: Optional[int] = None,
    ) -> None:
        self.config = config
        self.effective_seed = effective_seed
        self.total_prompts = 0
        self._entries: dict[tuple[Any, ...], _UsageEntry] = {}
        self._base_key_counts: dict[str, int] = defaultdict(int)

    def record(
        self,
        selection: Optional[GenerationParamsSelection],
        effective: dict[str, float | int],
    ) -> str:
        self.total_prompts += 1
        effective = _ordered_params(effective)

        if selection is None:
            if not effective:
                return "default"
            identity = ("effective", _canonical_params(effective))
            base_key = format_generation_params_key(
                effective,
                self.config.key_precision,
            )
        elif selection.option_name:
            if selection.option_has_ranges:
                identity = (
                    "named_ranged",
                    selection.option_name,
                    _canonical_params(effective),
                )
                base_key = (
                    f"{selection.option_name}:"
                    f"{format_generation_params_key(effective, self.config.key_precision)}"
                )
            else:
                identity = ("named", selection.option_name)
                base_key = selection.option_name
        else:
            identity = ("effective", _canonical_params(effective))
            base_key = format_generation_params_key(
                effective,
                self.config.key_precision,
            )

        entry = self._entries.get(identity)
        if entry is None:
            self._base_key_counts[base_key] += 1
            collision_index = self._base_key_counts[base_key]
            display_key = (
                base_key
                if collision_index == 1
                else f"{base_key}#{collision_index}"
            )
            entry = _UsageEntry(
                base_key=base_key,
                display_key=display_key,
                effective=effective,
            )
            self._entries[identity] = entry
        entry.count += 1
        return entry.display_key

    def summary(self) -> dict[str, Any]:
        ranked = sorted(
            self._entries.items(),
            key=lambda item: (
                -item[1].count,
                item[1].display_key,
                repr(item[0]),
            ),
        )
        top = [
            {
                "key": entry.display_key,
                "count": entry.count,
                "effective": entry.effective,
            }
            for _, entry in ranked[: self.config.usage_summary_top_n]
        ]
        result: dict[str, Any] = {
            "strategy": self.config.strategy,
            "total_prompts": self.total_prompts,
            "unique_count": len(self._entries),
            "top": top,
        }
        if self.config.enabled and self.effective_seed is not None:
            result["effective_seed"] = self.effective_seed
        return result


def derive_suite_generation_params_seed(
    config: GenerationParamsConfig,
    run_name: str,
    repeat_index: int,
) -> Optional[int]:
    """Derive a stable effective seed for one suite run."""
    if config.seed is None:
        return None
    if config.seed_scope == "same_per_suite":
        return config.seed
    digest = hashlib.sha256(run_name.encode("utf-8")).digest()
    run_hash = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return (config.seed + repeat_index + run_hash) % (2**63)


def format_generation_params_key(
    params: dict[str, float | int],
    precision: int,
) -> str:
    if not params:
        return "default"
    parts = []
    for parameter, value in _ordered_params(params).items():
        if isinstance(value, float):
            rendered = f"{value:.{precision}f}".rstrip("0").rstrip(".")
            if precision > 0 and "." not in rendered:
                rendered += ".0"
        else:
            rendered = str(value)
        parts.append(f"{parameter}={rendered}")
    return ",".join(parts)


def _parse_option(
    data: Any,
    index: int,
    warnings: list[str],
) -> GenerationParamOption:
    path = f"generation_params.options[{index}]"
    if not isinstance(data, dict):
        raise ValueError(f"{path} must be a mapping")
    for field in data:
        if field != "name" and field not in GENERATION_PARAM_NAMES:
            warnings.append(f"Unknown {path} field {field!r}; it will be ignored")

    name_raw = data.get("name")
    name = None if name_raw is None else str(name_raw).strip() or None
    values: dict[str, GenerationParamSpec] = {}
    for parameter in GENERATION_PARAM_NAMES:
        if parameter not in data:
            continue
        value = data[parameter]
        if isinstance(value, dict):
            values[parameter] = _parse_range(value, parameter, f"{path}.{parameter}")
        else:
            values[parameter] = _parse_param_value(
                value,
                parameter,
                f"{path}.{parameter}",
            )
    return GenerationParamOption(name=name, values=values)


def _parse_ranges(
    data: Any,
    path: str,
    warnings: list[str],
) -> dict[str, GenerationParamRange]:
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must be a mapping")
    for field in data:
        if field not in GENERATION_PARAM_NAMES:
            warnings.append(f"Unknown {path} field {field!r}; it will be ignored")
    return {
        parameter: _parse_range(data[parameter], parameter, f"{path}.{parameter}")
        for parameter in GENERATION_PARAM_NAMES
        if parameter in data
    }


def _parse_range(
    data: Any,
    parameter: str,
    path: str,
) -> GenerationParamRange:
    if not isinstance(data, dict):
        raise ValueError(f"{path} must be a mapping with min and max")
    unknown = set(data) - {"min", "max"}
    if unknown:
        raise ValueError(
            f"{path} contains unknown range fields: {', '.join(sorted(unknown))}"
        )
    if "min" not in data or "max" not in data:
        raise ValueError(f"{path} must define both min and max")
    minimum = _parse_param_value(data["min"], parameter, f"{path}.min")
    maximum = _parse_param_value(data["max"], parameter, f"{path}.max")
    if minimum > maximum:
        raise ValueError(f"{path}.min must be less than or equal to max")
    return GenerationParamRange(minimum=minimum, maximum=maximum)


def _parse_param_value(value: Any, parameter: str, path: str) -> float | int:
    if parameter == "top_k":
        parsed: float | int = _strict_int(value, path)
    else:
        if isinstance(value, bool):
            raise ValueError(f"{path} must be a number")
        try:
            parsed = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path} must be a number") from exc
        if not math.isfinite(parsed):
            raise ValueError(f"{path} must be finite")

    if parameter == "temperature" and parsed < 0:
        raise ValueError(f"{path} must be greater than or equal to 0")
    if parameter == "top_p" and not 0 <= parsed <= 1:
        raise ValueError(f"{path} must be between 0 and 1")
    if parameter == "top_k" and parsed < 1:
        raise ValueError(f"{path} must be greater than or equal to 1")
    return parsed


def _validate_unique_options(
    options: tuple[GenerationParamOption, ...],
    key_precision: int,
) -> None:
    names: set[str] = set()
    unnamed: set[tuple[Any, ...]] = set()
    for option in options:
        if option.name:
            if option.name in names:
                raise ValueError(
                    f"Duplicate generation_params option name: {option.name}"
                )
            names.add(option.name)
            continue

        identity = tuple(
            (parameter, _canonical_spec(option.values[parameter]))
            for parameter in GENERATION_PARAM_NAMES
            if parameter in option.values
        )
        if identity in unnamed:
            key = _format_option_definition_key(option, key_precision)
            raise ValueError(
                f"Duplicate unnamed generation_params option: {key}"
            )
        unnamed.add(identity)


def _format_option_definition_key(
    option: GenerationParamOption,
    precision: int,
) -> str:
    if not option.values:
        return "default"
    parts: list[str] = []
    for parameter in GENERATION_PARAM_NAMES:
        if parameter not in option.values:
            continue
        value = option.values[parameter]
        if isinstance(value, GenerationParamRange):
            bounds = format_generation_params_key(
                {parameter: value.minimum},
                precision,
            ).split("=", 1)[1]
            upper = format_generation_params_key(
                {parameter: value.maximum},
                precision,
            ).split("=", 1)[1]
            parts.append(f"{parameter}={bounds}..{upper}")
        else:
            parts.append(
                format_generation_params_key({parameter: value}, precision)
            )
    return ",".join(parts)


def _canonical_spec(value: GenerationParamSpec) -> tuple[Any, ...]:
    if isinstance(value, GenerationParamRange):
        return ("range", value.minimum, value.maximum)
    return ("value", value)


def _canonical_params(
    params: dict[str, float | int],
) -> tuple[tuple[str, float | int], ...]:
    return tuple(_ordered_params(params).items())


def _ordered_params(
    params: dict[str, float | int],
) -> dict[str, float | int]:
    ordered = {
        parameter: params[parameter]
        for parameter in GENERATION_PARAM_NAMES
        if parameter in params
    }
    for parameter in sorted(set(params) - set(GENERATION_PARAM_NAMES)):
        ordered[parameter] = params[parameter]
    return ordered


def _strict_int(value: Any, path: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{path} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path} must be an integer") from exc
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{path} must be an integer")
    if isinstance(value, str) and str(parsed) != value.strip():
        raise ValueError(f"{path} must be an integer")
    return parsed


def _optional_int(value: Any, path: str) -> Optional[int]:
    if value is None:
        return None
    return _strict_int(value, path)


def _optional_non_negative_int(value: Any, path: str) -> Optional[int]:
    if value is None:
        return None
    return _non_negative_int(value, path)


def _non_negative_int(value: Any, path: str) -> int:
    parsed = _strict_int(value, path)
    if parsed < 0:
        raise ValueError(f"{path} must be non-negative")
    return parsed


def _log_warnings(warnings: tuple[str, ...]) -> None:
    for warning in warnings:
        logger.warning(warning)
