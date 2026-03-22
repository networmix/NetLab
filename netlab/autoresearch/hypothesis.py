"""Hypothesis management for autoresearch.

Provides:
- ParamDef: dataclass defining a single parameter's type, range, and default.
- HypothesisTemplate: parses a hypothesis_template.yml, exposes param definitions.
- Hypothesis: holds param values, validates against template, computes deterministic hash.
- HypothesisMerger: substitutes ${{param}} placeholders in a base scenario YAML text.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional

import yaml


@dataclass
class ParamDef:
    name: str
    type: Literal["int", "float", "enum"]
    range: Optional[tuple[float, float]] = None  # for int/float
    step: Optional[float] = None  # for int/float
    values: Optional[list[str]] = None  # for enum
    default: Any = None
    description: str = ""


class HypothesisTemplate:
    """Parses a hypothesis_template.yml and exposes parameter definitions."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._params: dict[str, ParamDef] = {}
        self._parse(path)

    def _parse(self, path: Path) -> None:
        with open(path) as f:
            data = yaml.safe_load(f)

        params_data = data.get("params") or data.get("parameters") or {}
        for name, spec in params_data.items():
            ptype = spec["type"]
            prange = None
            step = None
            values = None

            if ptype in ("int", "float"):
                raw_range = spec.get("range")
                if raw_range is not None:
                    prange = (float(raw_range[0]), float(raw_range[1]))
                step = spec.get("step")
                if step is not None:
                    step = float(step)

            if ptype == "enum":
                values = list(spec["values"])

            default = spec.get("default")
            description = spec.get("description", "")

            self._params[name] = ParamDef(
                name=name,
                type=ptype,
                range=prange,
                step=step,
                values=values,
                default=default,
                description=description,
            )

    @property
    def params(self) -> dict[str, ParamDef]:
        return dict(self._params)

    def validate_hypothesis(self, params: dict[str, Any]) -> list[str]:
        """Returns list of error messages. Empty = valid."""
        errors: list[str] = []

        # Check for unknown params
        for name in params:
            if name not in self._params:
                errors.append(f"Unrecognized parameter: {name}")

        # Check for missing params (no default)
        for name, _pdef in self._params.items():
            if name not in params:
                errors.append(f"Missing required parameter: {name}")

        # Validate types and ranges for known params
        for name, value in params.items():
            if name not in self._params:
                continue
            pdef = self._params[name]

            if pdef.type == "int":
                if not isinstance(value, int) or isinstance(value, bool):
                    errors.append(
                        f"Parameter {name}: expected type int, got {type(value).__name__}"
                    )
                elif pdef.range is not None:
                    lo, hi = pdef.range
                    if value < lo or value > hi:
                        errors.append(
                            f"Parameter {name}: value {value} out of range [{lo}, {hi}]"
                        )

            elif pdef.type == "float":
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    errors.append(
                        f"Parameter {name}: expected type float, got {type(value).__name__}"
                    )
                elif pdef.range is not None:
                    lo, hi = pdef.range
                    if value < lo or value > hi:
                        errors.append(
                            f"Parameter {name}: value {value} out of range [{lo}, {hi}]"
                        )

            elif pdef.type == "enum":
                if pdef.values is not None and str(value) not in pdef.values:
                    errors.append(
                        f"Parameter {name}: value {value!r} not in allowed values {pdef.values}"
                    )

        return errors


class Hypothesis:
    """Holds parameter values, validates against template, computes deterministic hash."""

    def __init__(self, params: dict[str, Any], template: HypothesisTemplate) -> None:
        self._params = dict(params)
        self._template = template

    @property
    def params(self) -> dict[str, Any]:
        return dict(self._params)

    @property
    def params_hash(self) -> str:
        """Deterministic hash of normalized params. Same params in any order -> same hash."""
        # Sort keys for deterministic ordering
        normalized = json.dumps(self._params, sort_keys=True, default=str)
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def validate(self) -> list[str]:
        """Returns list of error messages. Empty = valid."""
        return self._template.validate_hypothesis(self._params)


# Regex matching ${{...}} placeholders (double curly braces with dollar sign).
# Does NOT match ${single_brace}.
_PLACEHOLDER_RE = re.compile(r"\$\{\{(\w+)\}\}")


class HypothesisMerger:
    """Template mode only. Not used in programmatic mode.

    Loads a base scenario as text, substitutes ${{param}} placeholders with
    hypothesis param values, and returns the parsed YAML dict.
    """

    def __init__(self, base_scenario_text: str, template: HypothesisTemplate) -> None:
        self._base_text = base_scenario_text
        self._template = template

    def validate_placeholders(self) -> list[str]:
        """Cross-check ${{...}} tokens against template params. Returns errors."""
        errors: list[str] = []
        placeholders_in_text = set(_PLACEHOLDER_RE.findall(self._base_text))
        template_params = set(self._template.params.keys())

        # Placeholders in text that are not in template
        for ph in sorted(placeholders_in_text - template_params):
            errors.append(
                f"Placeholder ${{{{{ph}}}}} in scenario not found in template params"
            )

        # Template params not referenced in text
        for p in sorted(template_params - placeholders_in_text):
            errors.append(
                f"Template parameter {p} has no corresponding placeholder in scenario"
            )

        return errors

    def merge(self, hypothesis: Hypothesis) -> dict:
        """Substitute params, parse YAML, return scenario dict.

        Raises ValueError on unreplaced tokens.
        """
        params = hypothesis.params
        text = self._base_text

        def _replacer(match: re.Match) -> str:
            name = match.group(1)
            if name not in params:
                raise ValueError(
                    f"Unreplaced placeholder: ${{{{{name}}}}} — "
                    f"parameter not found in hypothesis"
                )
            value = params[name]
            # Return the raw value as a string for YAML substitution.
            return str(value)

        text = _PLACEHOLDER_RE.sub(_replacer, text)

        # Post-substitution scan: reject if any ${{ remains
        remaining = _PLACEHOLDER_RE.findall(text)
        if remaining:
            raise ValueError(f"Unreplaced placeholders after merge: {remaining}")

        return yaml.safe_load(text)
