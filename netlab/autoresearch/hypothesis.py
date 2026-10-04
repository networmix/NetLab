"""Validate research parameters and substitute them into scenario templates."""

from __future__ import annotations

import hashlib
import json
import math
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
    values: Optional[list[str | int | float]] = None  # for enum
    default: Any = None
    description: str = ""


class HypothesisTemplate:
    """Load parameter types, ranges, and defaults from hypothesis_template.yml."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._params: dict[str, ParamDef] = {}
        self._parse(path)

    def _parse(self, path: Path) -> None:
        with open(path) as f:
            data = yaml.safe_load(f)

        params_data = data.get("params") or {}
        for name, spec in params_data.items():
            ptype = spec["type"]
            if ptype not in {"int", "float", "enum"}:
                raise ValueError(f"Parameter {name}: unsupported type {ptype!r}")
            prange = None
            step = None
            values = None

            if ptype in ("int", "float"):
                raw_range = spec.get("range")
                if raw_range is not None:
                    if len(raw_range) != 2:
                        raise ValueError(f"Parameter {name}: range needs two bounds")
                    prange = (float(raw_range[0]), float(raw_range[1]))
                    if (
                        not all(math.isfinite(v) for v in prange)
                        or prange[0] > prange[1]
                    ):
                        raise ValueError(f"Parameter {name}: invalid range")
                step = spec.get("step")
                if step is not None:
                    step = float(step)
                    if not math.isfinite(step) or step <= 0:
                        raise ValueError(
                            f"Parameter {name}: step must be finite and positive"
                        )

            if ptype == "enum":
                values = list(spec["values"])
                if not values or any(
                    type(v) not in {str, int, float}
                    or (isinstance(v, str) and "\n" in v)
                    or (isinstance(v, float) and not math.isfinite(v))
                    for v in values
                ):
                    raise ValueError(
                        f"Parameter {name}: enum values must be finite numbers or single-line strings"
                    )

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

        for name in params:
            if name not in self._params:
                errors.append(f"Unrecognized parameter: {name}")

        for name, _pdef in self._params.items():
            if name not in params:
                errors.append(f"Missing required parameter: {name}")

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
                elif not math.isfinite(value):
                    errors.append(f"Parameter {name}: value must be finite")
                elif pdef.range is not None:
                    lo, hi = pdef.range
                    if value < lo or value > hi:
                        errors.append(
                            f"Parameter {name}: value {value} out of range [{lo}, {hi}]"
                        )

            elif pdef.type == "enum":
                if not any(
                    type(value) is type(v) and value == v for v in (pdef.values or [])
                ):
                    errors.append(
                        f"Parameter {name}: value {value!r} not in allowed values {pdef.values}"
                    )

        return errors


class Hypothesis:
    """Parameter values with template validation and a stable hash."""

    def __init__(self, params: dict[str, Any], template: HypothesisTemplate) -> None:
        self._params = dict(params)
        self._template = template

    @property
    def params(self) -> dict[str, Any]:
        return dict(self._params)

    @property
    def params_hash(self) -> str:
        """Hash parameter values independently of dictionary key order."""
        normalized = json.dumps(self._params, sort_keys=True, allow_nan=False)
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def validate(self) -> list[str]:
        """Returns list of error messages. Empty = valid."""
        return self._template.validate_hypothesis(self._params)


# Regex matching ${{...}} placeholders (double curly braces with dollar sign).
# Does NOT match ${single_brace}.
_PLACEHOLDER_RE = re.compile(r"\$\{\{(\w+)\}\}")


class HypothesisMerger:
    """Substitute ``${{param}}`` values in YAML for template-mode research."""

    def __init__(self, base_scenario_text: str, template: HypothesisTemplate) -> None:
        self._base_text = base_scenario_text
        self._template = template

    def validate_placeholders(self) -> list[str]:
        """Cross-check ${{...}} tokens against template params. Returns errors."""
        errors: list[str] = []
        placeholders_in_text = set(_PLACEHOLDER_RE.findall(self._base_text))
        template_params = set(self._template.params.keys())

        for ph in sorted(placeholders_in_text - template_params):
            errors.append(
                f"Placeholder ${{{{{ph}}}}} in scenario not found in template params"
            )

        for p in sorted(template_params - placeholders_in_text):
            errors.append(
                f"Template parameter {p} has no corresponding placeholder in scenario"
            )

        return errors

    def merge(self, hypothesis: Hypothesis) -> dict:
        """Substitute parameters and parse YAML; reject unreplaced placeholders."""
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
            return str(value)

        text = _PLACEHOLDER_RE.sub(_replacer, text)

        remaining = _PLACEHOLDER_RE.findall(text)
        if remaining:
            raise ValueError(f"Unreplaced placeholders after merge: {remaining}")

        return yaml.safe_load(text)


def validate_template_workflow(text: str) -> None:
    """Require the current list-form workflow and its capacity-search step."""
    data = yaml.safe_load(text)
    workflow = data.get("workflow") if isinstance(data, dict) else None
    if not isinstance(workflow, list) or not any(
        isinstance(step, dict) and step.get("type") == "MaximumSupportedDemand"
        for step in workflow
    ):
        raise ValueError(
            "Base scenario must have a MaximumSupportedDemand workflow step in a workflow list"
        )
