from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd


def summarize_across_seeds(series_by_seed: Dict[int, Any], label: str) -> dict:
    """
    Generic cross-seed summarizer.
    If values are pandas Series (any length): compute per-position median & IQR
    using positional alignment (0..N-1). This avoids issues with duplicate or
    non-matching indexes across seeds.
    If values are scalars: compute median & IQR.
    """
    if not series_by_seed:
        return {}
    any_val = next(iter(series_by_seed.values()))
    if hasattr(any_val, "index") and hasattr(any_val, "values"):
        # Positional alignment: rows are iteration positions 0..max_len-1, columns are seeds
        max_len = max(len(v) for v in series_by_seed.values())
        data = {}
        for seed, series in series_by_seed.items():
            vals = pd.Series(series).astype(float).values
            if len(vals) < max_len:
                vals = np.pad(
                    vals.astype(float), (0, max_len - len(vals)), constant_values=np.nan
                )
            data[seed] = vals
        df = pd.DataFrame(data)  # index: 0..max_len-1, columns: seeds
        med = df.median(axis=1, numeric_only=True)
        q25 = df.quantile(0.25, axis=1, numeric_only=True)
        q75 = df.quantile(0.75, axis=1, numeric_only=True)
        return {
            "type": "series",
            "median": {int(k): float(v) for k, v in med.to_dict().items()},
            "q25": {int(k): float(v) for k, v in q25.to_dict().items()},
            "q75": {int(k): float(v) for k, v in q75.to_dict().items()},
        }
    else:
        arr = np.array(
            [float(v) for v in series_by_seed.values() if v is not None], dtype=float
        )
        if arr.size == 0:
            return {}
        return {
            "type": "scalar",
            "median": float(np.nanmedian(arr)),
            "q25": float(np.nanpercentile(arr, 25)),
            "q75": float(np.nanpercentile(arr, 75)),
        }


def write_json_atomic(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    tmp.replace(path)


def write_csv_atomic(path: Path, df_or_series) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(df_or_series, "to_csv"):
        df_or_series.to_csv(tmp)
    else:
        pd.DataFrame(df_or_series).to_csv(tmp)
    tmp.replace(path)
