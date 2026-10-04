from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from netlab.artifacts import write_csv_atomic, write_json_atomic
from netlab.metrics.aggregate import (
    summarize_across_seeds,
)


def test_summarize_across_seeds_scalars() -> None:
    series_by_seed = {1: 1.0, 2: 3.0, 3: 2.0}
    out = summarize_across_seeds(series_by_seed)
    assert out["type"] == "scalar"
    assert out["median"] == 2.0
    assert out["q25"] == 1.5
    assert out["q75"] == 2.5


def test_summarize_across_seeds_ignores_unavailable_scalars() -> None:
    assert summarize_across_seeds({1: np.nan, 2: np.inf}) == {}
    assert summarize_across_seeds({1: np.nan, 2: 2.0})["median"] == 2.0


def test_write_json_atomic(tmp_path: Path) -> None:
    path = tmp_path / "data.json"
    data = {"a": 1, "b": [1, 2, 3]}
    write_json_atomic(path, data)
    assert path.exists()
    with path.open("r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded == data


def test_write_csv_atomic_roundtrip_df(tmp_path: Path) -> None:
    path = tmp_path / "tab.csv"
    df = pd.DataFrame({"a": [1, 2], "b": [3.0, 4.0]})
    write_csv_atomic(path, df)
    assert path.exists()
    df2 = pd.read_csv(path, index_col=0)
    pd.testing.assert_frame_equal(df2, df)
