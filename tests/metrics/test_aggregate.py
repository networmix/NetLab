from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from metrics.aggregate import (
    summarize_across_seeds,
    write_csv_atomic,
    write_json_atomic,
)


def test_summarize_across_seeds_scalars() -> None:
    series_by_seed = {1: 1.0, 2: 3.0, 3: 2.0}
    out = summarize_across_seeds(series_by_seed, label="x")
    assert out["type"] == "scalar"
    assert out["median"] == 2.0
    assert out["q25"] == 1.5
    assert out["q75"] == 2.5


def test_summarize_across_seeds_series_positional_alignment() -> None:
    # Different lengths; later positions padded with NaN should be ignored in quantiles
    series_by_seed = {
        0: pd.Series([1.0, 2.0, 3.0]),
        1: pd.Series([4.0, 6.0]),
        2: pd.Series([2.0, 4.0, 6.0]),
    }
    out = summarize_across_seeds(series_by_seed, label="x")
    assert out["type"] == "series"
    # Position 0: values [1,4,2] → median 2, q25 1.5, q75 3.0
    assert out["median"][0] == 2.0
    assert out["q25"][0] == 1.5
    assert out["q75"][0] == 3.0
    # Position 1: values [2,6,4] → median 4, q25 3.0, q75 5.0
    assert out["median"][1] == 4.0
    assert out["q25"][1] == 3.0
    assert out["q75"][1] == 5.0
    # Position 2: values [3,NaN,6] → median 4.5, q25 3.75, q75 5.25 (linear percentile)
    assert out["median"][2] == 4.5
    assert np.isclose(out["q25"][2], 3.75)
    assert np.isclose(out["q75"][2], 5.25)


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
