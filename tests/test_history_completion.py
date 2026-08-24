from datetime import datetime
from pathlib import Path

import pandas as pd

from scripts import download_missing_histories as mod


def test_partial_history_is_not_marked_complete(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "DATA_DIR", tmp_path)
    monkeypatch.setattr(mod, "_expected_history_cutoff", lambda: datetime(2011, 1, 1))

    sparse_df = pd.DataFrame(
        {"time": pd.date_range("2011-01-01", periods=1000, freq="h")},
        index=pd.date_range("2011-01-01", periods=1000, freq="h"),
    )
    sparse_path = tmp_path / "TEST_1h_history.parquet"
    sparse_df.to_parquet(sparse_path)

    assert mod._history_file_complete("TEST", "1h") is False


def test_large_sufficient_history_is_marked_complete(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "DATA_DIR", tmp_path)
    monkeypatch.setattr(mod, "_expected_history_cutoff", lambda: datetime(2011, 1, 1))

    full_df = pd.DataFrame(
        {"time": pd.date_range("2011-01-01", periods=100000, freq="h")},
        index=pd.date_range("2011-01-01", periods=100000, freq="h"),
    )
    full_path = tmp_path / "TEST_1h_history.parquet"
    full_df.to_parquet(full_path)

    assert mod._history_file_complete("TEST", "1h") is True
