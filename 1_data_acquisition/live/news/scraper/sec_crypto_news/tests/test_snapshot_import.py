from copy import deepcopy
from pathlib import Path

import pandas as pd

from src.config import load_config, resolve_config_paths
from src.pipeline import import_verified_snapshot


def test_verified_snapshot_import_writes_csv_and_reports(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "scraper_config.yaml")
    config = deepcopy(config)
    for key, value in config["storage"].items():
        config["storage"][key] = str(tmp_path / Path(value).name)
    resolved = resolve_config_paths(config, root)
    frame = import_verified_snapshot(
        resolved,
        root / "data" / "snapshots" / "verified_bitcoin_press_releases.jsonl",
    )
    assert len(frame) == 16
    assert Path(resolved["storage"]["processed_csv"]).is_file()
    assert Path(resolved["storage"]["quality_report_csv"]).is_file()
    loaded = pd.read_csv(resolved["storage"]["processed_csv"])
    assert loaded["is_bitcoin_related"].all()
    assert loaded["source_row_id"].is_unique
