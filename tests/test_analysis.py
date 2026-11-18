"""Тести аналізу результатів і перевірки коректності (C1)."""
import math

import pytest

from satimg.analysis import correlation, markdown_table, run_analysis, stats, summary_table
from satimg.config import apply_overrides, load_config
from satimg.verify import check_p1_synthetic, check_p2_lines


def test_stats_by_hand():
    st = stats([1.0, 2.0, 3.0, float("inf")])  # inf не враховується
    assert st["n"] == 3 and st["mean"] == 2.0 and st["median"] == 2.0
    assert st["std"] == pytest.approx(1.0)                       # вибіркове std
    assert st["ci95"] == pytest.approx(1.96 / math.sqrt(3))


def test_summary_and_markdown():
    rows = [{"noise": "none", "m": 1.0}, {"noise": "none", "m": 3.0},
            {"noise": "gaussian_20", "m": 5.0}]
    table = summary_table(rows, [("m", "Метрика")])
    assert [r["noise"] for r in table] == ["none", "gaussian_20"]
    assert table[0]["m_mean"] == 2.0
    md = markdown_table(table, [("m", "Метрика")])
    assert "без шуму" in md and "2.000" in md


def test_correlation_by_hand():
    rows = [{"a": float(i), "b": 2.0 * i} for i in range(5)]
    assert correlation(rows, "a", "b") == pytest.approx(1.0)


def test_run_analysis_without_results(tmp_path):
    assert run_analysis(tmp_path, log=lambda *_: None) is False


def test_verification_scenarios_pass_on_reference_images():
    cfg = load_config("config.yaml")
    rows = []
    check_p1_synthetic(cfg, 0.85, rows)
    check_p2_lines(cfg, cfg["verify"]["thresholds"], rows)
    assert len(rows) == 12
    assert sum(r["passed"] for r in rows) >= 10


def test_run_analysis_end_to_end(tmp_path):
    from satimg.runner import run_experiment  # pylint: disable=import-outside-toplevel
    cfg = apply_overrides(load_config("config.yaml"), ["output.save_stages=false"])
    run_experiment(cfg, ["p1", "p2"], output_dir=tmp_path, noise_names=["none", "gaussian_20"],
                   per_class=1, limit=2, log=lambda *_: None)
    assert run_analysis(tmp_path, log=lambda *_: None)
    out = tmp_path / "analysis"
    assert (out / "summary.md").exists() and (out / "pipelines_stability.png").exists()
    assert (out / "pipelines_comparison.csv").exists()
