"""Smoke test for examples/01_plugin_effects.py with --quick (about 40 s): each plugin's measured effect holds."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("hylo.envs.mujoco", reason="the examples run on hylo-bench's MuJoCo environments")
pytest.importorskip("hylo.bench", reason="the effects are scored under hylo-bench's mujoco-reach protocol")
pytest.importorskip("hylo.evolve", reason="the online learner runs inside a campaign (hylo-evolve)")
pytest.importorskip("matplotlib", reason="the example renders a figure")

SCRIPT = Path(__file__).resolve().parents[3] / "examples" / "01_plugin_effects.py"


def test_plugin_effects_quick(tmp_path):
    proc = subprocess.run([sys.executable, str(SCRIPT), "--quick", "--out", str(tmp_path)], capture_output=True,
                          text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr[-3000:]
    r = json.loads((tmp_path / "result.json").read_text())
    g = r["guard"]
    assert g["norm_limit + slow_zone"]["zone_speed_mean_m_s"] < 0.5 * g["no guards"]["zone_speed_mean_m_s"]
    assert "slow_zone" in g["norm_limit + slow_zone"]["interventions_per_episode_by_guard"]
    d = r["gate_test"]["direct"]
    assert d["mcnemar"]["passes_alpha_0.05"] and not d["examples.min_effect 0.1"]["passes_alpha_0.05"]
    assert r["online_learner"]["verdict"] == "promoted" and r["online_learner"]["rollouts"]
    m = r["mixture"]
    assert m["mixture steered (config)"]["steps_mean"] < m["member B (gain 0.1, noise 0.3)"]["steps_mean"]
