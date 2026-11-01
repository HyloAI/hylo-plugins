"""The plugins repository's example configs, on MuJoCo planar reach (hylo-bench's hylo-env-mujoco): run when
the bench is in the workspace, skipped otherwise."""

from pathlib import Path

import pytest

pytest.importorskip("hylo.envs.mujoco", reason="the examples run on hylo-bench's MuJoCo environments")

from hylo.core.config import ExperimentConfig  # noqa: E402
from hylo.core.registry import registry  # noqa: E402
from hylo.core.task import TaskInstance  # noqa: E402
from hylo.core.trace import TraceStore  # noqa: E402
from hylo.eval.experiment import run_experiment  # noqa: E402
from hylo.testing.conformance import check_policy  # noqa: E402

EXAMPLES = Path(__file__).resolve().parents[3] / "examples"      # the plugins repository's examples/


def _env_and_task():
    env = registry.build("envs", {"type": "mujoco.planar_reach"})
    bench = registry.build("benchmarks", {"type": "mujoco/reach-v0"})
    return env, bench.instances("validation")[0]


def test_the_policy_examples_pass_the_policy_conformance_check():
    env, task = _env_and_task()
    for spec in ({"type": "examples.ik_reacher"},
                 {"type": "examples.mixture", "params": {"members": [{"type": "examples.ik_reacher"},
                                                                     {"type": "examples.ik_reacher", "params": {"gain": 0.1}}]}}):
        rep = check_policy(registry.build("policies", spec), env, task)
        assert rep.ok, rep.summary()


def test_a_steered_mixture_of_third_party_policies_reaches(tmp_path):
    r = run_experiment(ExperimentConfig.load(EXAMPLES / "configs/mujoco_mixture.yaml"), run_root=tmp_path)
    assert r.metrics["success_mean"] >= 0.9 and "mixture[sample]" in r.log["agent"]["components"]["policy"]


def test_the_slow_zone_slows_the_fingertip_near_the_target(tmp_path):
    cfg = ExperimentConfig.load(EXAMPLES / "configs/mujoco_guarded.yaml")
    guarded = run_experiment(cfg, run_root=tmp_path, label="zone")
    free = run_experiment(cfg.with_overrides({"safety.guards": []}), run_root=tmp_path, label="free")
    assert guarded.metrics["success_mean"] == 1.0 and guarded.metrics["steps_mean"] > free.metrics["steps_mean"]
    ep = next(s["episode_id"] for s in guarded.log["samples"] if s["task_id"].endswith("fixed"))
    reasons = [r for e in TraceStore(guarded.run_dir).read(ep).of_type("safety.review") for r in e["payload"]["reasons"]]
    assert any(r.startswith("slow_zone") for r in reasons)
