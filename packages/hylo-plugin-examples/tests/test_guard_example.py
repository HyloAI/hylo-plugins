import numpy as np

from hylo.eval.experiment import run_experiment
from hylo.core.config import ExperimentConfig
from hylo.core.trace import TraceStore

# Unit tests run on the test double (hylo.testing); the example configs (MuJoCo) are run by test_examples.py.
DOUBLE = ExperimentConfig(name="double", benchmark={"type": "testing/reach-v0", "params": {"tags": ["reach"]}},
                          env={"type": "testing.point_reach"}, agent={"type": "testing.p_controller", "params": {"kp": 2.0}},
                          safety={"max_delta": 0.25})

ZONE = {"type": "examples.slow_zone", "params": {"state_key": "pos", "center": [0.5, 0.5], "radius": 0.25, "factor": 0.25}}


def _cfg(guards):
    return DOUBLE.with_overrides({"safety.max_delta": 0.5, "safety.guards": guards})


def test_the_slow_zone_plugin_slows_the_robot_near_the_goal(tmp_path):
    fast = run_experiment(_cfg([]), run_root=tmp_path, label="free")
    slow = run_experiment(_cfg([ZONE]), run_root=tmp_path, label="zone")
    both = {s["key"]: s["steps"] for s in fast.log["samples"] if s["status"] == "success"}
    paired = [(both[s["key"]], s["steps"]) for s in slow.log["samples"] if s["status"] == "success" and s["key"] in both]
    assert paired and all(b >= a for a, b in paired) and any(b > a for a, b in paired)   # same task: never faster
    ep = slow.log["samples"][0]["episode_id"]
    reasons = [r for e in TraceStore(slow.run_dir).read(ep).of_type("safety.review") for r in e["payload"]["reasons"]]
    assert any(r.startswith("slow_zone") for r in reasons)
