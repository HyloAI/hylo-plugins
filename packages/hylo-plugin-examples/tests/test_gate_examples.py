from hylo.eval.experiment import run_experiment
from hylo.core.config import ExperimentConfig
from hylo.evolve.candidate import Candidate, CandidateRegistry
from hylo.evolve.protocol import Archive, ProtocolConfig, run_cycle
from hylo.evolve.steering import diagnose_traces
from hylo_examples.stats import MinEffectMcNemar

# Unit tests run on the test double (hylo.testing); the example configs (MuJoCo) are run by test_examples.py.
DOUBLE = ExperimentConfig(name="double", benchmark={"type": "testing/reach-v0", "params": {"tags": ["reach"]}},
                          env={"type": "testing.point_reach"}, agent={"type": "testing.p_controller", "params": {"kp": 2.0}},
                          safety={"max_delta": 0.25})

CFG = DOUBLE.with_overrides(
    {"benchmark.params": {"tags": ["reach"], "protocol": "rsi"}, "agent.params.kp": 2.0})


class Faster:
    name = "test.faster"

    def propose(self, diagnosis, cfg, parent_id, archive):
        return [(Candidate(patch={"agent.params.kp": 4.0}, rationale="", parent_id=parent_id, generator=self.name),
                 "a stiffer gain reaches the goal in fewer steps")]


def _cycle(tmp_path, gate_test):
    ev = lambda c, split, label: run_experiment(c, split=split, run_root=tmp_path / "runs", label=label)  # noqa: E731
    return run_cycle(CFG, CandidateRegistry(tmp_path / "reg"), Archive(tmp_path / "a.jsonl"), Faster(), ev, diagnose_traces,
                     protocol=ProtocolConfig(gate_test=gate_test))


def test_a_steps_gate_can_promote_what_success_alone_cannot_tell_apart(tmp_path):
    by_success = _cycle(tmp_path / "s", {"type": "mcnemar"})
    by_steps = _cycle(tmp_path / "t", {"type": "sign", "params": {"metric": "steps", "direction": "lower"}})
    assert by_success.promoted is None                            # both always succeed: no discordant pairs
    assert by_steps.promoted is not None and by_steps.entries[0].stages[-1].wins > 0


def test_min_effect_refuses_a_tiny_but_significant_advantage():
    parent = {f"k{i}": 0.0 for i in range(400)}
    cand = {**parent, **{f"k{i}": 1.0 for i in range(12)}}       # 12 wins, 0 losses of 400: p ~ 2e-4, net 3%
    assert MinEffectMcNemar(min_effect=0.0).compare(cand, parent).p_value < 0.001
    assert MinEffectMcNemar(min_effect=0.1).compare(cand, parent).p_value == 1.0
