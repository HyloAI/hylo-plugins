import json
import sys
from hylo.eval.experiment import run_experiment
from hylo.core.artifact import verify
from hylo.core.config import ExperimentConfig
from hylo.core.learning import TrainJob
from hylo.evolve.candidate import CandidateRegistry
from hylo.evolve.learning import LearningProposer
from hylo.evolve.protocol import Archive, run_cycle
from hylo.evolve.steering import diagnose_traces
from hylo.learn.jobs import run_job

# Unit tests run on the test double (hylo.testing); the example configs (MuJoCo) are run by test_examples.py.
DOUBLE = ExperimentConfig(name="double", benchmark={"type": "testing/reach-v0", "params": {"tags": ["reach"]}},
                          env={"type": "testing.point_reach"}, agent={"type": "testing.p_controller", "params": {"kp": 2.0}},
                          safety={"max_delta": 0.25})

CFG = DOUBLE.with_overrides(
    {"benchmark.params": {"tags": ["reach"], "protocol": "rsi"}, "agent.params.kp": 0.3})
SPACE = {"kp": {"path": "params.kp", "low": 0.1, "high": 6.0, "init": 0.5}}


def test_cem_learns_by_acting_on_the_search_split_and_seals_its_rollouts(tmp_path):
    job = TrainJob(learner="examples.cem", inputs=[], out=str(tmp_path / "cem"), splits=("search",),
                   params={"experiment": CFG.model_dump(mode="json"), "iterations": 3, "space": SPACE, "population": 5})
    m = run_job(job)
    assert m.kind == "policy_params" and m.provenance.splits() == {"search"} and len(m.provenance.sources) >= 5
    assert verify(tmp_path / "cem").digest == m.digest
    base = run_experiment(CFG, split="search", run_root=tmp_path / "x", label="base").metrics["success_mean"]
    tuned = run_experiment(CFG.with_overrides({"agent.params.kp": m.metrics["best"]["kp"]}), split="search",
                           run_root=tmp_path / "x", label="tuned").metrics["success_mean"]
    assert tuned > base + 0.3
    assert json.loads((tmp_path / "cem" / "history.json").read_text())[0]["iteration"] == 0


def test_an_online_learner_becomes_a_gated_candidate(tmp_path):
    prop = LearningProposer(learner="examples.cem", online=True, python=sys.executable, out_root=str(tmp_path / "learned"),
                            params={"space": SPACE, "population": 4, "iterations": 2},
                            patch={"agent.params.kp": "{metrics.best.kp}"}, hypothesis="CEM gain {metrics.best.kp}")
    ev = lambda c, split, label: run_experiment(c, split=split, run_root=tmp_path / "runs", label=label)  # noqa: E731
    rep = run_cycle(CFG, CandidateRegistry(tmp_path / "reg"), Archive(tmp_path / "a.jsonl"), prop, ev, diagnose_traces)
    [e] = rep.entries
    assert e.candidate.evidence["artifact"]["kind"] == "policy_params" and e.candidate.evidence["artifact"]["splits"] == ["search"]
    assert rep.promoted == e.candidate.id                        # judged on gate init states the search never saw
