"""Example online learner: the cross-entropy method over an agent's numeric parameters.

Each iteration samples a population of parameter vectors from a Gaussian, rolls each one out on the development
split (the runner does that), keeps the elite fraction by score and refits the Gaussian to the elites. The
sealed artifact holds the best parameters found; a LearningProposer turns them into a candidate, which the gate
then judges on init states the search never saw.

    proposer: {type: learn, params: {learner: examples.cem, online: true, python: <this venv>,
               params: {space: {kp: {path: params.kp, low: 0.1, high: 6.0}}, population: 6, iterations: 3},
               patch: {agent.params.kp: "{metrics.best.kp}"}}}

The same contract fits on-policy RL: `policies()` returns the current policy, `update()` reads the rollouts'
traces (run_dir) and takes a gradient step, `seal()` writes the weights.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np

from hylo.core.artifact import ArtifactManifest, Provenance, write_manifest


def _set(spec: dict[str, Any], path: str, value: float) -> None:
    node = spec
    *parents, leaf = path.split(".")
    for k in parents:
        node = node.setdefault(k, {})
    node[leaf] = value


class CEMLearner:
    name = "hylo_examples.cem@0.1"

    def start(self, job, agent: dict[str, Any]) -> None:
        p = dict(job.params)
        self.space: dict[str, dict[str, Any]] = p["space"]
        self.names = list(self.space)
        self.low = np.array([self.space[n]["low"] for n in self.names], float)
        self.high = np.array([self.space[n]["high"] for n in self.names], float)
        self.mean = np.array([self.space[n].get("init", (self.space[n]["low"] + self.space[n]["high"]) / 2) for n in self.names], float)
        self.std = (self.high - self.low) / 4
        self.pop, self.elite = int(p.get("population", 6)), max(1, int(round(p.get("elite_frac", 0.34) * int(p.get("population", 6)))))
        self.step_penalty = float(p.get("step_penalty", 1e-3))     # tie-break: fewer steps is better
        self.rng = np.random.default_rng(int(p.get("seed", 0)))
        self.base, self.job = agent, job
        self.best: tuple[float, np.ndarray] | None = None
        self.iter, self._cands = 0, []

    def _spec(self, x: np.ndarray) -> dict[str, Any]:
        spec = copy.deepcopy(self.base)
        for n, v in zip(self.names, x):
            _set(spec, self.space[n]["path"], float(round(v, 4)))
        return spec

    def policies(self) -> list[dict[str, Any]]:
        self._cands = [np.clip(self.rng.normal(self.mean, self.std), self.low, self.high) for _ in range(self.pop)]
        return [self._spec(x) for x in self._cands]

    def update(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        scores = []
        for r in results:
            steps = np.mean([s["steps"] for s in r["samples"]]) if r["samples"] else 0.0
            scores.append(float(r["metrics"]["success_mean"]) - self.step_penalty * steps)
        order = np.argsort(scores)[::-1]
        elites = np.stack([self._cands[i] for i in order[: self.elite]])
        self.mean, self.std = elites.mean(0), np.maximum(elites.std(0), (self.high - self.low) * 0.02)
        top = order[0]
        if self.best is None or scores[top] > self.best[0]:
            self.best = (scores[top], self._cands[top])
        self.iter += 1
        return {"best_score": round(self.best[0], 4), "iteration_best": round(scores[top], 4),
                "mean": dict(zip(self.names, np.round(self.mean, 3).tolist()))}

    def finished(self) -> bool:
        return bool(np.all(self.std <= (self.high - self.low) * 0.021))

    def seal(self, out: str, sources: list[Any]) -> ArtifactManifest:
        score, x = self.best
        best = dict(zip(self.names, np.round(x, 4).tolist()))
        Path(out).mkdir(parents=True, exist_ok=True)
        (Path(out) / "best.json").write_text(json.dumps({"params": best, "score": score, "agent": self._spec(x)}, indent=2))
        return write_manifest(out, kind="policy_params", role="behaviour", metrics={"best": best, "score": round(score, 4),
                              "iterations": self.iter},
                              provenance=Provenance(builder=self.name, sources=sources,
                                                    params={k: v for k, v in self.job.params.items() if k != "experiment"}))
