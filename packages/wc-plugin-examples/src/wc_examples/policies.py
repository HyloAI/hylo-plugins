"""Policy plugins (group "policies"): one written from scratch, one composed from other registered policies.

- examples.ik_reacher: a scripted chunk policy for MuJoCo planar reach, with seeded exploration noise.
- examples.mixture: several policies behind one name. mode "sample": each seed picks one member (with `weights`),
  so steering's sample-and-select chooses among the members' proposals; mode "mean": the members' chunks
  averaged (an ensemble).

    agent: {type: steered, params: {sampler: {type: examples.mixture, params: {members: [...]}},
                                    steering: {n_samples: 4, selector: medoid}}}
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from worldcurrent.core.agent import AgentSpec, EpisodeContext
from worldcurrent.core.policy import build_policy
from worldcurrent.core.types import ActionSemantics, Observation


class IKReacherPolicy:
    """For mujoco.planar_reach: analytic two-link IK to the target, a chunk of joint steps toward it, plus seeded
    exploration noise (so steering has different proposals to choose from)."""

    def __init__(self, gain: float = 0.3, horizon: int = 8, noise: float = 0.1, l1: float = 0.3, l2: float = 0.25,
                 max_step: float = 0.05) -> None:
        self.gain, self.horizon, self.noise, self.l1, self.l2, self.max_step = gain, horizon, noise, l1, l2, max_step

    def spec(self) -> AgentSpec:
        return AgentSpec(name="examples.ik_reacher", version="0.1", required_state=("qpos", "target_pos"),
                         action_semantics=ActionSemantics(control_mode="joint_delta", frame="base"), action_dim=2,
                         components={"policy": f"ik_reacher(gain={self.gain})@0.1"})

    def reset(self, ctx: EpisodeContext) -> None:
        pass

    def default_execution_steps(self) -> int:
        return max(1, self.horizon // 2)

    def _ik(self, x: float, y: float) -> np.ndarray:
        c2 = np.clip((x * x + y * y - self.l1 ** 2 - self.l2 ** 2) / (2 * self.l1 * self.l2), -1.0, 1.0)
        q2 = np.arccos(c2)
        return np.array([np.arctan2(y, x) - np.arctan2(self.l2 * np.sin(q2), self.l1 + self.l2 * np.cos(q2)), q2])

    def sample(self, observation: Observation, seeds: list[int] | None, noise_scale: float = 1.0) -> np.ndarray:
        s = observation.state
        err = (self._ik(*np.asarray(s["target_pos"], float)[:2]) - np.asarray(s["qpos"], float) + np.pi) % (2 * np.pi) - np.pi
        a = np.clip(self.gain * err / self.max_step, -1.0, 1.0)
        rngs = [np.random.default_rng(x) for x in seeds] if seeds else [np.random.default_rng()]
        return np.stack([np.clip(a + self.noise * noise_scale * r.standard_normal((self.horizon, 2)), -1.0, 1.0)
                         for r in rngs])


class MixturePolicy:
    def __init__(self, members: list[dict[str, Any]], weights: list[float] | None = None,
                 mode: Literal["sample", "mean"] = "sample") -> None:
        self.members = [build_policy(m) for m in members]
        if len({m.spec().action_dim for m in self.members}) != 1:
            raise ValueError("all members must have the same action dimension")
        w = np.asarray(weights or [1.0] * len(self.members), dtype=float)
        self.weights, self.mode = w / w.sum(), mode

    def spec(self) -> AgentSpec:
        specs = [m.spec() for m in self.members]
        return specs[0].model_copy(update={
            "name": "examples.mixture",
            "required_state": tuple(sorted({k for s in specs for k in s.required_state})),
            "components": {"policy": f"mixture[{self.mode}](" + ", ".join(s.components.get("policy", s.name) for s in specs) + ")"}})

    def reset(self, ctx: EpisodeContext) -> None:
        for m in self.members:
            m.reset(ctx)

    def default_execution_steps(self) -> int:
        return min(m.default_execution_steps() for m in self.members)

    def sample(self, observation: Observation, seeds: list[int] | None, noise_scale: float = 1.0) -> np.ndarray:
        chunks = [np.asarray(m.sample(observation, seeds, noise_scale)) for m in self.members]   # each (N, T_i, A)
        T = min(c.shape[1] for c in chunks)
        chunks = np.stack([c[:, :T] for c in chunks])                                           # (M, N, T, A)
        if self.mode == "mean":
            return np.tensordot(self.weights, chunks, axes=1)
        picks = [np.random.default_rng(s).choice(len(self.members), p=self.weights) for s in (seeds or [None])]
        return np.stack([chunks[k, i] for i, k in enumerate(picks)])
