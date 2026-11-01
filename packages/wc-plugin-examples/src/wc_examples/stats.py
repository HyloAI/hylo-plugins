"""Example gate-test plugin: statistical and practical significance.

McNemar alone can promote a candidate whose advantage is real but tiny (many pairs, a few more wins). This test
also requires the net win rate (wins - losses) / pairs to reach `min_effect`; below it the p-value is reported
as 1, so the gate does not promote.

    protocol: {gate_test: {type: examples.min_effect, params: {min_effect: 0.1}}}
"""

from __future__ import annotations

from worldcurrent.eval.scoring import PairedComparison, paired_compare


class MinEffectMcNemar:
    name = "min_effect"

    def __init__(self, min_effect: float = 0.1, metric: str = "success") -> None:
        self.min_effect, self.metric = float(min_effect), metric

    def compare(self, candidate, parent) -> PairedComparison:
        c = paired_compare(candidate, parent)
        net = (c.candidate_wins - c.parent_wins) / max(1, c.n_pairs)
        return c if net >= self.min_effect else c.model_copy(update={"p_value": 1.0})
