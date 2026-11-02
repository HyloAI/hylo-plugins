"""What each example plugin changes, measured with and without it on MuJoCo planar reach.

    uv run python examples/01_plugin_effects.py [--quick] [--out DIR]

1. Guard `examples.slow_zone` (configs/mujoco_guarded.yaml): the same IK controller with no guards, with the
   config's norm_limit only, and with norm_limit + slow_zone. Shield interventions per guard (from the traces'
   safety reviews), steps, and the fingertip speed inside the 10 cm zone around the fixed target.
2. Gate test `examples.min_effect`: two IK gains scored under bench's `mujoco-reach` protocol (100 held-out
   episodes) whose difference is real but small; the built-in McNemar test and min_effect at 0.1 and 0.05 decide.
   Full mode also runs the steps-gate campaign (evolve/mujoco_steps_gate.yaml) with three gate tests.
3. Online learner `examples.cem` (evolve/mujoco_online_cem.yaml): the campaign as shipped; every CEM rollout
   (gain, success and steps on the search split) and the gate's verdict on unseen init states.
4. Policies `examples.ik_reacher` and `examples.mixture` (configs/mujoco_mixture.yaml): the steered mixture
   against each member alone (plain and steered) and the mixture in "mean" mode, under `mujoco-reach`, with
   paired comparisons.

Typical runtime: --quick about 40 s, full about 50 s (128-core CPU). Requires hylo-plugin-examples, hylo-bench
(hylo-env-mujoco), hylo-evolve, matplotlib. Output: runs/examples/plugin-effects/ (result.json, cem.png).
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]          # the plugins repository
EX = REPO / "examples"
ZONE_CENTER, ZONE_RADIUS, CONTROL_HZ = np.array([0.35, 0.2]), 0.1, 20.0


def table(headers, rows) -> str:
    rows = [[str(c) for c in r] for r in rows]
    w = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    line = lambda cells: "| " + " | ".join(c.ljust(n) for c, n in zip(cells, w)) + " |"   # noqa: E731
    return "\n".join([line(headers), "|" + "|".join("-" * (n + 2) for n in w) + "|", *map(line, rows)])


# 1. guard ------------------------------------------------------------------------------------------------------

def guard_stats(result) -> dict:
    from hylo.core.trace import TraceStore

    store, per_guard, speeds, zone_steps = TraceStore(result.run_dir), {}, [], 0
    for s in result.log["samples"]:
        tr = store.read(s["episode_id"])
        for e in tr.of_type("safety.review"):
            for g in {r.split(":")[0].split(" ")[0] for r in e["payload"]["reasons"]}:
                per_guard[g] = per_guard.get(g, 0) + 1
        if s["task_id"].endswith("fixed"):                       # the zone sits on the fixed target
            tips = [np.asarray(o["payload"]["state"]["tip_pos"]) for o in tr.of_type("observation")]
            for a, b in zip(tips, tips[1:]):
                if np.linalg.norm(a - ZONE_CENTER) <= ZONE_RADIUS:
                    speeds.append(float(np.linalg.norm(b - a)) * CONTROL_HZ)
                    zone_steps += 1
    n = len(result.log["samples"])
    n_fixed = sum(1 for s in result.log["samples"] if s["task_id"].endswith("fixed"))
    return {"success": result.metrics["success_mean"], "steps_mean": result.metrics["steps_mean"],
            "interventions_per_episode": result.metrics["safety_interventions_mean"],
            "interventions_per_episode_by_guard": {g: round(c / n, 2) for g, c in sorted(per_guard.items())},
            "zone_speed_mean_m_s": round(float(np.mean(speeds)), 3) if speeds else None,
            "zone_speed_max_m_s": round(float(np.max(speeds)), 3) if speeds else None,
            "zone_steps_per_fixed_episode": round(zone_steps / max(1, n_fixed), 1), "episodes": n}


def guard(out: Path) -> dict:
    from hylo.core.config import ExperimentConfig
    from hylo.eval.experiment import run_experiment

    cfg = ExperimentConfig.load(EX / "configs" / "mujoco_guarded.yaml")
    guards = cfg.safety["guards"]
    variants = {"no guards": [], "norm_limit": guards[:1], "norm_limit + slow_zone": guards}
    res = {k: guard_stats(run_experiment(cfg.with_overrides({"safety.guards": g}), run_root=out / "guard",
                                         label=re.sub(r"\W+", "-", k))) for k, g in variants.items()}
    print("\n1. guard examples.slow_zone (validation split, 20 episodes; zone = 10 cm around the fixed target)")
    print(table(["guards", "success", "steps", "interventions/ep (by guard)", "speed in zone mean / max (m/s)",
                 "steps in zone"],
                [(k, f"{v['success']:.2f}", f"{v['steps_mean']:.1f}",
                  f"{v['interventions_per_episode']:.2f} {v['interventions_per_episode_by_guard'] or ''}",
                  f"{v['zone_speed_mean_m_s']} / {v['zone_speed_max_m_s']}", v["zone_steps_per_fixed_episode"])
                 for k, v in res.items()]))
    return res


# 2. gate test --------------------------------------------------------------------------------------------------

def gate(out: Path, quick: bool) -> dict:
    from hylo.bench.protocol import load_protocol
    from hylo.bench.run import run_protocol
    from hylo.core.registry import registry

    p = load_protocol("mujoco-reach")
    parent, _ = run_protocol(p, {"type": "baseline.planar_ik", "params": {"kp": 0.025}}, "ik-0.025", out=out / "gate",
                             overrides={"workers": 16})
    cand, _ = run_protocol(p, {"type": "baseline.planar_ik", "params": {"kp": 0.0265}}, "ik-0.0265", out=out / "gate",
                           overrides={"workers": 16})
    tests = {"mcnemar": {"type": "mcnemar"},
             "examples.min_effect 0.1": {"type": "examples.min_effect", "params": {"min_effect": 0.1}},
             "examples.min_effect 0.05": {"type": "examples.min_effect", "params": {"min_effect": 0.05}}}
    direct = {}
    for name, spec in tests.items():
        c = registry.build("gate_tests", spec).compare(cand.episodes, parent.episodes)
        direct[name] = {"n_pairs": c.n_pairs, "wins": c.candidate_wins, "losses": c.parent_wins,
                        "net_effect": (c.candidate_wins - c.parent_wins) / c.n_pairs, "p_value": c.p_value,
                        "passes_alpha_0.05": c.p_value < 0.05}
    print(f"\n2. gate test examples.min_effect: IK kp 0.0265 vs 0.025 under mujoco-reach "
          f"({cand.metrics['success_mean']:.2f} vs {parent.metrics['success_mean']:.2f}, 100 paired episodes)")
    print(table(["gate test", "wins–losses", "net effect", "p", "passes at 0.05"],
                [(k, f"{v['wins']}–{v['losses']}", f"{v['net_effect']:.2f}", f"{v['p_value']:.4g}",
                  "yes" if v["passes_alpha_0.05"] else "no") for k, v in direct.items()]))
    campaigns = {}
    if not quick:
        from hylo.evolve.campaign import EvolveConfig, run_campaign

        base = EvolveConfig.load(EX / "evolve" / "mujoco_steps_gate.yaml")
        for name, spec in {"sign on steps (as shipped)": {"type": "sign", "params": {"metric": "steps", "direction": "lower"}},
                           "mcnemar (success)": {"type": "mcnemar"},
                           "examples.min_effect 0.1": {"type": "examples.min_effect", "params": {"min_effect": 0.1}}}.items():
            d = out / "gate" / "campaigns" / re.sub(r"\W+", "-", name)
            shutil.rmtree(d, ignore_errors=True)
            cfg = base.model_copy(update={"out_dir": str(d), "protocol": base.protocol.model_copy(update={"gate_test": spec})})
            run_campaign(cfg, log=lambda _: None)
            rep = json.loads(next(d.glob("cycle-*/cycle_report.json")).read_text())
            e = rep["entries"][0]
            last = e["stages"][-1]
            campaigns[name] = {"candidate": e["candidate"]["patch"], "verdict": e["verdict"], "promoted": rep["promoted"],
                               "gate_wins": last["wins"], "gate_losses": last["losses"], "gate_p": last["p_value"],
                               "success": last["success"], "parent_success": last["parent_success"]}
        print("\n   steps-gate campaign (kp 0.1 -> 0.2, both always succeed), three gate tests:")
        print(table(["gate test", "verdict", "success cand / parent", "wins–losses", "p"],
                    [(k, v["verdict"], f"{v['success']} / {v['parent_success']}", f"{v['gate_wins']}–{v['gate_losses']}",
                      "–" if v["gate_p"] is None else f"{v['gate_p']:.3g}") for k, v in campaigns.items()]))
    return {"direct": direct, "parent_success": parent.metrics["success_mean"],
            "candidate_success": cand.metrics["success_mean"], "campaigns": campaigns}


# 3. online learner ---------------------------------------------------------------------------------------------

def cem(out: Path) -> dict:
    from hylo.evolve.campaign import EvolveConfig, run_campaign

    d = out / "cem"
    shutil.rmtree(d, ignore_errors=True)
    base = EvolveConfig.load(EX / "evolve" / "mujoco_online_cem.yaml")
    params = {**base.proposer.params, "out_root": str(d / "learned")}
    cfg = base.model_copy(update={"out_dir": str(d), "proposer": base.proposer.model_copy(update={"params": params})})
    t = time.perf_counter()
    run_campaign(cfg, log=lambda _: None)
    secs = time.perf_counter() - t
    learned = next((d / "learned").glob("*/history.json")).parent
    rollouts = []
    for it in sorted((learned / "rollouts").glob("iter-*")):
        for r in sorted(it.glob("*/evallog.json")):
            e = json.loads(r.read_text())
            kp = float(re.search(r"kp=([\d.]+)", e["agent"]["components"]["policy"]).group(1))
            rollouts.append({"iteration": int(it.name.split("-")[1]), "kp": kp, "success": e["results"]["success_mean"],
                             "steps_mean": e["results"]["steps_mean"]})
    history = json.loads((learned / "history.json").read_text())
    best = json.loads((learned / "best.json").read_text())
    rep = json.loads(next(d.glob("cycle-*/cycle_report.json")).read_text())
    e = rep["entries"][0]
    gate_stage = next((s for s in e["stages"] if s["stage"] == 3), e["stages"][-1])
    res = {"parent_kp": 0.02, "best_kp": best["params"]["kp"], "iterations": len(history), "rollouts": rollouts,
           "history": history, "verdict": e["verdict"], "gate": gate_stage, "heldout": rep["heldout"],
           "seconds": round(secs, 1)}
    print(f"\n3. online learner examples.cem: {len(rollouts)} rollouts (80 search-split episodes each) in "
          f"{len(history)} iterations, best kp {res['best_kp']} (start 0.05, parent 0.02)")
    print(table(["iteration", "kp", "success (search)", "mean steps"],
                [(r["iteration"], r["kp"], f"{r['success']:.2f}", f"{r['steps_mean']:.1f}") for r in rollouts]))
    h = rep["heldout"] or {}
    print(f"   gate (unseen init states): {gate_stage['success']} vs parent {gate_stage['parent_success']}, "
          f"wins–losses {gate_stage['wins']}–{gate_stage['losses']}, p {gate_stage['p_value']:.2g} -> {e['verdict']}; "
          f"held-out (report only): {h.get('candidate_success')} vs {h.get('parent_success')}; {secs:.0f} s")
    _cem_figure(rollouts, out / "cem.png")
    return res


def _cem_figure(rollouts: list[dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, surface = "#0b0b0b", "#52514e", "#fcfcfb"
    colors = ["#2a78d6", "#eb6834", "#1baf7a"]
    fig, ax = plt.subplots(figsize=(6.0, 2.6), dpi=110)
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)
    for it in sorted({r["iteration"] for r in rollouts}):
        rs = [r for r in rollouts if r["iteration"] == it]
        ax.scatter([r["kp"] for r in rs], [r["steps_mean"] for r in rs], s=40, color=colors[it % 3],
                   edgecolor=surface, linewidth=1.5, label=f"iteration {it}", zorder=3)
    ax.set_xscale("log")
    ax.set_xlabel("controller gain kp (log)", fontsize=8.5, color=muted)
    ax.set_ylabel("mean steps (search split)", fontsize=8.5, color=muted)
    ax.set_title("examples.cem: each rollout is one candidate gain on 80 search episodes", fontsize=9, color=ink, loc="left")
    ax.legend(fontsize=7.5, frameon=False)
    ax.grid(color="#e4e3df", linewidth=0.6, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=muted, labelsize=8)
    fig.tight_layout()
    fig.savefig(path, facecolor=surface)
    plt.close(fig)


# 4. mixture ----------------------------------------------------------------------------------------------------

def mixture(out: Path) -> dict:
    from hylo.bench.protocol import load_protocol
    from hylo.bench.run import compare, run_protocol
    from hylo.core.config import ExperimentConfig
    from hylo.core.registry import registry

    agent = ExperimentConfig.load(EX / "configs" / "mujoco_mixture.yaml").agent.model_dump(mode="json")
    sampler, steering = agent["params"]["sampler"], agent["params"]["steering"]
    a, b = sampler["params"]["members"]
    variants = {"member A (gain 0.3, noise 0.05)": {"type": "policy", "params": {"policy": a}},
                "member B (gain 0.1, noise 0.3)": {"type": "policy", "params": {"policy": b}},
                "A steered": {"type": "steered", "params": {"sampler": a, "steering": steering}},
                "B steered": {"type": "steered", "params": {"sampler": b, "steering": steering}},
                "mixture steered (config)": agent,
                "mixture mean": {"type": "policy", "params": {"policy": {**sampler, "params": {**sampler["params"], "mode": "mean"}}}}}
    p = load_protocol("mujoco-reach")
    cards = {k: run_protocol(p, v, re.sub(r"\W+", "-", k), out=out / "mixture", overrides={"workers": 16})[0]
             for k, v in variants.items()}
    steps = {k: {s["key"]: float(s["steps"]) for s in json.loads((Path(c.run_dir) / "evallog.json").read_text())["samples"]}
             for k, c in cards.items()}
    sign = registry.build("gate_tests", {"type": "sign", "params": {"metric": "steps", "direction": "lower"}})
    mk = "mixture steered (config)"
    rows, res = [], {}
    for k, c in cards.items():
        cmp = compare(cards[mk], c) if k != mk else None             # success: McNemar
        cms = sign.compare(steps[mk], steps[k]) if k != mk else None  # steps: sign test, fewer is better
        res[k] = {"success": c.metrics["success_mean"], "ci95": c.success_ci95, "steps_mean": c.metrics["steps_mean"],
                  "interventions_per_episode": c.metrics["safety_interventions_mean"],
                  "mixture_vs_this_success": cmp and cmp.model_dump(), "mixture_vs_this_steps": cms and cms.model_dump()}
        rows.append((k, f"{100 * c.metrics['success_mean']:.0f}", f"{100 * c.success_ci95[0]:.0f}–{100 * c.success_ci95[1]:.0f}",
                     f"{c.metrics['steps_mean']:.1f}",
                     "–" if cmp is None else f"{cmp.candidate_wins}–{cmp.parent_wins} (p {cmp.p_value:.2g})",
                     "–" if cms is None else f"{cms.candidate_wins}–{cms.parent_wins} (p {cms.p_value:.2g})"))
    print("\n4. policies examples.ik_reacher + examples.mixture under mujoco-reach (100 held-out episodes)")
    print(table(["agent", "success %", "95% CI", "mean steps", "mixture: success wins–losses",
                 "mixture: fewer–more steps (sign)"], rows))
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--quick", action="store_true", help="skip the three gate campaigns (< 60 s)")
    ap.add_argument("--out", type=Path, default=REPO / "runs" / "examples" / "plugin-effects")
    a = ap.parse_args()
    out = a.out
    for d in ("guard", "gate", "mixture"):
        shutil.rmtree(out / d, ignore_errors=True)
    t = time.perf_counter()
    result = {"example": "plugin-effects", "quick": a.quick, "guard": guard(out), "gate_test": gate(out, a.quick),
              "online_learner": cem(out), "mixture": mixture(out)}
    result["seconds_total"] = round(time.perf_counter() - t, 1)
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps(result, indent=2, default=str))
    print(f"\ntotal {result['seconds_total']:.0f} s; {out / 'result.json'}")


if __name__ == "__main__":
    main()
