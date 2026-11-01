# worldcurrent-plugins

Plugins written the way a third party would: separate packages that register entry points and never modify the
harness. They show that each extension point can be used from outside, and serve as templates.

| Package | Plugins |
|---|---|
| `packages/wc-plugin-examples` (`wc_examples`) | `examples.slow_zone` (guard), `examples.min_effect` (gate test), `examples.cem` (online learner), `examples.ik_reacher` and `examples.mixture` (policies) |

```bash
uv run wc run examples/configs/mujoco_guarded.yaml            # a guard that slows the fingertip near the target
uv run wc evolve run examples/evolve/mujoco_steps_gate.yaml   # a gate on steps instead of success
uv run wc evolve run examples/evolve/mujoco_online_cem.yaml   # an online learner as a candidate generator
uv run wc run examples/configs/mujoco_mixture.yaml            # a steered mixture of two policies
uv run wc plugin new policy my-policy --dir packages        # a new plugin package from a template
```

The examples run on MuJoCo planar reach, so they need worldcurrent-bench (wc-env-mujoco) installed next to the
harness; the unit tests use the harness's test double. Run from this repository. The extension points, their contracts and conformance checks: the harness's
docs/extending.md.
