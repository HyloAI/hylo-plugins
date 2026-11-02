<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset=".github/assets/banner-dark.svg">
    <img src=".github/assets/banner-light.svg" width="100%" alt="hylo-plugins: extend the kernel from outside: policies, environments, learners, guards and gate tests">
  </picture>
</p>

<p align="center">
  <a href="LICENSE"><img alt="License: Apache-2.0" src="https://img.shields.io/badge/license-Apache--2.0-c8442d?style=flat-square"></a>
  <img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10%2B-171c22?style=flat-square&logo=python&logoColor=white">
  <img alt="Registered through entry points" src="https://img.shields.io/badge/registered%20by-entry%20points-171c22?style=flat-square">
  <img alt="5 example plugins" src="https://img.shields.io/badge/examples-5%20plugins-5b6168?style=flat-square">
</p>

<p align="center">
  <a href="https://hyloai.github.io/"><b>Website</b></a> ·
  <a href="https://hyloai.github.io/hylo-docs/"><b>Docs</b></a> ·
  <a href="https://hyloai.github.io/hylo-docs/getting-started/quickstart/"><b>Quickstart</b></a> ·
  <a href="https://github.com/HyloAI"><b>All repositories</b></a>
</p>

Part of [Hylo](https://hyloai.github.io/), the stack for physical AI. Plugins extend the **kernel** layer:
policies, environments, learners, guards and gate tests, each registered through an entry point.

Plugins written the way a third party would: separate packages that register entry points and never modify the
harness. They show that each extension point can be used from outside, and serve as templates.

The same extension contracts support the [robot-skill lifecycle](https://hyloai.github.io/hylo-docs/user-guide/skills/):
an agent can use robot tools, and a generator can
propose program candidates for evaluation. Register implementations through the harness's entry points and keep
robot and artifact formats in core contracts. The [development guide](https://hyloai.github.io/hylo-docs/user-guide/developing-strategies/)
shows how these capabilities fit together; the examples here demonstrate the extension mechanism.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset=".github/assets/flow-dark.svg">
    <img src=".github/assets/flow-light.svg" width="100%" alt="A package declares an entry point in pyproject.toml; the plugin registry in hylo-core finds it by name with its contract version and maturity; configs use it by that name. This repository holds a guard, a gate test, an online learner and two policies">
  </picture>
</p>

| Package | Plugins |
|---|---|
| `packages/hylo-plugin-examples` (`hylo_examples`) | `examples.slow_zone` (guard), `examples.min_effect` (gate test), `examples.cem` (online learner), `examples.ik_reacher` and `examples.mixture` (policies) |

```bash
uv run hylo run examples/configs/mujoco_guarded.yaml            # a guard that slows the fingertip near the target
uv run hylo evolve run examples/evolve/mujoco_steps_gate.yaml   # a gate on steps instead of success
uv run hylo evolve run examples/evolve/mujoco_online_cem.yaml   # an online learner as a candidate generator
uv run hylo run examples/configs/mujoco_mixture.yaml            # a steered mixture of two policies
uv run hylo plugin new policy my-policy --dir packages        # a new plugin package from a template
uv run python examples/01_plugin_effects.py                 # each plugin with and without, measured (examples/README.md)
```

The examples run on MuJoCo planar reach, so they need hylo-bench (hylo-env-mujoco) installed next to the
harness; the unit tests use the harness's test double. Run from this repository. The extension points, their contracts and conformance checks: the harness's
docs/extending.md.

Documentation (hylo-docs, `uv run mkdocs serve -f docs/mkdocs.yml` from the workspace): [extending](../harness/docs/extending.md) (also on the site), the plugin catalogue (generated from `hylo plugins`).

## The Hylo stack

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset=".github/assets/stack-dark.svg">
    <img src=".github/assets/stack-light.svg" width="100%" alt="The six layers of the Hylo stack; hylo-plugins extends the kernel">
  </picture>
</p>

## License

Apache License 2.0 (see [LICENSE](LICENSE)). Copyright 2026 Hylo AI.
