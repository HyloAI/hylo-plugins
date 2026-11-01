"""Example worldcurrent plugins, written the way a third party would: own package name, entry points, no changes
to worldcurrent. One per extension point; see the harness's docs/extending.md."""

# contract versions and what is known about these plugins (worldcurrent.core.plugins)
META = {
    "contracts": {
        "learners": "wc.learner/v0.1",
        "gate_tests": "wc.gate_test/v0.1",
        "policies": "wc.policy/v0.1",
        "guards": "wc.guard/v0.1",
    },
    "maturity": "implemented",
}
