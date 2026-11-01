"""Example Hylo plugins, written the way a third party would: own package name, entry points, no changes
to Hylo. One per extension point; see the harness's docs/extending.md."""

# contract versions and what is known about these plugins (hylo.core.plugins)
META = {
    "contracts": {
        "learners": "hylo.learner/v0.1",
        "gate_tests": "hylo.gate_test/v0.1",
        "policies": "hylo.policy/v0.1",
        "guards": "hylo.guard/v0.1",
    },
    "maturity": "implemented",
}
