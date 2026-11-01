from worldcurrent.cli.scaffold import check_plugin


def test_the_example_plugins_pass_their_conformance_checks():
    assert check_plugin("guard", "examples.slow_zone", {"state_key": "pos", "center": [0, 0], "radius": 0.5}).ok
    assert check_plugin("gate_test", "examples.min_effect").ok
    assert check_plugin("online_learner", "examples.cem", {"space": {"kp": {"path": "params.kp", "low": 0.1, "high": 4}},
                                                            "population": 3}).ok
    # the policy examples target MuJoCo planar reach: checked in test_examples.py
