"""Generalisation: the engine is task-agnostic. A second task bundle (tasks/demo/)
loads through the SAME loaders + rules engine with no code change, and the default
task (face_support) is unaffected."""
import pytest
from prompt_config import load as load_prompts
from rule_config import load as load_rules
from harness_config import load as load_params
from rules_engine import decide
import task


def test_nonexistent_task_fails_loudly():
    with pytest.raises(RuntimeError):
        task.task_dir("definitely_not_a_real_task")


def test_failsafe_default_is_never_the_passing_verdict():
    # the demo's fail-safe default must be conservative (absent facts must not pass)
    assert load_rules("demo")["ppe"]["default"] != "COMPLIANT"
    # and face_support's coverage defaults are the non-compliant verdict
    fs = load_rules("face_support")
    assert fs["coverage_full"]["default"] == "NOT SUPPORTED"
    assert fs["coverage_overlap"]["default"] == "NOT SUPPORTED"


def test_active_task_default_is_face_support():
    assert task.active_task() == "face_support"


def test_bundles_are_independent_per_task():
    # different prompts per task
    assert load_prompts("demo")["system"] != load_prompts("face_support")["system"]
    # different rules per task
    fs, demo = load_rules("face_support"), load_rules("demo")
    assert "ppe" in demo and "ppe" not in fs
    assert "operator_entry" in fs and "operator_entry" not in demo
    # params: each bundle's overrides merge over the SAME safe DEFAULTS
    assert load_params("demo")["operator"]["boom_motion_thresh"] == 0.05
    assert load_params("face_support")["operator"]["boom_motion_thresh"] == 0.035
    # demo inherits an unspecified default unchanged
    assert load_params("demo")["coverage"]["mesh_gap"] == 240


def test_engine_runs_a_different_task_unchanged():
    ppe = load_rules("demo")["ppe"]
    assert decide(ppe, {"helmet": False, "vest": True}) == "VIOLATION"
    assert decide(ppe, {"helmet": True, "vest": False}) == "VIOLATION"
    assert decide(ppe, {"helmet": True, "vest": True}) == "COMPLIANT"
