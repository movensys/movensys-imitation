"""Smoke tests that run without torch (CI) and verify the package skeleton."""

import importlib

import movensys_imitation


def test_version():
    assert movensys_imitation.__version__


def test_policy_registry_is_importable_without_lerobot():
    policies = importlib.import_module("movensys_imitation.policies")
    assert hasattr(policies, "register_all")
