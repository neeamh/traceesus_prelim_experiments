"""Focused contracts for the root command-line orchestration."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

import run

def test_figures_command_points_to_notebook_without_mutating_outputs(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Keep the compatibility command read-only and direct users to the notebook."""

    before = tuple(tmp_path.iterdir())
    status = run._figures(
        argparse.Namespace(experiment="endotype_discovery", out=tmp_path)
    )
    assert status == 0
    assert tuple(tmp_path.iterdir()) == before
    assert "notebooks/figures.ipynb" in capsys.readouterr().out


def test_cli_inventory_is_exactly_the_three_retained_experiments() -> None:
    """Prevent archived or extension-only studies from re-entering the root CLI."""

    expected = {"endotype_discovery", "transportability", "counterfactual"}
    assert set(run.EXPERIMENTS) == expected
    assert set(run.FACTORIES) == expected
