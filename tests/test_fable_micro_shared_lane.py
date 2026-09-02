"""scripts/fable/micro_shared_lane.py — CLI surface and byte model, no Metal.

The micro is the falsifier for program row 9, so the parts of it that decide a
verdict are checked on the CPU: the byte model it prices against (which must
reproduce the retained-stack census's ``MoE shared`` row), the arm list, the
hand-counted dispatches per arm, and the argument validation that stops a
typo'd sweep from quietly measuring the default arm.  MLX is imported lazily by
the script, so nothing here touches the GPU.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "fable" / "micro_shared_lane.py"


def _load():
    spec = importlib.util.spec_from_file_location("micro_shared_lane", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def micro():
    return _load()


def test_the_script_does_not_import_mlx_at_module_scope(micro):
    """Lazy import: --plan must work on a box with the GPU held by someone else."""

    assert micro.mx is None
    assert micro._shared_lane is None


def test_byte_model_reproduces_the_census_row(micro):
    shared = micro.shared_bytes_per_layer()
    assert shared["shared_gu"] == 3_481_600
    assert shared["shared_down"] == 1_740_800
    assert shared["shared_branch"] == 5_222_400
    # The census's "MoE shared 251.7 MB/cyc", less the scalar gate.
    assert shared["shared_branch"] * 48 == 250_675_200


def test_routed_issued_bytes_are_the_forty_lanes(micro):
    # q4/group-32: 0.625 B/weight.  gu 1280x2560, down 2560x640, 40 lanes.
    per_lane = int((2 * 640 * 2560 + 2560 * 640) * 0.625)
    assert micro.routed_bytes_per_layer() == 40 * per_lane


def test_resident_bank_fits_the_window(micro):
    total = micro.bank_bytes()["total"]
    # One 512-expert q4 bank plus 48 shared packs. Well under any memory knob;
    # the point of the assertion is that a shape drift cannot silently turn
    # this micro into a 60 GB run inside someone else's window.
    assert total < 2 * 1024**3


def test_plan_is_json_serialisable_and_states_the_floor(micro):
    plan = micro.plan(48)
    json.dumps(plan)
    assert plan["byte_floor_ms_per_cycle_at_600GBs"] == pytest.approx(0.4178, abs=1e-4)
    assert plan["census_apportioned_ms_per_cycle"]["control"] == 1.238
    assert "fitted apportionment" in plan["note"]


def test_dispatch_counts_cover_every_arm(micro):
    assert set(micro.DISPATCHES_PER_LAYER) == set(micro.ARMS)
    # The lane removes no dispatch; it adds MLX's two fence crossings.
    assert micro.DISPATCHES_PER_LAYER["lane"] == (
        micro.DISPATCHES_PER_LAYER["stock"] + 4
    )
    assert micro.DISPATCHES_PER_LAYER["shared_branch"] == 3


def test_plan_mode_exits_clean_without_mlx(micro, capsys):
    assert micro.main(["--plan", "--layers", "12"]) == 0
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["layers"] == 12
    assert micro.mx is None


@pytest.mark.parametrize(
    "argv",
    [
        ["--arms", "shared_gu,nope"],
        ["--layers", "0"],
        ["--reps", "0"],
    ],
)
def test_bad_arguments_are_rejected(micro, argv):
    with pytest.raises(SystemExit):
        micro.main(argv + ["--plan"])


def test_derive_reports_the_four_decisive_numbers(micro):
    timings = {
        "shared_gu": {"median_ms": 1.0},
        "shared_down": {"median_ms": 1.0},
        "shared_branch": {"median_ms": 1.0},
        "routed": {"median_ms": 10.0},
        "stock": {"median_ms": 11.0},
        "lane": {"median_ms": 10.4},
    }
    derived = micro.derive(timings, 48)
    assert derived["exposed_shared_cost_today_ms"] == pytest.approx(1.0)
    assert derived["exposed_shared_cost_with_lane_ms"] == pytest.approx(0.4)
    assert derived["lane_delta_ms_per_cycle"] == pytest.approx(-0.6)
    assert derived["exposed_shared_cost_today_us_per_layer"] == pytest.approx(
        1000.0 / 48
    )
    assert derived["achieved_GBs"]["shared_branch"] == pytest.approx(
        5_222_400 * 48 / 1e-3 / 1e9
    )
    assert "close row 9" in derived["verdict_rule"]
