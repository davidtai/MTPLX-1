#!/usr/bin/env python3
"""Preflight: will the native QSA extension's types resolve against mlx.core?

NO GPU, NO BUILD.  Pure file inspection.

WHY THIS EXISTS
---------------
W68's first guarded run built the extension cleanly and then failed at the
first call with::

    TypeError: qsa_sparse_gqa_decode(): incompatible function arguments.
      1. qsa_sparse_gqa_decode(queries: mlx::core::array, ...)
      Invoked with types: mlx.core.array, mlx.core.array, ...

The invoked type IS ``mlx.core.array``, and the signature still refuses it,
because nanobind rendered the parameter as the raw C++ name ``mlx::core::array``
rather than the registered ``array`` -- the signal that the extension cannot
see mlx.core's type registry at all.  It is not a ``stream=None`` problem: the
same failure reproduces with ``stream`` omitted, and on W50's
``qsa_sparse_gqa_unsupported_reason``, which is a pure host predicate.

Two nanobind modules share a type registry only when they agree on the capsule
key ``__nb_internals_<abi_tag>_<domain>__`` (nanobind ``src/nb_internals.cpp``,
``nb_module_exec``).  The domain is right on both sides (``NB_DOMAIN=mlx``).
The ABI TAG is not:

    mlx.core (wheel 0.32.2) ......... v21_system_libcpp_abi1
    _ext (venv nanobind 2.12.0) ..... v19_system_libcpp_abi1

``NB_ABI_TAG`` is ``"v" NB_INTERNALS_VERSION ... "_" NB_PLATFORM_ABI_TAG``
(nanobind ``src/nb_abi.h``), and ``NB_INTERNALS_VERSION`` is 19 in nanobind
2.12.0 and 21 in 2.15.0.  Different key, separate registries, so every function
taking an ``mx::array`` rejects every call.

Do NOT "fix" this by defining ``NB_INTERNALS_VERSION=21`` on the v19 sources.
That macro guards the LAYOUT of the shared ``nb_internals`` struct; forcing the
tag to match while the struct differs makes two modules write to each other's
memory through the same capsule.  Build against a nanobind whose real internals
version matches.

USAGE
-----
    python scripts/fable/check_native_qsa_abi.py \
        --python /path/to/venv/bin/python [--nanobind /path/to/nanobind]

Exit 0 when the two will share a registry, 1 when they will not.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

#: nanobind writes its ABI tag as one NUL-terminated literal, e.g.
#: ``v21_system_libcpp_abi1``.  Match the version and the platform half.
_ABI_TAG_RE = re.compile(rb"v(\d+)((?:[0-9a-zA-Z.\-]*)_[0-9a-zA-Z_]*(?:libcpp|libstdcpp|ms)[0-9a-zA-Z_]*)")
_INTERNALS_RE = re.compile(r"define\s+NB_INTERNALS_VERSION\s+(\d+)")


def abi_tags_in_binary(path: Path) -> set[str]:
    """Every nanobind ABI tag literal embedded in a shared object."""

    data = path.read_bytes()
    return {
        f"v{m.group(1).decode()}{m.group(2).decode()}"
        for m in _ABI_TAG_RE.finditer(data)
    }


def abi_version_of_binary(path: Path) -> int | None:
    """The nanobind internals version a shared object was built against."""

    versions = {int(tag[1:].split("_", 1)[0]) for tag in abi_tags_in_binary(path)}
    if len(versions) != 1:
        return None
    return versions.pop()


def internals_version_of_nanobind(nanobind_dir: Path) -> int | None:
    """``NB_INTERNALS_VERSION`` from a nanobind source tree.

    2.13+ keeps it in ``src/nb_abi.h``; older releases keep it in
    ``src/nb_internals.h``.  Both are checked so this works across the
    versions actually installed on this box.
    """

    for name in ("src/nb_abi.h", "src/nb_internals.h"):
        candidate = nanobind_dir / name
        if not candidate.exists():
            continue
        match = _INTERNALS_RE.search(candidate.read_text())
        if match:
            return int(match.group(1))
    return None


def _run(python: Path, *args: str) -> str:
    return subprocess.run(
        [str(python), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def mlx_package_dir(python: Path) -> Path:
    return Path(_run(python, "-m", "mlx", "--cmake-dir"))


def nanobind_dir_for(python: Path) -> Path:
    # ``--cmake_dir`` points at <nanobind>/cmake; the sources are one up.
    return Path(_run(python, "-m", "nanobind", "--cmake_dir")).parent


def mlx_core_binary(mlx_dir: Path) -> Path | None:
    matches = sorted(mlx_dir.glob("core*.so"))
    return matches[0] if matches else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--python",
        type=Path,
        default=Path(sys.executable),
        help="interpreter whose mlx wheel and nanobind the build will use",
    )
    parser.add_argument(
        "--nanobind",
        type=Path,
        default=None,
        help="override the nanobind source tree the build will use",
    )
    parser.add_argument(
        "--ext",
        type=Path,
        default=None,
        help="optional: an already-built _ext*.so to report on",
    )
    args = parser.parse_args()

    mlx_dir = mlx_package_dir(args.python)
    core = mlx_core_binary(mlx_dir)
    if core is None:
        print(f"FAIL: no core*.so under {mlx_dir}", file=sys.stderr)
        return 1
    mlx_tags = abi_tags_in_binary(core)
    mlx_version = abi_version_of_binary(core)
    print(f"mlx.core            {core.name}")
    print(f"  abi tag           {', '.join(sorted(mlx_tags)) or '(none found)'}")

    nanobind = args.nanobind or nanobind_dir_for(args.python)
    build_version = internals_version_of_nanobind(nanobind)
    print(f"nanobind            {nanobind}")
    print(f"  internals version {build_version}")

    if args.ext is not None and args.ext.exists():
        print(f"built extension     {args.ext.name}")
        print(
            f"  abi tag           "
            f"{', '.join(sorted(abi_tags_in_binary(args.ext))) or '(none found)'}"
        )

    if mlx_version is None or build_version is None:
        print(
            "FAIL: could not read both versions; refusing to guess",
            file=sys.stderr,
        )
        return 1
    if mlx_version == build_version:
        print(f"\nOK: both are nanobind internals v{mlx_version}; "
              "the extension will share mlx.core's type registry.")
        return 0

    print(
        f"\nFAIL: mlx.core is nanobind internals v{mlx_version}, the build "
        f"would use v{build_version}.\n"
        "      The two get separate __nb_internals_<tag>_mlx__ capsules, so "
        "the\n"
        "      extension cannot resolve mlx::core::array and EVERY call "
        "raises\n"
        "      TypeError. Point the build at a matching nanobind:\n"
        "\n"
        "        cmake -S . -B build -Dnanobind_ROOT=<nanobind with "
        f"internals v{mlx_version}> ...\n"
        "\n"
        "      Do NOT define NB_INTERNALS_VERSION to force the tag: it guards "
        "the\n"
        "      shared nb_internals struct layout.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
