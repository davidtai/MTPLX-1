#!/usr/bin/env python3
"""Run PR335 DFlash2 with the current matrix's frozen prompt contract."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PROMPT_ARM = ROOT / "scripts/qwen38_native_mtp_matrix_arm.py"


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _workload(args: Any) -> str:
    if args.prompt_kind == "is_palindrome":
        return "vanity"
    if args.reasoning_effort in {"low", "xhigh"}:
        return str(args.reasoning_effort)
    raise ValueError("coding comparator requires low or xhigh reasoning effort")


def _generate_or_skip(generate: Any, runtime: Any, prompt_ids: list[int], args: Any) -> Any:
    if int(args.max_tokens) == 0:
        return None
    return generate(runtime, prompt_ids, args)


def main() -> int:
    prompt_arm = _load_module("qwen38_matrix_prompt_contract", PROMPT_ARM)
    source_root = Path(sys.argv[sys.argv.index("--source-root") + 1]).resolve(strict=True)
    expected_commit = sys.argv[sys.argv.index("--source-commit") + 1]
    observed_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=source_root, text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--short"], cwd=source_root, text=True
    ).splitlines()
    if observed_commit != expected_commit or status:
        raise RuntimeError(
            f"DFlash2 source must be clean at {expected_commit}; "
            f"found commit={observed_commit} status={status}"
        )

    upstream = _load_module(
        "qwen38_pr335_final_benchmark_arm",
        source_root / "scripts/qwen38_final_benchmark_arm.py",
    )

    def load_frozen_prompt(args: Any, tokenizer: Any) -> tuple[str, list[int]]:
        _prompt_id, instruction = prompt_arm._read_prompt(args.prompt_file)
        return prompt_arm.build_prompt(
            tokenizer,
            workload=_workload(args),
            instruction=instruction,
            context=args.context_file.read_text(encoding="utf-8"),
            target_tokens=int(args.prompt_tokens),
        )

    original_generate = upstream._generate_dflash
    upstream._load_prompt = load_frozen_prompt
    upstream._sha256_tokens = prompt_arm._token_hash
    upstream._generate_dflash = (
        lambda runtime, prompt_ids, args: _generate_or_skip(
            original_generate, runtime, prompt_ids, args
        )
    )
    exit_code = int(upstream.main())

    output = Path(sys.argv[sys.argv.index("--output") + 1])
    receipt = json.loads(output.read_text(encoding="utf-8"))
    workload = _workload(upstream._parse_args())
    receipt["kind"] = "qwen38_dflash2_frozen_matrix_arm"
    receipt["harness_commit"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    receipt["workload"]["workload"] = workload
    receipt["workload"]["enable_thinking"] = workload in {"low", "xhigh"}
    receipt["workload"]["prompt_format"] = {
        "vanity": "qwen_chat_template_non_thinking",
        "low": "qwen_chat_template_thinking_low",
        "xhigh": "qwen_chat_template_thinking_xhigh",
    }[workload]
    receipt["workload"]["prompt_artifact_sha256"] = prompt_arm._sha256(
        Path(receipt["workload"].get("prompt_file", ""))
    ) if receipt["workload"].get("prompt_file") else prompt_arm._sha256(
        Path(sys.argv[sys.argv.index("--prompt-file") + 1])
    )
    receipt["workload"]["context_artifact_sha256"] = prompt_arm._sha256(
        Path(sys.argv[sys.argv.index("--context-file") + 1])
    )
    receipt["mlx_metal_version"] = importlib.metadata.version("mlx-metal")
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
