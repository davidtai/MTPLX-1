#!/usr/bin/env python3
"""Guarded, deliberately explicit 0731 candidate promotion workflow.

This is an operator workflow, not an auto-promotion hook.  It has no default
action, takes an exclusive nonblocking GPU lock, and refuses receipts that can
contain request content or process secrets.  The lock spans both cutover and
rollback, so an unrelated GPU user cannot be interrupted or raced.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import plistlib
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


LOCK_PATH = Path("/tmp/mtplx-gpu-exclusive.lock")
CANDIDATE_LABEL = "com.tea.deepseek-v4-0731.candidate"
CANDIDATE_PORT = 8081
LIVE_PORT = 8080
SENSITIVE_KEY = re.compile(r"(?:prompt|message|tool|secret|token|authorization|argv|env|stdout|stderr)", re.I)
SENSITIVE_VALUE = re.compile(
    r"(?:-----BEGIN [A-Z ]*PRIVATE KEY-----|\bBearer\s+\S+|\bsk-[A-Za-z0-9_-]{12,}|"
    r"[\"']?(?:prompt|messages|tools|secret|authorization)[\"']?\s*[:=])",
    re.I,
)
ALLOWED_SIGNERS = Path("/Users/davidtai/.config/mtplx/deepseek-v4-0731-allowed-signers")
SIGNING_IDENTITY = "mtplx-deepseek-v4-0731-candidate"
SIGNING_NAMESPACE = "mtplx-deepseek-v4-0731"
CANDIDATE_WORKTREE = Path("/Users/davidtai/projects/OpenSourceWTF/.worktrees/dsv4-0731-service")
REVIEWED_REF = "refs/tags/mtplx-dsv4-0731-reviewed"
CANDIDATE_PLIST_SHA256 = "93eac0d4eaac491c7f2f1d3a293ba38a3144ade59ee3afdf52b35cc9ec9bb101"
ENCODING_ASSET_SET_SHA256 = "6758dfda8a39afdd00d907606c42c1a268289c463351b9628ac07f4f916d7d0a"
MODEL_CONFIG_SHA256 = "c8ff87fd5ee5c9587d0c937e9bfd3193e1a1621141aa367848a9610b3291fa6f"
MODEL_INDEX_SHA256 = "c84d2b369f5d5023d0f2d183fc36a935a3981751414996243b65f069983e43d8"


class PromotionError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    if not path.is_file() or path.is_symlink():
        raise PromotionError("attested plist is missing or unsafe")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _command(*argv: str) -> str:
    result = subprocess.run(argv, check=False, capture_output=True, text=True)
    if result.returncode:
        raise PromotionError(f"required identity probe failed: {argv[0]}")
    return result.stdout


def _http_json(url: str) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(url, timeout=3) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, ValueError, urllib.error.URLError) as error:
        raise PromotionError("required HTTP readiness probe failed") from error
    if not isinstance(payload, dict):
        raise PromotionError("required HTTP readiness response is malformed")
    return payload


def _smoke_stop(model_id: str) -> None:
    """Run a real, unrecorded readiness completion and require normal stop."""
    body = json.dumps(
        {
            "model": model_id,
            "messages": [{"role": "user", "content": "Reply with exactly READY."}],
            "temperature": 0,
            "max_tokens": 8,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"http://127.0.0.1:{LIVE_PORT}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            payload = json.loads(response.read().decode("utf-8"))
        choice = payload["choices"][0]
        content = choice["message"]["content"]
        if choice.get("finish_reason") != "stop" or not isinstance(content, str) or "READY" not in content:
            raise ValueError("required READY/stop evidence absent")
    except (KeyError, OSError, TypeError, ValueError, urllib.error.URLError) as error:
        raise PromotionError("service smoke did not return READY with finish_reason=stop") from error


def _listener_pid(port: int) -> int:
    output = _command("/usr/sbin/lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-Fpn")
    pids = {int(line[1:]) for line in output.splitlines() if line.startswith("p") and line[1:].isdigit()}
    if len(pids) != 1:
        raise PromotionError("listener identity is absent or ambiguous")
    return pids.pop()


def _launchctl_pid(label: str) -> int:
    domain = f"gui/{os.getuid()}/{label}"
    output = _command("/bin/launchctl", "print", domain)
    match = re.search(r"\bpid = (\d+)", output)
    if not match:
        raise PromotionError("launchd service has no single running PID")
    return int(match.group(1))


def attest_live(*, label: str, plist: Path) -> dict[str, Any]:
    """Capture exact live identity without sending a generation prompt."""
    launch_pid = _launchctl_pid(label)
    listener_pid = _listener_pid(LIVE_PORT)
    if launch_pid != listener_pid:
        raise PromotionError("launchd PID and 8080 listener PID differ")
    models = _http_json(f"http://127.0.0.1:{LIVE_PORT}/v1/models")
    model_ids = [item.get("id") for item in models.get("data", []) if isinstance(item, dict)]
    if not model_ids or not all(isinstance(model_id, str) for model_id in model_ids):
        raise PromotionError("live /v1/models is not a valid service identity")
    return {
        "schema": "mtplx.live-identity.v1",
        "label": label,
        "pid": launch_pid,
        "listener_port": LIVE_PORT,
        "plist_sha256": _sha256(plist),
        "model_ids": model_ids,
    }


def _read_json_bytes(path: Path) -> tuple[dict[str, Any], bytes]:
    if not path.is_file() or path.is_symlink():
        raise PromotionError("receipt is missing or unsafe")
    try:
        raw = path.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, ValueError) as error:
        raise PromotionError("receipt is not valid JSON") from error
    if not isinstance(payload, dict):
        raise PromotionError("receipt root must be an object")
    return payload, raw


def _read_json(path: Path) -> dict[str, Any]:
    return _read_json_bytes(path)[0]


def _verify_candidate_signature(receipt_bytes: bytes, signature: Path) -> None:
    if not ALLOWED_SIGNERS.is_file() or ALLOWED_SIGNERS.is_symlink():
        raise PromotionError("pinned candidate allowed-signers file is missing or unsafe")
    if not signature.is_file() or signature.is_symlink():
        raise PromotionError("detached candidate signature is missing or unsafe")
    result = subprocess.run(
        [
            "/usr/bin/ssh-keygen",
            "-Y",
            "verify",
            "-f",
            str(ALLOWED_SIGNERS),
            "-I",
            SIGNING_IDENTITY,
            "-n",
            SIGNING_NAMESPACE,
            "-s",
            str(signature),
        ],
        input=receipt_bytes,
        check=False,
        capture_output=True,
    )
    if result.returncode:
        raise PromotionError("candidate receipt signature verification failed")


def _contains_sensitive(value: Any) -> bool:
    if isinstance(value, dict):
        return any(SENSITIVE_KEY.search(str(key)) or _contains_sensitive(item) for key, item in value.items())
    if isinstance(value, list):
        return any(_contains_sensitive(item) for item in value)
    if isinstance(value, str):
        if any(marker in value for marker in ("/Users/", "/private/", "/tmp/")):
            return True
        if SENSITIVE_VALUE.search(value):
            return True
    return False


def _require_exact_keys(value: Any, expected: set[str], context: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise PromotionError(f"{context} fields do not match the strict receipt schema")
    return value


def assert_candidate_receipt(payload: dict[str, Any]) -> None:
    """Accept only a previously passing, scrubbed candidate preflight+smoke receipt."""
    if _contains_sensitive(payload):
        raise PromotionError("candidate receipt includes prohibited sensitive capture")
    _require_exact_keys(payload, {"schema", "candidate_preflight", "candidate_smoke"}, "candidate receipt")
    if payload["schema"] != "mtplx.dsv4-0731-candidate.v1":
        raise PromotionError("candidate receipt schema is not pinned")
    preflight = _require_exact_keys(
        payload["candidate_preflight"],
        {
            "ok", "label", "port", "plist_sha256", "encoding_source_revision",
            "encoding_asset_set_sha256", "reviewed_commit", "model_config_sha256",
            "model_index_sha256", "promotion_target",
        },
        "candidate preflight",
    )
    smoke = _require_exact_keys(
        payload["candidate_smoke"],
        {"ok", "models_ok", "ready", "finish_reason", "candidate_model_ids"},
        "candidate smoke",
    )
    if preflight.get("ok") is not True or smoke.get("ok") is not True:
        raise PromotionError("candidate preflight and smoke must already pass")
    if preflight.get("label") != CANDIDATE_LABEL or preflight.get("port") != CANDIDATE_PORT:
        raise PromotionError("candidate identity does not match the pinned isolated service")
    if smoke.get("models_ok") is not True or smoke.get("ready") is not True or smoke.get("finish_reason") != "stop":
        raise PromotionError("candidate smoke receipt lacks models/READY/stop evidence")
    if preflight.get("encoding_source_revision") != "7872f01b1d1fe23eabc4c98b48bffcef5a386062":
        raise PromotionError("candidate encoding source revision changed")
    for field in (
        "plist_sha256", "encoding_asset_set_sha256", "model_config_sha256", "model_index_sha256"
    ):
        if not isinstance(preflight.get(field), str) or not re.fullmatch(r"[0-9a-f]{64}", preflight[field]):
            raise PromotionError(f"candidate preflight has invalid {field}")
    if not isinstance(preflight.get("reviewed_commit"), str) or not re.fullmatch(r"[0-9a-f]{40}", preflight["reviewed_commit"]):
        raise PromotionError("candidate preflight has invalid reviewed_commit")
    target = _require_exact_keys(preflight["promotion_target"], {"label", "plist_sha256"}, "promotion target")
    if not isinstance(target.get("label"), str):
        raise PromotionError("candidate preflight lacks a separately reviewed promotion target")
    digest = target.get("plist_sha256")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise PromotionError("candidate preflight lacks a valid promotion plist digest")
    candidate_model_ids = smoke.get("candidate_model_ids")
    if (
        not isinstance(candidate_model_ids, list)
        or not candidate_model_ids
        or not all(isinstance(model_id, str) and model_id for model_id in candidate_model_ids)
    ):
        raise PromotionError("candidate smoke lacks nonempty candidate_model_ids")


def assert_live_identity(expected: dict[str, Any], current: dict[str, Any]) -> None:
    _require_exact_keys(
        expected,
        {"schema", "label", "pid", "listener_port", "plist_sha256", "model_ids"},
        "live attestation",
    )
    fields = ("schema", "label", "pid", "listener_port", "plist_sha256", "model_ids")
    if any(expected.get(field) != current.get(field) for field in fields):
        raise PromotionError("live service identity changed since its attestation")


@contextmanager
def exclusive_gpu_lock() -> Iterator[None]:
    """Take the shared lock once, nonblocking, and retain it through rollback."""
    fd = os.open(LOCK_PATH, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise PromotionError("GPU lock is already held; no service action was taken") from error
        yield
    finally:
        os.close(fd)


def _bootstrap(plist: Path) -> None:
    _command("/bin/launchctl", "bootstrap", f"gui/{os.getuid()}", str(plist))


def _bootout(label: str) -> None:
    _command("/bin/launchctl", "bootout", f"gui/{os.getuid()}/{label}")


def _verify_live_ready(expected_model_ids: list[str]) -> None:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        try:
            payload = _http_json(f"http://127.0.0.1:{LIVE_PORT}/v1/models")
            ids = [item.get("id") for item in payload.get("data", []) if isinstance(item, dict)]
            if ids == expected_model_ids:
                _smoke_stop(expected_model_ids[0])
                return
        except PromotionError:
            pass
        time.sleep(0.5)
    raise PromotionError("restored service did not recover its exact /v1/models identity")


def promote(args: argparse.Namespace) -> None:
    if args.promote is not True:
        raise PromotionError("refusing promotion without --promote")
    candidate, candidate_bytes = _read_json_bytes(args.candidate_receipt)
    _verify_candidate_signature(candidate_bytes, args.candidate_signature)
    expected_live = _read_json(args.live_attestation)
    if _contains_sensitive(expected_live):
        raise PromotionError("live attestation includes prohibited sensitive capture")
    assert_candidate_receipt(candidate)
    preflight = candidate["candidate_preflight"]
    reviewed_commit = _command(
        "/usr/bin/git",
        "-C",
        str(CANDIDATE_WORKTREE),
        "rev-parse",
        "--verify",
        f"{REVIEWED_REF}^{{commit}}",
    ).strip()
    pinned_candidate = {
        "reviewed_commit": reviewed_commit,
        "plist_sha256": CANDIDATE_PLIST_SHA256,
        "encoding_asset_set_sha256": ENCODING_ASSET_SET_SHA256,
        "model_config_sha256": MODEL_CONFIG_SHA256,
        "model_index_sha256": MODEL_INDEX_SHA256,
    }
    if any(preflight[field] != expected for field, expected in pinned_candidate.items()):
        raise PromotionError("signed candidate receipt does not match the reviewed installation")
    # The production target must be a separately reviewed 8080 plist.  The
    # candidate plist stays isolated on 8081 and is never edited in place.
    target = args.production_plist
    if target == args.live_plist or not target.is_absolute():
        raise PromotionError("an absolute separately reviewed production plist is required")
    if not target.is_file() or target.is_symlink():
        raise PromotionError("production plist is missing or unsafe")
    promotion_target = preflight["promotion_target"]
    if promotion_target["label"] != args.production_label or promotion_target["plist_sha256"] != _sha256(target):
        raise PromotionError("production plist identity does not match the passing candidate preflight")
    try:
        target_label = plistlib.loads(target.read_bytes()).get("Label")
    except (plistlib.InvalidFileException, OSError) as error:
        raise PromotionError("production plist is not valid") from error
    if target_label != args.production_label or args.production_label == str(expected_live.get("label")):
        raise PromotionError("production label is unsafe or does not match its plist")

    prior_plist = args.live_plist
    if not prior_plist.is_absolute():
        raise PromotionError("live attestation does not name an absolute prior plist")
    with exclusive_gpu_lock():
        current = attest_live(label=str(expected_live.get("label", "")), plist=prior_plist)
        assert_live_identity(expected_live, current)
        # No service is stopped until every receipt and identity check above has
        # passed under the lock.  Any post-cutover exception restores the exact
        # attested plist before releasing that same lock.
        try:
            _bootout(current["label"])
            _bootstrap(target)
            _verify_live_ready(candidate["candidate_smoke"]["candidate_model_ids"])
        except BaseException:
            try:
                _bootout(args.production_label)
            finally:
                _bootstrap(prior_plist)
                _verify_live_ready(current["model_ids"])
            raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--promote", action="store_true", help="explicitly authorize guarded service action")
    parser.add_argument("--candidate-receipt", type=Path, required=True)
    parser.add_argument("--candidate-signature", type=Path, required=True)
    parser.add_argument("--live-attestation", type=Path, required=True)
    parser.add_argument("--live-plist", type=Path, required=True)
    parser.add_argument("--production-plist", type=Path, required=True)
    parser.add_argument("--production-label", required=True)
    args = parser.parse_args(argv)
    try:
        promote(args)
    except PromotionError as error:
        print(f"promotion refused: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
