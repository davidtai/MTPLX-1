# DeepSeek V4 0731 isolated service candidate

This directory is a deliberately separate candidate surface.  It does not
change MTPLX's live service code, does not start a process on installation, and
its launchd plist has a distinct label on `127.0.0.1:8081`.

`encoding/` holds the review-gated DeepSeek 0731 chat-encoding asset slot.  The
attribution names source revision `7872f01b1d1fe23eabc4c98b48bffcef5a386062`;
because that exact revision did not resolve from public upstream history during
implementation, this candidate is intentionally promotion-blocked until the
official bytes replace the review fixture and its manifest.  Once installed,
the manifest hash is verified before the candidate process is exec'd;
`render.py` then uses the installed encoding without per-request integrity work.

`launch_candidate.sh` has no service-management commands.  It accepts no
arguments or caller command overrides; its only exception is the non-starting
`MTPLX_DSV4_0731_TEST_FIXTURE=1 ... --print-command` test seam.  Its fixed
environment and absolute executable/model arguments are intentionally boring.

`promote_cutover.py` is not an automatic promotion command.  Its `--promote`
action requires an already-passing, scrubbed candidate preflight/smoke receipt;
a separately reviewed production plist digest; and a live identity attestation.
It nonblockingly acquires `/tmp/mtplx-gpu-exclusive.lock`, rechecks the exact
live launchd PID/listener/plist hash before stopping anything, and holds that
lock through rollback.  The receipt must contain no local paths, prompts,
messages, tools, secrets, argv/env, or captured process output.  It verifies
`/v1/models` and an unrecorded `READY` completion with `finish_reason=stop`
after a cutover or rollback.

No script here is a permission to start, stop, or promote a service without an
operator explicitly supplying the required current receipts and `--promote`.
