# DeepSeek-V4-Flash-0731 isolated candidate service

This directory owns a separate candidate only. It does not change the loaded
production service, and its launchd identity is
`com.tea.deepseek-v4-0731.candidate` on `127.0.0.1:8081`.

The `encoding/` directory vendors the exact official Python encoder and four
input/output vectors from
`deepseek-ai/DeepSeek-V4-Flash-0731@7872f01b1d1fe23eabc4c98b48bffcef5a386062`.
`candidate_entry.py` verifies all nine assets and runs every official vector at
construction. It then installs the encoder directly at MTPLX's prompt-ID call
site and installs the official DSML parser at the nonstream and streaming
response call sites. The stream translator retains ordinary preamble text while
continuing to scan later chunks and holding any suffix that could grow into a
split DSML marker; raw markup is never released. The complete turn still passes
through the official parser. Plain no-tool turns also use that parser and report
its engagement. No tokenizer-template, stock prompt, or stock completion-parser
fallback remains in the enabled 0731 lane. Per-request observability reports
`backend_chat_encoding=deepseek-v4-flash-0731-official`.

`launch_candidate.sh` accepts no production arguments. Its only test seam is
`MTPLX_DSV4_0731_TEST_FIXTURE=1 ... --print-command`, which cannot start the
service. A real launch requires:

- the exact commit referenced by `refs/tags/mtplx-dsv4-0731-reviewed`;
- a completely clean worktree;
- the pinned interpreter, model config/index, manifest, encoder, and official
  vector hashes;
- the exact reviewed artifact validator at commit `bbf02944`, which hashes the
  0731 tokenizer and all 20 Safetensors shards and checks their index closure,
  topology, and quantization assignment;
- the separately pinned official `tokenizer_config.json`; and
- the fixed, absolute entrypoint and minimal `env -i` environment.

`promote_cutover.py` remains an explicit operator action. Before it can stop a
service it requires `--promote`, a detached SSH signature over a strict-schema
candidate receipt, a passing 8081 preflight/smoke, a separately hashed
production plist, the nonblocking GPU lock, and exact current launchd
label/PID/listener/plist identity. Candidate model IDs are taken only from the
signed receipt for cutover verification; the prior model IDs are used only to
verify rollback. The same lock remains held through restoration and the real
`/v1/models` plus exact `content.strip() == "READY"`/`finish_reason=stop` smoke.
After the new plist is bootstrapped, its hash, launchd PID, and ownership of the
8080 listener are reattested under the lock before any HTTP identity or readiness
probe is allowed.

Receipts have an exact allowlist and recursively reject local paths, request
content, tool schemas, secrets, argv/env, and captured process output.
