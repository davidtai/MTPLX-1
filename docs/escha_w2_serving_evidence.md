# Escha-W2 serving evidence (real hardware)

`EschaLabs/Qwen3.6-35B-A3B-Escha-W2` (2-bit A3B) served through the **real mtplx runtime** —
`mtplx.runtime.load(path, mtp=True)` → `MTPLXRuntime.forward_ar` → serving decode — on an M5 Max
(128 GB), not a bench harness. Recommended serving path is **autoregressive (AR)**; see the MTP note
at the end. Captured 2026-08-05.

## Load + validate

```
mtplx.runtime.load(escha, mtp=True) -> MTPLXRuntime
  load: 5.2 s   resident: 12.4 GiB   mtp_enabled: True (validate passed)
```

## Programming prompt → output (the model works)

Prompt (chat template, thinking off): *"Write a Python function `is_palindrome(s)` … ignoring case,
spaces, and punctuation. Include a docstring and two example asserts."* Output (94 tok, clean EOS):

```python
def is_palindrome(s):
    """Returns True if the string is a palindrome, ignoring case, spaces, and punctuation."""
    cleaned = ''.join(char.lower() for char in s if char.isalnum())
    return cleaned == cleaned[::-1]

assert is_palindrome("A man, a plan, a canal: Panama") == True
assert is_palindrome("racecar") == True
```

The function is correct (both assertions hold).

## Decode throughput (AR, 1024 ctx)

The 2-bit eschamoe decode is host-encode-bound at batch=1, so two lossless levers matter:
`ESCHA_COMPILE` (mx.compile the decode kernel chain) and `MTPLX_AR_ASYNC_PIPELINE` (overlap the
next token's host encode with the current GPU step).

| config | decode tok/s | note |
| --- | --- | --- |
| sync, eager | ~49 | baseline |
| + async pipeline | ~56 | overlap host encode |
| + `mx.compile` decode | **~58** | bit-identical to eager (same output SHA) |

Compute is **bf16** (the checkpoint's native dtype); the earlier fp32 round-trip was a wasteful cast
with identical greedy output. int8 non-experts use the fused matvec at decode (no dequant) and a
dequant→GEMM at prefill (M>32).

## Prefill tok/s, decode tok/s, memory vs context (chunked prefill)

Prefill is chunked (2048-tok blocks, full cache-state eval per block) so activation memory stays
bounded with context — a single whole-context forward is not:

| ctx | prefill tok/s | decode tok/s | peak mem |
| --- | --- | --- | --- |
| 1024 | ~432 | ~58 | 16.2 GiB |
| 16384 | ~445 | ~56 | 20.7 GiB |
| 32768 | ~427 | ~53 | 21.7 GiB |

Memory is flat across context (20.7→21.7 GiB from 16k→32k) — the model resident (12.4 GiB) plus a
small, bounded prefill/KV working set.

## MTP status (experimental; AR is the default)

The native MTP draft head binds and drafts (`validate` passes; ~short-context spec-decode is
bit-exact). MTP **speculative decode is not the recommended serving path yet**: at long context its
verify uses a different `scaled_dot_product_attention` kernel (masked, L≥2) than single-token decode
(`mask=None`, L=1), and those two kernels are not bit-identical in bf16 — an occasional near-tie
greedy flip makes spec-decode output diverge from greedy AR. Root cause and fix (an L-invariant
attention path for the escha trunk + capture-based accept-prefix rewind) are identified for a
follow-up. Until then, **serve AR** (lossless, ~58 tok/s).

## Reproduce (AR)

```python
import mtplx.runtime as R
rt = R.load("<Escha-W2 path>", mtp=True)          # mtp_enabled=True
from mlx_lm.generate import generate_step         # async-pipelined AR decode
# or the runtime serving path with MTPLX_AR_ASYNC_PIPELINE=1
```
