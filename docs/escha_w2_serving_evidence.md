# Escha-W2 serving evidence (real hardware)

`EschaLabs/Qwen3.6-35B-A3B-Escha-W2` (2-bit A3B) served through the **real mtplx runtime** —
`mtplx.runtime.load(path, mtp=True)` → `MTPLXRuntime.forward_ar` → `batched_decode` speculative
decode — on an M5 Max (128 GB), not a bench harness. Captured 2026-08-05.

## Load + validate

```
mtplx.runtime.load(escha, mtp=True) -> MTPLXRuntime
  load: 5.5 s
  resident: 12.5 GiB
  mtp_enabled: True   (validate_mtp_support passed — native MTP draft head bound)
```

## Programming prompt → output (proof the LLM works)

**Prompt** (chat template, thinking off, 60 tokens):

> Write a Python function `is_palindrome(s)` that returns True if the string is a palindrome,
> ignoring case, spaces, and punctuation. Include a short docstring and two example assert
> statements. Answer with the code directly, no explanation.

**Generated output** (94 tokens, `finish_reason=stop` — clean EOS):

```python
def is_palindrome(s):
    """
    Returns True if the string is a palindrome, ignoring case, spaces, and punctuation.
    """
    cleaned = ''.join(char.lower() for char in s if char.isalnum())
    return cleaned == cleaned[::-1]

assert is_palindrome("A man, a plan, a canal: Panama") == True
assert is_palindrome("racecar") == True
```

The generated function is correct (both example assertions hold).

## Speculative decode is lossless

The same prompt decoded two ways through `generate_greedy_batched`:

| mode                         | decode tok/s | output sha         |
| ---------------------------- | ------------ | ------------------ |
| plain autoregressive (AR)    | 47.2         | `2b8d0698d2db430a` |
| MTP speculative (depth-2)    | 44.1         | `2b8d0698d2db430a` |

Identical output SHA → the MTP draft/verify path is **bit-for-bit lossless** vs greedy AR.
Depth-2 acceptance on this run: 23/48 all-accept cycles (47.9%), 73 verify+repair forwards for 94
committed tokens. (Speculative wins on latency-bound single-stream shapes with higher acceptance;
here it tracks AR because this short single stream is dominated by the per-cycle repair forward. The
serving win banked separately is the int8 non-expert matvec: AR decode 25.1 → 59.6 tok/s.)

## Reproduce

```python
import mtplx.runtime as R
rt = R.load("<EschaLabs/Qwen3.6-35B-A3B-Escha-W2 path>", mtp=True)
assert rt.mtp_enabled
from mtplx.batched_decode import generate_greedy_batched
res = generate_greedy_batched(rt, [prompt_ids], max_new_tokens=220, use_mtp_draft=True, decode_mode="spec")
print(res.aggregate_decode_tokps, res.streams[0].sha)
```
