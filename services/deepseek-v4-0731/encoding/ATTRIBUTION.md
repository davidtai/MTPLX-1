# DeepSeek 0731 encoding attribution

This directory reserves the pinned chat-encoding asset slot for the isolated
`deepseek-v4-0731` candidate.  The requested official DeepSeek 0731 source
revision is
`7872f01b1d1fe23eabc4c98b48bffcef5a386062`.

Intended upstream attribution: DeepSeek AI, DeepSeek-V3.1,
`assets/chat_template.jinja`.  At implementation time that exact revision did
not resolve from the public upstream history, so `chat_template.jinja` is a
minimal review fixture, **not a claim of a byte-for-byte retrieved upstream
file**.  Do not promote this service until an operator replaces it with the
retrieved official bytes and updates this note plus `SHA256SUMS` in the same
reviewed commit.

The asset is retained locally for review and reproducibility; it is not fetched
at service start.  `SHA256SUMS` is the authority at installation time.  A
missing, symlinked, malformed, or mismatched asset is a hard error, never a
fallback to a local tokenizer template.

The small renderer is intentionally kept separate from MTPLX's existing chat
template routes until this candidate has a promotion receipt.
