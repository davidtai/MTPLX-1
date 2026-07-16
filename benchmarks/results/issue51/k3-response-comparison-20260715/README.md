# Issue 51 K3 response comparison

Both files contain the exact decoded 1,028-token K3 output, including special tokens.

- [Exact original 1,024-token prompt](./original-prompt.txt)
- [Prompt builder metadata and token hash](./prompt-metadata.json)
- [B0 stock-router response](./stock-response.txt)
- [C0 authoritative-MPP response](./mpp-response.txt)
- [Side-by-side HTML view](./comparison.html)
- [Machine-readable summary](./summary.json)

The reconstructed prompt token SHA-256 is
`2a269f7f7b17b10fdca1bd72d61fa7b50f329744ac2023da1086f6ac13a44829`,
an exact match for both benchmark artifacts.

## Headline comparison

| Property | B0 stock router | C0 authoritative MPP + #58 |
|---|---:|---:|
| Output tokens | 1,028 | 1,028 |
| Characters | 4,453 | 4,171 |
| Lines | 103 | 137 |
| Common token prefix | 3 tokens | 3 tokens |
| First divergent token | index 3: `#` | index 3: `import` |
| EOS positions | 500, 1,010 | none |
| Python blocks | 4 | 1 |
| Python syntax | 3 pass, 1 fail | 1 fail |

The benchmark intentionally disabled stop-token termination to force exactly 1,028 output tokens. The stock response would normally stop at token 500; everything after that first EOS is forced continuation. The MPP response never emitted EOS and ends at the hard token limit.

## Prompt-aware interpretation

The final request asks for one code-only Python file implementing ten ordered
sections: prompt loading, validation, an LRU cache, metrics, records, sampling,
an event log, a run registry, a CLI, and a self-test. The authoritative-MPP
response follows that requested structure directly and does not loop, although
it remains incomplete and contains syntax errors at the hard token cutoff.

The stock response instead latches onto the filler repository's `JobStore`,
adds prose despite the code-only instruction, emits EOS at token 500, and then
repeats the same answer because the throughput benchmark deliberately ignores
stop tokens. On instruction following and repetition behavior, the MPP response
is the better of these two samples.

## Source artifacts

- B0: `/tmp/issue51-b0-rebased-device-k-k0to7-c1024-o1028-r1.json`
- C0: `/tmp/issue51-c0-router58-devicek-k0to7-c1024-o1028-r1.json`
