# Issue #68: Hy3-Q2 expert layout/coalescing campaign

Status: **hold; do not integrate into decode**.

This is a construction-time layout and raw-consumption microbenchmark. It is
not the mandatory 1024-input/1024-output K0-K7 generation qualification. The
primary point is R=4 (MTP K=3). A ratio above 1.0 means the packed candidate is
faster than the fastest source-layout control selected in the same run.

## Actual Hy3-Q2 geometry

- Quantization: affine Q2, group size 64.
- Eight experts, one layer: 47,185,920 bytes loaded; 5,898,240 bytes/expert.
- Gate/up: weight `(1536, 256)` U32 and scale/bias `(1536, 64)` BF16 each.
- Down: weight `(4096, 96)` U32 and scale/bias `(4096, 24)` BF16 each.
- Packed group: four Q2 U32 words plus one U32 containing the exact BF16
  scale/bias pair. Source and packed steady-state byte counts are equal.

## Search and correctness

The campaign screened 30 candidates: output-packet and group-output-packet
layouts plus both tile-major orders at block sizes 16/32/64/128 and consumer
vector widths 1/2/4. It refined the six best screen results over R1-R8.

- 30/30 candidates: exact raw-record scan checksum and exact component-byte
  reconstruction.
- 6/6 finalists: exact stock expert output; source and reconstructed output SHA
  are both `1cb9eb86006622710aa5af59887f30660a1bed016b16ac5b53e22b574a80ef76`.
- Fixed `(slot, expert, generation)` replacement checks pass fail-closed.
- Resident-byte overhead and steady-state duplicate bytes: 0.
- Best candidate construction cost for eight experts: 11.802 ms mean.
- Analytical 32-lane/128-byte-line model for the best candidate: 536 source
  cache lines versus 15 packed cache lines per modeled transaction, and six
  versus three metadata loads per output group.

## Primary R=4 / MTP K=3 finalists

| Candidate | Source ms | Packed ms | Source/packed | Paired bootstrap 95% CI | Gate |
|---|---:|---:|---:|---:|---|
| `tiled_field_projection_b64_v1` | 0.445642 | 0.417775 | 1.0667x | [0.9609, 1.1835] | hold |
| `tiled_projection_field_b64_v1` | 0.402986 | 0.386206 | 1.0435x | [0.9673, 1.1306] | hold |
| `tiled_projection_field_b16_v1` | 0.422995 | 0.420422 | 1.0061x | [0.9048, 1.1235] | hold |
| `output_packet_flat_v2` | 0.421950 | 0.421778 | 1.0004x | [0.8954, 1.1160] | hold |
| `tiled_field_projection_b32_v1` | 0.417775 | 0.417631 | 1.0003x | [0.9138, 1.0937] | hold |
| `tiled_projection_field_b16_v4` | 0.413428 | 0.941172 | 0.4393x | [0.4160, 0.4655] | reject |

No finalist has a K3 interval wholly above 1.0. Wider vector consumers are
especially poor for this access pattern; V4's extra indexing and lane work more
than erase the theoretical coalescing benefit.

## Higher-repeat K3 confirmation

A second complete 30-candidate screen used ten paired screen repeats, then
refined eight finalists at K3 with 50 paired repeats, three warmups, and 20,000
paired bootstrap resamples. It independently selected the fastest source
control (`source_components_v4`) for that run.

| Candidate | Source ms | Packed ms | Source/packed | Paired bootstrap 95% CI | Gate |
|---|---:|---:|---:|---:|---|
| `tiled_field_projection_b32_v1` | 0.442036 | 0.433329 | 1.0201x | [0.9656, 1.0765] | hold |
| `output_packet_flat_v1` | 0.480697 | 0.482236 | 0.9968x | [0.9415, 1.0561] | hold |
| `output_packet_flat_v4` | 0.403458 | 0.405044 | 0.9961x | [0.9302, 1.0572] | hold |
| `tiled_projection_field_b32_v1` | 0.454313 | 0.469167 | 0.9683x | [0.9239, 1.0113] | hold |
| `tiled_projection_field_b128_v1` | 0.421043 | 0.444871 | 0.9464x | [0.8964, 0.9984] | reject |
| `tiled_field_projection_b16_v1` | 0.445099 | 0.470831 | 0.9453x | [0.9063, 0.9846] | reject |
| `tiled_projection_field_b64_v1` | 0.441179 | 0.475702 | 0.9274x | [0.8785, 0.9733] | reject |
| `group_output_packet_flat_v1` | 0.463588 | 0.500478 | 0.9263x | [0.8863, 0.9704] | reject |

The higher-powered confirmation narrows the best observed K3 mean gain from
6.7% to 2.0%, still with an interval crossing parity. All 30 screen candidates
and all eight finalists again passed byte, checksum, expert-output, generation,
and zero-overhead gates.

## R1-R8 scaling of the best refined candidate

Candidate: `tiled_field_projection_b64_v1`.

| MTP K | Rows | Source ms | Packed ms | Source/packed | Paired bootstrap 95% CI |
|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 0.505089 | 0.452389 | 1.1165x | [1.0306, 1.2020] |
| 1 | 2 | 0.323736 | 0.302180 | 1.0713x | [1.0045, 1.1398] |
| 2 | 3 | 0.375836 | 0.362103 | 1.0379x | [0.9765, 1.1096] |
| 3 | 4 | 0.445642 | 0.417775 | 1.0667x | [0.9609, 1.1835] |
| 4 | 5 | 0.458970 | 0.458128 | 1.0018x | [0.9369, 1.0722] |
| 5 | 6 | 0.485911 | 0.493850 | 0.9839x | [0.9123, 1.0594] |
| 6 | 7 | 0.537897 | 0.529514 | 1.0158x | [0.9431, 1.1019] |
| 7 | 8 | 0.586861 | 0.574639 | 1.0213x | [0.9620, 1.0832] |

The narrow R1/R2 result does not carry to the primary R4 point or larger row
groups. The raw scan appears dominated by dispatch/address/checksum work once
the source reads are already cache-resident, so fewer modeled transactions do
not translate into a robust K3 gain. This result does not measure the full QMM
and therefore cannot establish a decode-speed claim.

## Reproducibility

- Full artifact: `issue68-hy3-q2-layout-coalescing-20260715.json`
- Artifact SHA-256:
  `5d0e9016e78b17abdfeb97ab5957febb13c46e65b86323eff2756cf652ccd57e`
- High-repeat K3 artifact: `issue68-hy3-q2-layout-r4-confirm.json`
- High-repeat artifact SHA-256:
  `37df568364a28b18b5e48760468e0b278468311299f8733d5cb1c6bea48fd6d2`
- Manifest SHA-256:
  `086fcce170efdc1b6840bd1f9f1003e4bbebbe812641bd5a1a9c76c562cc4950`
- Sidecar SHA-256:
  `e62791b52c824efa2b50d7fb00607a031be62a31b330184bf6c91a27a4c1a1c9`
- Five paired screen repeats, 15 paired refinement repeats, two warmups, and
  10,000 paired bootstrap resamples.
- Exclusive lock receipt:
  `5eaac53024797b7852cc3a693b5aa8673fa74c6bfc08f3fe207bebd99e962328`.
- High-repeat lock receipt:
  `020998720ed7e92de73453bd72191be6a08e0a58b46624a5d36b3cbb8c7c8691`.
- `/tmp/mtplx-gpu-exclusive.lock` was held through Qwen restoration; before and
  after state matched exactly.

Decision: keep this as a separately benchmarkable experiment, but do not add a
runtime selector or spend the 1024/1024 K0-K7 qualification lane until a K3
microbenchmark interval clears parity. The full generation matrix remains a
required gate for any future promotable revision.
