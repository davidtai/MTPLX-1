# Table 5 — Long-context verdict at 255K

The 255K context is 261,120 tokens. The 262,144 context is not run: it is
above the knob. The knob is 107.374 GB. The table shows the peak memory at
255K for each arm. Every arm is an OOM at 255K. The cause is the QSA indexer
prefill transient (issue #393): the steady peak is near 100.79 GB, but a
command-buffer allocation in the prefill exceeds the GPU memory and the
request fails. The steady peak is below the knob; the transient is not. The
cell is void: it is not measured.

| arm | 255K steady peak GB | result |
| --- | ---: | --- |
| A: release 2.11.2 | 100.82 | void — OOM (prefill transient) |
| C: +#475 aux | 100.79 | void — OOM (prefill transient) |
| D: +#478 @0.2 | 100.79 | void — OOM (prefill transient) |
| E: +#478 @0.09 | 100.79 | void — OOM (prefill transient) |

## W1 and W2 bisect probes on C at 255K

W1 sets MTPLX_QSA_SCORE_TILE_ROWS=256 to tile the QSA score prefill. W2 turns
the session cache off and tries to set the bank to zero. Each probe is one
cold seed. W1 runs its other two seeds only if the first seed fits. Neither
probe fit. Both cells are void by the allocation-failure rule.

| probe | cell | steady peak GB | engaged? | classification |
| --- | --- | ---: | --- | --- |
| W1 | error after 3 tok | 100.79 | UNOBSERVABLE — reader is read-at-use (qwen4_exp.py:1599) but the tiled path emits no log or /health marker; inconclusive | void — OOM (exceeds knob, allocation-failure) |
| W2 | error after 3 tok | 100.79 | cache off = confirmed (/health enabled=False); bank 0 = REJECTED (invalid; fell back to 25769803776 bytes) | void — OOM (exceeds knob, allocation-failure) |

W1 is INCONCLUSIVE, not a negative: the tiler may have run, but nothing
observable proves it, and the cell OOM'd regardless. W2's cache-off half
engaged and the bank-zero half was rejected; the cell OOM'd. The verdict
stands: 255K is not measurable under the 107.374 GB knob on this pack.
