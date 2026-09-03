"""A compiled GDN run must not silently drop ragged-batch metadata.

``Qwen4ExpTextModel._compiled_run_fn`` hands each GDN layer a THROWAWAY
``ArraysCache(size=2)`` that carries neither ``lengths`` nor ``left_padding``,
and neither field is an input to the traced step.  ``GatedDeltaNet.__call__``
reads both OUTSIDE the mask -- the ragged conv-state write branches on
``cache.lengths``, and all three ``_fused_*_applies`` predicates refuse on it --
so inside a compiled run the layer would take a different branch from the eager
path on the same cache.  Not a decline: different arithmetic, silently.  That is
The cache-identity failure mode, one field over.

Nothing reaches it today: ``_forward`` enters the compiled path only when
``create_ssm_mask(h, cache[self.ssm_idx])`` is None, and ``make_mask`` returns
an array whenever either field is set on that entry.  The producers set the
fields uniformly across the cache list, so the ssm entry stands in for all of
them -- a coincidence, not a contract.  These tests pin the guard that makes it
a contract, driven through the REAL ``_decode_layers_compiled`` and the REAL
``_compiled_run_fn``:

1. the plain (metadata-free) path is unchanged and still runs compiled;
2. ``lengths`` or ``left_padding`` on an entry a run would re-wrap raises,
   naming the layer and the field;
3. metadata on a layer the compiled path runs EAGERLY (a PLE/attention layer,
   or a run that falls back for want of state) is not the hazard and does not
   raise;
4. an entry that is not the run's head is caught at trace time.

CPU-only: the default device is pinned to the CPU for the whole file, so no
Metal kernel is ever built.
"""

from __future__ import annotations

import pytest

import mlx.core as mx

from mtplx import cache_identity as ci


@pytest.fixture(autouse=True)
def _cpu_device():
    # set_default_device leaks into every later-collected module (pytest
    # imports share one process), so restore it on the way out.
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)


# --------------------------------------------------------------------------
# The predicate
# --------------------------------------------------------------------------


class _Entry:
    """Stand-in for a GDN cache entry: two state slots plus the metadata."""

    def __init__(self, *, state=True, lengths=None, left_padding=None) -> None:
        self.cache = [mx.zeros((1, 2)), mx.zeros((1, 2))] if state else [None, None]
        self.lengths = lengths
        self.left_padding = left_padding

    def __getitem__(self, idx):
        return self.cache[idx]

    def __setitem__(self, idx, value):
        self.cache[idx] = value


def test_a_plain_entry_carries_no_ragged_fields():
    assert ci.ragged_metadata_fields(_Entry()) == ()
    assert ci.ragged_metadata_fields(None) == ()
    # An object with neither attribute (a lane's own container) is plain too.
    assert ci.ragged_metadata_fields(object()) == ()
    ci.assert_no_ragged_metadata(_Entry(), 0, label="unit")


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({"lengths": mx.array([3, 5])}, ("lengths",)),
        ({"left_padding": mx.array([0, 2])}, ("left_padding",)),
        (
            {"lengths": mx.array([3, 5]), "left_padding": mx.array([0, 2])},
            ("lengths", "left_padding"),
        ),
    ],
)
def test_ragged_fields_are_reported_in_field_order(kwargs, expected):
    assert ci.ragged_metadata_fields(_Entry(**kwargs)) == expected


def test_the_assertion_names_the_layer_and_the_field():
    entry = _Entry(left_padding=mx.array([0, 2]))
    with pytest.raises(ci.RaggedCacheInCompiledRunError, match="layer 7") as excinfo:
        ci.assert_no_ragged_metadata(entry, 7, label="unit run")
    message = str(excinfo.value)
    assert "left_padding" in message
    assert "unit run" in message
    # A sibling of the error, so an existing handler still catches it.
    assert isinstance(excinfo.value, ci.CacheIdentityContractError)


# --------------------------------------------------------------------------
# The real call site: _decode_layers_compiled over fake layers
# --------------------------------------------------------------------------


class _Layer:
    """A GDN-shaped layer: writes both state slots, adds one to the stream."""

    def __init__(self, index: int, *, linear: bool = True, ple: bool = False) -> None:
        self.index = index
        self.is_linear = linear
        self._ple = ple

    def __contains__(self, name: str) -> bool:  # `"ple" in layer`
        return name == "ple" and self._ple

    def __call__(self, h, *, input_ids, ssm_mask, cache):
        # A real GDN layer seeds an empty slot rather than failing; the eager
        # fallback below is exactly the path that hands it empty state.
        zero = mx.zeros((1, 2))
        cache[0] = zero if cache[0] is None else cache[0] + 1
        cache[1] = zero if cache[1] is None else cache[1] + 1
        return h + 1


class _Model:
    """The three methods ``_decode_layers_compiled`` reaches through ``self``."""

    def __init__(self, layers):
        from mtplx.models import qwen4_exp as qm

        self.layers = layers
        self._decode_runs = None
        self._decode_run_fns = {}
        self._cls = qm.Qwen4ExpTextModel

    def _build_decode_runs(self):
        return self._cls._build_decode_runs(self)

    def _compiled_run_fn(self, idxs, capture: bool = False):
        return self._cls._compiled_run_fn(self, idxs, capture=capture)

    def _get_run_fn(self, idxs, capture: bool):
        return self._cls._get_run_fn(self, idxs, capture)

    def decode(self, cache):
        return self._cls._decode_layers_compiled(
            self, mx.zeros((1, 2)), None, cache
        )


def _drive(cache, layers=None, monkeypatch=None):
    """Run the REAL ``_decode_layers_compiled`` over ``cache``."""

    model = _Model(layers if layers is not None else [_Layer(i) for i in range(3)])
    out = model.decode(cache)
    mx.eval(out)
    return out


def test_the_plain_path_runs_compiled_and_is_unchanged(monkeypatch):
    cache = [_Entry() for _ in range(3)]
    out = _drive(cache, monkeypatch=monkeypatch)
    # One run over all three layers: the stream saw every layer...
    assert out.tolist() == [[3.0, 3.0]]
    # ...and every layer's state came back out of the compiled outputs.
    for entry in cache:
        assert entry[0].tolist() == [[1.0, 1.0]]
        assert entry[1].tolist() == [[1.0, 1.0]]


@pytest.mark.parametrize("field", ["lengths", "left_padding"])
def test_ragged_metadata_on_the_head_entry_raises(monkeypatch, field):
    cache = [_Entry() for _ in range(3)]
    setattr(cache[0], field, mx.array([2, 4]))
    with pytest.raises(ci.RaggedCacheInCompiledRunError, match="layer 0") as excinfo:
        _drive(cache, monkeypatch=monkeypatch)
    assert field in str(excinfo.value)


@pytest.mark.parametrize("layer_index", [1, 2])
def test_ragged_metadata_deeper_in_the_run_is_caught_at_trace_time(
    monkeypatch, layer_index
):
    """The head entry is clean, so only the traced body can see this one."""

    cache = [_Entry() for _ in range(3)]
    cache[layer_index].lengths = mx.array([2, 4])
    with pytest.raises(
        ci.RaggedCacheInCompiledRunError, match=f"layer {layer_index}"
    ) as excinfo:
        _drive(cache, monkeypatch=monkeypatch)
    assert "trace" in str(excinfo.value)


def test_metadata_on_an_eagerly_run_layer_does_not_raise(monkeypatch):
    """A PLE / attention layer is handed the REAL entry, so nothing is dropped."""

    layers = [_Layer(0), _Layer(1, ple=True), _Layer(2)]
    cache = [_Entry() for _ in range(3)]
    cache[1].lengths = mx.array([2, 4])
    out = _drive(cache, layers=layers, monkeypatch=monkeypatch)
    assert out.tolist() == [[3.0, 3.0]]
    # The eager layer still ran, with its own container.
    assert cache[1][0].tolist() == [[1.0, 1.0]]


def test_a_run_that_falls_back_to_eager_does_not_raise(monkeypatch):
    """No state -> the run is executed layer by layer on the real entries."""

    cache = [_Entry() for _ in range(3)]
    cache[0].lengths = mx.array([2, 4])
    cache[2].cache = [None, None]
    out = _drive(cache, monkeypatch=monkeypatch)
    mx.eval(out)
    assert out.tolist() == [[3.0, 3.0]]


def test_the_guard_is_off_the_plain_path_entirely(monkeypatch):
    """An entry class with no metadata attributes at all still runs compiled."""

    class _Bare:
        def __init__(self):
            self.cache = [mx.zeros((1, 2)), mx.zeros((1, 2))]

        def __getitem__(self, idx):
            return self.cache[idx]

        def __setitem__(self, idx, value):
            self.cache[idx] = value

    cache = [_Bare() for _ in range(3)]
    out = _drive(cache, monkeypatch=monkeypatch)
    assert out.tolist() == [[3.0, 3.0]]


# --------------------------------------------------------------------------
# The gate that keeps it unreachable today
# --------------------------------------------------------------------------


def test_make_mask_is_what_keeps_the_compiled_path_off_a_ragged_cache():
    """``_forward``'s entry condition, pinned as the reason this cannot fire.

    ``create_ssm_mask`` returns ``cache.make_mask(S)``, and ``ArraysCache``
    returns an array from either field -- so a ragged ssm entry makes
    ``ssm_mask`` non-None and ``_forward`` takes the eager path.  If this ever
    stops holding, the guard above is the only thing left.
    """

    from mlx_lm.models.base import create_ssm_mask
    from mlx_lm.models.cache import ArraysCache

    h = mx.zeros((2, 1, 4))

    plain = ArraysCache(size=2)
    assert create_ssm_mask(h, plain) is None

    with_lengths = ArraysCache(size=2)
    with_lengths.prepare(lengths=[3, 5])
    assert create_ssm_mask(h, with_lengths) is not None

    with_padding = ArraysCache(size=2, left_padding=[0, 2])
    assert create_ssm_mask(h, with_padding) is not None


@pytest.mark.parametrize("batch", [1, 3])
def test_merging_empty_arrays_caches_seeds_left_padding(batch):
    """The live producer: mlx-lm's all-empty merge (ar_batch admission).

    ``ArraysCache.merge`` stamps ``left_padding = [0] * B`` on every merged
    all-empty cache and nothing clears it -- at B=1 too -- so the AR batch
    lane decodes with the metadata live, ``create_ssm_mask`` returns an array
    every step, and the compiled GDN path (with every ``mask is None`` fused
    kernel) is off for that batch.  Which is exactly why the ssm-mask gate,
    not the absence of producers, is what keeps the compiled path clean.
    """

    from mlx_lm.models.cache import ArraysCache

    merged = ArraysCache.merge([ArraysCache(size=2) for _ in range(batch)])
    assert merged.left_padding is not None
    assert ci.ragged_metadata_fields(merged) == ("left_padding",)
    assert merged.make_mask(1) is not None
