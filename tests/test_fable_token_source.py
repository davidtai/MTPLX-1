"""Per-token provenance: the recorder, its wiring, and the receipt it makes.

No MLX import and no model load.  The recorder is pure Python plus NumPy, so
a stub accept loop that calls the SAME methods ``generate_mtpk`` calls -- in
the same order, at the same points -- exercises the real assignment logic; the
wiring itself is guarded by an AST tripwire over ``mtplx/generation.py``, which
is how the rest of this lane is checked (see ``tests/test_fable_k20_log.py``).
"""

from __future__ import annotations

import ast
import base64
import importlib.util
import json
import sys
import unittest
from pathlib import Path

import numpy as np

from mtplx import fable_token_source as ts


ROOT = Path(__file__).resolve().parents[1]


def _load_driver_module():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    path = ROOT / "scripts" / "fable" / "abba_driver.py"
    spec = importlib.util.spec_from_file_location("_fable_abba_driver_ts", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


driver = _load_driver_module()


def stub_accept_loop(recorder: ts.TokenSourceLog, script) -> list[int]:
    """``generate_mtpk``'s emission sites, in the order the real loop hits them.

    Every branch here mirrors one place in ``generate_mtpk`` that mutates
    ``tokens``; the mapping is the table in ``mtplx/fable_token_source.py``.
    A cycle that ended on a bonus or a correction leaves that token as the
    next cycle's ``pending_primary``, so the next cycle emits NO fresh
    primary -- the scripts below respect that, because a recorder that
    double-counted primaries would still look right on a script that did not.
    """

    tokens: list[int] = []
    recorder.begin_request()
    for step in script:
        kind = step["kind"]
        if kind == "primary":
            # tokens.append(primary)
            tokens.append(int(step["token"]))
            recorder.primary()
        elif kind == "copy":
            # tokens.extend(_cc_acc) / tokens.extend(_cb_acc)
            accepted = [int(token) for token in step["accepted"]]
            tokens.extend(accepted)
            recorder.copy_block(len(accepted))
            if step.get("correction") is not None:
                # tokens.append(int(_cc_correction))
                tokens.append(int(step["correction"]))
                recorder.copy_correction()
        elif kind == "all_accept":
            # tokens.extend(draft_tokens)
            drafts = [int(token) for token in step["drafts"]]
            tokens.extend(drafts)
            recorder.draft_run(len(drafts))
            if step.get("bonus") is not None:
                # tokens.append(bonus)
                tokens.append(int(step["bonus"]))
                recorder.bonus()
        elif kind == "reject":
            # tokens.extend(committed[1:]) at a rejection boundary
            accepted = [int(token) for token in step["accepted"]]
            correction = step.get("correction")
            tokens.extend(accepted)
            if correction is not None:
                tokens.append(int(correction))
            recorder.mtp_commit(
                len(accepted), correction=correction is not None
            )
        else:  # pragma: no cover - a typo in a script, not a behaviour
            raise AssertionError(f"unknown stub step {kind!r}")
    return tokens


#: One request that touches every lane: a fresh primary, an all-accept block
#: with its bonus, a context-copy round that ends on a correction, a partial
#: accept that ends on a correction, and a total rejection.
FULL_SCRIPT = (
    {"kind": "primary", "token": 900},
    {"kind": "all_accept", "drafts": [901, 902, 903], "bonus": 904},
    # bonus was the pending primary -> no fresh primary this cycle
    {"kind": "copy", "accepted": [905, 906, 907, 908], "correction": 909},
    # copy correction was the pending primary -> no fresh primary
    {"kind": "reject", "accepted": [910, 911], "correction": 912},
    # correction was the pending primary -> no fresh primary
    {"kind": "reject", "accepted": [], "correction": 913},
)

D1 = ts.SOURCE_DRAFT_BASE + 1
D2 = ts.SOURCE_DRAFT_BASE + 2
D3 = ts.SOURCE_DRAFT_BASE + 3

EXPECTED_SOURCES = [
    ts.SOURCE_PRIMARY,
    D1, D2, D3,
    ts.SOURCE_BONUS,
    ts.SOURCE_COPY, ts.SOURCE_COPY, ts.SOURCE_COPY, ts.SOURCE_COPY,
    ts.SOURCE_COPY_CORRECTION,
    D1, D2,
    ts.SOURCE_CORRECTION,
    ts.SOURCE_CORRECTION,
]


class TestSourceAssignment(unittest.TestCase):
    def test_every_lane_lands_on_its_own_token(self):
        recorder = ts.TokenSourceLog(enabled=True)
        tokens = stub_accept_loop(recorder, FULL_SCRIPT)
        self.assertEqual(len(tokens), len(EXPECTED_SOURCES))
        sources = recorder.expand(len(tokens))
        self.assertEqual(list(sources), EXPECTED_SOURCES)
        self.assertEqual(recorder.recorded_tokens, len(tokens))

    def test_copy_lane_commit_and_its_correction_are_distinguishable(self):
        recorder = ts.TokenSourceLog(enabled=True)
        tokens = stub_accept_loop(
            recorder,
            [
                {"kind": "primary", "token": 1},
                {"kind": "copy", "accepted": [2, 3, 4], "correction": 5},
            ],
        )
        sources = list(recorder.expand(len(tokens)))
        self.assertEqual(
            sources,
            [
                ts.SOURCE_PRIMARY,
                ts.SOURCE_COPY,
                ts.SOURCE_COPY,
                ts.SOURCE_COPY,
                ts.SOURCE_COPY_CORRECTION,
            ],
        )
        # A copy-lane correction must NOT read as an MTP correction: they come
        # from different verifies and the whole point of the column is telling
        # them apart.
        self.assertNotIn(ts.SOURCE_CORRECTION, sources)

    def test_correction_after_a_partial_accept(self):
        recorder = ts.TokenSourceLog(enabled=True)
        tokens = stub_accept_loop(
            recorder,
            [
                {"kind": "primary", "token": 1},
                {"kind": "reject", "accepted": [2], "correction": 3},
            ],
        )
        self.assertEqual(
            list(recorder.expand(len(tokens))),
            [ts.SOURCE_PRIMARY, D1, ts.SOURCE_CORRECTION],
        )

    def test_draft_depths_are_readable_off_the_code(self):
        recorder = ts.TokenSourceLog(enabled=True)
        recorder.begin_request()
        recorder.draft_run(4)
        codes = list(recorder.expand(4))
        self.assertEqual(
            [code - ts.SOURCE_DRAFT_BASE for code in codes], [1, 2, 3, 4]
        )
        self.assertEqual(ts.source_name(codes[2]), "draft_d3")

    def test_a_stop_trim_clips_the_spans_and_stays_complete(self):
        # Every trim generate_mtpk performs is a PREFIX trim, so a recorder
        # that recorded more than survived is correct, not broken.
        recorder = ts.TokenSourceLog(enabled=True)
        tokens = stub_accept_loop(recorder, FULL_SCRIPT)
        trimmed = tokens[:10]
        sources = recorder.expand(len(trimmed))
        self.assertEqual(list(sources), EXPECTED_SOURCES[:10])
        receipt = recorder.receipt(trimmed)
        self.assertTrue(receipt["complete"])
        self.assertEqual(receipt["recorded_tokens"], len(tokens))
        self.assertEqual(receipt["tokens"], 10)

    def test_an_uninstrumented_lane_reads_unknown_not_invented(self):
        recorder = ts.TokenSourceLog(enabled=True)
        recorder.begin_request()
        recorder.primary()
        receipt = recorder.receipt([1, 2, 3])
        _, sources = ts.decode_receipt(receipt)
        self.assertEqual(
            list(sources),
            [ts.SOURCE_PRIMARY, ts.SOURCE_UNKNOWN, ts.SOURCE_UNKNOWN],
        )
        self.assertFalse(receipt["complete"])

    def test_disabled_recorder_costs_nothing_and_says_so(self):
        recorder = ts.TokenSourceLog(enabled=False)
        tokens = stub_accept_loop(recorder, FULL_SCRIPT)
        self.assertEqual(recorder.recorded_tokens, 0)
        self.assertEqual(recorder.spans, ())
        receipt = recorder.receipt(tokens)
        self.assertFalse(receipt["available"])
        self.assertFalse(receipt["complete"])
        # The ids are still real: only the source column is unknown.
        ids, _ = ts.decode_receipt(receipt)
        self.assertEqual(list(ids), tokens)

    def test_begin_request_does_not_leak_across_requests(self):
        recorder = ts.TokenSourceLog(enabled=True)
        stub_accept_loop(recorder, FULL_SCRIPT)
        second = stub_accept_loop(
            recorder, [{"kind": "primary", "token": 7}]
        )
        self.assertEqual(recorder.recorded_tokens, 1)
        self.assertEqual(
            list(recorder.expand(len(second))), [ts.SOURCE_PRIMARY]
        )

    def test_depth_beyond_the_uint8_encoding_is_refused(self):
        recorder = ts.TokenSourceLog(enabled=True)
        recorder.begin_request()
        with self.assertRaises(ValueError):
            recorder.draft_run(ts.MAX_DRAFT_DEPTH + 1)


class TestReceiptRoundTrip(unittest.TestCase):
    def test_ids_and_sources_survive_json(self):
        recorder = ts.TokenSourceLog(enabled=True)
        tokens = stub_accept_loop(recorder, FULL_SCRIPT)
        receipt = json.loads(json.dumps(recorder.receipt(tokens)))
        ids, sources = ts.decode_receipt(receipt)
        self.assertEqual(list(ids), tokens)
        self.assertEqual(list(sources), EXPECTED_SOURCES)
        self.assertEqual(receipt["schema"], ts.SCHEMA)
        self.assertTrue(receipt["available"])

    def test_digest_is_over_the_raw_uint32_bytes(self):
        recorder = ts.TokenSourceLog(enabled=True)
        tokens = stub_accept_loop(recorder, FULL_SCRIPT)
        receipt = recorder.receipt(tokens)
        self.assertEqual(receipt["output_ids_sha256"], ts.sha256_ids(tokens))
        # Reproducible from the base64 alone -- no textual formatting of the
        # ids is in the loop, so a reader cannot get a different answer.
        import hashlib

        raw = base64.b64decode(receipt["output_ids_b64"])
        self.assertEqual(
            hashlib.sha256(raw).hexdigest(), receipt["output_ids_sha256"]
        )
        self.assertEqual(len(raw), 4 * len(tokens))

    def test_a_single_changed_token_changes_the_digest(self):
        # This is the property the 600-char head/tail hash did not have.
        recorder = ts.TokenSourceLog(enabled=True)
        tokens = stub_accept_loop(recorder, FULL_SCRIPT)
        other = list(tokens)
        other[len(other) // 2] += 1
        self.assertNotEqual(ts.sha256_ids(tokens), ts.sha256_ids(other))

    def test_large_ids_survive_the_uint32_encoding(self):
        recorder = ts.TokenSourceLog(enabled=True)
        recorder.begin_request()
        recorder.draft_run(2)
        tokens = [151_665, 4_294_967_295]
        ids, _ = ts.decode_receipt(recorder.receipt(tokens))
        self.assertEqual(list(ids), tokens)

    def test_counts_add_up_to_the_stream(self):
        recorder = ts.TokenSourceLog(enabled=True)
        tokens = stub_accept_loop(recorder, FULL_SCRIPT)
        receipt = recorder.receipt(tokens)
        self.assertEqual(sum(receipt["counts"].values()), len(tokens))
        self.assertEqual(receipt["counts"]["copy"], 4)
        self.assertEqual(receipt["counts"]["copy_correction"], 1)
        self.assertEqual(receipt["counts"]["correction"], 2)
        self.assertEqual(receipt["counts"]["bonus"], 1)
        self.assertEqual(receipt["counts"]["primary"], 1)
        self.assertEqual(receipt["counts"]["draft_d1"], 2)


class TestDriverReceiptBlock(unittest.TestCase):
    """``abba_driver.token_provenance`` -- what actually lands on the receipt."""

    def setUp(self):
        # addCleanup, not tearDown: the recorder is a process-global, and a
        # leaked `enabled = True` would arm it for every later test in the
        # run.  Registered before the mutation so it fires even if setUp
        # itself raises after this point.
        previous, spans, recorded = (
            ts.token_source.enabled,
            list(ts.token_source.spans),
            ts.token_source.recorded_tokens,
        )

        def restore():
            ts.token_source.enabled = previous
            ts.token_source._spans = spans
            ts.token_source._recorded = recorded

        self.addCleanup(restore)
        ts.token_source.enabled = True

    def test_block_carries_ids_digest_and_sources(self):
        tokens = stub_accept_loop(ts.token_source, FULL_SCRIPT)
        block = json.loads(json.dumps(driver.token_provenance(tokens)))
        self.assertEqual(
            sorted(block),
            ["output_ids_b64", "output_ids_sha256", "token_sources"],
        )
        self.assertEqual(block["output_ids_sha256"], ts.sha256_ids(tokens))
        # The ids are NOT duplicated inside the nested block.
        self.assertNotIn("output_ids_b64", block["token_sources"])
        ids, sources = ts.decode_receipt(
            {
                "output_ids_b64": block["output_ids_b64"],
                "token_sources_b64": block["token_sources"][
                    "token_sources_b64"
                ],
            }
        )
        self.assertEqual(list(ids), tokens)
        self.assertEqual(list(sources), EXPECTED_SOURCES)
        self.assertTrue(block["token_sources"]["available"])
        self.assertTrue(block["token_sources"]["complete"])

    def test_receipt_size_for_a_1024_token_arm(self):
        recorder = ts.token_source
        recorder.begin_request()
        recorder.draft_run(3)
        tokens = list(range(1024))
        block = driver.token_provenance(tokens)
        added = len(json.dumps(block))
        # ~7 KiB: the ids at 4 bytes each and one byte of provenance per
        # token, both base64.  Small enough that this is never a reason to
        # drop the evidence.
        self.assertLess(added, 9 * 1024)
        self.assertGreater(added, 6 * 1024)


class TestGenerationWiring(unittest.TestCase):
    """Every ``tokens`` mutation in ``generate_mtpk`` must be recorded.

    A new emission site added without a recorder call is exactly the failure
    the K20 log hit on ``worker/w51b-shadow-segments`` (00ac2690): the
    context-copy rounds committed tokens the log never saw, and the gap read
    as a request boundary.  This test is the tripwire for the same mistake
    here, so a future lane cannot silently produce ``unknown`` tokens.
    """

    GENERATION = ROOT / "mtplx" / "generation.py"

    @classmethod
    def setUpClass(cls):
        cls.tree = ast.parse(cls.GENERATION.read_text())
        cls.function = next(
            node
            for node in ast.walk(cls.tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == "generate_mtpk"
        )

    @staticmethod
    def _is_token_mutation(node):
        # `tokens += [...]` grows the list exactly like `.extend` and would
        # otherwise slip past this tripwire.
        if isinstance(node, ast.AugAssign):
            return (
                isinstance(node.target, ast.Name)
                and node.target.id == "tokens"
            )
        if not isinstance(node, ast.Expr):
            return False
        call = node.value
        if not isinstance(call, ast.Call):
            return False
        func = call.func
        return (
            isinstance(func, ast.Attribute)
            and func.attr in {"append", "extend", "insert"}
            and isinstance(func.value, ast.Name)
            and func.value.id == "tokens"
        )

    @staticmethod
    def _guards_the_recorder(node):
        if not isinstance(node, ast.If):
            return False
        if not (
            isinstance(node.test, ast.Name)
            and node.test.id == "_track_token_sources"
        ):
            return False
        return any(
            isinstance(inner, ast.Attribute)
            and isinstance(inner.value, ast.Name)
            and inner.value.id == "_fable_token_source"
            for inner in ast.walk(node)
        )

    def test_every_emission_site_records_a_source(self):
        unguarded = []
        for parent in ast.walk(self.function):
            for field in ("body", "orelse", "finalbody"):
                body = getattr(parent, field, None)
                if not isinstance(body, list):
                    continue
                for index, statement in enumerate(body):
                    if not self._is_token_mutation(statement):
                        continue
                    following = body[index + 1 : index + 4]
                    if not any(
                        self._guards_the_recorder(node) for node in following
                    ):
                        unguarded.append(statement.lineno)
        self.assertEqual(
            sorted(set(unguarded)),
            [],
            "mtplx/generation.py mutates `tokens` at these lines without a "
            "`_fable_token_source` call within three statements; those tokens "
            "would land in a receipt as source `unknown`",
        )

    def test_the_recorder_is_reset_per_request(self):
        source = self.GENERATION.read_text()
        self.assertIn("_fable_token_source.begin_request()", source)
        self.assertIn(
            "_track_token_sources = bool(_fable_token_source.enabled)", source
        )

    def test_all_three_stop_trims_are_prefix_trims(self):
        # The span model is only correct because every trim keeps a PREFIX.
        # If one of these ever becomes a middle-splice the expansion silently
        # misattributes everything after it.
        trims = [
            node
            for node in ast.walk(self.function)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "tokens"
                for target in node.targets
            )
        ]
        self.assertTrue(trims)
        for node in trims:
            value = node.value
            if isinstance(value, ast.Call):
                self.assertEqual(
                    getattr(value.func, "id", None), "_truncate_after_first_stop"
                )
                continue
            self.assertIsInstance(value, ast.Subscript)
            self.assertIsInstance(value.slice, ast.Slice)
            # tokens[:N] -- no lower bound, so the head always survives.
            self.assertIsNone(value.slice.lower)


class TestCodeTable(unittest.TestCase):
    def test_named_codes_do_not_collide_with_draft_depths(self):
        named = set(ts.SOURCE_NAMES)
        for depth in range(1, 8):
            self.assertNotIn(ts.draft_code(depth), named)

    def test_codes_fit_uint8(self):
        codes = [*ts.SOURCE_NAMES, ts.draft_code(ts.MAX_DRAFT_DEPTH)]
        self.assertTrue(all(0 <= code <= 255 for code in codes))
        self.assertEqual(np.uint8(ts.draft_code(3)), ts.draft_code(3))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
