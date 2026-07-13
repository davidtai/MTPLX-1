# Project Map
_Generated: 2026-07-12 19:59 CDT | Git: fb9268a_

## Directory Structure
.github/ — CI, build, release, hygiene, and Apple-Silicon kernel validation workflows.
apps/MTPLXApp/ — Native macOS SwiftUI application, shared core, resources, and Swift tests.
benchmarks/ — Reproducible prompts plus curated benchmark summaries; raw run artifacts are ignored.
bin/ — Installed command shim used by packaged/local launch flows.
dashboard/ — React/Vite source for the browser dashboard embedded into the Python package.
docs/ — Architecture, operational guides, release notes, experimental designs, and implementation plans.
examples/ — Integration examples for OpenAI-compatible clients and local tools.
mtplx/ — Primary Python package: CLI, runtime, generation, serving, model adapters, kernels, and UI helpers.
native_extensions/ — Optional nanobind/C++/Metal extensions for expert I/O and fused verification work.
scripts/ — Benchmark, artifact, install, release, model-conversion, and validation entry points.
templates/ — Packaged integration templates.
tests/ — Python unit, contract, regression, and fixture-driven integration tests.
tools/ — Repository maintenance utilities.
vllm_metal/ — Vendored/adapted paged-attention Metal operators and Python build bindings.

## Key Files
pyproject.toml — Python package metadata, dependencies, console entry points, test configuration, and packaged assets.
mtplx/cli.py — Lightweight argparse surface; must remain usable for non-model commands without importing MLX.
mtplx/commands/public.py — Product-facing command handlers and lazy dispatch into model, server, benchmark, and integration flows.
mtplx/backends/registry.py — Compatibility inspection and architecture selection registry.
mtplx/runtime.py — Central model loading and runtime contract assembly; highest Python import fan-in.
mtplx/generation.py — Reference AR/MTP generation, cache restore/commit, session, and streamed-expert orchestration.
mtplx/server/openai.py — OpenAI/Anthropic-compatible server, streaming translation, sessions, metrics, and dashboard endpoints.
mtplx/engine_session.py — Long-lived engine/session lifecycle, concurrency, and request coordination.
mtplx/sampling.py — Exact sampling and backend-independent speculative verification semantics.
mtplx/cache_state.py — KV/cache ownership, paged-attention routing, snapshots, rollback, and long-context state.
mtplx/graphbank.py — Compiled verification graph eligibility, parity checks, and cache-state promotion.
mtplx/expert_manifest.py — Pinned streamed-expert artifact inventory, validation, sidecar building, and verification.
mtplx/expert_runtime.py — Streamed-expert admission, route lifecycle, I/O coordination, telemetry, and failure handling.
mtplx/expert_slots.py — Fixed-capacity expert slot banks, generations, leases, eviction, and cache simulation.
mtplx/expert_streaming.py — Memory planning, route policy, and bounded expert-streaming configuration.
mtplx/models/expert_mlx.py — Shared streamed-expert MLX execution primitives used by Hy3/GLM overlays.
scripts/benchmark_streamed_generation.py — Reproducible end-to-end streamed-MoE benchmark and saturation harness.
benchmarks/results/moe-runtime-gate-matrix.md — Current pass/fail evidence and retained performance/safety tradeoffs for the experimental runtime stack.
apps/MTPLXApp/Package.swift — Swift package targets and external app dependencies.
dashboard/package.json — Dashboard build/typecheck surface and frontend dependencies.

## Critical Constraints
- The fork default branch at generation time is `experiment/moe-pr13-pr14-stack`, not `main`; this map describes that experimental snapshot.
- Apple Silicon/macOS is the execution target, but inspection, doctor, configuration, and other non-model CLI paths must work without MLX installed.
- Exact sampling and router selections are correctness boundaries; a performance change cannot alter accepted tokens, routing IDs, weights, or residual correction.
- `HotExpertSwitchGLU` all-hit decode must stay inside `runtime.route_waves`; raw PR #15 flattened it and broke multi-wave ordering, counters, and pin/error cleanup.
- Streamed expert memory is fixed-budget: slot capacity must not grow, and a slot generation cannot be overwritten before its last Metal consumer completes.
- Hardware-only kernels, opt-in probes, dynamic backend imports, and closed experimental lanes can look unused statically; classify them before deletion.
- Performance changes require matched runs against the immediate predecessor with model/artifact identity, bytes, cache state, thermals, memory, and tokens/s recorded.
- Raw benchmark artifacts belong under ignored `benchmarks/raw/`; only curated reproducible summaries belong in `benchmarks/results/`.
- Fan-controlled runs are not valid product headline evidence.
- The built dashboard under `mtplx/dashboard/_static/` and native/Metal sources are package data, not disposable generated clutter.
- Auxiliary worktrees belong directly under the workspace-level `.worktrees/` directory; do not move worktrees owned by another process or agent.

## Hot Files
mtplx/commands/public.py, mtplx/server/openai.py, mtplx/generation.py, mtplx/runtime.py, mtplx/cache_state.py, mtplx/graphbank.py, mtplx/expert_runtime.py, mtplx/expert_slots.py
