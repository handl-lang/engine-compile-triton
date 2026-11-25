# tool-compile-triton

kernel IR `Term` → Triton Python source (ADR-0165/0171) — `tool-compile-triton`
crate (`tool_compile_triton` lib + `tool-compile-triton` bin), Rust 2024,
`publish = false` (git-install only).  A term JSON on stdin becomes a
`.py` Triton kernel on stdout; no hand-edited templates.

## Layout

| path | role |
| --- | --- |
| `src/term.rs` | wire intake — `{"op": …, "args": […]}` terms, `{"leaf": …}` leaves, literals → `Node`; the 21-op `OPS` table; `as_qgemv` (qgemv contract check) + `as_elementwise` (ADR-0171 pointwise/layout fragment) |
| `src/codegen.rs` | the emitters — `emit` (packed qgemv kernel) + `emit_elementwise` (one flat elementwise kernel); `py_ident`/`py_str` Python-safe naming |
| `src/kernel_term.rs` | desired-state half (PL-004 T-094) — `KernelTerm` as `IrModule`, `TritonEmitter` as `Emitter<KernelTerm>`; blake3 artifact slug `kernel_triton_<hash>.py` identical to kernelopt's `lower.py` |
| `src/main.rs` | CLI — stdin term JSON → stdout `.py`; `--block-n`/`--block-k` tile knobs (defaults 64/128) |
| `tests/codegen_test.rs` | the emitted-source contract — lowering-path fixtures, refusal cases, artifact-slug stability |
| `xtask/`, `bupha.toml`, `workflows/gate.yaml` | vendored repo gate + CI wiring (sibling convention) |

## Contract

- **Input**: `lib-kernel-ir` wire terms — `{"op": …, "args": […]}` with
  `{"leaf": {"name", "kind", "div"}}` leaves and int/float/str literals.
  `KernelTerm::parse` takes the *canonical* `lib_kernel_ir.to_json`
  bytes: the artifact slug hashes them, so re-formatted JSON parses but
  hashes differently (different bytes, different artifact — honest).
- **Two lowering paths** (`compile` dispatches qgemv first):
  1. **qgemv** — `store(split(k, parts, pipeline?(dot(dequant(unpack_i4(w, shifts), z, s, group), x))), out)`
     → the packed qgemv Triton kernel with signature-compatible
     `run(x, qw, qz, sc, n, k, bn, bk, warps)`.
  2. **elementwise** — `store(body, out)` over the pointwise ops
     (`mul`, `sub`, `neg`, `relu`, `exp`, `sigmoid`, `tanh`, `gelu`,
     `silu`, `bias`) and leaf operands → one flat elementwise kernel.
     `transpose`/`permute` are torch *views* replayed host-side
     (priced at zero); `tile_load`/`pipeline` are identities.
- **Refusals are named, never silent**: a term fitting neither path
  fails with both reasons (`not a lowerable term:\n  qgemv: …\n  elementwise: …`).
  A view over a computed tile, broadcast rank > 4, a qgemv op outside
  its path, or an op outside the 21-op table is refused loudly — no
  miscompile.
- `Emitter::verify` is structural only (exactly one `Put` on the
  promised artifact): there is no Triton/CUDA runner in this repo, so
  artifact-level verify stays producer-side (`KernelSink.verify` /
  `verify_lowered` in kernelopt).

## Conventions

- `hnl-tool-codegen` is a **git dep on `branch = "main"`** — no `[patch]`
  tables in the manifest; local sibling overrides live in the
  gitignored workspace-root `.cargo/config.toml` (engine-public-
  readiness D7).
- Edition 2024, `#![forbid(unsafe_code)]`, `#![warn(missing_docs)]`,
  `rust-version = "1.85"`, `publish = false`.
- Crate/bin names match the repo (`tool-compile-triton` /
  `tool_compile_triton`) — sibling git deps and kernelopt's
  `lower.py` reference them by name; rename both sides together.

## Verification

```bash
cargo run --manifest-path xtask/Cargo.toml -- gate   # fmt → clippy → test → doc
cargo test --test codegen_test                       # emitted-source contract
echo '<term.json>' | cargo run --quiet -- --block-n 64 --block-k 128   # stdin term → .py
```

The bupha/fleet runner invokes `workflows/gate.yaml` (the peregrin rust
family campaign) — see `bupha.toml`; there is no `ci.sh`.

<!-- myco:agent-awareness -->

## Agent awareness

- Session start / any work discussion: run `myco status --prompt`
  before proposing next steps — know the board first.
- User describes new work → offer to capture it (`myco todo add`,
  `issue new`, `proposal new`). Don't wait to be asked.
- Picking up a task: `myco resume <ID>` is the briefing.
- Wrapping up: `myco handoff <ID>` anything resumable; keep statuses
  honest (claimed → in-progress → blocked/done).

<!-- /myco:agent-awareness -->
