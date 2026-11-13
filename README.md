# compile-triton

Lowers a kernel IR `Term` (ADR-0165, `lib-kernel-ir`) to Triton source.

The term is the input, the source is the output, and neither is a
template someone edited by hand for one shape:

```bash
echo '{"op":"store","args":[...]}' | cargo run --quiet -- --block-n 64 --block-k 128 > k.py
```

The emitted module exposes `run(x, qw, qz, sc, n, k, bn, bk, warps)` —
signature-compatible with the hand-written `gemv_split2`, so the equality
gate and the referee can judge both without special-casing either.

## Develop

```bash
cargo test                 # tests/codegen_test.rs — the emitted-source contract
cargo run --quiet -- --block-n 64 --block-k 128   # stdin term JSON -> .py
bash ci.sh                 # the bupha gate: cargo check + cargo test
```
