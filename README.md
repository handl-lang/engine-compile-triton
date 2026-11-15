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

A second lowering path covers the ADR-0171 fragment: any
`store(body, out)` term built from the pointwise ops (`mul`, `sub`,
`neg`, `relu`, `exp`, `sigmoid`, `tanh`, `gelu`, `silu`, `bias`) over
leaf operands lowers to one flat Triton elementwise kernel. `transpose`
and `permute` are torch *views* replayed host-side before launch (the IR
prices them at zero); `tile_load`/`pipeline` are identities. A view over
a computed tile, a broadcast rank above 4, or a qgemv op (`dot`,
`dequant`, `split`, …) outside its own path is refused loudly, not
miscompiled.

## Develop

```bash
cargo test                 # tests/codegen_test.rs — the emitted-source contract
cargo run --quiet -- --block-n 64 --block-k 128   # stdin term JSON -> .py
bash ci.sh                 # the bupha gate: cargo check + cargo test
```
