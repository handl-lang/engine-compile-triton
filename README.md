# tool-compile-triton

Lowers a kernel IR `Term` to Triton source: a term JSON on stdin becomes
a `.py` Triton kernel on stdout — signature-compatible launchers, no
hand-edited templates.

```bash
echo '{"op":"store","args":[...]}' | cargo run --quiet -- --block-n 64 --block-k 128 > k.py
```

The emitted module exposes `run(x, qw, qz, sc, n, k, bn, bk, warps)` —
signature-compatible with the hand-written `gemv_split2`, so the equality
gate and the referee can judge both without special-casing either.

A second lowering path covers the elementwise fragment: any
`store(body, out)` term built from the pointwise ops (`mul`, `sub`,
`neg`, `relu`, `exp`, `sigmoid`, `tanh`, `gelu`, `silu`, `bias`) over
leaf operands lowers to one flat Triton elementwise kernel. `transpose`
and `permute` are torch *views* replayed host-side before launch (the IR
prices them at zero); `tile_load`/`pipeline` are identities. A view over
a computed tile, a broadcast rank above 4, or a qgemv op (`dot`,
`dequant`, `split`, …) outside its own path is refused loudly, not
miscompiled.

## Development

```bash
cargo run --manifest-path xtask/Cargo.toml -- gate   # fmt → clippy → test → doc
cargo test                                 # tests/codegen_test.rs — the emitted-source contract
cargo run --quiet -- --block-n 64 --block-k 128   # stdin term JSON -> .py
```

`xtask/` provides the gate tasks (`fmt`, `clippy`, `test`, `doc`, `check`,
`gate`); bupha/the fleet runner executes the `workflows/gate.yaml`
peregrin campaign (see `bupha.toml`) — there is no `ci.sh`. Deps
are git deps on `branch = "main"`; locally the workspace-root
`.cargo/config.toml` patches them to sibling checkouts (dev-only, never
committed).

## License

MIT.
