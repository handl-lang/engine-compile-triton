//! `KernelTerm` — the `lib-kernel-ir` wire term as an `IrModule`, plus
//! the `Emitter<KernelTerm>` impls that materialize it to desired-state
//! `Change`s (PL-004 T-094).
//!
//! `lib-kernel-ir` is Python-only, so the IR this crate consumes is the
//! *wire format* the Python side serializes (`{"op": ..., "args":
//! [...]}` terms, `{"leaf": {...}}` leaves, literals) — the same bytes
//! `tool-compile-triton`'s stdin protocol already parses into [`Node`].
//! [`KernelTerm`] keeps those raw bytes because the artifact slug is
//! `kernel_<target>_<blake3(raw)[:12]>` — hashing the canonical
//! `lib_kernel_ir.to_json` output is what makes a Rust-materialized
//! `Change` name-identical to the Python `lower.py` one. Feed
//! [`KernelTerm::parse`] canonical term JSON (what `to_json` emits) and
//! the slugs agree; re-formatted JSON parses fine but hashes
//! differently — that is honest (different bytes, different artifact).
//!
//! `Emitter::verify` stays structural, mirroring `tool-fmt`'s
//! `FmtEmitter`: the change set must be exactly the one `Put` this
//! emitter promised. There is no Triton/CUDA runner in this repo, so
//! artifact-level verify remains the producer-side term-denotation
//! gate (`KernelSink.verify` / `verify_lowered` in kernelopt).

use hnl_tool_codegen::{
    ArtifactRef, Change, CodegenError, ContentHash, EmitContext, Emitter, IrKind, IrModule,
};

use crate::term::Node;
use crate::{as_elementwise, as_qgemv, emit, emit_elementwise};

/// IR family tag for the `lib-kernel-ir` wire term.
pub const KERNEL_TERM_KIND: &str = "kernel-term";

/// A `lib-kernel-ir` `Term` as it arrives on the wire.
///
/// Carries the raw JSON text (for provenance hashing) alongside the
/// parsed [`Node`]. Construct via [`KernelTerm::parse`]; there is no
/// other honest source — building a `Node` by hand loses the bytes the
/// artifact slug is computed over.
#[derive(Debug, Clone)]
pub struct KernelTerm {
    raw: String,
    node: Node,
}

impl KernelTerm {
    /// Parse canonical term JSON (what `lib_kernel_ir.to_json` emits).
    ///
    /// Fails on malformed JSON or a shape [`Node`]'s `Deserialize`
    /// rejects; *lowerability* is the emitter's call, not the parser's.
    pub fn parse(raw: &str) -> Result<Self, String> {
        let node: Node =
            serde_json::from_str(raw).map_err(|e| format!("term is not a kernel-ir node: {e}"))?;
        Ok(Self {
            raw: raw.to_string(),
            node,
        })
    }

    /// The parsed term tree.
    pub fn node(&self) -> &Node {
        &self.node
    }

    /// The raw wire bytes — the provenance-hash input.
    pub fn raw(&self) -> &str {
        &self.raw
    }
}

impl IrModule for KernelTerm {
    fn kind(&self) -> IrKind {
        IrKind::new(KERNEL_TERM_KIND)
    }
}

/// `Emitter<KernelTerm>` for the Triton backend.
///
/// Stateless apart from the tile knobs the CLI exposes as
/// `--block-n`/`--block-k` (defaults 64/128 — what the emitter itself
/// uses when callers don't tune).
#[derive(Debug, Clone)]
pub struct TritonEmitter {
    /// `BLOCK_N` — output tile rows.
    pub block_n: usize,
    /// `BLOCK_K` — reduction tile width.
    pub block_k: usize,
}

impl Default for TritonEmitter {
    fn default() -> Self {
        Self {
            block_n: 64,
            block_k: 128,
        }
    }
}

/// The deterministic artifact name for a lowered term —
/// `kernel_<target>_<blake3(raw)[:12]>.<ext>`, identical to kernelopt's
/// `lower.py` `_slug` because both hash the same canonical term bytes.
pub fn artifact_name(target: &str, raw_term_json: &str, ext: &str) -> ArtifactRef {
    let digest = ContentHash::of(raw_term_json.as_bytes()).to_hex();
    ArtifactRef::new(format!("kernel_{target}_{}.{ext}", &digest[..12]))
}

/// Shared lowering dispatch: qgemv fragment first, then the ADR-0171
/// elementwise/layout fragment; both refusals named on failure.
/// Used by the `tool-compile-triton` CLI and [`TritonEmitter`].
pub fn compile(node: &Node, block_n: usize, block_k: usize) -> Result<String, String> {
    match as_qgemv(node) {
        Ok(q) => Ok(emit(&q, block_n, block_k)),
        Err(qerr) => match as_elementwise(node) {
            Ok(e) => Ok(emit_elementwise(&e)),
            Err(eerr) => Err(format!(
                "not a lowerable term:\n  qgemv: {qerr}\n  elementwise: {eerr}"
            )),
        },
    }
}

impl Emitter<KernelTerm> for TritonEmitter {
    fn materialize(
        &self,
        ir: &KernelTerm,
        _req: &EmitContext,
    ) -> Result<Vec<Change>, CodegenError> {
        let source = compile(ir.node(), self.block_n, self.block_k).map_err(CodegenError::Emit)?;
        Ok(vec![Change::text(
            artifact_name("triton", ir.raw(), "py"),
            source,
        )])
    }

    /// Structural verify: exactly one `Put` on the artifact this
    /// emitter would have named. Content-level verify needs a Triton
    /// runner — none exists yet (see module docs).
    fn verify(&self, ir: &KernelTerm, changes: &[Change]) -> Result<(), CodegenError> {
        let want = artifact_name("triton", ir.raw(), "py");
        match changes {
            [change] if change.artifact() == &want && matches!(change, Change::Put { .. }) => {
                Ok(())
            }
            _ => Err(CodegenError::Verify(format!(
                "triton emitter produced {} change(s), expected one Put on {want}",
                changes.len()
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const QGEMV: &str = r#"{"op":"store","args":[{"op":"split","args":[256,2,
        {"op":"dot","args":[{"op":"dequant","args":[
          {"op":"unpack_i4","args":[{"leaf":{"name":"w","kind":"k"}},
            {"leaf":{"name":"shifts"}}]},
          {"leaf":{"name":"z","kind":"g","div":32}},
          {"leaf":{"name":"s","kind":"g","div":32}},
          32]},
         {"leaf":{"name":"x","kind":"k"}}]}]},
        {"leaf":{"name":"out"}}]}"#;

    const ELEMENTWISE: &str = r#"{"op":"store","args":[{"op":"neg","args":[{"leaf":{"name":"x"}}]},
            {"leaf":{"name":"out"}}]}"#;

    #[test]
    fn kernel_term_kind_and_parse() {
        let t = KernelTerm::parse(ELEMENTWISE).unwrap();
        assert_eq!(t.kind(), IrKind::new(KERNEL_TERM_KIND));
        assert_eq!(t.raw(), ELEMENTWISE);
        assert!(KernelTerm::parse("{not json").is_err());
    }

    #[test]
    fn materialize_qgemv_put_shape() {
        let t = KernelTerm::parse(QGEMV).unwrap();
        let emitter = TritonEmitter::default();
        let changes = emitter
            .materialize(&t, &EmitContext::new())
            .expect("qgemv fixture lowers");
        assert_eq!(changes.len(), 1);
        let Change::Put {
            artifact, content, ..
        } = &changes[0]
        else {
            panic!("expected a Put");
        };
        assert!(artifact.as_str().starts_with("kernel_triton_"));
        assert!(artifact.as_str().ends_with(".py"));
        let text = String::from_utf8(content.clone()).unwrap();
        assert!(text.contains("triton"));
        emitter.verify(&t, &changes).unwrap();
    }

    #[test]
    fn materialize_elementwise() {
        let t = KernelTerm::parse(ELEMENTWISE).unwrap();
        let emitter = TritonEmitter::default();
        let changes = emitter
            .materialize(&t, &EmitContext::new())
            .expect("elementwise fixture lowers");
        assert!(changes[0].artifact().as_str().ends_with(".py"));
    }

    #[test]
    fn materialize_refusal_is_named() {
        // `dot` of a computed lhs fits neither fragment.
        let bad = r#"{"op":"store","args":[{"op":"dot","args":[
            {"op":"neg","args":[{"leaf":{"name":"a"}}]},{"leaf":{"name":"b"}}]},
            {"leaf":{"name":"out"}}]}"#;
        let t = KernelTerm::parse(bad).unwrap();
        let err = TritonEmitter::default()
            .materialize(&t, &EmitContext::new())
            .unwrap_err();
        assert!(err.to_string().contains("not a lowerable term"));
    }

    #[test]
    fn artifact_slug_is_stable_and_content_keyed() {
        let a = artifact_name("triton", ELEMENTWISE, "py");
        let b = artifact_name("triton", ELEMENTWISE, "py");
        assert_eq!(a, b);
        let c = artifact_name("triton", QGEMV, "py");
        assert_ne!(a, c);
        // 12 hex chars of the blake3-256 digest, kernel_ prefix + ext.
        let name = a.as_str();
        assert!(name.starts_with("kernel_triton_"));
        assert!(name.ends_with(".py"));
        assert_eq!(name.len(), "kernel_triton_".len() + 12 + ".py".len());
    }
}
