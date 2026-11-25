#![forbid(unsafe_code)]
#![deny(missing_docs)]

//! Lower a kernel IR `Term` to Triton source (ADR-0165/0171).
//!
//! ```text
//! term (JSON from lib-kernel-ir)
//!   -> as_qgemv      -> emit              (the packed qgemv kernel)
//!   -> as_elementwise -> emit_elementwise (the ADR-0171 pointwise/layout
//!                                        fragment: one flat Triton kernel)
//! ```
//!
//! [`kernel_term`] adds the desired-state half (PL-004 T-094):
//! [`KernelTerm`] is the wire term as an `IrModule` and
//! [`TritonEmitter`] its `Emitter<KernelTerm>` — `materialize`
//! produces the same `kernel_triton_<hash>.py` `Change` kernelopt's
//! Python `lower.py` declares, natively, no subprocess.

pub mod codegen;
pub mod kernel_term;
pub mod term;

pub use codegen::{emit, emit_elementwise};
pub use kernel_term::{KERNEL_TERM_KIND, KernelTerm, TritonEmitter, artifact_name, compile};
pub use term::{Elementwise, Expr, Node, Operand, Qgemv, View, as_elementwise, as_qgemv};
