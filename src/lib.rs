//! Lower a kernel IR `Term` to Triton source (ADR-0165/0171).
//!
//! ```text
//! term (JSON from lib-kernel-ir)
//!   -> as_qgemv      -> emit              (the packed qgemv kernel)
//!   -> as_elementwise -> emit_elementwise (the ADR-0171 pointwise/layout
//!                                        fragment: one flat Triton kernel)
//! ```

pub mod codegen;
pub mod term;

pub use codegen::{emit, emit_elementwise};
pub use term::{as_elementwise, as_qgemv, Elementwise, Expr, Node, Operand, Qgemv, View};
