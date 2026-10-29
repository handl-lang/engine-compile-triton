//! Lower a kernel IR `Term` to Triton source (ADR-0165).
//!
//! ```text
//! term (JSON from lib-kernel-ir)  ->  as_qgemv  ->  emit  ->  Triton module
//! ```

pub mod codegen;
pub mod term;

pub use codegen::emit;
pub use term::{as_qgemv, Node, Qgemv};
