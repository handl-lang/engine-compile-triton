//! The IR term, as it arrives over the wire from `lib-kernel-ir`.
//!
//! A term is `{"op": ..., "args": [...]}`, a leaf is
//! `{"leaf": {"name": ..., "kind": ..., "div": ...}}`, and everything else
//! is a literal. The Rust side does not invent structure: it *reads* the
//! term and refuses anything that is not one of the nine ops.

use serde::Deserialize;

/// One of the nine IR ops (ADR-0165 rule 2).
pub const OPS: [&str; 9] = [
    "tile_load",
    "unpack_i4",
    "dequant",
    "dot",
    "split",
    "stage",
    "reduce",
    "store",
    "bias",
];

#[derive(Deserialize, Debug, Clone, PartialEq)]
pub struct Leaf {
    pub name: String,
    #[serde(default = "default_kind")]
    pub kind: String,
    #[serde(default = "one")]
    pub div: i64,
}

fn default_kind() -> String {
    "out".to_string()
}

fn one() -> i64 {
    1
}

#[derive(Deserialize, Debug, Clone, PartialEq)]
#[serde(untagged)]
pub enum Node {
    /// `{"op": ..., "args": [...]}`
    Term { op: String, args: Vec<Node> },
    /// `{"leaf": {...}}`
    Leaf { leaf: Leaf },
    /// a literal
    Int(i64),
    Float(f64),
    Str(String),
}

impl Node {
    pub fn op(&self) -> Option<&str> {
        match self {
            Node::Term { op, .. } => Some(op),
            _ => None,
        }
    }

    pub fn args(&self) -> Option<&[Node]> {
        match self {
            Node::Term { args, .. } => Some(args),
            _ => None,
        }
    }

    pub fn leaf_name(&self) -> Option<&str> {
        match self {
            Node::Leaf { leaf } => Some(&leaf.name),
            _ => None,
        }
    }

    pub fn as_int(&self) -> Option<i64> {
        match self {
            Node::Int(i) => Some(*i),
            _ => None,
        }
    }
}

/// What a `qgemv` term has to say: the geometry and the split.
#[derive(Debug, Clone, PartialEq)]
pub struct Qgemv {
    pub k: i64,
    pub split: i64,
    pub group: i64,
}

/// Read a term as a `qgemv`, or say precisely why it is not one.
pub fn as_qgemv(root: &Node) -> Result<Qgemv, String> {
    check_ops(root)?;
    if root.op() != Some("store") {
        return Err(format!("expected a store term, got {:?}", root.op()));
    }
    let args = root.args().ok_or("store has no args")?;
    if args.len() != 2 {
        return Err("store wants (tile, out)".into());
    }
    if args[1].leaf_name() != Some("out") {
        return Err("store's second argument must be the `out` leaf".into());
    }
    let split = &args[0];
    if split.op() != Some("split") {
        return Err(format!(
            "store's tile must be a split term, got {:?}",
            split.op()
        ));
    }
    let sargs = split.args().ok_or("split has no args")?;
    if sargs.len() != 3 {
        return Err("split wants (k, parts, body)".into());
    }
    let k = sargs[0].as_int().ok_or("split's k must be an integer")?;
    let parts = sargs[1]
        .as_int()
        .ok_or("split's parts must be an integer")?;
    if k <= 0 || parts <= 0 {
        return Err(format!("split geometry must be positive, got {k}/{parts}"));
    }

    // body: dot(dequant(unpack_i4(load w, shifts), load z, load s, g), x)
    let body = &sargs[2];
    if body.op() != Some("dot") {
        return Err(format!("split's body must be a dot, got {:?}", body.op()));
    }
    let dargs = body.args().ok_or("dot has no args")?;
    if dargs.len() != 2 || dargs[1].leaf_name() != Some("x") {
        return Err("dot wants (dequant(...), x)".into());
    }
    let deq = &dargs[0];
    if deq.op() != Some("dequant") {
        return Err("dot's first argument must be a dequant".into());
    }
    let deargs = deq.args().ok_or("dequant has no args")?;
    if deargs.len() != 4 {
        return Err("dequant wants (q, zero, scale, group)".into());
    }
    let group = deargs[3]
        .as_int()
        .ok_or("dequant's group must be an integer")?;
    if group <= 0 {
        return Err(format!("group must be positive, got {group}"));
    }
    let unp = &deargs[0];
    if unp.op() != Some("unpack_i4") {
        return Err("dequant's first argument must be an unpack_i4".into());
    }
    let uargs = unp.args().ok_or("unpack_i4 has no args")?;
    if uargs.len() != 2 || uargs[0].leaf_name() != Some("w") {
        return Err("unpack_i4 wants (w, shifts)".into());
    }
    if deargs[1].leaf_name() != Some("z") || deargs[2].leaf_name() != Some("s")
    {
        return Err("dequant wants the `z` and `s` leaves".into());
    }
    Ok(Qgemv { k, split: parts, group })
}

fn check_ops(node: &Node) -> Result<(), String> {
    match node {
        Node::Term { op, args } => {
            if !OPS.contains(&op.as_str()) {
                return Err(format!("{op:?} is not one of the nine IR ops"));
            }
            for arg in args {
                check_ops(arg)?;
            }
            Ok(())
        }
        _ => Ok(()),
    }
}
