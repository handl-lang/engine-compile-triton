//! The IR term, as it arrives over the wire from `lib-kernel-ir`.
//!
//! A term is `{"op": ..., "args": [...]}`, a leaf is
//! `{"leaf": {"name": ..., "kind": ..., "div": ...}}`, and everything else
//! is a literal. The Rust side does not invent structure: it *reads* the
//! term and refuses anything that is not one of the twenty-one ops.

use serde::Deserialize;

/// The twenty-one IR ops (ADR-0165/0166 kernel + ADR-0171 layout/pointwise).
pub const OPS: [&str; 21] = [
    "tile_load",
    "unpack_i4",
    "dequant",
    "dot",
    "split",
    "stage",
    "reduce",
    "store",
    "bias",
    "pipeline",
    // ADR-0171
    "transpose",
    "permute",
    "mul",
    "sub",
    "neg",
    "relu",
    "exp",
    "sigmoid",
    "tanh",
    "gelu",
    "silu",
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
    Term {
        op: String,
        args: Vec<Node>,
    },
    /// `{"leaf": {...}}`
    Leaf {
        leaf: Leaf,
    },
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
    /// `pipeline` depth: 1 means the term declares no prefetching.
    pub depth: i64,
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

    // body: an optional pipeline annotation around
    // dot(dequant(unpack_i4(load w, shifts), load z, load s, g), x)
    let mut depth = 1i64;
    let mut body = &sargs[2];
    if body.op() == Some("pipeline") {
        let pargs = body.args().ok_or("pipeline has no args")?;
        if pargs.len() != 2 {
            return Err("pipeline wants (depth, body)".into());
        }
        depth = pargs[0]
            .as_int()
            .ok_or("pipeline's depth must be an integer")?;
        if depth < 1 {
            return Err(format!("pipeline depth must be >= 1, got {depth}"));
        }
        body = &pargs[1];
    }
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
    if deargs[1].leaf_name() != Some("z") || deargs[2].leaf_name() != Some("s") {
        return Err("dequant wants the `z` and `s` leaves".into());
    }
    Ok(Qgemv {
        k,
        split: parts,
        group,
        depth,
    })
}

fn check_ops(node: &Node) -> Result<(), String> {
    match node {
        Node::Term { op, args } => {
            if !OPS.contains(&op.as_str()) {
                return Err(format!("{op:?} is not one of the twenty-one IR ops"));
            }
            for arg in args {
                check_ops(arg)?;
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

// -- ADR-0171: the elementwise/layout fragment ------------------------------
//
// A second, independent lowering path. Where `as_qgemv` recognises exactly
// one kernel shape, `as_elementwise` recognises the fragment the pointwise
// and layout ops actually form: `store(body, out)` where `body` is a tree
// of elementwise ops over leaf operands. The ops that stay qgemv-only —
// `dot`, `dequant`, `unpack_i4`, `split`, `stage`, `reduce` — are refused
// here with a message naming the fragment, never silently miscompiled.
//
// Layout ops (`transpose`/`permute`) denote torch *views* (Cost() zero),
// so they lower as host-side `Tensor.transpose`/`Tensor.permute` calls on
// the operand before launch — the kernel itself only ever sees flat,
// stride-described loads. A view over a *computed* tile (e.g.
// `transpose(mul(a, b), 0, 1)`) has no honest transcription in this
// fragment and is refused.

/// One layout op applied to a leaf operand, in application order
/// (innermost first — the order the host replays on the tensor).
#[derive(Debug, Clone, PartialEq)]
pub enum View {
    Transpose(i64, i64),
    Permute(Vec<i64>),
}

/// A read operand: a leaf name plus the view ops wrapped around it.
#[derive(Debug, Clone, PartialEq)]
pub struct Operand {
    pub name: String,
    pub views: Vec<View>,
}

/// A scalar literal inside the expression.
#[derive(Debug, Clone, PartialEq)]
pub enum Lit {
    Int(i64),
    Float(f64),
}

/// The lowered body: a tree of elementwise ops over operand loads.
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    /// index into `Elementwise::operands`
    Load(usize),
    Lit(Lit),
    /// neg relu exp sigmoid tanh gelu silu
    Unary(&'static str, Box<Expr>),
    /// mul sub bias
    Binary(&'static str, Box<Expr>, Box<Expr>),
}

/// A `store(body, out)` term in the elementwise/layout fragment.
#[derive(Debug, Clone, PartialEq)]
pub struct Elementwise {
    pub expr: Expr,
    /// unique (name, views) read operands, in first-appearance order
    pub operands: Vec<Operand>,
    /// (leaf name, python variable) bindings for `run`, read leaves first
    /// then the store target
    pub env: Vec<(String, String)>,
    /// python variable holding the store target (may be `None` -> allocate)
    pub out_var: String,
    /// true when the store target is not also a read leaf, so `run` gives
    /// it a `=None` default and allocates the output itself
    pub out_nullable: bool,
}

const UNARY_OPS: [&str; 7] = ["neg", "relu", "exp", "sigmoid", "tanh", "gelu", "silu"];
const BINARY_OPS: [&str; 3] = ["mul", "sub", "bias"];

/// Read a term as an elementwise/layout kernel body, or say why not.
pub fn as_elementwise(root: &Node) -> Result<Elementwise, String> {
    check_ops(root)?;
    if root.op() != Some("store") {
        return Err(format!("expected a store term, got {:?}", root.op()));
    }
    let args = root.args().ok_or("store has no args")?;
    if args.len() != 2 {
        return Err("store wants (tile, out)".into());
    }
    let out_name = args[1]
        .leaf_name()
        .ok_or("store's second argument must be a leaf")?
        .to_string();

    // `pipeline` denotes the identity; the elementwise kernel is a single
    // launch, so the depth annotation carries nothing here.
    let mut body = &args[0];
    while body.op() == Some("pipeline") {
        let pargs = body.args().ok_or("pipeline has no args")?;
        if pargs.len() != 2 {
            return Err("pipeline wants (depth, body)".into());
        }
        body = &pargs[1];
    }

    let mut operands: Vec<Operand> = Vec::new();
    let mut leaf_order: Vec<String> = Vec::new();
    let expr = build_expr(body, &mut operands, &mut leaf_order)?;

    // env: read leaves in first-appearance order, then the store target
    // if its name is not already bound.
    let mut used: Vec<String> = Vec::new();
    let mut env: Vec<(String, String)> = Vec::new();
    for name in &leaf_order {
        let py = py_ident(name, &mut used);
        env.push((name.clone(), py));
    }
    let out_var = if let Some((_, py)) = env.iter().find(|(n, _)| *n == out_name) {
        py.clone()
    } else {
        py_ident(&out_name, &mut used)
    };
    let out_nullable = !env.iter().any(|(n, _)| *n == out_name);
    if out_nullable {
        env.push((out_name.clone(), out_var.clone()));
    }

    Ok(Elementwise {
        expr,
        operands,
        env,
        out_var,
        out_nullable,
    })
}

const PY_KEYWORDS: [&str; 35] = [
    "False", "None", "True", "and", "as", "assert", "async", "await", "break", "class", "continue",
    "def", "del", "elif", "else", "except", "finally", "for", "from", "global", "if", "import",
    "in", "is", "lambda", "nonlocal", "not", "or", "pass", "raise", "return", "try", "while",
    "with", "yield",
];

fn py_ident(name: &str, used: &mut Vec<String>) -> String {
    let mut s: String = name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    if s.is_empty() || !(s.chars().next().unwrap().is_ascii_alphabetic() || s.starts_with('_')) {
        s = format!("a_{s}");
    }
    while used.contains(&s) || s == "block" || s == "warps" || PY_KEYWORDS.contains(&s.as_str()) {
        s.push('_');
    }
    used.push(s.clone());
    s
}

/// Build the elementwise expression tree, registering operands.
fn build_expr(
    node: &Node,
    operands: &mut Vec<Operand>,
    leaf_order: &mut Vec<String>,
) -> Result<Expr, String> {
    match node {
        Node::Leaf { leaf } => {
            if !leaf_order.contains(&leaf.name) {
                leaf_order.push(leaf.name.clone());
            }
            // dedupe on (name, no views); a viewed occurrence registers
            // its own operand below via view_operand
            if let Some(i) = operands
                .iter()
                .position(|o| o.name == leaf.name && o.views.is_empty())
            {
                return Ok(Expr::Load(i));
            }
            operands.push(Operand {
                name: leaf.name.clone(),
                views: Vec::new(),
            });
            Ok(Expr::Load(operands.len() - 1))
        }
        Node::Int(i) => Ok(Expr::Lit(Lit::Int(*i))),
        Node::Float(f) => Ok(Expr::Lit(Lit::Float(*f))),
        Node::Str(s) => Err(format!("string literal {s:?} is not a tensor operand")),
        Node::Term { op, args } => match op.as_str() {
            // identity wrappers — denote the same tensor
            "tile_load" => {
                if args.len() != 1 {
                    return Err("tile_load wants 1 argument".into());
                }
                build_expr(&args[0], operands, leaf_order)
            }
            "pipeline" => {
                if args.len() != 2 {
                    return Err("pipeline wants (depth, body)".into());
                }
                build_expr(&args[1], operands, leaf_order)
            }
            "transpose" | "permute" => {
                let (name, views) = view_operand(node)?;
                if !leaf_order.contains(&name) {
                    leaf_order.push(name.clone());
                }
                if let Some(i) = operands
                    .iter()
                    .position(|o| o.name == name && o.views == views)
                {
                    return Ok(Expr::Load(i));
                }
                operands.push(Operand { name, views });
                Ok(Expr::Load(operands.len() - 1))
            }
            _ if UNARY_OPS.contains(&op.as_str()) => {
                if args.len() != 1 {
                    return Err(format!("{op} wants 1 argument"));
                }
                let name: &'static str = match op.as_str() {
                    "neg" => "neg",
                    "relu" => "relu",
                    "exp" => "exp",
                    "sigmoid" => "sigmoid",
                    "tanh" => "tanh",
                    "gelu" => "gelu",
                    _ => "silu",
                };
                Ok(Expr::Unary(
                    name,
                    Box::new(build_expr(&args[0], operands, leaf_order)?),
                ))
            }
            _ if BINARY_OPS.contains(&op.as_str()) => {
                if args.len() != 2 {
                    return Err(format!("{op} wants 2 arguments"));
                }
                let name: &'static str = match op.as_str() {
                    "mul" => "mul",
                    "sub" => "sub",
                    _ => "bias",
                };
                Ok(Expr::Binary(
                    name,
                    Box::new(build_expr(&args[0], operands, leaf_order)?),
                    Box::new(build_expr(&args[1], operands, leaf_order)?),
                ))
            }
            _ => Err(format!(
                "{op:?} does not lower in the elementwise fragment \
                 (the qgemv ops need the qgemv path)"
            )),
        },
    }
}

/// Peel a chain of `transpose`/`permute`/`tile_load` wrappers down to a
/// leaf. Returns the leaf name and the views in application order
/// (innermost first). A chain bottoming out on anything but a leaf is a
/// view of a computed tile — refused, not approximated.
fn view_operand(node: &Node) -> Result<(String, Vec<View>), String> {
    let mut collected: Vec<View> = Vec::new();
    let mut cur = node;
    loop {
        match cur {
            Node::Leaf { leaf } => {
                collected.reverse();
                return Ok((leaf.name.clone(), collected));
            }
            Node::Term { op, args } => match op.as_str() {
                "tile_load" if args.len() == 1 => cur = &args[0],
                "transpose" => {
                    if args.len() != 3 {
                        return Err("transpose wants (tensor, dim0, dim1)".into());
                    }
                    let (Some(d0), Some(d1)) = (args[1].as_int(), args[2].as_int()) else {
                        return Err("transpose dims must be integer literals".into());
                    };
                    collected.push(View::Transpose(d0, d1));
                    cur = &args[0];
                }
                "permute" => {
                    if args.len() < 2 {
                        return Err("permute wants (tensor, dims...) with >= 1 dim".into());
                    }
                    let mut dims = Vec::new();
                    for a in &args[1..] {
                        match a.as_int() {
                            Some(d) => dims.push(d),
                            None => return Err("permute dims must be integer literals".into()),
                        }
                    }
                    collected.push(View::Permute(dims));
                    cur = &args[0];
                }
                _ => {
                    return Err(format!(
                        "a view chain bottoms out on {op:?}, not a leaf — \
                         layout ops lower only over leaf operands"
                    ))
                }
            },
            _ => return Err("a layout op must sit on a leaf operand, not a literal".into()),
        }
    }
}
