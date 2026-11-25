use compile_triton::term::Node;
use compile_triton::{as_qgemv, emit};

fn term(k: i64, split: i64, group: i64) -> String {
    term_with_depth(k, split, group, 1)
}

fn term_with_depth(k: i64, split: i64, group: i64, depth: i64) -> String {
    let dot = format!(
        r#"{{"op":"dot","args":[{{"op":"dequant","args":[
          {{"op":"unpack_i4","args":[{{"leaf":{{"name":"w","kind":"k"}}}},
            {{"leaf":{{"name":"shifts"}}}}]}},
          {{"leaf":{{"name":"z","kind":"g","div":{group}}}}},
          {{"leaf":{{"name":"s","kind":"g","div":{group}}}}},
          {group}]}},
         {{"leaf":{{"name":"x","kind":"k"}}}}]}}"#
    );
    let body = if depth > 1 {
        format!(r#"{{"op":"pipeline","args":[{depth},{dot}]}}"#)
    } else {
        dot
    };
    format!(
        r#"{{"op":"store","args":[{{"op":"split","args":[{k},{split},{body}]}},
        {{"leaf":{{"name":"out"}}}}]}}"#
    )
}

#[allow(dead_code)]
fn _unused_term(k: i64, split: i64, group: i64) -> String {
    format!(
        r#"{{"op":"store","args":[{{"op":"split","args":[{k},{split},
        {{"op":"dot","args":[{{"op":"dequant","args":[
          {{"op":"unpack_i4","args":[{{"leaf":{{"name":"w","kind":"k"}}}},
            {{"leaf":{{"name":"shifts"}}}}]}},
          {{"leaf":{{"name":"z","kind":"g","div":{group}}}}},
          {{"leaf":{{"name":"s","kind":"g","div":{group}}}}},
          {group}]}},
         {{"leaf":{{"name":"x","kind":"k"}}}}]}}]}},
        {{"leaf":{{"name":"out"}}}}]}}"#
    )
}

fn parse(raw: &str) -> Node {
    serde_json::from_str(raw).expect("the test term must parse")
}

#[test]
fn reads_geometry_from_the_term() {
    let q = as_qgemv(&parse(&term(4864, 4, 128))).expect("a qgemv term");
    assert_eq!(q.k, 4864);
    assert_eq!(q.split, 4);
    assert_eq!(q.group, 128);
}

#[test]
fn emits_one_variant_per_split() {
    for split in [1, 2, 4, 8] {
        let q = as_qgemv(&parse(&term(896, split, 128))).unwrap();
        let src = emit(&q, 64, 128);
        assert!(src.contains(&format!("SPLIT = {split}")), "split {split}");
        assert!(src.contains("K = 896"));
        assert!(src.contains("GROUP = 128"));
        assert!(src.contains("@triton.jit"));
        assert!(src.contains("def _stage1("));
        assert!(src.contains("def _reduce("));
        assert!(src.contains("def run("));
        assert!(src.contains("BLOCK_N_DEFAULT = 64"));
        assert!(src.contains("PIPELINE_DEPTH = 1"));
        assert!(src.contains("num_stages=PIPELINE_DEPTH"));
        // P13 retracted the atomic reduction: it must not come back.
        assert!(!src.contains("atomic"));
        // the program must be a module, not a fragment
        assert!(src.starts_with("\"\"\""));
    }
}

#[test]
fn refuses_a_term_that_is_not_a_qgemv() {
    let bad = r#"{"op":"store","args":[{"op":"dot","args":[]},{"leaf":{"name":"out"}}]}"#;
    let err = as_qgemv(&parse(bad)).unwrap_err();
    assert!(err.contains("split"), "{err}");
}

#[test]
fn refuses_an_op_outside_the_twenty_one() {
    let bad = r#"{"op":"matmul","args":[]}"#;
    let err = as_qgemv(&parse(bad)).unwrap_err();
    assert!(err.contains("not one of the twenty-one"), "{err}");
}

#[test]
fn refuses_wrong_leaf_names() {
    let bad = term(4864, 4, 128).replace(r#""name":"x""#, r#""name":"y""#);
    let err = as_qgemv(&parse(&bad)).unwrap_err();
    assert!(err.contains("dot wants"), "{err}");
}

#[test]
fn refuses_non_positive_geometry() {
    let err = as_qgemv(&parse(&term(4864, 0, 128))).unwrap_err();
    assert!(err.contains("positive"), "{err}");
}

#[test]
fn reads_pipeline_depth_from_the_term() {
    let q = as_qgemv(&parse(&term_with_depth(4864, 4, 128, 3))).unwrap();
    assert_eq!(q.depth, 3);
    let src = emit(&q, 64, 128);
    assert!(src.contains("PIPELINE_DEPTH = 3"), "{src}");
    assert!(src.contains("num_stages=PIPELINE_DEPTH"));
}

#[test]
fn depth_defaults_to_one_without_the_annotation() {
    let q = as_qgemv(&parse(&term(896, 8, 128))).unwrap();
    assert_eq!(q.depth, 1);
}

#[test]
fn refuses_wrong_leaf_kinds() {
    // `x` sliced as a group row denotes a different tensor — refuse.
    let bad = term(4864, 4, 128).replace(
        r#"{"leaf":{"name":"x","kind":"k"}}"#,
        r#"{"leaf":{"name":"x","kind":"g","div":128}}"#,
    );
    let err = as_qgemv(&parse(&bad)).unwrap_err();
    assert!(err.contains("kind"), "{err}");
    // `w` must be k-indexed too.
    let bad = term(4864, 4, 128).replace(
        r#"{"leaf":{"name":"w","kind":"k"}}"#,
        r#"{"leaf":{"name":"w","kind":"out"}}"#,
    );
    let err = as_qgemv(&parse(&bad)).unwrap_err();
    assert!(err.contains("kind"), "{err}");
}

#[test]
fn refuses_z_s_with_wrong_group_div() {
    // z/s are group-indexed with div == group; a div of 64 under
    // group=128 denotes per-64 rows the kernel cannot produce.
    let bad = term(4864, 4, 128).replace(r#""div":128"#, r#""div":64"#);
    let err = as_qgemv(&parse(&bad)).unwrap_err();
    assert!(err.contains("div"), "{err}");
}

#[test]
fn refuses_a_computed_shifts_argument() {
    // unpack_i4's shifts must be a leaf — the kernel hardcodes the
    // nibble order, so anything else denotes different values.
    let bad = term(4864, 4, 128).replace(
        r#"{"leaf":{"name":"shifts"}}"#,
        r#"{"op":"neg","args":[{"leaf":{"name":"shifts"}}]}"#,
    );
    let err = as_qgemv(&parse(&bad)).unwrap_err();
    assert!(err.contains("shifts"), "{err}");
}

#[test]
fn refuses_sliced_shifts_leaf() {
    let bad = term(4864, 4, 128).replace(
        r#"{"leaf":{"name":"shifts"}}"#,
        r#"{"leaf":{"name":"shifts","kind":"k"}}"#,
    );
    let err = as_qgemv(&parse(&bad)).unwrap_err();
    assert!(err.contains("shifts"), "{err}");
}

#[test]
fn emitted_source_parses_as_python() {
    // Both lowering paths must emit syntactically valid Python. Skips
    // silently where no interpreter is on PATH (the gate host has one).
    let q = as_qgemv(&parse(&term(4864, 4, 128))).unwrap();
    let elem_body = r#"{"op":"silu","args":[{"op":"mul","args":[
        {"op":"transpose","args":[{"leaf":{"name":"w"}},0,1]},
        {"leaf":{"name":"x"}}]}]}"#;
    let e = as_elementwise(&parse(&elem_term(elem_body))).unwrap();
    for (name, src) in [
        ("qgemv", emit(&q, 64, 128)),
        ("elementwise", emit_elementwise(&e)),
    ] {
        let status = std::process::Command::new("python3")
            .args(["-c", "import ast,sys; ast.parse(sys.stdin.read())"])
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .and_then(|mut child| {
                use std::io::Write;
                child
                    .stdin
                    .as_mut()
                    .unwrap()
                    .write_all(src.as_bytes())
                    .unwrap();
                child.wait()
            });
        match status {
            Ok(s) if s.success() => {}
            Ok(s) => panic!("{name} output is not valid Python: {s}"),
            Err(_) => eprintln!("skipping ast.parse check: no python3"),
        }
    }
}

// -- ADR-0171: elementwise/layout fragment ----------------------------------

use compile_triton::term::{Expr, Lit, View};
use compile_triton::{as_elementwise, emit_elementwise};

fn leaf(name: &str) -> String {
    format!(r#"{{"leaf":{{"name":"{name}"}}}}"#)
}

fn elem_term(body: &str) -> String {
    format!(r#"{{"op":"store","args":[{body},{}]}}"#, leaf("out"))
}

#[test]
fn reads_a_pointwise_body() {
    // silu(mul(transpose(w, 0, 1), x)) — a decoder-tail spelling
    let body = format!(
        r#"{{"op":"silu","args":[{{"op":"mul","args":[
            {{"op":"transpose","args":[{},0,1]}}, {}]}}]}}"#,
        leaf("w"),
        leaf("x")
    );
    let e = as_elementwise(&parse(&elem_term(&body))).expect("elementwise");
    assert_eq!(e.operands.len(), 2);
    assert_eq!(e.operands[0].name, "w");
    assert_eq!(e.operands[0].views, vec![View::Transpose(0, 1)]);
    assert_eq!(e.operands[1].name, "x");
    assert!(e.operands[1].views.is_empty());
    assert!(e.out_nullable);
    assert_eq!(e.out_var, "out");
}

#[test]
fn emits_the_elementwise_kernel() {
    let body = r#"{"op":"silu","args":[{"op":"mul","args":[
        {"op":"transpose","args":[{"leaf":{"name":"w"}},0,1]},
        {"leaf":{"name":"x"}}]}]}"#;
    let e = as_elementwise(&parse(&elem_term(body))).unwrap();
    let src = emit_elementwise(&e);
    assert!(src.contains("@triton.jit"));
    assert!(src.contains("def _elem_kernel("));
    assert!(src.contains("def run("));
    assert!(src.contains("tl.exp"));
    // silu = x * sigmoid(x): one exp inside the sigmoid form
    assert!(!src.contains("0.7071067811865476"));
    // the view replays host-side, never inside the kernel
    assert!(src.contains("t.transpose(v[1], v[2])"));
    assert!(src.contains(r#"("transpose", 0, 1)"#));
    // broadcast machinery
    assert!(src.contains("torch.broadcast_shapes"));
    assert!(src.contains("_es(t, cs)"));
    assert!(src.contains("res.to(out_p.dtype.element_ty)"));
    // launch guards: pow2 block, store-target shape compatibility
    assert!(src.contains("block must be a power of two"));
    assert!(src.contains("cannot hold"));
}

#[test]
fn covers_every_pointwise_op() {
    for (op, needle) in [
        ("mul", "*"),
        ("sub", "-"),
        ("neg", "(-("),
        ("relu", "tl.maximum"),
        ("exp", "tl.exp"),
        ("sigmoid", "1.0 / (1.0 + tl.exp"),
        ("tanh", "libdevice.tanh"),
        ("gelu", "tl.math.erf"),
        ("silu", "tl.exp"),
        ("bias", "+"),
    ] {
        let args = if op == "neg"
            || op == "relu"
            || op == "exp"
            || op == "sigmoid"
            || op == "tanh"
            || op == "gelu"
            || op == "silu"
        {
            format!(r#"{{"op":"{op}","args":[{}]}}"#, leaf("x"))
        } else {
            format!(r#"{{"op":"{op}","args":[{},{}]}}"#, leaf("x"), leaf("b"))
        };
        let e =
            as_elementwise(&parse(&elem_term(&args))).unwrap_or_else(|err| panic!("{op}: {err}"));
        let src = emit_elementwise(&e);
        assert!(src.contains(needle), "{op}: missing {needle}\n{src}");
    }
}

#[test]
fn permute_collects_dims_in_order() {
    let body = format!(r#"{{"op":"permute","args":[{},2,0,1]}}"#, leaf("x"));
    let e = as_elementwise(&parse(&elem_term(&body))).unwrap();
    assert_eq!(e.operands[0].views, vec![View::Permute(vec![2, 0, 1])]);
    let src = emit_elementwise(&e);
    assert!(src.contains("t.permute(*v[1])"));
}

#[test]
fn composed_views_replay_innermost_first() {
    // permute(transpose(w, 0, 1), 1, 0): host applies transpose then permute
    let body = format!(
        r#"{{"op":"permute","args":[{{"op":"transpose","args":[{},0,1]}},1,0]}}"#,
        leaf("w")
    );
    let e = as_elementwise(&parse(&elem_term(&body))).unwrap();
    assert_eq!(
        e.operands[0].views,
        vec![View::Transpose(0, 1), View::Permute(vec![1, 0])]
    );
}

#[test]
fn tile_load_and_pipeline_are_identities() {
    let body = format!(
        r#"{{"op":"pipeline","args":[2,{{"op":"tile_load","args":[{}]}}]}}"#,
        leaf("x")
    );
    let e = as_elementwise(&parse(&elem_term(&body))).unwrap();
    assert!(matches!(e.expr, Expr::Load(_)));
    assert_eq!(e.operands.len(), 1);
}

#[test]
fn literals_become_kernel_constants() {
    let body = format!(r#"{{"op":"mul","args":[{},2.5]}}"#, leaf("x"));
    let e = as_elementwise(&parse(&elem_term(&body))).unwrap();
    let has_float_lit = match &e.expr {
        Expr::Binary(_, a, b) => {
            matches!(&**a, Expr::Lit(Lit::Float(_))) || matches!(&**b, Expr::Lit(Lit::Float(_)))
        }
        _ => false,
    };
    assert!(has_float_lit, "{:?}", e.expr);
    let src = emit_elementwise(&e);
    assert!(src.contains("2.5"));
}

#[test]
fn refuses_a_view_of_a_computed_tile() {
    let body = format!(
        r#"{{"op":"transpose","args":[{{"op":"mul","args":[{},{}]}},0,1]}}"#,
        leaf("x"),
        leaf("y")
    );
    let err = as_elementwise(&parse(&elem_term(&body))).unwrap_err();
    assert!(err.contains("leaf operands"), "{err}");
}

#[test]
fn refuses_qgemv_ops_in_the_fragment() {
    for op in ["dot", "dequant", "unpack_i4", "split", "stage", "reduce"] {
        let body = format!(r#"{{"op":"{op}","args":[]}}"#);
        let err = as_elementwise(&parse(&elem_term(&body))).unwrap_err();
        assert!(err.contains("elementwise fragment"), "{op}: {err}");
    }
}

#[test]
fn refuses_a_viewed_store_target_alias() {
    // store(mul(transpose(w, 0, 1), x), w): `w` is read through a view
    // and written flat — kernel lanes would race where torch's copy_
    // is sequential. Refused, not approximated.
    let body = format!(
        r#"{{"op":"mul","args":[{{"op":"transpose","args":[{},0,1]}},{}]}}"#,
        leaf("w"),
        leaf("x")
    );
    let raw = format!(r#"{{"op":"store","args":[{body},{}]}}"#, leaf("w"));
    let err = as_elementwise(&parse(&raw)).unwrap_err();
    assert!(err.contains("layout view"), "{err}");
}

#[test]
fn refuses_a_bad_pipeline_depth() {
    for depth in ["0", "-1", "\"deep\""] {
        let body = format!(r#"{{"op":"pipeline","args":[{depth},{}]}}"#, leaf("x"));
        let err = as_elementwise(&parse(&elem_term(&body))).unwrap_err();
        assert!(err.contains("pipeline depth"), "{depth}: {err}");
    }
}

#[test]
fn refuses_a_non_store_root() {
    let err = as_elementwise(&parse(&leaf("x"))).unwrap_err();
    assert!(err.contains("store"), "{err}");
}

#[test]
fn refuses_bad_view_dims() {
    // permute with no dims
    let body = format!(r#"{{"op":"permute","args":[{}]}}"#, leaf("x"));
    let err = as_elementwise(&parse(&elem_term(&body))).unwrap_err();
    assert!(err.contains("permute"), "{err}");
    // transpose with a float dim
    let body = format!(r#"{{"op":"transpose","args":[{},0.5,1]}}"#, leaf("x"));
    let err = as_elementwise(&parse(&elem_term(&body))).unwrap_err();
    assert!(err.contains("integer literals"), "{err}");
}

#[test]
fn out_leaf_can_be_a_read_leaf_too() {
    // store(x, x): copy through the same binding — no =None default
    let body = leaf("x");
    let raw = format!(r#"{{"op":"store","args":[{body},{}]}}"#, leaf("x"));
    let e = as_elementwise(&parse(&raw)).unwrap();
    assert!(!e.out_nullable);
    let src = emit_elementwise(&e);
    assert!(src.contains("def run(x, block=1024, warps=4):"), "{src}");
}

#[test]
fn odd_leaf_names_become_safe_params() {
    let raw = format!(
        r#"{{"op":"store","args":[{{"op":"relu","args":[{}]}},{{"leaf":{{"name":"out-put"}}}}]}}"#,
        r#"{"leaf":{"name":"in put"}}"#
    );
    let e = as_elementwise(&parse(&raw)).unwrap();
    let src = emit_elementwise(&e);
    assert!(src.contains("def run("), "{src}");
    // the mangled names must be valid python identifiers
    for line in src.lines() {
        if line.starts_with("def run(") {
            assert!(!line.contains("in put"), "{line}");
            assert!(!line.contains("out-put"), "{line}");
        }
    }
}

#[test]
fn keyword_leaf_names_become_safe_params() {
    let raw = format!(
        r#"{{"op":"store","args":[{{"op":"relu","args":[{}]}},{{"leaf":{{"name":"out"}}}}]}}"#,
        r#"{"leaf":{"name":"class"}}"#
    );
    let e = as_elementwise(&parse(&raw)).unwrap();
    let src = emit_elementwise(&e);
    assert!(
        src.contains("def run(class_, out=None, block=1024, warps=4):"),
        "{src}"
    );
}
