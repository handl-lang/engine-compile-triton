use compile_triton::term::Node;
use compile_triton::{as_qgemv, emit};

fn term(k: i64, split: i64, group: i64) -> String {
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
fn refuses_an_op_outside_the_nine() {
    let bad = r#"{"op":"matmul","args":[]}"#;
    let err = as_qgemv(&parse(bad)).unwrap_err();
    assert!(err.contains("not one of the nine"), "{err}");
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
