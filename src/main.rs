//! CLI: a term on stdin, a Triton module on stdout.

use std::io::Read;

use compile_triton::{as_qgemv, emit};
use compile_triton::term::Node;

fn main() -> std::process::ExitCode {
    let mut args = std::env::args().skip(1);
    let (mut block_n, mut block_k) = (64usize, 128usize);
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--block-n" => block_n = args.next().and_then(|v| v.parse().ok()).unwrap_or(64),
            "--block-k" => block_k = args.next().and_then(|v| v.parse().ok()).unwrap_or(128),
            other => {
                eprintln!("unknown flag {other}");
                return std::process::ExitCode::FAILURE;
            }
        }
    }

    let mut raw = String::new();
    if let Err(err) = std::io::stdin().read_to_string(&mut raw) {
        eprintln!("cannot read the term: {err}");
        return std::process::ExitCode::FAILURE;
    }
    let node: Node = match serde_json::from_str(&raw) {
        Ok(node) => node,
        Err(err) => {
            eprintln!("the term is not valid JSON: {err}");
            return std::process::ExitCode::FAILURE;
        }
    };
    match as_qgemv(&node) {
        Ok(q) => {
            print!("{}", emit(&q, block_n, block_k));
            std::process::ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("not a lowerable qgemv term: {err}");
            std::process::ExitCode::FAILURE
        }
    }
}
