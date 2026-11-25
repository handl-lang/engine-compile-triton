//! CLI: a term on stdin, a Triton module on stdout.

#![forbid(unsafe_code)]

use std::io::Read;

use tool_compile_triton::compile;
use tool_compile_triton::term::Node;

fn main() -> std::process::ExitCode {
    let mut args = std::env::args().skip(1);
    let (mut block_n, mut block_k) = (64usize, 128usize);
    while let Some(flag) = args.next() {
        let mut value = |name: &str| -> Result<usize, std::process::ExitCode> {
            match args.next().and_then(|v| v.parse::<usize>().ok()) {
                Some(v) if v > 0 => Ok(v),
                _ => {
                    eprintln!("{name} wants a positive integer");
                    Err(std::process::ExitCode::FAILURE)
                }
            }
        };
        match flag.as_str() {
            "--block-n" => match value("--block-n") {
                Ok(v) => block_n = v,
                Err(code) => return code,
            },
            "--block-k" => match value("--block-k") {
                Ok(v) => block_k = v,
                Err(code) => return code,
            },
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
            eprintln!("the term did not parse as a kernel-ir node: {err}");
            return std::process::ExitCode::FAILURE;
        }
    };
    // Two lowering paths: the packed qgemv kernel, then the ADR-0171
    // elementwise/layout fragment. A term that fits neither is refused
    // with both reasons. `compile` is the same dispatch
    // `TritonEmitter::materialize` runs.
    match compile(&node, block_n, block_k) {
        Ok(src) => {
            print!("{src}");
            std::process::ExitCode::SUCCESS
        }
        Err(reason) => {
            eprintln!("{reason}");
            std::process::ExitCode::FAILURE
        }
    }
}
