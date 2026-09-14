/// Run with `cargo run --bin stub_gen` (extension-module must be off, which
/// it is by default here since that feature only turns on for
/// `maturin develop`/`maturin build`). Writes csgpy.pyi.
fn main() -> pyo3_stub_gen::Result<()> {
    let stub = csgpy::stub_info()?;
    stub.generate()?;
    Ok(())
}
