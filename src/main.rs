use anyhow::{Context, Result};

fn main() -> Result<()> {
    let path = std::env::args().nth(1).context("usage: brume <file.laz>")?;

    let reader = las::Reader::from_path(path)?;

    let header = reader.header();

    println!("{}", header.number_of_points());

    Ok(())
}
