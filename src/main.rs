use std::time::Instant;

use anyhow::{Context, Result};
use glam::DVec3;

use crate::loader::read_file;

mod cloud;
mod loader;

fn main() -> Result<()> {
    let path = std::env::args().nth(1).context("usage: brume <file.laz>")?;

    let mut reader = las::Reader::from_path(path)?;

    let header = reader.header();

    let n_points = header.number_of_points();
    println!("Number of points -> {}", n_points);

    let _ = read_file(&mut reader)?;

    Ok(())
}
