use std::time::Instant;

use anyhow::{Context, Result};
use glam::DVec3;

use crate::loader::load;

mod cloud;
mod loader;

fn main() -> Result<()> {
    let path_str = std::env::args().nth(1).context("usage: brume <file.laz>")?;
    let path = std::path::Path::new(&path_str);

    let _ = load(&path)?;

    Ok(())
}
