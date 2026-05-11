from __future__ import annotations

from pathlib import Path

import requests
from loguru import logger

from src.config.settings import DataConfig


def download_dataset(url: str, dest: Path, force: bool = False) -> Path:
    """Download a file from url to dest.

    Skips if dest already exists and force is False.
    Returns the path to the downloaded file.
    """
    if dest.exists() and not force:
        logger.info(f"Skipping download — already exists: {dest.name}")
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading {url} → {dest}")

    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    downloaded = 0
    chunk_size = 8192

    with dest.open("wb") as fh:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                fh.write(chunk)
                downloaded += len(chunk)

    logger.info(f"Downloaded {downloaded:,} bytes → {dest.name}")

    return dest


def download_all_datasets(
    data_dir: Path,
    config: DataConfig | None = None,
    force: bool = False,
) -> dict[str, Path]:
    """Download results, goalscorers, and shootouts CSVs to data_dir/raw/.

    URLs are pulled from config (defaults to DataConfig()).
    Returns a mapping of dataset name → local path.
    """
    if config is None:
        config = DataConfig()

    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    datasets = {
        "results": config.results_url,
        "goalscorers": config.goalscorers_url,
        "shootouts": config.shootouts_url,
    }

    paths: dict[str, Path] = {}
    for name, url in datasets.items():
        dest = raw_dir / f"{name}.csv"
        paths[name] = download_dataset(url, dest, force=force)

    logger.info(f"All datasets available in {raw_dir}")
    return paths
