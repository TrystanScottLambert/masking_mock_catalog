"""
Adds the metadata to the appropriate parquet file.
"""

import polars as pl
from pathlib import Path


def add_maml(root_name: str) -> None:
    """
    Adds the maml contents to the metadata of the corresponding parquet file.
    """
    maml_text = Path(f"{root_name}.maml").read_text()
    df = pl.read_parquet(f"{root_name}.parquet")
    df.write_parquet(f"{root_name}.parquet", metadata={"maml": maml_text})


if __name__ == "__main__":
    GROUP_NAME = "groups_shark"
    GALAXY_NAME = "galaxies_shark"
    add_maml(GROUP_NAME)
    add_maml(GALAXY_NAME)
