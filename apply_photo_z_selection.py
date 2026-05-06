"""
Script that will apply the photo-z selection to the final mocks.
"""

from pathlib import Path
import polars as pl


def read_in_photo_z_selection(file_name: Path) -> pl.DataFrame:
    df = pl.read_csv(file_name)
    # reading in the empiriacl photo-z selection
    df = df.select(["id_galaxy_sky", "wide_photoz_selected_empirical"])
    return df


def main():
    first_file_name = "combined.parquet"
    photo_z_file_name = Path("photo-z-selection/sharks_wide_completeness_selection.csv")
    df_shark = pl.read_parquet(first_file_name)
    df = read_in_photo_z_selection(photo_z_file_name)
    df_shark = df_shark.join(df, on="id_galaxy_sky", how="full")
    df_shark = df_shark.with_columns(
        pl.col("wide_photoz_selected_empirical").fill_null(True)
    )
    df_shark = df_shark.rename({"wide_photoz_selected_empirical": "photo_z_selected"})
    df_shark.write_parquet(first_file_name)


if __name__ == "__main__":
    main()
