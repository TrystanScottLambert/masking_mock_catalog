"""
Applying the mask to the mock catalogue data.
"""

import glob
import datetime
import polars as pl
import numpy as np
from astropy.coordinates import SkyCoord
import pyvista as pv
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM
from regionx import Aperture, Polygon, apply_apertures, apply_polygons


FAINTEST_STAR_TO_BE_MASKED = 18
BRIGHTEST_STAR_TO_BE_MASKED = 3.5


# Reading in the masks
def calculate_radii(magnitudes: np.ndarray) -> np.ndarray:
    """
    Calculates the radii based on the magnitude.
    """
    cut = np.where(magnitudes > BRIGHTEST_STAR_TO_BE_MASKED)[0]
    raddii = np.repeat(7, len(magnitudes))
    raddii[cut] = 10 ** (1.3 - 0.13 * magnitudes[cut])
    return raddii / 60


def read_gaia_star_mask(file_name: str) -> list[Aperture]:
    ra, dec, mag = np.loadtxt(file_name, skiprows=1, unpack=True, delimiter=",")
    cut = np.where(mag < FAINTEST_STAR_TO_BE_MASKED)[0]
    ra, dec, mag = ra[cut], dec[cut], mag[cut]
    radii = calculate_radii(mag)
    return [Aperture(_ra, _dec, _rad) for _ra, _dec, _rad in zip(ra, dec, radii)]


def read_ghosts(file_name: str) -> list[Aperture]:
    ra, dec, radius = np.loadtxt(
        file_name, unpack=True, skiprows=1, usecols=(1, 2, 6), delimiter=","
    )
    return [Aperture(_ra, _dec, _rad / 60) for _ra, _dec, _rad in zip(ra, dec, radius)]


def read_polygons(files: list[str]) -> list[Polygon]:
    polies = []
    for file in files:
        ra_verticies, dec_verticies = np.loadtxt(
            file, unpack=True, delimiter=",", skiprows=1
        )
        polies.append(Polygon(ra_verticies, dec_verticies))
    return polies


# Reading in the mocks
def read_waves_wide(file_name: str) -> pl.DataFrame:
    df = pl.read_parquet(file_name)
    df = df.filter(pl.col("zobs") < 0.2)
    df = df.filter(pl.col("total_ap_dust_Z_VISTA") < 21.2)
    return df


def read_waves_deep(file_name: str) -> pl.DataFrame:
    df = pl.read_parquet(file_name)
    df = df.filter(pl.col("zobs") < 0.8)
    df = df.filter(pl.col("total_ap_dust_Z_VISTA") < 21.25)
    return df


def create_pv_cloud(
    ra: np.ndarray, dec: np.ndarray, redshift: np.ndarray
) -> pv.PolyData:
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    distances = cosmo.comoving_distance(redshift)
    c = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, distance=distances)
    x = c.cartesian.x.value
    y = c.cartesian.y.value
    z = c.cartesian.z.value
    points = np.column_stack((x, y, z))
    return pv.PolyData(points)


def split_deep_and_wide(
    total_file: str,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    df = pl.read_parquet(total_file)
    waves_wide = df.filter(pl.col("zobs") < 0.2)
    waves_wide = waves_wide.filter(pl.col("total_ap_dust_Z_VISTA") < 21.2)
    waves_deep = df.filter((pl.col("ra") > 360 - 21) & (pl.col("ra") < 360 - 9))
    waves_deep = waves_deep.filter((pl.col("dec") > -35) & (pl.col("dec") < -30))
    waves_deep = waves_deep.filter(pl.col("zobs") < 0.8)
    waves_deep = waves_deep.filter(pl.col("total_ap_dust_Z_VISTA") < 21.25)
    combined = waves_wide.vstack(waves_deep)
    combined = combined.unique()
    return waves_wide, waves_deep, combined


if __name__ == "__main__":
    INFILE_GAIA_MASKS = "masks/gaiastarmaskwaves.csv"
    INFILE_GHOST_MASKS = "masks/GhostLocations_v0.csv"
    INFILE_MOCK_WIDE = (
        "~/Desktop/mock_catalogs/custom_waves_requests/waves_wide_gals.parquet"
    )

    polygon_files = glob.glob("masks/MaskPolygons_v1/*.csv")

    polygons = read_polygons(polygon_files)
    ghosts = read_ghosts(INFILE_GHOST_MASKS)
    stars = read_gaia_star_mask(INFILE_GAIA_MASKS)

    waves_wide, waves_deep, combined = split_deep_and_wide(INFILE_MOCK_WIDE)

    ras_wide, decs_wide, zobs_wide, zcos_wide = (
        waves_wide["ra"].to_numpy(),
        waves_wide["dec"].to_numpy(),
        waves_wide["zobs"].to_numpy(),
        waves_wide["zcos"].to_numpy(),
    )
    ras_deep, decs_deep, zobs_deep, zcos_deep = (
        waves_deep["ra"].to_numpy(),
        waves_deep["dec"].to_numpy(),
        waves_deep["zobs"].to_numpy(),
        waves_deep["zcos"].to_numpy(),
    )
    ras_comb, decs_comb, zobs_comb, zcos_comb = (
        combined["ra"].to_numpy(),
        combined["dec"].to_numpy(),
        combined["zobs"].to_numpy(),
        combined["zcos"].to_numpy(),
    )

    tic = datetime.datetime.now()
    mask_ghosts = np.array(apply_apertures(ras_comb, decs_comb, ghosts))
    mask_stars = np.array(apply_apertures(ras_comb, decs_comb, stars))
    mask_polygons = np.array(apply_polygons(ras_comb, decs_comb, polygons))
    toc = datetime.datetime.now()
    print(f"time: {toc - tic}")

    regions = np.array([mask_stars, mask_ghosts, mask_polygons])
    masked = np.array([any(region) for region in regions.T])
    not_masked = ~masked

    cloud_unmasked = create_pv_cloud(
        ras_comb[not_masked], decs_comb[not_masked], zcos_comb[not_masked]
    )
    cloud_masked = create_pv_cloud(
        ras_comb[masked], decs_comb[masked], zcos_comb[masked]
    )
    cloud_deep = create_pv_cloud(ras_deep, decs_deep, zcos_deep)

    plotter = pv.Plotter()
    plotter.add_points(
        cloud_unmasked,
        color="black",
        point_size=1.0,
        render_points_as_spheres=False,
        label="Unmasked",
    )
    plotter.add_points(
        cloud_masked,
        color="red",
        point_size=1.0,
        render_points_as_spheres=False,
        label="Masked",
    )
    plotter.add_title("Galaxy Distribution (Black=Unmasked, Red=Masked)", font_size=12)
    plotter.add_legend()
    plotter.show_axes()
    plotter.show()

    plotter = pv.Plotter()
    plotter.add_points(
        cloud_masked,
        color="red",
        point_size=3,
        render_points_as_spheres=False,
        label="Masked",
    )
    plotter.show()

    plotter = pv.Plotter()
    plotter.add_points(
        cloud_unmasked,
        color="black",
        point_size=1.0,
        render_points_as_spheres=False,
        label="Unmasked",
    )
    plotter.add_points(
        cloud_deep,
        color="red",
        point_size=1.0,
        render_points_as_spheres=False,
        label="Masked",
    )
    plotter.add_title("Galaxy Distribution (Black=Unmasked, Red=Masked)", font_size=12)
    plotter.show()
