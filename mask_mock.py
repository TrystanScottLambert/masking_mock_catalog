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


# Applying masks

if __name__ == "__main__":
    INFILE_GAIA_MASKS = "masks/gaiastarmaskwaves.csv"
    INFILE_GHOST_MASKS = "masks/GhostLocations_v0.csv"
    INFILE_MOCK_WIDE = (
        "~/Desktop/mock_catalogs/offical_waves_mocks/v0.4.0/waves_wide_gals.parquet"
    )
    polygon_files = glob.glob("masks/MaskPolygons_v1/*.csv")

    polygons = read_polygons(polygon_files)
    ghosts = read_ghosts(INFILE_GHOST_MASKS)
    stars = read_gaia_star_mask(INFILE_GAIA_MASKS)

    waves_wide = read_waves_wide(INFILE_MOCK_WIDE)

    ras, decs, zobs, zcos = (
        waves_wide["ra"].to_numpy(),
        waves_wide["dec"].to_numpy(),
        waves_wide["zobs"].to_numpy(),
        waves_wide["zcos"].to_numpy(),
    )
    tic = datetime.datetime.now()
    mask_ghosts = np.array(apply_apertures(ras, decs, ghosts))
    mask_stars = np.array(apply_apertures(ras, decs, stars))
    mask_polygons = np.array(apply_polygons(ras, decs, polygons))
    toc = datetime.datetime.now()
    print(f"time: {toc - tic}")

    regions = np.array([mask_stars, mask_ghosts, mask_polygons])
    masked = np.array([any(region) for region in regions.T])
    not_masked = ~masked
    print("here 1")
    # plot
    # plt.scatter(ras[not_masked], decs[not_masked], color="k", s=0.1)
    # plt.show()
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    c_obs = SkyCoord(
        ra=ras[not_masked] * u.deg,
        dec=decs[not_masked] * u.deg,
        distance=cosmo.comoving_distance(zobs[not_masked]),
    )
    c_obs_masked = SkyCoord(
        ra=ras[masked] * u.deg,
        dec=decs[masked] * u.deg,
        distance=cosmo.comoving_distance(zobs[masked]),
    )
    c_cos = SkyCoord(
        ra=ras[not_masked] * u.deg,
        dec=decs[not_masked] * u.deg,
        distance=cosmo.comoving_distance(zcos[not_masked]),
    )
    x_obs = c_obs.cartesian.x.value
    y_obs = c_obs.cartesian.y.value
    z_obs = c_obs.cartesian.z.value

    x_obs_masked = c_obs_masked.cartesian.x.value
    y_obs_masked = c_obs_masked.cartesian.y.value
    z_obs_masked = c_obs_masked.cartesian.z.value

    x_cos = c_cos.cartesian.x.value
    y_cos = c_cos.cartesian.y.value
    z_cos = c_cos.cartesian.z.value
    print("here 2")
    # Create PyVista 3D scatter plot with both masked and unmasked
    points_unmasked = np.column_stack((x_obs, y_obs, z_obs))
    cloud_unmasked = pv.PolyData(points_unmasked)

    points_masked = np.column_stack((x_obs_masked, y_obs_masked, z_obs_masked))
    cloud_masked = pv.PolyData(points_masked)

    print("here 3")
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
    print("here 4")
    plotter.add_title("Galaxy Distribution (Black=Unmasked, Red=Masked)", font_size=12)
    plotter.add_legend()
    plotter.show_axes()
    plotter.show()

    plotter = pv.Plotter()
    plotter.add_points(
        cloud_masked,
        color="red",
        point_size=10,
        render_points_as_spheres=False,
        label="Masked",
    )
    plotter.show()
