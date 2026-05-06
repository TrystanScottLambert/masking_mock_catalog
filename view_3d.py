"""
PyVista visualization of WAVES wide selection.
Targets with wide_photoz_selected_empirical == False are shown in red.
"""

import polars as pl
import numpy as np
from astropy.coordinates import SkyCoord
import pyvista as pv
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM


def read_waves_wide(file_name: str) -> pl.DataFrame:
    """Apply WAVES wide selection cuts."""
    df = pl.read_parquet(file_name)
    df = df.filter(pl.col("redshift_observed") < 0.2)
    df = df.filter(pl.col("mag_Z_VISTA") < 21.1)
    return df


def create_pv_cloud(
    ra: np.ndarray, dec: np.ndarray, redshift: np.ndarray
) -> pv.PolyData:
    """Convert sky coordinates and redshift to a 3D point cloud."""
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    distances = cosmo.comoving_distance(redshift)
    c = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, distance=distances)
    x = c.cartesian.x.value
    y = c.cartesian.y.value
    z = c.cartesian.z.value
    points = np.column_stack((x, y, z))
    return pv.PolyData(points)


if __name__ == "__main__":
    INFILE = "delete.parquet"

    # Load and apply WAVES wide selection
    waves_wide = read_waves_wide(INFILE)

    ra = waves_wide["ra"].to_numpy()
    dec = waves_wide["dec"].to_numpy()
    zcos = waves_wide["redshift_cosmological"].to_numpy()
    selected = waves_wide["wide_photoz_selected_empirical"].to_numpy()

    not_selected = ~selected

    print(f"Total WAVES wide targets: {len(selected)}")
    print(f"Selected (True):     {selected.sum()}")
    print(f"Not selected (False): {not_selected.sum()}")

    # Build point clouds
    cloud_selected = create_pv_cloud(ra[selected], dec[selected], zcos[selected])
    cloud_not_selected = create_pv_cloud(
        ra[not_selected], dec[not_selected], zcos[not_selected]
    )

    # Plot
    plotter = pv.Plotter()
    plotter.add_points(
        cloud_selected,
        color="black",
        point_size=1.0,
        render_points_as_spheres=False,
        label="Selected (True)",
    )
    plotter.add_points(
        cloud_not_selected,
        color="red",
        point_size=2.0,
        render_points_as_spheres=False,
        label="Not selected (False)",
    )
    plotter.add_title("WAVES Wide: photo-z selection (Red = False)", font_size=12)
    plotter.add_legend()
    plotter.show_axes()
    plotter.show()
