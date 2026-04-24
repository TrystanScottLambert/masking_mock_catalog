"""
Script built to convert the Shark-standard mock names to WAVES-standard column names
"""

import polars as pl


def convert_log(data_frame: pl.DataFrame, column_name: str) -> pl.DataFrame:
    """
    Takes the column name and coverts it into
    """
    return data_frame.with_columns((10 ** pl.col(column_name)).alias(column_name))


COLUMN_NAME_MAP_GALAXIES = {
    "zobs": "redshift_observed",
    "zcos": "redshift_cosmological",
    "pa": "position_angle",
    "mstars_disk": "mass_stellar_disk",
    "mstars_bulge": "mass_stellar_bulge",
    "mvir_hosthalo": "mass_virial_hosthalo",
    "mvir_subhalo": "mass_virial_subhalo",
    "rstar_disk_apparent": "radius_disk_star_apparent",
    "rstar_bulge_apparent": "radius_bulge_star_apparent",
    "rstar_disk_intrinsic": "radius_disk_star_instrinsic",
    "vpec_x": "vel_peculiar_x",
    "vpec_y": "vel_peculiar_y",
    "vpec_z": "vel_peculiar_z",
    "vpec_r": "vel_peculiar_radial",
    "log_mstar_total": "mass_stellar_total",
    "log_sfr_total": "sfr_total",
    "total_ap_dust_FUV_GALEX": "mag_FUV_GALEX",
    "total_ap_dust_NUV_GALEX": "mag_NUV_GALEX",
    "total_ap_dust_u_SDSS": "mag_u_SDSS",
    "total_ap_dust_g_SDSS": "mag_g_SDSS",
    "total_ap_dust_r_SDSS": "mag_r_SDSS",
    "total_ap_dust_i_SDSS": "mag_i_SDSS",
    "total_ap_dust_z_SDSS": "mag_z_SDSS",
    "total_ap_dust_u_VST": "mag_u_VST",
    "total_ap_dust_g_VST": "mag_g_VST",
    "total_ap_dust_r_VST": "mag_r_VST",
    "total_ap_dust_i_VST": "mag_i_VST",
    "total_ap_dust_Z_VISTA": "mag_Z_VISTA",
    "total_ap_dust_Y_VISTA": "mag_Y_VISTA",
    "total_ap_dust_J_VISTA": "mag_J_VISTA",
    "total_ap_dust_H_VISTA": "mag_H_VISTA",
    "total_ap_dust_K_VISTA": "mag_K_VISTA",
    "total_ap_dust_W1_WISE": "mag_W1_WISE",
    "total_ap_dust_I1_Spitzer": "mag_I1_Spitzer",
    "total_ap_dust_I2_Spitzer": "mag_I2_Spitzer",
    "total_ap_dust_W2_WISE": "mag_W2_WISE",
    "total_ap_dust_I3_Spitzer": "mag_I3_Spitzer",
    "total_ap_dust_I4_Spitzer": "mag_I4_Spitzer",
    "total_ap_dust_W3_WISE": "mag_W3_WISE",
    "total_ap_dust_W4_WISE": "mag_W4_WISE",
    "total_ap_dust_M24_Spitzer": "mag_M24_Spitzer",
    "total_ap_dust_M70_Spitzer": "mag_M70_Spitzer",
    "total_ap_dust_P70_Herschel": "mag_P70_Herschel",
    "total_ap_dust_P100_Herschel": "mag_P100_Herschel",
    "total_ap_dust_P160_Herschel": "mag_P160_Herschel",
    "total_ap_dust_S250_Herschel": "mag_S250_Herschel",
    "total_ap_dust_S350_Herschel": "mag_S350_Herschel",
    "total_ap_dust_S450_JCMT": "mag_S450_JCMT",
    "total_ap_dust_S500_Herschel": "mag_S500_Herschel",
    "total_ap_dust_S850_JCMT": "mag_S850_JCMT",
    "total_ap_dust_Band_ionising_photons": "mag_Band_ionising_photons",
    "total_ap_dust_Band9_ALMA": "mag_Band9_ALMA",
    "total_ap_dust_Band8_ALMA": "mag_Band8_ALMA",
    "total_ap_dust_Band7_ALMA": "mag_Band7_ALMA",
    "total_ap_dust_Band6_ALMA": "mag_Band6_ALMA",
    "total_ap_dust_Band5_ALMA": "mag_Band5_ALMA",
    "total_ap_dust_Band4_ALMA": "mag_Band4_ALMA",
    "total_ap_dust_Band3_ALMA": "mag_Band3_ALMA",
    "total_ap_dust_BandX_VLA": "mag_BandX_VLA",
    "total_ap_dust_BandC_VLA": "mag_BandC_VLA",
    "total_ap_dust_BandS_VLA": "mag_BandS_VLA",
    "total_ap_dust_BandL_VLA": "mag_BandL_VLA",
    "total_ap_dust_Band_610MHz": "mag_Band_610MHz",
    "total_ap_dust_Band_325MHz": "mag_Band_325MHz",
    "total_ap_dust_Band_150MHz": "mag_Band_150MHz",
    "total_ab_dust_FUV_GALEX": "mag_abs_FUV_GALEX",
    "total_ab_dust_NUV_GALEX": "mag_abs_NUV_GALEX",
    "total_ab_dust_u_SDSS": "mag_abs_u_SDSS",
    "total_ab_dust_g_SDSS": "mag_abs_g_SDSS",
    "total_ab_dust_r_SDSS": "mag_abs_r_SDSS",
    "total_ab_dust_i_SDSS": "mag_abs_i_SDSS",
    "total_ab_dust_z_SDSS": "mag_abs_z_SDSS",
    "total_ab_dust_u_VST": "mag_abs_u_VST",
    "total_ab_dust_g_VST": "mag_abs_g_VST",
    "total_ab_dust_r_VST": "mag_abs_r_VST",
    "total_ab_dust_i_VST": "mag_abs_i_VST",
    "total_ab_dust_Z_VISTA": "mag_abs_Z_VISTA",
    "total_ab_dust_Y_VISTA": "mag_abs_Y_VISTA",
    "total_ab_dust_J_VISTA": "mag_abs_J_VISTA",
    "total_ab_dust_H_VISTA": "mag_abs_H_VISTA",
    "total_ab_dust_K_VISTA": "mag_abs_K_VISTA",
    "total_ab_dust_W1_WISE": "mag_abs_W1_WISE",
    "total_ab_dust_I1_Spitzer": "mag_abs_I1_Spitzer",
    "total_ab_dust_I2_Spitzer": "mag_abs_I2_Spitzer",
    "total_ab_dust_W2_WISE": "mag_abs_W2_WISE",
    "total_ab_dust_I3_Spitzer": "mag_abs_I3_Spitzer",
    "total_ab_dust_I4_Spitzer": "mag_abs_I4_Spitzer",
    "total_ab_dust_W3_WISE": "mag_abs_W3_WISE",
    "total_ab_dust_W4_WISE": "mag_abs_W4_WISE",
    "total_ab_dust_M24_Spitzer": "mag_abs_M24_Spitzer",
    "total_ab_dust_M70_Spitzer": "mag_abs_M70_Spitzer",
    "total_ab_dust_P70_Herschel": "mag_abs_P70_Herschel",
    "total_ab_dust_P100_Herschel": "mag_abs_P100_Herschel",
    "total_ab_dust_P160_Herschel": "mag_abs_P160_Herschel",
    "total_ab_dust_S250_Herschel": "mag_abs_S250_Herschel",
    "total_ab_dust_S350_Herschel": "mag_abs_S350_Herschel",
    "total_ab_dust_S450_JCMT": "mag_abs_S450_JCMT",
    "total_ab_dust_S500_Herschel": "mag_abs_S500_Herschel",
    "total_ab_dust_S850_JCMT": "mag_abs_S850_JCMT",
    "total_ab_dust_Band_ionising_photons": "mag_abs_Band_ionising_photons",
    "total_ab_dust_Band9_ALMA": "mag_abs_Band9_ALMA",
    "total_ab_dust_Band8_ALMA": "mag_abs_Band8_ALMA",
    "total_ab_dust_Band7_ALMA": "mag_abs_Band7_ALMA",
    "total_ab_dust_Band6_ALMA": "mag_abs_Band6_ALMA",
    "total_ab_dust_Band5_ALMA": "mag_abs_Band5_ALMA",
    "total_ab_dust_Band4_ALMA": "mag_abs_Band4_ALMA",
    "total_ab_dust_Band3_ALMA": "mag_abs_Band3_ALMA",
    "total_ab_dust_BandX_VLA": "mag_abs_BandX_VLA",
    "total_ab_dust_BandC_VLA": "mag_abs_BandC_VLA",
    "total_ab_dust_BandS_VLA": "mag_abs_BandS_VLA",
    "total_ab_dust_BandL_VLA": "mag_abs_BandL_VLA",
    "total_ab_dust_Band_610MHz": "mag_abs_Band_610MHz",
    "total_ab_dust_Band_325MHz": "mag_abs_Band_325MHz",
    "total_ab_dust_Band_150MHz": "mag_abs_Band_150MHz",
}

COLUMN_NAME_MAP_GROUPS = {
    "zobs": "redshift_observed",
    "zcos": "redshift_cosmological",
    "zcmb": "redshift_cmb",
    "vpec_x": "vel_peculiar_x",
    "vpec_y": "vel_peculiar_y",
    "vpec_z": "vel_peculiar_z",
    "vpec_r": "vel_peculiar_radial",
    "mvir": "mass_virial",
    "n_galaxies_total": "number_galaxies_total",
    "n_galaxies_selected": "number_galaxies_selected",
}

if __name__ == "__main__":
    df_galaxies = pl.read_parquet("combined.parquet")
    df_galaxies = convert_log(df_galaxies, "log_mstar_total")
    df_galaxies = convert_log(df_galaxies, "log_sfr_total")
    print(df_galaxies["log_mstar_total"])
    df_galaxies = df_galaxies.rename(COLUMN_NAME_MAP_GALAXIES)
    df_galaxies = df_galaxies.with_columns(
        [
            pl.when(pl.col(c) == -999).then(None).otherwise(pl.col(c)).alias(c)
            for c in df_galaxies.columns
        ]
    )
    df_galaxies = df_galaxies.with_columns(pl.col("ra").replace(360, 0).alias("ra"))
    df_galaxies.write_parquet("galaxies_shark.parquet")

    df_groups = pl.read_parquet(
        "../mock_catalogs/custom_waves_requests/waves_wide_groups.parquet"
    )
    df_groups = df_groups.rename(COLUMN_NAME_MAP_GROUPS)
    df_groups = df_groups.with_columns(
        [
            pl.when(pl.col(c) == -999).then(None).otherwise(pl.col(c)).alias(c)
            for c in df_groups.columns
        ]
    )
    df_groups = df_groups.with_columns(pl.col("ra").replace(360, 0).alias("ra"))
    df_groups.write_parquet("groups_shark.parquet")
