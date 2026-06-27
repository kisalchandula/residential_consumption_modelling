import configparser
import pandas as pd
import geopandas as gpd
from sqlalchemy import create_engine


# =====================================================
# CONFIG LOADER
# =====================================================
def load_config(config_path: str):
    """
    Load INI configuration file.

    Parameters
    ----------
    config_path : str
        Path to .ini config file

    Returns
    -------
    configparser.ConfigParser
    """
    config = configparser.ConfigParser()
    config.read(config_path)

    return config


# =====================================================
# DATABASE CONNECTION ENGINE
# =====================================================
from sqlalchemy import create_engine

def create_db_engine(config, section: str = "database"):
    """
    Create SQLAlchemy engine from config section.

    Parameters
    ----------
    config : ConfigParser
        Loaded INI file
    section : str
        Section name in INI (e.g. 'database', 'bkg')
    """

    db_host = config[section]["host"]
    db_name = config[section]["dbname"]
    db_user = config[section]["user"]
    db_pass = config[section]["password"]
    db_port = config[section]["port"]

    db_url = f"postgresql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"

    engine = create_engine(db_url)
    return engine


# =====================================================
# HOUSEHOLD PV ALLOCATED DATA LOADER
# =====================================================
def load_household_pv_data(config, schema: str, table: str):
    """
    Load household PV rooftop allocation data from PostgreSQL.

    Parameters
    ----------
    config : ConfigParser
        Loaded configuration object
    schema : str
        Database schema name (e.g. 'eam_projekt')
    table : str
        Table or materialized view name

    Returns
    -------
    pd.DataFrame
        Household PV dataset with lat/lon coordinates
    """

    engine = create_db_engine(config)

    query = f"""
    SELECT 
        an_fid,
        building_id,
        gid_unique,
        ea_p_pv,
        "uw_fid",
        slope,
        orientation,
        roof_area,
        ST_X(ST_Transform(household_geom, 4326)) AS lon,
        ST_Y(ST_Transform(household_geom, 4326)) AS lat
    FROM {schema}.{table}
    """

    df = pd.read_sql(query, engine)

    print(f"Loaded {len(df)} rows from {schema}.{table}")

    return df



# =====================================================
# HEAT PUMP HOUSEHOLD LOADER
# =====================================================
def load_household_data(config, schema: str, table: str):
    """
    Load households with heat pumps (W_WP > 0).

    Parameters
    ----------
    config : ConfigParser
        Loaded configuration object
    schema : str
        Database schema name
    table : str
        Table name

    Returns
    -------
    pd.DataFrame
        Household IDs and annual heat pump consumption.
    """

    engine = create_db_engine(config)

    query = f"""
    SELECT
    *,
    ST_X(ST_Transform("AN_geom", 4326)) AS lon,
    ST_Y(ST_Transform("AN_geom", 4326)) AS lat
    FROM eam_projekt."Kundenstruktur_NS_Anschluesse"
    WHERE "W_H0" > 0
    """

    df = pd.read_sql(query, engine)

    print(f"Loaded {len(df)} W_H0 households from {schema}.{table}")

    return df





# =====================================================
# HEAT PUMP HOUSEHOLD LOADER
# =====================================================
def load_heatpump_data(config, schema: str, table: str):
    """
    Load households with heat pumps (W_WP > 0).

    Parameters
    ----------
    config : ConfigParser
        Loaded configuration object
    schema : str
        Database schema name
    table : str
        Table name

    Returns
    -------
    pd.DataFrame
        Household IDs and annual heat pump consumption.
    """

    engine = create_db_engine(config)

    query = f"""
    SELECT
    *,
    ST_X(ST_Transform("AN_geom", 4326)) AS lon,
    ST_Y(ST_Transform("AN_geom", 4326)) AS lat
    FROM eam_projekt."Kundenstruktur_NS_Anschluesse"
    WHERE "W_WP" > 0
    """

    df = pd.read_sql(query, engine)

    print(f"Loaded {len(df)} heat pump households from {schema}.{table}")

    return df


# =====================================================
# STORAGE HEATING HOUSEHOLD LOADER
# =====================================================

def load_storage_heating_data(config, schema: str, table: str):
    """
    Load households with storage heating (W_SH > 0).

    Parameters
    ----------
    config : ConfigParser
        Database configuration
    schema : str
        Database schema name
    table : str
        Table name

    Returns
    -------
    pd.DataFrame
        Household IDs, coordinates, and annual storage heating consumption (W_SH).
    """

    engine = create_db_engine(config)

    query = f"""
    SELECT
        *,
        ST_X(ST_Transform("AN_geom", 4326)) AS lon,
        ST_Y(ST_Transform("AN_geom", 4326)) AS lat
    FROM {schema}."{table}"
    WHERE "W_SH" > 0 AND "W_H0" > 0
    """

    df = pd.read_sql(query, engine)

    print(f"Loaded {len(df)} storage heating households from {schema}.{table}")

    return df



# =====================================================
# HOUSEHOLD W_H0 MAPPING LOADER
# =====================================================
def load_household_wh0_mapping(config, schema: str, table: str):
    """
    Load household W_H0 and CWGAN cluster mapping from PostgreSQL.

    Returns
    -------
    dict
        {
            hh_id: {
                "W_H0": W_H0,
                "id_cwgan_adjusted": id_cwgan_adjusted
            }
        }
    """

    engine = create_db_engine(config)

    query = f"""
    SELECT
        hh_id,
        "W_H0",
        id_cwgan
    FROM "{schema}"."{table}"
    """

    df = pd.read_sql(query, engine)

    print(f"Loaded {len(df)} rows from {schema}.{table}")

    # clean data
    df = df.dropna(
        subset=["hh_id", "W_H0", "id_cwgan"]
    )

    # handle duplicate hh_id safely
    df = (
        df.groupby("hh_id", as_index=False)
          .agg({
              "W_H0": "mean",
              "id_cwgan": "first"
          })
    )

    mapping = (
        df.set_index("hh_id")
          [["W_H0", "id_cwgan"]]
          .to_dict("index")
    )

    return mapping


# =====================================================
# HOUSEHOLD PV LOADER
# =====================================================

def load_households_with_pv(config, schema: str, table: str):
    """
    Load households with PV systems.

    Returns
    -------
    pd.DataFrame
        Household locations and installed PV capacity.
    """

    engine = create_db_engine(config)

    query = f"""
    SELECT
        "AN_FID" AS an_fid,
        "EA_P_PV" AS ea_p_pv,
        "UW_FID" AS uw_fid,
        "AN_geom" AS geom,
        ST_X(ST_Transform("AN_geom", 4326)) AS lon,
        ST_Y(ST_Transform("AN_geom", 4326)) AS lat
    FROM {schema}."{table}"
    WHERE "EA_P_PV" > 0
    """

    df = pd.read_sql(query, engine)

    print(f"Loaded {len(df)} PV households from {schema}.{table}")

    return df



# =====================================================
# PROPERTY POLYGON LOADER
# =====================================================

def load_property_for_households(config, households, schema, table):
    """
    Load only property polygons containing households.
    """

    engine = create_db_engine(config)

    household_ids = tuple(
        households["an_fid"].tolist()
    )

    query = f"""
    SELECT
        h."AN_FID" AS an_fid,
        p.gid AS fs_gid,
        p.geom
    FROM {schema}."{table}" p
    JOIN {schema}."Kundenstruktur_NS_Anschluesse" h
        ON ST_Within(h."AN_geom", p.geom)
    WHERE h."AN_FID" IN {household_ids}
    """

    gdf = gpd.read_postgis(
        query,
        engine,
        geom_col="geom"
    )

    return gdf



# =====================================================
# PV ROOF CANDIDATE LOADER
# =====================================================

def load_roofs_for_households(config, schema: str):
    """
    Load roof parts belonging to properties containing PV households.

    Uses:
        household -> FS polygon -> building_id -> roof parts

    Returns
    -------
    GeoDataFrame
        Roof parts with household PV information.
    """

    import geopandas as gpd

    engine = create_db_engine(config)

    query = f"""
    WITH household_fs AS MATERIALIZED (

        SELECT
            h."AN_FID" AS an_fid,
            h."EA_P_PV" AS ea_p_pv,
            h."UW_FID" AS uw_fid,
            fs.gid AS fs_gid,
            fs.geom AS fs_geom

        FROM {schema}."Kundenstruktur_NS_Anschluesse" h

        JOIN {schema}.fs_tab fs
            ON ST_Within(
                h."AN_geom",
                fs.geom
            )

        WHERE h."EA_P_PV" > 0
    ),

    candidate_buildings AS MATERIALIZED (

        SELECT DISTINCT
            h.an_fid,
            h.ea_p_pv,
            h.uw_fid,
            r.building_id

        FROM household_fs h

        JOIN {schema}.roofs_centroids_tab r
            ON ST_Within(
                r.roof_centroid,
                h.fs_geom
            )

        WHERE r.building_id IS NOT NULL
    )

    SELECT

        cb.an_fid,
        cb.ea_p_pv,
        cb.uw_fid,

        r.building_id,
        r.cityobject_id,
        r.roof_type,
        r.slope,
        r.orientation,
        r.roof_area,

        r.geom

    FROM candidate_buildings cb

    JOIN {schema}.roofs_mv r
        ON cb.building_id = r.building_id

    WHERE r.roof_area IS NOT NULL
      AND r.roof_area >= 8
    """

    roofs = gpd.read_postgis(
        query,
        engine,
        geom_col="geom"
    )

    print(f"Loaded {len(roofs)} PV roof candidates")

    return roofs