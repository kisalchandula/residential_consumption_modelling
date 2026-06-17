import os
# Aktuellen Skriptpfad erhalten
script_dir = os.path.dirname(os.path.abspath(__file__))
# In das Verzeichnis wechseln
os.chdir(script_dir)
import pandas as pd
import numpy as np
import torch
import time
from sklearn.preprocessing import LabelEncoder
from sqlalchemy import create_engine, text
from models.Models import BasicEmbeddingCManyToManyHybridLSTMWFFGAN
import configparser
from pathlib import Path
import requests
from utils import dataprep_utils
import matplotlib.pyplot as plt
import io

from concurrent.futures import ThreadPoolExecutor, as_completed


class TimeSeriesGenerator:
    """
    A class to generate time series data using a pre-trained model and weather data.
    Attributes:
    -----------
    seq_length : int
        The length of the sequence to generate.
    start_time : str
        The start time for the time series index.
    end_time : str
        The end time for the time series index.
    frequency : str
        The frequency of the time series index.
    index : pd.DatetimeIndex
        The datetime index for the time series.
    index_len : int
        The length of the datetime index.
    device : str
        The device to run the model on (e.g., "cpu" or "cuda").
    model : torch.nn.Module
        The pre-trained model for generating time series data.
    engine : sqlalchemy.engine.Engine
        The SQLAlchemy engine for database connection.
    all_series_df : pd.DataFrame
        DataFrame to store all generated time series.
    Methods:
    --------
    load_model():
        Loads the pre-trained model from a file.
    create_engine():
        Creates a SQLAlchemy engine for database connection.
    wetterapi(latitude, longitude, year, model='era5land', height=None):
        Fetches weather data from an API.
    generate_time_series(look_up_table='public.near_open_meter_03241011'):
        Generates time series data for multiple locations and stores them in a DataFrame.
    get_series_by_id(id):
        Retrieves the generated time series data for a specific ID.
    """
    def __init__(self, seq_length=48, start_time="2021-01-01 00:00:00", end_time="2021-01-31 23:30:00", frequency="30min"):
        self.seq_length = seq_length
        self.start_time = start_time
        self.end_time = end_time
        self.frequency = frequency
        self.index = pd.date_range(start=self.start_time, end=self.end_time, freq=self.frequency)
        self.index_len = len(self.index)
        self.device = "cpu"
        self.model = self.load_model()
        self.engine = self.create_engine()
        self.all_series_df = pd.DataFrame(index=self.index)
        self.year = pd.to_datetime(self.start_time).year

    def load_model(self):
        path = Path(r"C:\Users\kwijeyasekera\Documents\IEE-Kisal\Masterarbeit - Kisal\src\utils\artifacts\BasicEmbeddingCManyToManyHybridLSTMWFFGAN20-02-2024-05-52-33_private.pt")
        model = BasicEmbeddingCManyToManyHybridLSTMWFFGAN.loadModel(path, device=self.device)
        return model

    def create_engine(self, db = 'bkg256'):
        config = configparser.ConfigParser()
        config.read(Path(Path.home(), "iee.ini"))
        pgsql_info = dict(config[db])
        engine = create_engine(f'postgresql://{pgsql_info["user"]}:{pgsql_info["password"]}@{pgsql_info["host"]}/{pgsql_info["database"]}')
        return engine

    def wetterapi(self, latitude, longitude, year, model="era5", height=100,
                url="https://maps.iee.fraunhofer.de/antsapi/api/data/v1/query",
                token=None):
        if token is None:
            token = os.getenv("ANTS_API_TOKEN")
        if not token:
            token = ''

        headers = {"Authorization": f"Bearer {token}"}

        requested_year = int(year)
        fallback_years = [requested_year - i for i in range(0, 4)]
        last_error = None

        for current_year in fallback_years:
            params = {
                "x": float(longitude),
                "y": float(latitude),
                "crs": "EPSG:4326",
                "spatial_interp": "nearest",
                "spatial_interp_samples": 3,
                "height": height,
                "height_interp": "nearest",
                "begin": f"{current_year}-01-01",
                "end": f"{current_year + 1}-01-01",
                "resample": "",
                "resample_method": "nearest",
                "t2m:var": "t2m",
                "format": "application/json",
                "model": model,
            }

            try:
                response = requests.get(url=url, params=params, headers=headers, timeout=60)
                response.raise_for_status()
                data = response.json()

                times = pd.to_datetime(data["data"]["coords"]["time/0"])
                t2m = np.array(data["data"]["data"][0]["data"]).squeeze()
                t2m_c = t2m - 273.15
                return pd.DataFrame(t2m_c, index=times)
            except requests.exceptions.HTTPError as exc:
                # Some years are not available; retry with older year.
                if exc.response is not None and exc.response.status_code == 400:
                    last_error = exc
                    continue
                raise

        raise RuntimeError(
            f"Weather API failed for requested year {requested_year} and fallback years {fallback_years}. "
            f"Last error: {last_error}"
        )
        

    def count_table_lines(self,look_up_table='last.near_open_meter_03241011'):
        
        query_get = text(f"SELECT count(hh_id) from {look_up_table}")

        with self.engine.connect() as connection:
            value = pd.read_sql(query_get, connection)

        return value


    def generate_time_series(self, limit = 1000, offset=0,
                             look_up_table='last.near_open_meter_03241011',
                             token=None, torken=None):
        
        #if run_in_loop:
        self.all_series_df = pd.DataFrame(index=self.index)
        
        if limit == None or limit == 0:
            limit_query = ""
        else:
            limit_query = f""" LIMIT {limit} """

        offset_query = f""" OFFSET {offset} """

        query_get = text(f"""
            SELECT id_cwgan, hh_id::integer,
            ST_X(ST_Centroid(ST_Transform(geom,4326))), 
            ST_Y(ST_Centroid(ST_Transform(geom,4326))) from {look_up_table}
            INNER JOIN fs_de.ni as b  
            ON fs_id = gid 
            ORDER BY hh_id
            {offset_query} 
            {limit_query}  
        """)


        connection = self.engine.connect()
        values = pd.read_sql(query_get, connection)
        values = values.rename(columns={"st_x": "longitude", "st_y": "latitude"})
        values = values.set_index("id_cwgan")

        for idx, row in values.iterrows():
            contCond = np.hstack([
                dataprep_utils.getDayPeriodicEmbedding(self.index),
                dataprep_utils.getWeekPeriodicEmbedding(self.index),
                dataprep_utils.getYearPeriodicEmbedding(self.index),
                dataprep_utils.getSummertimes(self.index).reshape(-1, 1)
            ])
            latitude = row.latitude
            longitude = row.longitude
            id_cwgan = idx
            id = row.hh_id
            # Keep backward compatibility for misspelled argument name "torken".
            request_token = token if token is not None else torken
            temperature_loaded = self.wetterapi(latitude, longitude, self.year, token=request_token)
            temperature_loaded = temperature_loaded.reindex(self.index).interpolate(method='time')
            temperature_filtered = temperature_loaded.loc[self.index]
            weather_data = temperature_filtered[0].to_numpy(dtype=float)
            temp_mean = np.mean(weather_data)
            temp_std = np.std(weather_data)
            tempdata = (weather_data - temp_mean) / temp_std
            contCond = np.hstack([contCond, tempdata.reshape(-1, 1)])
            contCond = torch.from_numpy(contCond.reshape((-1, self.seq_length, contCond.shape[1]))).to(torch.float32).to(self.device)
            le = LabelEncoder()
            le.classes_ = np.load("clusteridsprivate.labelencoder.npy")
            cluster_labels_csv = pd.read_csv("sorted_cluster_ids_maxes_private.csv")
            cluster_id = cluster_labels_csv.label[id_cwgan]
            maxes = cluster_labels_csv.max_val[id_cwgan]
            catCond = np.hstack([
                np.full((self.index_len, 1), le.transform([cluster_id])),
                np.full((self.index_len, 1), le.transform([cluster_id])),
                np.full((self.index_len, 1), id)
            ])
            catCond = torch.from_numpy(catCond.reshape(-1, self.seq_length, catCond.shape[1])).to(torch.int32).to(self.device)

            # Generate data and convert to DataFrame
            series = self.model.generator.getSample(catCond.shape[0], self.seq_length, catCond, contCond).detach().cpu().flatten()
            series_df = pd.DataFrame({"timeseries": series.numpy().tolist()}, index=self.index) * maxes / 1000  # for KW
            series_df[series_df < 0] = 0 # set values under zero to zero

            # Collect all series in a DataFrame
            self.all_series_df[id] = series_df['timeseries']

    def get_series_by_id(self, id):
        if id in self.all_series_df.columns:
            return self.all_series_df[id]
        else:
            raise ValueError(f"ID '{id}' not found in the DataFrame columns.")

    def save_data(self, destination='csv', table_name='time_series_data', db='timescale256'):
        """
        Saves the generated time series data to a CSV file or PostgreSQL database with TimescaleDB extension.
        Parameters:
        -----------
        destination : str
            The destination to save the data ('csv' or 'postgres').
        table_name : str
            The name of the table to save the data in PostgreSQL.
        db : str
            The database configuration to use for saving the data.
        """
        if destination == 'csv':
            self.all_series_df.to_csv(f'{table_name}.csv')
            print(f"Data saved to {table_name}.csv")
        elif destination == 'postgres':
            engine_save = self.create_engine(db)
            connection = engine_save.connect()
            
            # Create table schema based on DataFrame columns with prefix 'fs_de_'
            columns = ', '.join([f'fs_de_{col} DOUBLE PRECISION' for col in self.all_series_df.columns])
            create_table_query = text(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    time TIMESTAMPTZ NOT NULL,
                    {columns}
                );
            """)
            #SELECT create_hypertable('{table_name}', 'time', if_not_exists => TRUE);
            connection.execute(create_table_query)
            
            # Rename DataFrame columns with prefix 'fs_de_'
            renamed_df = self.all_series_df.rename(columns=lambda x: f'fs_de_{x}')
            
            # Convert DataFrame to long format for insertion
            long_df = renamed_df.reset_index().melt(id_vars=['index'], var_name='fs_de_column_name', value_name='timeseries')
            long_df.rename(columns={'index': 'time'}, inplace=True)
            
            # Insert data into PostgreSQL using batch inserts
            batch_size = 1000
            for start in range(0, len(long_df), batch_size):
                end = start + batch_size
                batch_df = long_df.iloc[start:end]
                batch_df.to_sql(table_name, engine_save, if_exists='append', index=False)
            
            print(f"Data saved to PostgreSQL table {table_name}")
        else:
            raise ValueError("Invalid destination. Use 'csv' or 'postgres'.")

    def save_data_fast(
                self,
                destination: str = "postgres",
                table_name: str = "time_series_data",
                db: str = "timescale256",
                mode: str = "long",                 # "long" (empfohlen) oder "wide"
                create_hypertable: bool = True,
                chunk_time_interval: str | None = None,  # z.B. "7 days" (optional)
                dropna: bool = True,
            ):
        """
        Schnelle Alternative zu save_data(): nutzt COPY statt to_sql/INSERT.

        mode="long" (empfohlen):
            Tabelle: (time, series_id, value) und COPY von stacked DataFrame.
            -> sehr gut für Timescale/Hypertables und typische Abfragen.

        mode="wide":
            Tabelle: (time, fs_de_<id1>, fs_de_<id2>, ...) und COPY im Wide-Format.
            -> nur sinnvoll, wenn du wirklich wide abfragen willst.
        """
        if destination != "postgres":
            raise ValueError("save_data_fast unterstützt nur destination='postgres'.")

    
        engine_save = self.create_engine(db)

        # 1) Wide DataFrame vorbereiten
        wide_df = self.all_series_df.copy()
        wide_df.index.name = "time"

        # Timescale/PG: lieber naive NaNs vermeiden (optional)
        if dropna:
            # Zeilen löschen, die komplett NaN sind (sollte bei dir selten passieren)
            wide_df = wide_df.dropna(how="all")

        with engine_save.begin() as conn:
            if mode == "long":
                # 2a) Schema (long)
                conn.execute(text(f"""
                    CREATE TABLE IF NOT EXISTS {table_name} (
                        time TIMESTAMPTZ NOT NULL,
                        series_id INTEGER NOT NULL,
                        value DOUBLE PRECISION
                    );
                """))

            if create_hypertable:
                # Timescale: hypertable anlegen (wenn Extension vorhanden)
                # optional chunk_time_interval setzen
                if chunk_time_interval:
                    conn.execute(text(f"""
                        SELECT create_hypertable(
                            '{table_name}', 'time',
                            chunk_time_interval => INTERVAL '{chunk_time_interval}',
                            if_not_exists => TRUE
                        );
                    """))
                else:
                    conn.execute(text(f"""
                        SELECT create_hypertable(
                            '{table_name}', 'time',
                            if_not_exists => TRUE
                        );
                    """))

                # 3a) Long-DF bauen (Achtung: kann riesig werden)
                # series_id = Spaltenname (id) als int
                long_df = (
                    wide_df
                    .stack(dropna=dropna)
                    .reset_index()
                    .rename(columns={"level_1": "series_id", 0: "value"})
                )
                # Sicherstellen, dass series_id int ist (bei dir hh_id)
                long_df["series_id"] = long_df["series_id"].astype(int)

                # 4a) COPY
                raw_conn = conn.connection  # DBAPI connection (psycopg2)
                buf = io.StringIO()
                long_df.to_csv(buf, index=False, header=False)
                buf.seek(0)

                with raw_conn.cursor() as cur:
                    cur.copy_expert(
                        f"COPY {table_name} (time, series_id, value) FROM STDIN WITH (FORMAT csv)",
                        buf
                    )

            elif mode == "wide":
                # 2b) Schema (wide)
                # Spalten im DB-Table mit Prefix fs_de_
                col_map = {c: f"fs_de_{int(c)}" for c in wide_df.columns}
                wide_db_df = wide_df.rename(columns=col_map)

                # CREATE TABLE Statement
                cols_sql = ",\n".join([f'"{c}" DOUBLE PRECISION' for c in wide_db_df.columns])
                conn.execute(text(f"""
                    CREATE TABLE IF NOT EXISTS {table_name} (
                        time TIMESTAMPTZ NOT NULL,
                        {cols_sql}
                    );
                """))

                if create_hypertable:
                    if chunk_time_interval:
                        conn.execute(text(f"""
                            SELECT create_hypertable(
                                '{table_name}', 'time',
                                chunk_time_interval => INTERVAL '{chunk_time_interval}',
                                if_not_exists => TRUE
                            );
                        """))
                    else:
                        conn.execute(text(f"""
                            SELECT create_hypertable(
                                '{table_name}', 'time',
                                if_not_exists => TRUE
                            );
                        """))

                # 3b) COPY wide (time + alle Spalten)
                export_df = wide_db_df.reset_index()  # time als Spalte

                raw_conn = conn.connection
                buf = io.StringIO()
                export_df.to_csv(buf, index=False, header=False)
                buf.seek(0)

                # Spaltenreihenfolge muss exakt passen
                cols = ["time"] + list(wide_db_df.columns)
                cols_list = ", ".join([f'"{c}"' for c in cols])

                with raw_conn.cursor() as cur:
                    cur.copy_expert(
                        f"COPY {table_name} ({cols_list}) FROM STDIN WITH (FORMAT csv)",
                        buf
                    )
                    

    def get_grid_id(self, lat, lon, resolution=0.25):
        return (
            round(lat / resolution) * resolution,
            round(lon / resolution) * resolution
        )

    def generate_time_series_fast(
        self,
        limit=1000,
        offset=0,
        look_up_table='last.near_open_meter_03241011',
        token=None,
        max_workers=8
    ):

        # ----------------------------
        # SQL
        # ----------------------------
        limit_query = "" if (limit is None or limit == 0) else f" LIMIT {int(limit)} "
        offset_query = f" OFFSET {int(offset)} "

        query_get = text(f"""
            SELECT
                id_cwgan,
                hh_id::integer,
                ST_X(ST_Centroid(ST_Transform(geom,4326))) AS longitude,
                ST_Y(ST_Centroid(ST_Transform(geom,4326))) AS latitude
            FROM {look_up_table}
            INNER JOIN fs_de.ni AS b
                ON fs_id = gid
            ORDER BY hh_id
            {offset_query}
            {limit_query}
        """)

        with self.engine.connect() as connection:
            values = pd.read_sql(query_get, connection)

        values = values.set_index("id_cwgan")

        # ----------------------------
        # FAST GRID ID (vectorized)
        # ----------------------------
        values["grid_id"] = [
            self.get_grid_id(lat, lon)
            for lat, lon in zip(values["latitude"], values["longitude"])
        ]

        # ----------------------------
        # STATIC FEATURES (precompute ONCE)
        # ----------------------------
        cont_base = np.hstack([
            dataprep_utils.getDayPeriodicEmbedding(self.index),
            dataprep_utils.getWeekPeriodicEmbedding(self.index),
            dataprep_utils.getYearPeriodicEmbedding(self.index),
            dataprep_utils.getSummertimes(self.index).reshape(-1, 1)
        ])

        cont_base = cont_base.astype(np.float32)

        # ----------------------------
        # LOAD MODELS ONCE
        # ----------------------------
        le = LabelEncoder()
        le.classes_ = np.load(r"C:\Users\kwijeyasekera\Documents\IEE-Kisal\Masterarbeit - Kisal\src\utils\artifacts\clusteridsprivate.labelencoder.npy")
        cluster_labels_csv = pd.read_csv(r"C:\Users\kwijeyasekera\Documents\IEE-Kisal\Masterarbeit - Kisal\src\utils\artifacts\sorted_cluster_ids_maxes_private.csv")

        # ----------------------------
        # STEP 1: UNIQUE GRIDS
        # ----------------------------
        unique_grids = values["grid_id"].unique()

        # ----------------------------
        # STEP 2: PARALLEL WEATHER FETCH
        # ----------------------------
        grid_temperature_data = {}

        def fetch_grid(grid):
            lat, lon = grid
            temp_df = self.wetterapi(lat, lon, self.year, token=token)

            temp_df = temp_df.reindex(self.index).interpolate(method="time")
            series = temp_df.iloc[:, 0].to_numpy(dtype=np.float32)

            return grid, series

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(fetch_grid, g) for g in unique_grids]

            for f in as_completed(futures):
                grid, data = f.result()
                grid_temperature_data[grid] = data

        # ----------------------------
        # STEP 3: GENERATION
        # ----------------------------
        all_series = {}

        for row in values.itertuples():
            id_cwgan = row.Index
            hh_id = row.hh_id
            grid_id = row.grid_id

            weather_data = grid_temperature_data[grid_id]

            # normalize
            std = weather_data.std() or 1.0
            tempdata = (weather_data - weather_data.mean()) / std

            # continuous condition
            contCond_np = np.hstack([cont_base, tempdata.reshape(-1, 1)])

            contCond = torch.from_numpy(
                contCond_np.reshape((-1, self.seq_length, contCond_np.shape[1]))
            ).float().to(self.device)

            # labels
            cluster_id = cluster_labels_csv.loc[id_cwgan, "label"]
            maxes = float(cluster_labels_csv.loc[id_cwgan, "max_val"])

            cluster_encoded = le.transform([cluster_id])[0]

            # categorical condition
            catCond_np = np.column_stack([
                np.full(self.index_len, cluster_encoded),
                np.full(self.index_len, cluster_encoded),
                np.full(self.index_len, hh_id),
            ])

            catCond = torch.from_numpy(
                catCond_np.reshape((-1, self.seq_length, catCond_np.shape[1]))
            ).int().to(self.device)

            # inference
            series = (
                self.model.generator
                .getSample(catCond.shape[0], self.seq_length, catCond, contCond)
                .detach()
                .cpu()
                .numpy()
                .ravel()
            )

            series = series * (maxes / 1000.0)
            series = np.clip(series, 0, None)

            all_series[hh_id] = series

        # ----------------------------
        # FINAL DF
        # ----------------------------
        self.all_series_df = pd.DataFrame(all_series, index=self.index)
        self.all_series_df = self.all_series_df.reindex(
            sorted(self.all_series_df.columns), axis=1
        )

    
if __name__ == "__main__":
    # Example usage
    generator = TimeSeriesGenerator(start_time="2020-01-01 00:00:00", end_time="2020-12-31 23:59:59")
    generator.generate_time_series_fast(look_up_table='last.near_open_meter_08215059_mv', limit=None)
    print(generator.all_series_df.head())

    # generator.save_data(destination='postgres', table_name='time_series_data', db='timescale256')
    generator.save_data_fast(
        destination="postgres",
        table_name="load_time_series_08201509",
        db="timescale256",
        mode="long",
        create_hypertable=True,
        chunk_time_interval="7 days"
    )

    # Plot the generated time series
    # generator.all_series_df.plot()
    # plt.show()
