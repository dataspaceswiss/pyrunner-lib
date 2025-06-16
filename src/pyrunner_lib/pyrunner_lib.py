import importlib.util
import os
import time
import json
import sys
from typing import Dict, Any, Union, List
from pathlib import Path
from pydantic import BaseModel, ValidationError
import polars as pl
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
from . import input_source_tracer as ist
import warnings

warnings.filterwarnings(
    action='ignore',
    message="'json' serialization format of LazyFrame is deprecated",
    category=UserWarning
)

# JSON Schema Models
class Config(BaseModel):
    python: str
    dfType: str
    packages: List[str]


class Connection(BaseModel):
    id: str
    path: str
    inputs: List[str]


# Load and validate JSON files
def load_json(file_path: str, model: BaseModel) -> Any:
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        return model(**data) if isinstance(data, dict) else [model(**item) for item in data]
    except (FileNotFoundError, json.JSONDecodeError, ValidationError) as e:
        print(f"Failed to load or validate {file_path}: {e}", file=sys.stderr)
        sys.exit(1)


CONFIG_FILE = "_config.json"
CONNECTIONS_FILE = "_connections.json"

CONFIG: Config = load_json(CONFIG_FILE, Config)
CONNECTIONS: List[Connection] = load_json(CONNECTIONS_FILE, Connection)

DF_TYPE = CONFIG.dfType


# Utility functions
def write_column_trace_file(lf: pl.LazyFrame, transform_id: str, base_path: str) -> None:
    serial_plan = lf.serialize(format="json")
    origins = ist.trace_input_sources(serial_plan)

    try:
        trace_file_path = f'{base_path}/data/{transform_id}/meta/columns.json'
        os.makedirs(os.path.dirname(trace_file_path), exist_ok=True)
        with open(trace_file_path, "w") as f:
            json.dump(origins, f, indent=4)
    except Exception as e:
        print(f"Failed to write column trace file: {e}", file=sys.stderr)


def write_df_to_parquet(df: Union[pl.DataFrame, pl.LazyFrame, pd.DataFrame], file_path: str, transform_id: str, base_path: str) -> None:
    """
    Writes a DataFrame to a Parquet file with automatic type detection.
    
    Args:
        df: Can be a Polars DataFrame, Polars LazyFrame, or Pandas DataFrame
        file_path: Path where to save the parquet file
    """
    try:
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Check dataframe type and handle accordingly
        if isinstance(df, pl.LazyFrame):
            # If it's a LazyFrame, collect it first
            write_column_trace_file(df, transform_id, base_path)
            df.sink_parquet(file_path)
        elif isinstance(df, pl.DataFrame):
            # If it's a Polars DataFrame, write directly
            df.write_parquet(file_path)
        elif isinstance(df, pd.DataFrame):
            # If it's a Pandas DataFrame, use pyarrow
            table = pa.Table.from_pandas(df, preserve_index=False)
            pq.write_table(table, file_path)
        else:
            raise TypeError(f"Unsupported dataframe type: {type(df)}")
            
    except Exception as e:
        print(f"Failed to write Parquet file: {e}", file=sys.stderr)
        sys.exit(1)


def path_to_name(path: str) -> str:
    """Extracts the filename (without extension) from a given path."""
    return Path(path).stem


def load_module(module_name: str, file_path: str) -> Any:
    """Dynamically loads a module from a given file path."""
    try:
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        print(f"Failed to load module from {file_path}: {e}", file=sys.stderr)
        sys.exit(1)


def read_parquet_files(connections: Dict[str, Connection], params: List[str], base_path: str) -> Dict[str, Union[pl.LazyFrame, pd.DataFrame]]:
    """Reads input Parquet files based on the provided connections."""
    data_dict = {}
    for param in params:
        conn = connections.get(param)
        if conn:
            file_path = f'{base_path}/data/{conn.id}/datasets/data.parquet'
            try:
                if DF_TYPE == "polars":
                    data_dict[param] = pl.scan_parquet(file_path)
                else:
                    data_dict[param] = pd.read_parquet(file_path, engine='pyarrow')
            except Exception as e:
                print(f"Failed to load {param}. Ensure the input dataset exists!", file=sys.stderr)
                sys.exit(1)
        else:
            print(f"Input parameter '{param}' not found in dataset connections", file=sys.stderr)
            sys.exit(1)
    return data_dict


def transform(transform_id: str, base_path: str = "") -> None:
    """Main function to execute the transformation process."""

    connections = {path_to_name(conn.path): conn for conn in CONNECTIONS}
    transform_conn = next((c for c in CONNECTIONS if c.id == transform_id), None)

    if not transform_conn:
        print(f"Transform with ID '{transform_id}' not found.", file=sys.stderr)
        sys.exit(1)

    module_name = path_to_name(transform_conn.path)
    transform_func = load_module(module_name, transform_conn.path)

    if not hasattr(transform_func, 'transform'):
        print(f"Module {transform_conn.path} does not contain a 'transform' function.", file=sys.stderr)
        sys.exit(1)

    params = list(transform_func.transform.__code__.co_varnames[:transform_func.transform.__code__.co_argcount])

    # Read input data
    start_time = time.time()
    data_dict = read_parquet_files(connections, params, base_path)
    read_time = time.time() - start_time

    # Execute transformation
    start_time = time.time()
    result = transform_func.transform(**data_dict)
    transform_time = time.time() - start_time

    # Write output
    start_time = time.time()
    output_path = f'{base_path}/data/{transform_conn.id}/datasets/data.parquet'
    write_df_to_parquet(result, output_path, transform_conn.id, base_path)
    write_time = time.time() - start_time

    print("--------- Build successful ---------")
    print(f"Read Time:      {read_time:.2f} sec")
    print(f"Transform Time: {transform_time:.2f} sec")
    print(f"Write Time:     {write_time:.2f} sec")
