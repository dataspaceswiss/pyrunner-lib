import importlib.util
import os
import time
import json
import sys
from typing import Dict, Any, Union, List, Optional
from pathlib import Path
from pydantic import BaseModel, ValidationError
import polars as pl
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
from . import input_source_tracer as ist
import warnings


# Custom exceptions
class PyrunnerError(Exception):
    """Base exception for pyrunner library."""
    pass


class ConfigurationError(PyrunnerError):
    """Raised when configuration files are invalid or missing."""
    pass


class TransformNotFoundError(PyrunnerError):
    """Raised when a transform ID is not found."""
    pass


class ModuleLoadError(PyrunnerError):
    """Raised when a transform module cannot be loaded."""
    pass


class DataLoadError(PyrunnerError):
    """Raised when input data cannot be loaded."""
    pass


class DataWriteError(PyrunnerError):
    """Raised when output data cannot be written."""
    pass

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
    """Load and validate JSON files with proper error handling."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return model(**data) if isinstance(data, dict) else [model(**item) for item in data]
    except FileNotFoundError as e:
        raise ConfigurationError(f"Configuration file not found: {file_path}") from e
    except json.JSONDecodeError as e:
        raise ConfigurationError(f"Invalid JSON in {file_path}: {e}") from e
    except ValidationError as e:
        raise ConfigurationError(f"Invalid configuration in {file_path}: {e}") from e


CONFIG_FILE = "_config.json"
CONNECTIONS_FILE = "_connections.json"

# Global configuration - will be loaded when needed
CONFIG: Optional[Config] = None
CONNECTIONS: Optional[List[Connection]] = None
DF_TYPE: Optional[str] = None


def load_configuration() -> None:
    """Load configuration files with error handling."""
    global CONFIG, CONNECTIONS, DF_TYPE
    try:
        CONFIG = load_json(CONFIG_FILE, Config)
        CONNECTIONS = load_json(CONNECTIONS_FILE, Connection)
        DF_TYPE = CONFIG.dfType
    except ConfigurationError:
        raise
    except Exception as e:
        raise ConfigurationError(f"Unexpected error loading configuration: {e}") from e


# Utility functions
def write_column_trace_file(lf: pl.LazyFrame, transform_id: str, base_path: str) -> None:
    """Write column trace file with proper error handling."""
    try:
        serial_plan = lf.serialize(format="json")
        origins = ist.trace_input_sources(serial_plan)

        trace_file_path = f'{base_path}/data/{transform_id}/meta/columns.json'
        os.makedirs(os.path.dirname(trace_file_path), exist_ok=True)
        with open(trace_file_path, "w", encoding="utf-8") as f:
            json.dump(origins, f, indent=4)
    except Exception as e:
        # Don't raise an error here since this is not a critical error
        print(f"Failed to write column trace file: {e}", file=sys.stderr)


def write_df_to_parquet(df: Union[pl.DataFrame, pl.LazyFrame, pd.DataFrame], file_path: str, transform_id: str, base_path: str) -> None:
    """
    Writes a DataFrame to a Parquet file with automatic type detection.
    
    Args:
        df: Can be a Polars DataFrame, Polars LazyFrame, or Pandas DataFrame
        file_path: Path where to save the parquet file
        transform_id: ID of the transform for tracing
        base_path: Base path for the project
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
        raise DataWriteError(f"Failed to write Parquet file: {e}") from e


def path_to_name(path: str) -> str:
    """Extracts the filename (without extension) from a given path."""
    return Path(path).stem


def load_module(module_name: str, file_path: str) -> Any:
    """Dynamically loads a module from a given file path."""
    try:
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            raise ModuleLoadError(f"Could not create module spec for {file_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        raise ModuleLoadError(f"Failed to load module from {file_path}: {e}") from e


def read_parquet_files(connections: Dict[str, Connection], params: List[str], base_path: str, transform_id: str = None) -> Dict[str, Union[pl.LazyFrame, pd.DataFrame]]:
    """Reads input Parquet files based on the provided connections."""
    if DF_TYPE is None:
        raise ConfigurationError("Configuration not loaded. Call load_configuration() first.")
    
    data_dict = {}
    for param in params:
        conn = connections.get(param)
        if conn:
            file_path = f'{base_path}/data/{conn.id}/datasets/data.parquet'
            try:
                if DF_TYPE == "polars":
                    df = pl.scan_parquet(file_path)
                else:
                    df = pd.read_parquet(file_path, engine='pyarrow')
                
                # Inject ds_meta attribute into the dataframe
                if transform_id is not None:
                    # Create a lazy property for previous_lf that only scans when accessed
                    class LazyPreviousLF:
                        def __init__(self, transform_id, base_path):
                            self.transform_id = transform_id
                            self.base_path = base_path
                            self._lf = None
                        
                        def _load_previous(self):
                            if self._lf is None:
                                previous_path = f'{self.base_path}/data/{self.transform_id}/datasets/data.parquet'
                                try:
                                    self._lf = pl.scan_parquet(previous_path)
                                except Exception:
                                    # Return None if previous parquet doesn't exist
                                    self._lf = None
                            return self._lf
                        
                        @property
                        def value(self):
                            return self._load_previous()
                    
                    # Create a custom class for ds_meta with dot notation access
                    class DSMeta:
                        def __init__(self, transform_id, base_path):
                            self.transform_id = transform_id
                            self.artifact_dir = f"/data/{transform_id}/artifacts"
                            self._lazy_previous = LazyPreviousLF(transform_id, base_path)
                        
                        @property
                        def previous_lf(self):
                            return self._lazy_previous.value
                    
                    df.ds_meta = DSMeta(transform_id, base_path)
                
                data_dict[param] = df
            except Exception as e:
                raise DataLoadError(f"Failed to load {param}. Ensure the input dataset exists at {file_path}: {e}") from e
        else:
            raise DataLoadError(f"Input parameter '{param}' not found in dataset connections")
    return data_dict


def transform(transform_id: str, base_path: str = "") -> None:
    """Main function to execute the transformation process."""
    try:
        # Load configuration
        load_configuration()
        
        if CONNECTIONS is None:
            raise ConfigurationError("Connections not loaded")
        
        connections = {path_to_name(conn.path): conn for conn in CONNECTIONS}
        transform_conn = next((c for c in CONNECTIONS if c.id == transform_id), None)

        if not transform_conn:
            raise TransformNotFoundError(f"Transform with ID '{transform_id}' not found.")

        module_name = path_to_name(transform_conn.path)
        transform_func = load_module(module_name, transform_conn.path)

        if not hasattr(transform_func, 'transform'):
            raise ModuleLoadError(f"Module {transform_conn.path} does not contain a 'transform' function.")

        params = list(transform_func.transform.__code__.co_varnames[:transform_func.transform.__code__.co_argcount])

        # Read input data
        start_time = time.time()
        data_dict = read_parquet_files(connections, params, base_path, transform_id)
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
        
    except PyrunnerError:
        # Re-raise our custom exceptions
        raise
    except Exception as e:
        # Catch any unexpected errors
        raise PyrunnerError(f"Unexpected error during transformation: {e}") from e
