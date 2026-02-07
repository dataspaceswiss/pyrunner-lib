import importlib.util
import os
import time
import json
import sys
import inspect
from typing import Dict, Any, Union, List, Optional, Callable
from pathlib import Path
from pydantic import BaseModel, ValidationError
import polars as pl
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
from .input_source_tracer import trace_input_sources
from .health_check import run_health_checks
import warnings


class Transform:
    """Class to represent a transformation input with an RID."""
    def __init__(self, rid: str):
        self.rid = rid

    def __repr__(self):
        return f"Transform('{self.rid}')"


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
DS_META_DATA: Optional[Dict[str, Dict[str, Any]]] = None


def load_ds_meta() -> None:
    """Load DS_META data from environment variable."""
    global DS_META_DATA
    try:
        ds_meta_json = os.environ.get('DS_META')
        if ds_meta_json:
            ds_meta_list = json.loads(ds_meta_json)
            # Convert list to dict keyed by TransformId for easy lookup
            DS_META_DATA = {item['TransformId']: item for item in ds_meta_list}
        else:
            DS_META_DATA = {}
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Warning: Failed to parse DS_META environment variable: {e}", file=sys.stderr)
        DS_META_DATA = {}
    except Exception as e:
        print(f"Warning: Unexpected error loading DS_META: {e}", file=sys.stderr)
        DS_META_DATA = {}


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
        origins = trace_input_sources(serial_plan)

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


from functools import wraps

def transform(*args, **kwargs):
    """
    Acts as both a decorator and the main runner function for transformations.
    
    Decorator usage:
        @transform
        def my_func(input): ...
        
        @transform(input=Transform("rid-xxx"))
        def my_func(input): ...
        
    Runner usage:
        transform("transform_id", base_path="")
    """
    if len(args) == 1 and callable(args[0]):
        # Case: @transform
        func = args[0]
        func._is_pyrunner_transform = True
        return func
    elif len(args) == 0 and kwargs:
        # Case: @transform(input=Transform("..."))
        def wrapper(func):
            func._is_pyrunner_transform = True
            func._pyrunner_metadata = kwargs
            return func
        return wrapper
    else:
        # Case: transform("id", "path") - Runner mode
        return _execute_transform(*args, **kwargs)

# Mark as the runner function to avoid picking it up during discovery
transform._is_pyrunner_runner = True


def _execute_transform(transform_id: str, base_path: str = "") -> None:
    """Internal function to execute the transformation process."""
    try:
        # Load configuration and DS_META data
        load_configuration()
        load_ds_meta()
        
        if CONNECTIONS is None:
            raise ConfigurationError("Connections not loaded")
        
        connections = {path_to_name(conn.path): conn for conn in CONNECTIONS}
        transform_conn = next((c for c in CONNECTIONS if c.id == transform_id), None)

        if not transform_conn:
            raise TransformNotFoundError(f"Transform with ID '{transform_id}' not found.")

        module_name = path_to_name(transform_conn.path)
        module = load_module(module_name, transform_conn.path)

        # Find the transform function
        # 1. Look for a function named 'transform' (that isn't our own runner)
        # 2. Look for functions decorated with @transform
        main_func = None
        
        if hasattr(module, 'transform') and callable(module.transform) and not getattr(module.transform, '_is_pyrunner_runner', False):
            main_func = module.transform
        else:
            # Prefer decorated functions
            decorated_funcs = [
                obj for name, obj in inspect.getmembers(module)
                if callable(obj) and getattr(obj, '_is_pyrunner_transform', False) is True
            ]
            if decorated_funcs:
                # If multiple, pick the first one found.
                main_func = decorated_funcs[0]

        if not main_func:
            raise ModuleLoadError(f"Module {transform_conn.path} does not contain a 'transform' function or @transform decorated function.")

        # Inspect function signature
        sig = inspect.signature(main_func)
        params = sig.parameters
        
        # Determine mapping for each parameter
        metadata = getattr(main_func, '_pyrunner_metadata', {})
        param_mapping = {} # parameter_name -> RID or legacy_name
        explicit_params = set()
        
        for name, param in params.items():
            # Skip *args and **kwargs
            if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                continue
                
            # Check decorator metadata first
            if name in metadata:
                val = metadata[name]
                if isinstance(val, Transform):
                    param_mapping[name] = val.rid
                    explicit_params.add(name)
                else:
                    param_mapping[name] = val # Assume it's an RID string or legacy name
            # Check type annotation
            elif isinstance(param.annotation, Transform):
                param_mapping[name] = param.annotation.rid
                explicit_params.add(name)
            else:
                # Default to parameter name
                param_mapping[name] = name

        # Read input data
        start_time = time.time()
        data_dict = read_parquet_files(connections, param_mapping, base_path, transform_id, explicit_params)
        read_time = time.time() - start_time

        # Execute transformation
        start_time = time.time()
        transform_result = main_func(**data_dict)
        transform_time = time.time() - start_time

        # Handle optional health_checks return
        if isinstance(transform_result, tuple) and len(transform_result) == 2:
            result, health_checks = transform_result
        else:
            result = transform_result
            health_checks = None

        # Execute health check if provided
        health_check_time = None
        if health_checks and isinstance(result, (pl.LazyFrame, pl.DataFrame, pd.DataFrame)):
            start_time = time.time()
            run_health_checks(result, health_checks)
            health_check_time = time.time() - start_time

        # Write output
        start_time = time.time()
        output_path = f'{base_path}/data/{transform_conn.id}/datasets/data.parquet'
        write_df_to_parquet(result, output_path, transform_conn.id, base_path)
        write_time = time.time() - start_time

        print("--------- Build successful ---------")
        print(f"Read Time:      {read_time:.2f} sec")
        print(f"Transform Time: {transform_time:.2f} sec")
        if health_check_time is not None:
            print(f"Health Check Time: {health_check_time:.2f} sec")
        print(f"Write Time:     {write_time:.2f} sec")
        
    except PyrunnerError:
        # Re-raise our custom exceptions
        raise
    except Exception as e:
        # Catch any unexpected errors
        raise PyrunnerError(f"Unexpected error during transformation: {e}") from e


def read_parquet_files(connections: Dict[str, Connection], params: Union[List[str], Dict[str, str]], base_path: str, transform_id: str = None, explicit_params: Optional[set] = None) -> Dict[str, Union[pl.LazyFrame, pd.DataFrame]]:
    """Reads input Parquet files based on parameter mapping (RID or legacy name)."""
    if DF_TYPE is None:
        raise ConfigurationError("Configuration not loaded. Call load_configuration() first.")
    
    data_dict = {}
    
    # Convert list of params to identity mapping for backward compatibility
    if isinstance(params, list):
        param_mapping = {p: p for p in params}
    else:
        param_mapping = params

    # Create internal lookup for connections by ID (RID) and by legacy name (path stem)
    conn_by_id = {conn.id: conn for conn in connections.values()}
    conn_by_name = connections # Already keyed by path stem
    
    for param_name, mapping_key in param_mapping.items():
        # Try finding by ID first (RID), then by name
        conn = conn_by_id.get(mapping_key) or conn_by_name.get(mapping_key)
        
        if not conn:
            # Fallback to mapping_key as RID only if it was explicitly provided via Transform()
            is_explicit = explicit_params is not None and param_name in explicit_params
            if not is_explicit:
                raise DataLoadError(f"Input parameter '{param_name}' not found in dataset connections")
            rid = mapping_key
        else:
            rid = conn.id
        
        file_path = f'{base_path}/data/{rid}/datasets/data.parquet'
        try:
            if DF_TYPE == "polars":
                df = pl.scan_parquet(file_path)
            else:
                df = pd.read_parquet(file_path, engine='pyarrow')
            
            # Inject ds_meta attribute into the dataframe
            if transform_id is not None:                                                                                       
                # Create a custom class for ds_meta with dot notation access
                class DSMeta:
                    def __init__(self, rid, ds_meta_data=None):
                        self.transform_id = rid
                        self.artifact_dir = f"/data/{rid}/artifacts"
                        
                        # Add DS_META data if available
                        if ds_meta_data:
                            self.data_snapshot_id = ds_meta_data.get('Id')
                            self.build_id = ds_meta_data.get('BuildId')
                            self.row_count = ds_meta_data.get('RowCount')
                            self.column_count = ds_meta_data.get('ColumnCount')
                            self.file_size = ds_meta_data.get('FileSize')
                            self.schema = ds_meta_data.get('Schema')
                            self.creation_date = ds_meta_data.get('CreationDate')
                            
                            # Add schema columns for easy access
                            if self.schema and 'Columns' in self.schema:
                                self.columns = self.schema['Columns']
                            else:
                                self.columns = []
                        else:
                            # Default values when DS_META is not available
                            self.data_snapshot_id = None
                            self.build_id = None
                            self.row_count = None
                            self.column_count = None
                            self.file_size = None
                            self.schema = None
                            self.creation_date = None
                            self.columns = []
                
                # Get DS_META data for this RID (data snapshot)
                ds_meta_data = DS_META_DATA.get(rid) if DS_META_DATA else None
                df.ds_meta = DSMeta(rid, ds_meta_data)
            
            data_dict[param_name] = df
        except Exception as e:
            raise DataLoadError(f"Failed to load {param_name}. Ensure the input dataset exists at {file_path}: {e}") from e
    return data_dict
