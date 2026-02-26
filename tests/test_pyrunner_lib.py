"""Tests for pyrunner_lib module."""

import json
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
import polars as pl
import pandas as pd

from pyrunner_lib.pyrunner_lib import (
    Config,
    Connection,
    load_json,
    load_configuration,
    write_column_trace_file,
    write_df_to_parquet,
    path_to_name,
    load_module,
    read_parquet_files,
    transform,
    load_ds_meta,
    DS_META_DATA,
    PyrunnerError,
    ConfigurationError,
    TransformNotFoundError,
    ModuleLoadError,
    DataLoadError,
    DataWriteError,
)


class TestConfig:
    """Test Config model."""
    
    def test_config_creation(self):
        """Test Config model creation."""
        config = Config(
            python="3.8",
            dfType="polars",
            packages=["polars", "pandas"]
        )
        assert config.python == "3.8"
        assert config.dfType == "polars"
        assert config.packages == ["polars", "pandas"]


class TestConnection:
    """Test Connection model."""
    
    def test_connection_creation(self):
        """Test Connection model creation."""
        conn = Connection(
            id="test_transform",
            path="/path/to/transform.py",
            inputs=["input1", "input2"]
        )
        assert conn.id == "test_transform"
        assert conn.path == "/path/to/transform.py"
        assert conn.inputs == ["input1", "input2"]


class TestLoadJson:
    """Test load_json function."""
    
    def test_load_json_success(self):
        """Test successful JSON loading."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({
                "python": "3.8",
                "dfType": "polars",
                "packages": ["polars"]
            }, f)
            f.flush()
            
            try:
                result = load_json(f.name, Config)
                assert isinstance(result, Config)
                assert result.python == "3.8"
                assert result.dfType == "polars"
            finally:
                os.unlink(f.name)
    
    def test_load_json_file_not_found(self):
        """Test JSON file not found error."""
        with pytest.raises(ConfigurationError, match="Configuration file not found"):
            load_json("nonexistent.json", Config)
    
    def test_load_json_invalid_json(self):
        """Test invalid JSON error."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("invalid json content")
            f.flush()
            
            try:
                with pytest.raises(ConfigurationError, match="Invalid JSON"):
                    load_json(f.name, Config)
            finally:
                os.unlink(f.name)
    
    def test_load_json_validation_error(self):
        """Test validation error."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({"invalid": "data"}, f)
            f.flush()
            
            try:
                with pytest.raises(ConfigurationError, match="Invalid configuration"):
                    load_json(f.name, Config)
            finally:
                os.unlink(f.name)


class TestPathToName:
    """Test path_to_name function."""
    
    def test_path_to_name(self):
        """Test path to name conversion."""
        assert path_to_name("/path/to/file.py") == "file"
        assert path_to_name("file.py") == "file"
        assert path_to_name("/path/to/file") == "file"


class TestWriteColumnTraceFile:
    """Test write_column_trace_file function."""
    
    def test_write_column_trace_file_success(self):
        """Test successful column trace file writing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a simple LazyFrame
            df = pl.LazyFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
            
            with patch('pyrunner_lib.pyrunner_lib.trace_input_sources') as mock_trace:
                mock_trace.return_value = {"a": ["source1"], "b": ["source2"]}
                
                write_column_trace_file(df, "test_transform", temp_dir)
                
                # Check that the file was created
                trace_file = Path(temp_dir) / "data" / "test_transform" / "meta" / "columns.json"
                assert trace_file.exists()
                
                # Check file contents
                with open(trace_file) as f:
                    data = json.load(f)
                    assert data == {"a": ["source1"], "b": ["source2"]}
    

class TestWriteDfToParquet:
    """Test write_df_to_parquet function."""
    
    def test_write_polars_lazyframe(self):
        """Test writing Polars LazyFrame."""
        with tempfile.TemporaryDirectory() as temp_dir:
            df = pl.LazyFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
            file_path = os.path.join(temp_dir, "test.parquet")
            
            with patch('pyrunner_lib.pyrunner_lib.write_column_trace_file') as mock_trace:
                write_df_to_parquet(df, file_path, "test_transform", temp_dir)
                
                # Check that the file was created
                assert os.path.exists(file_path)
                mock_trace.assert_called_once()
    
    def test_write_polars_dataframe(self):
        """Test writing Polars DataFrame."""
        with tempfile.TemporaryDirectory() as temp_dir:
            df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
            file_path = os.path.join(temp_dir, "test.parquet")
            
            write_df_to_parquet(df, file_path, "test_transform", temp_dir)
            
            # Check that the file was created
            assert os.path.exists(file_path)
    
    def test_write_pandas_dataframe(self):
        """Test writing Pandas DataFrame."""
        with tempfile.TemporaryDirectory() as temp_dir:
            df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
            file_path = os.path.join(temp_dir, "test.parquet")
            
            write_df_to_parquet(df, file_path, "test_transform", temp_dir)
            
            # Check that the file was created
            assert os.path.exists(file_path)
    
    def test_write_unsupported_type(self):
        """Test writing unsupported dataframe type."""
        with tempfile.TemporaryDirectory() as temp_dir:
            df = "not a dataframe"
            file_path = os.path.join(temp_dir, "test.parquet")
            
            with pytest.raises(DataWriteError, match="Failed to write Parquet file"):
                write_df_to_parquet(df, file_path, "test_transform", temp_dir)
    
    def test_write_error(self):
        """Test write error handling."""
        with tempfile.TemporaryDirectory() as temp_dir:
            df = pl.DataFrame({"a": [1, 2, 3]})
            # Use invalid path to cause error
            file_path = "/invalid/path/test.parquet"
            
            with pytest.raises(DataWriteError, match="Failed to write Parquet file"):
                write_df_to_parquet(df, file_path, "test_transform", temp_dir)


class TestLoadModule:
    """Test load_module function."""
    
    def test_load_module_success(self):
        """Test successful module loading."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("""
def transform(df):
    return df
""")
            f.flush()
            
            try:
                module = load_module("test_module", f.name)
                assert hasattr(module, 'transform')
                assert callable(module.transform)
            finally:
                os.unlink(f.name)
    
    def test_load_module_file_not_found(self):
        """Test module loading with non-existent file."""
        with pytest.raises(ModuleLoadError, match="Failed to load module"):
            load_module("test_module", "nonexistent.py")
    
    def test_load_module_syntax_error(self):
        """Test module loading with syntax error."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("invalid python syntax")
            f.flush()
            
            try:
                with pytest.raises(ModuleLoadError, match="Failed to load module"):
                    load_module("test_module", f.name)
            finally:
                os.unlink(f.name)


class TestReadParquetFiles:
    """Test read_parquet_files function."""
    
    def test_read_parquet_files_polars(self):
        """Test reading parquet files with Polars."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test parquet file
            df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
            parquet_path = os.path.join(temp_dir, "data.parquet")
            df.write_parquet(parquet_path)
            
            # Create connection
            conn = Connection(
                id="test_conn",
                path="/path/to/transform.py",
                inputs=["input1"]
            )
            connections = {"input1": conn}
            
            with patch('pyrunner_lib.pyrunner_lib.DF_TYPE', 'polars'):
                with patch('pyrunner_lib.pyrunner_lib.pl.scan_parquet') as mock_scan:
                    mock_scan.return_value = pl.LazyFrame({"a": [1, 2, 3]})
                    
                    result = read_parquet_files(connections, ["input1"], temp_dir)
                    
                    assert "input1" in result
                    mock_scan.assert_called_once()
    
    def test_read_parquet_files_pandas(self):
        """Test reading parquet files with Pandas."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test parquet file
            df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
            parquet_path = os.path.join(temp_dir, "data.parquet")
            df.to_parquet(parquet_path)
            
            # Create connection
            conn = Connection(
                id="test_conn",
                path="/path/to/transform.py",
                inputs=["input1"]
            )
            connections = {"input1": conn}
            
            with patch('pyrunner_lib.pyrunner_lib.DF_TYPE', 'pandas'):
                with patch('pyrunner_lib.pyrunner_lib.pd.read_parquet') as mock_read:
                    mock_read.return_value = pd.DataFrame({"a": [1, 2, 3]})
                    
                    result = read_parquet_files(connections, ["input1"], temp_dir)
                    
                    assert "input1" in result
                    mock_read.assert_called_once()
    
    def test_read_parquet_files_missing_connection(self):
        """Test reading parquet files with missing connection."""
        connections = {}
        
        with patch('pyrunner_lib.pyrunner_lib.DF_TYPE', 'polars'):
            with pytest.raises(DataLoadError, match="Input parameter 'missing' not found"):
                read_parquet_files(connections, ["missing"], "/tmp")

    def test_read_parquet_files_draft_build_limit(self):
        """Test that read_parquet_files limits rows to 1,000,000 when IS_DRAFT_BUILD is true."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test parquet file with some dummy data
            # Use 5 rows for simplicity in testing the .head() call
            df = pl.DataFrame({"a": range(10)})
            rid = "test-rid"
            os.makedirs(os.path.join(temp_dir, "data", rid, "datasets"), exist_ok=True)
            parquet_path = os.path.join(temp_dir, "data", rid, "datasets", "data.parquet")
            df.write_parquet(parquet_path)
            
            conn = Connection(id=rid, path="test.py", inputs=[])
            connections = {"input1": conn}
            
            # Case 1: IS_DRAFT_BUILD is "true", limit to 2 rows for test
            with patch.dict(os.environ, {"IS_DRAFT_BUILD": "true"}):
                with patch('pyrunner_lib.pyrunner_lib.DF_TYPE', 'polars'):
                    # We need to mock the 1,000_000 to something smaller for the test 
                    # OR just verify that .head() was called.
                    # Actually, the implementation uses a hardcoded 1_000_000.
                    # To test it properly with a smaller dataset, we can check if it returns all rows if < 1M.
                    result = read_parquet_files(connections, {"input1": rid}, temp_dir)
                    df_result = result["input1"].collect()
                    assert len(df_result) == 10
            
            # Case 2: Verify it limits if we had more than 1M rows (mocking .head)
            with patch.dict(os.environ, {"IS_DRAFT_BUILD": "true"}):
                with patch('pyrunner_lib.pyrunner_lib.DF_TYPE', 'polars'):
                    with patch('polars.LazyFrame.head') as mock_head:
                        mock_head.return_value = pl.LazyFrame({"a": [1]})
                        read_parquet_files(connections, {"input1": rid}, temp_dir)
                        mock_head.assert_called_once_with(1_000_000)

            # Case 3: Verify it does NOT limit if IS_DRAFT_BUILD is not set
            with patch.dict(os.environ, {}, clear=True):
                with patch('pyrunner_lib.pyrunner_lib.DF_TYPE', 'polars'):
                    with patch('polars.LazyFrame.head') as mock_head:
                        read_parquet_files(connections, {"input1": rid}, temp_dir)
                        mock_head.assert_not_called()

            # Case 4: Verify Pandas limit
            with patch.dict(os.environ, {"IS_DRAFT_BUILD": "true"}):
                with patch('pyrunner_lib.pyrunner_lib.DF_TYPE', 'pandas'):
                    with patch('pandas.read_parquet') as mock_read:
                        mock_df = MagicMock()
                        mock_read.return_value = mock_df
                        read_parquet_files(connections, {"input1": rid}, temp_dir)
                        mock_df.head.assert_called_once_with(1_000_000)

    def test_read_parquet_files_explicit_rid_fallback(self):
        """Test that explicit RIDs can fall back while others cannot."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test parquet file for the "explicit" RID
            df = pl.DataFrame({"a": [1, 2, 3]})
            rid = "explicit-rid"
            os.makedirs(os.path.join(temp_dir, "data", rid, "datasets"), exist_ok=True)
            df.write_parquet(os.path.join(temp_dir, "data", rid, "datasets", "data.parquet"))
            
            connections = {} # Empty connections
            
            with patch('pyrunner_lib.pyrunner_lib.DF_TYPE', 'polars'):
                # 1. Explicit RID should work even if not in connections
                param_mapping = {"df": rid}
                explicit_params = {"df"}
                result = read_parquet_files(connections, param_mapping, temp_dir, explicit_params=explicit_params)
                assert "df" in result
                
                # 2. Non-explicit RID should fail if not in connections
                param_mapping = {"other": "some-rid"}
                with pytest.raises(DataLoadError, match="Input parameter 'other' not found"):
                    read_parquet_files(connections, param_mapping, temp_dir, explicit_params=set())
    
    def test_read_parquet_files_configuration_not_loaded(self):
        """Test reading parquet files without configuration."""
        with patch('pyrunner_lib.pyrunner_lib.DF_TYPE', None):
            with pytest.raises(ConfigurationError, match="Configuration not loaded"):
                read_parquet_files({}, [], "/tmp")


class TestTransform:
    """Test transform function."""
    
    def test_transform_success(self):
        """Test successful transformation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test files
            config_file = os.path.join(temp_dir, "_config.json")
            connections_file = os.path.join(temp_dir, "_connections.json")
            
            with open(config_file, 'w') as f:
                json.dump({
                    "python": "3.8",
                    "dfType": "polars",
                    "packages": ["polars"]
                }, f)
            
            # Create test transform module
            transform_file = os.path.join(temp_dir, "transform.py")
            with open(transform_file, 'w') as f:
                f.write("""
def transform(input1):
    return input1
""")
            
                with open(connections_file, 'w') as f:
                    json.dump([{
                        "id": "test_transform",
                        "path": transform_file,
                        "inputs": ["input1"]
                    }], f)
            
            # Mock the file paths
            with patch('pyrunner_lib.pyrunner_lib.CONFIG_FILE', config_file):
                with patch('pyrunner_lib.pyrunner_lib.CONNECTIONS_FILE', connections_file):
                    with patch('pyrunner_lib.pyrunner_lib.read_parquet_files') as mock_read:
                        mock_read.return_value = {"input1": pl.LazyFrame({"a": [1, 2, 3]})}
                        
                        with patch('pyrunner_lib.pyrunner_lib.write_df_to_parquet') as mock_write:
                            transform("test_transform", temp_dir)
                            
                            mock_read.assert_called_once()
                            mock_write.assert_called_once()
    
    def test_transform_not_found(self):
        """Test transform with non-existent transform ID."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = os.path.join(temp_dir, "_config.json")
            connections_file = os.path.join(temp_dir, "_connections.json")
            
            with open(config_file, 'w') as f:
                json.dump({
                    "python": "3.8",
                    "dfType": "polars",
                    "packages": ["polars"]
                }, f)
            
            with open(connections_file, 'w') as f:
                json.dump([{
                    "id": "other_transform",
                    "path": "/path/to/transform.py",
                    "inputs": []
                }], f)
            
            with patch('pyrunner_lib.pyrunner_lib.CONFIG_FILE', config_file):
                with patch('pyrunner_lib.pyrunner_lib.CONNECTIONS_FILE', connections_file):
                    with pytest.raises(TransformNotFoundError, match="Transform with ID 'test_transform' not found"):
                        transform("test_transform", temp_dir)
    
    def test_transform_module_no_transform_function(self):
        """Test transform with module missing transform function."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = os.path.join(temp_dir, "_config.json")
            connections_file = os.path.join(temp_dir, "_connections.json")
            
            with open(config_file, 'w') as f:
                json.dump({
                    "python": "3.8",
                    "dfType": "polars",
                    "packages": ["polars"]
                }, f)
            
            with open(connections_file, 'w') as f:
                json.dump([{
                    "id": "test_transform",
                    "path": "/path/to/transform.py",
                    "inputs": []
                }], f)
            
            with patch('pyrunner_lib.pyrunner_lib.CONFIG_FILE', config_file):
                with patch('pyrunner_lib.pyrunner_lib.CONNECTIONS_FILE', connections_file):
                    with patch('pyrunner_lib.pyrunner_lib.load_module') as mock_load:
                        mock_module = MagicMock()
                        # Remove transform attribute
                        del mock_module.transform
                        mock_load.return_value = mock_module
                        
                        with pytest.raises(ModuleLoadError, match="does not contain a 'transform' function"):
                            transform("test_transform", temp_dir)


