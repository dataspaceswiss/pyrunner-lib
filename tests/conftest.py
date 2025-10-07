"""Shared test fixtures and configuration."""

import json
import tempfile
import os
import pytest
import polars as pl
import pandas as pd


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def sample_config():
    """Sample configuration data."""
    return {
        "python": "3.8",
        "dfType": "polars",
        "packages": ["polars", "pandas"]
    }


@pytest.fixture
def sample_connections():
    """Sample connections data."""
    return [
        {
            "id": "test_transform",
            "path": "/path/to/transform.py",
            "inputs": ["input1", "input2"]
        },
        {
            "id": "input_transform",
            "path": "/path/to/input.py",
            "inputs": []
        }
    ]


@pytest.fixture
def sample_parquet_file(temp_dir):
    """Create a sample parquet file for testing."""
    df = pl.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
        "age": [25, 30, 35, 40, 45],
        "salary": [50000, 60000, 70000, 80000, 90000]
    })
    
    parquet_path = os.path.join(temp_dir, "data.parquet")
    df.write_parquet(parquet_path)
    return parquet_path


@pytest.fixture
def sample_pandas_parquet_file(temp_dir):
    """Create a sample pandas parquet file for testing."""
    df = pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
        "age": [25, 30, 35, 40, 45],
        "salary": [50000, 60000, 70000, 80000, 90000]
    })
    
    parquet_path = os.path.join(temp_dir, "data.parquet")
    df.to_parquet(parquet_path)
    return parquet_path


@pytest.fixture
def sample_transform_module(temp_dir):
    """Create a sample transform module for testing."""
    transform_code = '''
def transform(input1, input2=None):
    """Sample transform function."""
    if input2 is not None:
        return input1.join(input2, on="id")
    return input1
'''
    
    transform_path = os.path.join(temp_dir, "transform.py")
    with open(transform_path, 'w') as f:
        f.write(transform_code)
    
    return transform_path


@pytest.fixture
def sample_lazy_query_plan():
    """Sample lazy query plan for testing."""
    return {
        "Scan": {
            "sources": {
                "Paths": ["/path/to/rid.transform123/data.parquet"]
            }
        }
    }


@pytest.fixture
def sample_complex_query_plan():
    """Sample complex query plan for testing."""
    return {
        "Select": {
            "expr": [
                {"Column": "id"},
                {"Column": "name"},
                {
                    "Alias": [
                        {"Column": "age"},
                        "user_age"
                    ]
                }
            ]
        },
        "input": {
            "Filter": {
                "predicate": {
                    "BinaryExpr": {
                        "left": {"Column": "age"},
                        "right": {"Literal": {"value": 30}}
                    }
                },
                "input": {
                    "Scan": {
                        "sources": {
                            "Paths": ["/path/to/rid.transform123/data.parquet"]
                        }
                    }
                }
            }
        }
    }
