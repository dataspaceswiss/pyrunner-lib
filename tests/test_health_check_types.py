"""Tests for health_check module with different dataframe types."""

import os
import tempfile

import pandas as pd
import polars as pl

from pyrunner_lib.health_check import Check, run_health_checks


def written(frame, directory):
    """Write a transform output the way the runner does and return its path."""
    path = os.path.join(directory, "data.parquet")
    if isinstance(frame, pl.LazyFrame):
        frame.sink_parquet(path)
    elif isinstance(frame, pl.DataFrame):
        frame.write_parquet(path)
    else:
        frame.to_parquet(path, index=False)
    return path

class TestHealthCheckTypes:
    """Test health checks on different DataFrame types."""
    
    def test_polars_dataframe(self):
        """Test that health checks work on eager Polars DataFrame."""
        with tempfile.TemporaryDirectory() as temp_dir:
            os.environ["META_FOLDER"] = temp_dir
            
            df = pl.DataFrame({
                "col": [1, 2, 3]
            })
            
            checks = [
                Check("col").no_nulls()
            ]
            
            # Should not raise
            report = run_health_checks(written(df, temp_dir), checks)
            
            assert report["summary"]["passed"] == 1
            assert report["summary"]["failed"] == 0

    def test_pandas_dataframe(self):
        """Test that health checks work on Pandas DataFrame."""
        with tempfile.TemporaryDirectory() as temp_dir:
            os.environ["META_FOLDER"] = temp_dir
            
            df = pd.DataFrame({
                "col": [1, 2, 3]
            })
            
            checks = [
                Check("col").no_nulls()
            ]
            
            # Should not raise
            report = run_health_checks(written(df, temp_dir), checks)
            
            assert report["summary"]["passed"] == 1
            assert report["summary"]["failed"] == 0

    def test_polars_lazyframe(self):
        """Test that health checks work on Polars LazyFrame (existing functionality)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            os.environ["META_FOLDER"] = temp_dir
            
            lf = pl.LazyFrame({
                "col": [1, 2, 3]
            })
            
            checks = [
                Check("col").no_nulls()
            ]
            
            # Should not raise
            report = run_health_checks(written(lf, temp_dir), checks)
            
            assert report["summary"]["passed"] == 1
            assert report["summary"]["failed"] == 0
