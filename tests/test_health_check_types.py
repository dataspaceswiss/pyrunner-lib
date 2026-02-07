"""Tests for health_check module with different dataframe types."""

import os
import tempfile
import pytest
import polars as pl
import pandas as pd
from pyrunner_lib.health_check import Check, run_health_checks

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
            report = run_health_checks(df, checks)
            
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
            report = run_health_checks(df, checks)
            
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
            report = run_health_checks(lf, checks)
            
            assert report["summary"]["passed"] == 1
            assert report["summary"]["failed"] == 0
