"""Tests for input_source_tracer module."""

import json
from pathlib import Path

import pytest
import polars as pl
import tempfile
import os
from unittest.mock import patch, MagicMock

from pyrunner_lib.input_source_tracer import (
    InputSourceTracer,
    _ColumnLineage,
    _PlanState,
    trace_lazyframe_sources,
)
from pyrunner_lib.pyrunner_lib import write_column_trace_file


class TestInputSourceTracer:
    """Test InputSourceTracer class."""
    
    def _create_minimal_tracer(self):
        """Create a minimal tracer for testing without file operations."""
        tracer = InputSourceTracer.__new__(InputSourceTracer)
        tracer.column_sources = {}
        return tracer
    
    def _create_test_parquet_file(self, data: dict, filename: str = "test.parquet") -> str:
        """Create a temporary parquet file for testing."""
        df = pl.DataFrame(data)
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
            df.write_parquet(f.name)
            return f.name
    
    def test_init_with_valid_plan(self):
        """Test initialization with a live LazyFrame."""
        # Create a real parquet file and scan it
        parquet_file = self._create_test_parquet_file({
            "col1": [1, 2, 3],
            "col2": [4, 5, 6]
        })
        
        try:
            lf = pl.scan_parquet(parquet_file)
            
            tracer = InputSourceTracer(lf)
            
            assert set(tracer.column_sources) == {"col1", "col2"}
        finally:
            os.unlink(parquet_file)
    
    def test_scan_operation(self):
        """Test Scan operation handling."""
        # Create a real parquet file
        parquet_file = self._create_test_parquet_file({
            "col1": [1, 2, 3],
            "col2": [4, 5, 6]
        })
        
        try:
            # Create a LazyFrame that scans the parquet file
            lf = pl.scan_parquet(parquet_file)
            
            tracer = InputSourceTracer(lf)
            
            # Check that columns are mapped correctly
            assert "col1" in tracer.column_sources
            assert "col2" in tracer.column_sources
            
            # Check that sources contain the file path
            for col in ["col1", "col2"]:
                sources = tracer.column_sources[col]
                assert any(parquet_file in source for source in sources)
        finally:
            # Clean up
            os.unlink(parquet_file)
    
    def test_scan_operation_no_transform_id(self):
        """Test Scan operation with no transform ID in path."""
        # Create a real parquet file with a simple path (no transform ID)
        parquet_file = self._create_test_parquet_file({
            "col1": [1, 2, 3]
        })
        
        try:
            # Create a LazyFrame that scans the parquet file
            lf = pl.scan_parquet(parquet_file)
            
            tracer = InputSourceTracer(lf)
            
            # Should still work with unknown_transform
            assert "col1" in tracer.column_sources
        finally:
            # Clean up
            os.unlink(parquet_file)
    
    def test_rename_operation(self):
        """Test Rename operation handling."""
        # Create a real parquet file and apply rename operation
        parquet_file = self._create_test_parquet_file({
            "old_col": [1, 2, 3],
            "other_col": [4, 5, 6]
        })
        
        try:
            lf = pl.scan_parquet(parquet_file).rename({"old_col": "new_col"})
            
            tracer = InputSourceTracer(lf)
            
            # Check that new column is tracked
            assert "new_col" in tracer.column_sources
            # The source should contain the original column name
            sources = tracer.column_sources["new_col"]
            assert any("old_col" in source for source in sources)
        finally:
            os.unlink(parquet_file)
    
    def test_alias_operation(self):
        """Test Alias operation handling."""
        # Create a real parquet file and apply alias operation
        parquet_file = self._create_test_parquet_file({
            "source_col": [1, 2, 3],
            "other_col": [4, 5, 6]
        })
        
        try:
            lf = pl.scan_parquet(parquet_file).select(pl.col("source_col").alias("alias_col"))
            
            tracer = InputSourceTracer(lf)
            
            # Check that alias is tracked with source column
            assert "alias_col" in tracer.column_sources
            # The source should reference the original column
            sources = tracer.column_sources["alias_col"]
            assert any("source_col" in source for source in sources)
        finally:
            os.unlink(parquet_file)
    
    def test_select_operation(self):
        """Test Select operation handling."""
        # Create a real parquet file and apply select operation
        parquet_file = self._create_test_parquet_file({
            "col1": [1, 2, 3],
            "col2": [4, 5, 6],
            "col3": [7, 8, 9]
        })
        
        try:
            lf = pl.scan_parquet(parquet_file).select(["col1", "col2"])
            
            tracer = InputSourceTracer(lf)
            
            # Select should preserve columns
            assert "col1" in tracer.column_sources
            assert "col2" in tracer.column_sources
        finally:
            os.unlink(parquet_file)
    
    def test_filter_operation(self):
        """Test Filter operation handling."""
        # Create a real parquet file and apply filter operation
        parquet_file = self._create_test_parquet_file({
            "col1": [1, 2, 3, 4, 5],
            "col2": [6, 7, 8, 9, 10]
        })
        
        try:
            lf = pl.scan_parquet(parquet_file).filter(pl.col("col1") > 3)
            
            tracer = InputSourceTracer(lf)
            
            # Filter should preserve all columns
            assert "col1" in tracer.column_sources
            assert "col2" in tracer.column_sources
        finally:
            os.unlink(parquet_file)
    
    def test_join_operation(self):
        """Test Join operation handling."""
        # Create real parquet files for join operation
        parquet_file1 = self._create_test_parquet_file({
            "id": [1, 2, 3],
            "left_col": [10, 20, 30]
        })
        parquet_file2 = self._create_test_parquet_file({
            "id": [1, 2, 3],
            "right_col": [100, 200, 300]
        })
        
        try:
            lf1 = pl.scan_parquet(parquet_file1)
            lf2 = pl.scan_parquet(parquet_file2)
            lf = lf1.join(lf2, on="id")
            
            tracer = InputSourceTracer(lf)
            
            # Join should preserve columns from both sides
            assert "id" in tracer.column_sources
            assert "left_col" in tracer.column_sources
            assert "right_col" in tracer.column_sources
        finally:
            os.unlink(parquet_file1)
            os.unlink(parquet_file2)
    
    def test_groupby_operation(self):
        """Test GroupBy operation handling."""
        # Create a real parquet file and apply groupby operation
        parquet_file = self._create_test_parquet_file({
            "group_col": ["A", "A", "B", "B"],
            "value_col": [10, 20, 30, 40]
        })
        
        try:
            lf = pl.scan_parquet(parquet_file).group_by("group_col").agg(pl.col("value_col").sum())
            
            tracer = InputSourceTracer(lf)
            
            # GroupBy should preserve grouping columns and aggregated columns
            assert "group_col" in tracer.column_sources
            assert "value_col" in tracer.column_sources
        finally:
            os.unlink(parquet_file)
    
    def test_aggregate_operation(self):
        """Test Aggregate operation handling."""
        # Create a real parquet file and apply aggregate operation
        parquet_file = self._create_test_parquet_file({
            "group_col": ["A", "A", "B", "B"],
            "value_col": [10, 20, 30, 40]
        })
        
        try:
            lf = pl.scan_parquet(parquet_file).group_by("group_col").agg([
                pl.col("value_col").sum().alias("sum_value"),
                pl.col("value_col").mean().alias("mean_value")
            ])
            
            tracer = InputSourceTracer(lf)
            
            # Only final output names are exposed, and both aggregate aliases
            # trace back to the physical value_col input.
            assert set(tracer.column_sources) == {
                "group_col",
                "sum_value",
                "mean_value",
            }
            assert all(
                "value_col" in source
                for source in tracer.column_sources["sum_value"]
            )
            assert all(
                "value_col" in source
                for source in tracer.column_sources["mean_value"]
            )
        finally:
            os.unlink(parquet_file)
    
    def test_sort_operation(self):
        """Test Sort operation handling."""
        # Create a real parquet file and apply sort operation
        parquet_file = self._create_test_parquet_file({
            "sort_col": [3, 1, 4, 2],
            "other_col": [10, 20, 30, 40]
        })
        
        try:
            lf = pl.scan_parquet(parquet_file).sort("sort_col")
            
            tracer = InputSourceTracer(lf)
            
            # Sort should preserve all columns
            assert "sort_col" in tracer.column_sources
            assert "other_col" in tracer.column_sources
        finally:
            os.unlink(parquet_file)
    
    def test_drop_operation(self):
        """Test Drop operation handling."""
        # Create a real parquet file and apply drop operation
        parquet_file = self._create_test_parquet_file({
            "col1": [1, 2, 3],
            "col2": [4, 5, 6],
            "col3": [7, 8, 9]
        })
        
        try:
            lf = pl.scan_parquet(parquet_file).drop("col1")
            
            tracer = InputSourceTracer(lf)
            
            # Drop operation is implemented as Select with Difference selector
            # All columns are still in the source, but only col2 and col3 are selected
            assert "col2" in tracer.column_sources
            assert "col3" in tracer.column_sources
        finally:
            os.unlink(parquet_file)
    
    def test_with_columns_operation(self):
        """Test WithColumns operation handling."""
        # Create a real parquet file and apply with_columns operation
        parquet_file = self._create_test_parquet_file({
            "source_col": [1, 2, 3],
            "other_col": [4, 5, 6]
        })
        
        try:
            lf = pl.scan_parquet(parquet_file).with_columns([
                pl.col("source_col").alias("new_col"),
                (pl.col("source_col") * 2).alias("doubled_col")
            ])
            
            tracer = InputSourceTracer(lf)
            
            # WithColumns should preserve source columns and add new ones
            assert "source_col" in tracer.column_sources
            assert "other_col" in tracer.column_sources
        finally:
            os.unlink(parquet_file)
    
    def test_unnest_operation(self):
        """Test Unnest operation handling."""
        # Create a real parquet file with a list column
        parquet_file = self._create_test_parquet_file({
            "id": [1, 2, 3],
            "values": [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
        })
        
        try:
            lf = pl.scan_parquet(parquet_file).explode("values")
            
            tracer = InputSourceTracer(lf)
            
            # Unnest should preserve all columns including the unnested one
            assert "id" in tracer.column_sources
            assert "values" in tracer.column_sources
        finally:
            os.unlink(parquet_file)
    
    def test_unnest_operation_with_struct(self):
        """Test Unnest operation with struct columns."""
        # Create a real parquet file with a struct column
        df = pl.DataFrame({
            "id": [1, 2, 3],
            "person": [
                {"name": "Alice", "age": 25},
                {"name": "Bob", "age": 30},
                {"name": "Charlie", "age": 35}
            ]
        })
        
        parquet_file = None
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
            df.write_parquet(f.name)
            parquet_file = f.name
        
        try:
            lf = pl.scan_parquet(parquet_file).unnest("person")
            
            tracer = InputSourceTracer(lf)
            sources = tracer.get_column_sources()
            
            # Unnest should track id and the unnested struct fields (name, age)
            # The struct's fields should trace back to the source file
            assert "id" in sources, "id column not tracked"
            assert "name" in sources, "unnested name column not tracked"
            assert "age" in sources, "unnested age column not tracked"
            
            # Verify the sources reference the parquet file
            all_sources = str(sources)
            assert parquet_file in all_sources, "Source file not in column sources"
        finally:
            if parquet_file:
                os.unlink(parquet_file)
    
    def test_unnest_operation_multiple_columns(self):
        """Test Unnest operation with multiple list columns."""
        # Create a real parquet file with multiple list columns
        parquet_file = self._create_test_parquet_file({
            "id": [1, 2],
            "list1": [[1, 2], [3, 4]],
            "list2": [["a", "b"], ["c", "d"]]
        })
        
        try:
            lf = pl.scan_parquet(parquet_file).explode(["list1", "list2"])
            
            tracer = InputSourceTracer(lf)
            
            # All columns should be preserved
            assert "id" in tracer.column_sources
            assert "list1" in tracer.column_sources
            assert "list2" in tracer.column_sources
        finally:
            os.unlink(parquet_file)
    
    def test_expression_sources(self):
        """Test that expressions track source columns correctly."""
        parquet_file = self._create_test_parquet_file({
            "col1": [1, 2, 3],
            "col2": [4, 5, 6],
            "col3": [7, 8, 9]
        })
        
        try:
            # Create a query with various expression types
            lf = pl.scan_parquet(parquet_file).select([
                pl.col("col1"),  # Direct column
                pl.col("col2").alias("renamed"),  # Alias
                (pl.col("col1") + pl.col("col3")).alias("sum_cols"),  # Binary expression
            ])
            
            tracer = InputSourceTracer(lf)
            sources = tracer.get_column_sources()
            
            # col1 should be tracked
            assert "col1" in sources
            # renamed should trace back to col2
            assert "renamed" in sources
            assert any("col2" in s for s in sources["renamed"])
        finally:
            os.unlink(parquet_file)
    
    def test_get_column_sources(self):
        """Test get_column_sources method."""
        tracer = self._create_minimal_tracer()
        
        # Add some test sources
        tracer.column_sources = {
            "col1": {"source1", "source2"},
            "col2": {"source3"}
        }
        
        sources = tracer.get_column_sources()
        
        assert "col1" in sources
        assert "col2" in sources
        assert isinstance(sources["col1"], list)
        assert "source1" in sources["col1"]
        assert "source2" in sources["col1"]


class TestTraceLazyFrameSources:
    """Test trace_lazyframe_sources function."""
    
    def _create_test_parquet_file(self, data: dict, filename: str = "test.parquet") -> str:
        """Create a temporary parquet file for testing."""
        df = pl.DataFrame(data)
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
            df.write_parquet(f.name)
            return f.name
    
    def test_trace_lazyframe_sources(self):
        """Test trace_lazyframe_sources function."""
        # Create a real parquet file and scan it
        parquet_file = self._create_test_parquet_file({
            "col1": [1, 2, 3],
            "col2": [4, 5, 6]
        })
        
        try:
            lf = pl.scan_parquet(parquet_file)
            
            result = trace_lazyframe_sources(lf)
            
            assert isinstance(result, dict)
            assert "col1" in result
            assert "col2" in result
        finally:
            os.unlink(parquet_file)
    
    def test_hconcat_operation(self):
        """Test HConcat operation handling."""
        # Create real parquet files for horizontal concatenation
        parquet_file1 = self._create_test_parquet_file({
            "id": [1, 2, 3],
            "col1": [10, 20, 30]
        })
        parquet_file2 = self._create_test_parquet_file({
            "id": [4, 5, 6],
            "col2": [40, 50, 60]
        })
        
        try:
            lf1 = pl.scan_parquet(parquet_file1)
            lf2 = pl.scan_parquet(parquet_file2).rename({"id": "right_id"})
            lf = pl.concat([lf1, lf2], how="horizontal")
            
            tracer = InputSourceTracer(lf)
            
            # HConcat should preserve columns from both sides
            assert "id" in tracer.column_sources
            assert "right_id" in tracer.column_sources
            assert "col1" in tracer.column_sources
            assert "col2" in tracer.column_sources
            
            # Check that sources from both files are tracked
            sources = tracer.get_column_sources()
            all_sources = []
            for source_list in sources.values():
                all_sources.extend(source_list)
            
            file1_found = any(parquet_file1 in source for source in all_sources)
            file2_found = any(parquet_file2 in source for source in all_sources)
            
            assert file1_found, "File1 not found in any source"
            assert file2_found, "File2 not found in any source"
        finally:
            os.unlink(parquet_file1)
            os.unlink(parquet_file2)
    
    def test_union_operation_same_columns(self):
        """Test Union operation with same columns from both inputs."""
        # Create real parquet files with the SAME columns
        parquet_file1 = self._create_test_parquet_file({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "age": [25, 30, 35]
        })
        parquet_file2 = self._create_test_parquet_file({
            "id": [4, 5, 6],
            "name": ["David", "Eve", "Frank"],
            "age": [40, 45, 50]
        })
        
        try:
            lf1 = pl.scan_parquet(parquet_file1)
            lf2 = pl.scan_parquet(parquet_file2)
            lf = pl.concat([lf1, lf2], how="vertical")  # This creates a Union
            
            tracer = InputSourceTracer(lf)
            
            # Union should preserve columns from both sides
            assert "id" in tracer.column_sources
            assert "name" in tracer.column_sources
            assert "age" in tracer.column_sources
            
            # Check that sources from BOTH files are tracked for EACH column
            sources = tracer.get_column_sources()
            expected_cols = ["id", "name", "age"]
            
            for col in expected_cols:
                assert col in sources, f"Column {col} not found in sources"
                source_list = sources[col]
                
                file1_found = any(parquet_file1 in source for source in source_list)
                file2_found = any(parquet_file2 in source for source in source_list)
                assert file1_found, f"File1 not found in sources for column {col}"
                assert file2_found, f"File2 not found in sources for column {col}"
                assert len(source_list) >= 2, f"Expected at least 2 sources for column {col}, got {len(source_list)}"
        finally:
            os.unlink(parquet_file1)
            os.unlink(parquet_file2)
    
    def test_new_operations_preserve_sources(self):
        """Test that new operations preserve column sources correctly."""
        # Create a real parquet file with various data types
        parquet_file = self._create_test_parquet_file({
            "id": [1, 2, 3, 4, 5],
            "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
            "age": [25, 30, 35, 40, 45],
            "salary": [50000, 60000, 70000, 80000, 90000],
            "scores": [[1, 2, 3], [4, 5], [6, 7, 8, 9], [10], [11, 12]]
        })
        
        try:
            lf = pl.scan_parquet(parquet_file)
            
            # Test operations that should preserve column sources
            test_cases = [
                ("cast", lambda lf: lf.cast({"age": pl.Float64})),
                ("explode", lambda lf: lf.explode("scores")),
                ("fill_nan", lambda lf: lf.fill_nan(0.0)),
                ("gather_every", lambda lf: lf.gather_every(2)),
                ("interpolate", lambda lf: lf.interpolate()),
                ("quantile", lambda lf: lf.quantile(0.5)),
                ("std", lambda lf: lf.std()),
                ("var", lambda lf: lf.var()),
                ("mean", lambda lf: lf.mean()),
                ("sum", lambda lf: lf.sum()),
                ("min", lambda lf: lf.min()),
                ("max", lambda lf: lf.max()),
                ("count", lambda lf: lf.count()),
                ("null_count", lambda lf: lf.null_count()),
            ]
            
            for op_name, op_func in test_cases:
                result_lf = op_func(lf)
                sources = trace_lazyframe_sources(result_lf)

                # Every listed operation must retain at least one proven edge.
                all_sources = []
                for source_list in sources.values():
                    all_sources.extend(source_list)

                file_found = any(parquet_file in source for source in all_sources)
                assert file_found, f"Original file not found in sources for {op_name}"
                    
        finally:
            os.unlink(parquet_file)
    
    def test_incremental_join_with_delta(self):
        """Test incremental build pattern - joining main data with delta data and itself."""
        # Create main and delta parquet files
        parquet_main = self._create_test_parquet_file({
            "id": [1, 2, 3],
            "value": [10, 20, 30]
        })
        parquet_delta = self._create_test_parquet_file({
            "id": [4, 5],
            "value": [40, 50]
        })
        
        try:
            # Simulate incremental pattern: union main+delta, then join with main
            main_lf = pl.scan_parquet(parquet_main)
            delta_lf = pl.scan_parquet(parquet_delta)
            
            # Union main and delta
            union_lf = pl.concat([main_lf, delta_lf])
            # Join with main (self-join pattern for lookups)
            result = union_lf.join(
                main_lf.rename({"value": "old_value"}),
                on="id",
                how="left"
            )
            
            sources = trace_lazyframe_sources(result)
            
            # Verify we have column sources
            assert "id" in sources, "id column not in sources"
            assert "value" in sources, "value column not in sources"
            
            # Convert all sources to a string for checking
            all_sources_str = str(sources)
            
            # Both files should be tracked
            assert parquet_main in all_sources_str, "Main file not found in sources"
            assert parquet_delta in all_sources_str, "Delta file not found in sources"
            
        finally:
            os.unlink(parquet_main)
            os.unlink(parquet_delta)
    
    def test_self_join(self):
        """Test self-join - joining a dataset with itself."""
        parquet_file = self._create_test_parquet_file({
            "id": [1, 2, 3],
            "value": [10, 20, 30]
        })
        
        try:
            lf1 = pl.scan_parquet(parquet_file)
            lf2 = pl.scan_parquet(parquet_file).rename({"value": "value2"})
            
            result = lf1.join(lf2, on="id")
            
            sources = trace_lazyframe_sources(result)
            
            # Both value columns should trace back to the same file
            assert "value" in sources, "value column not tracked"
            assert "value2" in sources, "value2 column not tracked"
            
            # Both should reference the same file
            assert any(parquet_file in s for s in sources["value"])
            assert any(parquet_file in s for s in sources["value2"])
            
        finally:
            os.unlink(parquet_file)


def _write_source(root: Path, rid: str, data: dict) -> tuple[Path, str]:
    path = root / "data" / rid / "datasets" / "data.parquet"
    path.parent.mkdir(parents=True)
    pl.DataFrame(data).write_parquet(path)
    return path, rid


def _trace(
    lazyframe: pl.LazyFrame, *, schema_first: bool = False
) -> dict[str, list[str]]:
    if schema_first:
        lazyframe.collect_schema()
    return trace_lazyframe_sources(lazyframe)


def _source(rid: str, column: str) -> str:
    return f"{rid}|{column}"


@pytest.fixture
def sources(tmp_path: Path):
    left_path, left_rid = _write_source(
        tmp_path,
        "rid.transform.left-123",
        {
            "id": [1, 2, 3],
            "name": ["a", "b", "c"],
            "value": [10, 20, 30],
            "other": [2, 1, 3],
            "items": [[1, 2], [3], [4, 5]],
        },
    )
    right_path, right_rid = _write_source(
        tmp_path,
        "rid.transform.right-456",
        {
            "id": [1, 2, 4],
            "name": ["x", "y", "z"],
            "score": [100, 200, 400],
        },
    )
    return {
        "left": pl.scan_parquet(left_path),
        "right": pl.scan_parquet(right_path),
        "left_path": left_path,
        "right_path": right_path,
        "left_rid": left_rid,
        "right_rid": right_rid,
    }


def test_scan_maps_each_column_to_its_physical_source(sources):
    result = _trace(sources["left"])
    assert result["id"] == [_source(sources["left_rid"], "id")]
    assert result["name"] == [_source(sources["left_rid"], "name")]


def test_multiple_scan_paths_contribute_to_each_output(tmp_path: Path):
    path_a, rid_a = _write_source(
        tmp_path, "rid.transform.a-1", {"id": [1], "value": [10]}
    )
    path_b, rid_b = _write_source(
        tmp_path, "rid.transform.b-2", {"id": [2], "value": [20]}
    )
    result = _trace(pl.scan_parquet([path_a, path_b]))
    assert result["value"] == sorted([_source(rid_a, "value"), _source(rid_b, "value")])


def test_alias_uses_expression_source_not_same_named_input(sources):
    # Regression: the old tracer attached source.name because the output was
    # called "name", despite the physical expression reading source.value.
    result = _trace(sources["left"].select(pl.col("value").alias("name")))
    assert result == {"name": [_source(sources["left_rid"], "value")]}


def test_literal_replacement_erases_same_named_lineage(sources):
    result = _trace(sources["left"].with_columns(pl.lit("generated").alias("name")))
    assert "name" not in result
    assert result["value"] == [_source(sources["left_rid"], "value")]


def test_with_columns_replacement_uses_its_expression_dependencies(sources):
    result = _trace(
        sources["left"].with_columns((pl.col("value") + pl.col("other")).alias("name"))
    )
    assert set(result["name"]) == {
        _source(sources["left_rid"], "value"),
        _source(sources["left_rid"], "other"),
    }


def test_projection_contains_final_outputs_only(sources):
    result = _trace(
        sources["left"].select(
            pl.col("value").alias("renamed"), pl.lit(1).alias("constant")
        )
    )
    assert result == {"renamed": [_source(sources["left_rid"], "value")]}


def test_drop_does_not_leak_removed_columns(sources):
    result = _trace(sources["left"].drop("name", "items"))
    assert set(result) == {"id", "value", "other"}


@pytest.mark.parametrize(
    ("expression", "output", "dependencies"),
    [
        (
            (pl.col("value") + pl.col("other")).alias("calculated"),
            "calculated",
            {"value", "other"},
        ),
        (
            pl.when(pl.col("id") > 1)
            .then(pl.col("value"))
            .otherwise(pl.col("other"))
            .alias("chosen"),
            "chosen",
            {"id", "value", "other"},
        ),
        (
            pl.col("value").sum().over("name").alias("windowed"),
            "windowed",
            {"value", "name"},
        ),
        (pl.col("name").str.to_uppercase().alias("upper"), "upper", {"name"}),
        (pl.col("value").cast(pl.Float64).alias("float"), "float", {"value"}),
        (
            pl.col("value").sort_by("other").alias("sorted"),
            "sorted",
            {"value", "other"},
        ),
    ],
)
def test_expression_dependencies(sources, expression, output, dependencies):
    result = _trace(sources["left"].select(expression))
    assert set(result[output]) == {
        _source(sources["left_rid"], column) for column in dependencies
    }


def test_horizontal_function_tracks_all_value_dependencies(sources):
    result = _trace(
        sources["left"].select(
            pl.concat_str("name", "value", separator=":").alias("combined")
        )
    )
    assert set(result["combined"]) == {
        _source(sources["left_rid"], "name"),
        _source(sources["left_rid"], "value"),
    }


@pytest.mark.parametrize(
    ("projection", "expected"),
    [
        (lambda lf: lf.select(pl.selectors.numeric()), {"id", "value", "other"}),
        (lambda lf: lf.select(pl.col("^.*e$")), {"name", "value"}),
        (
            lambda lf: lf.select(pl.all().exclude("name", "items")),
            {"id", "value", "other"},
        ),
        (lambda lf: lf.select(pl.nth(2).alias("third")), {"third"}),
        (
            lambda lf: lf.select(pl.all().name.suffix("_copy")),
            {"id_copy", "name_copy", "value_copy", "other_copy", "items_copy"},
        ),
    ],
)
def test_selector_and_multi_output_projections(sources, projection, expected):
    result = _trace(projection(sources["left"]))
    assert set(result) == expected
    if "third" in result:
        assert result["third"] == [_source(sources["left_rid"], "value")]
    if "value_copy" in result:
        assert result["value_copy"] == [_source(sources["left_rid"], "value")]


def test_rename_preserves_original_source(sources):
    result = _trace(sources["left"].rename({"name": "customer"}))
    assert result["customer"] == [_source(sources["left_rid"], "name")]
    assert "name" not in result


def test_groupby_tracks_derived_key_and_aggregate_inputs(sources):
    result = _trace(
        sources["left"]
        .group_by((pl.col("id") % 2).alias("bucket"))
        .agg((pl.col("value") + pl.col("other")).sum().alias("total"))
    )
    assert result["bucket"] == [_source(sources["left_rid"], "id")]
    assert set(result["total"]) == {
        _source(sources["left_rid"], "value"),
        _source(sources["left_rid"], "other"),
    }
    assert "value" not in result


def test_rolling_groupby_tracks_key_and_aggregate(sources):
    result = _trace(
        sources["left"]
        .sort("id")
        .rolling("id", period="2i")
        .agg(pl.col("value").sum().alias("rolling_value"))
    )
    assert result["id"] == [_source(sources["left_rid"], "id")]
    assert result["rolling_value"] == [_source(sources["left_rid"], "value")]


def test_join_duplicate_names_are_kept_separate(sources):
    result = _trace(sources["left"].join(sources["right"], on="id", suffix="_lookup"))
    assert result["name"] == [_source(sources["left_rid"], "name")]
    assert result["name_lookup"] == [_source(sources["right_rid"], "name")]
    assert result["id"] == [_source(sources["left_rid"], "id")]


def test_right_join_uses_right_join_key(sources):
    result = _trace(
        sources["left"].join(sources["right"], on="id", how="right", suffix="_lookup")
    )
    assert result["id"] == [_source(sources["right_rid"], "id")]
    assert result["name"] == [_source(sources["left_rid"], "name")]


def test_full_coalesced_join_key_has_both_sources(sources):
    result = _trace(
        sources["left"].join(sources["right"], on="id", how="full", coalesce=True)
    )
    assert set(result["id"]) == {
        _source(sources["left_rid"], "id"),
        _source(sources["right_rid"], "id"),
    }


def test_full_noncoalesced_join_keeps_both_keys(sources):
    result = _trace(
        sources["left"].join(
            sources["right"], on="id", how="full", coalesce=False, suffix="_r"
        )
    )
    assert result["id"] == [_source(sources["left_rid"], "id")]
    assert result["id_r"] == [_source(sources["right_rid"], "id")]


@pytest.mark.parametrize("how", ["semi", "anti"])
def test_semi_and_anti_joins_only_emit_left_columns(sources, how):
    result = _trace(sources["left"].join(sources["right"], on="id", how=how))
    assert set(result) == {"id", "name", "value", "other", "items"}
    assert all(sources["right_rid"] not in str(edges) for edges in result.values())


def test_cross_join_suffixes_right_collisions(sources):
    result = _trace(sources["left"].join(sources["right"], how="cross", suffix="_r"))
    assert result["id"] == [_source(sources["left_rid"], "id")]
    assert result["id_r"] == [_source(sources["right_rid"], "id")]


def test_join_with_different_key_names_tracks_physical_keys(sources):
    result = _trace(
        sources["left"].join(
            sources["right"],
            left_on="value",
            right_on="score",
            how="full",
            coalesce=True,
        )
    )
    assert set(result["value"]) == {
        _source(sources["left_rid"], "value"),
        _source(sources["right_rid"], "score"),
    }


def test_non_equi_join_keeps_same_named_columns_separate(sources):
    result = _trace(
        sources["left"].join_where(
            sources["right"], pl.col("value") < pl.col("score"), suffix="_r"
        )
    )
    assert result["name"] == [_source(sources["left_rid"], "name")]
    assert result["name_r"] == [_source(sources["right_rid"], "name")]


def test_vertical_union_combines_sources_by_output_position(tmp_path: Path):
    path_a, rid_a = _write_source(
        tmp_path, "rid.transform.union-a", {"id": [1], "name": ["a"]}
    )
    path_b, rid_b = _write_source(
        tmp_path, "rid.transform.union-b", {"id": [2], "name": ["b"]}
    )
    result = _trace(
        pl.concat([pl.scan_parquet(path_a), pl.scan_parquet(path_b)], how="vertical")
    )
    assert set(result["name"]) == {
        _source(rid_a, "name"),
        _source(rid_b, "name"),
    }


def test_diagonal_union_does_not_invent_missing_column_sources(tmp_path: Path):
    path_a, rid_a = _write_source(
        tmp_path, "rid.transform.diagonal-a", {"id": [1], "left": [10]}
    )
    path_b, rid_b = _write_source(
        tmp_path, "rid.transform.diagonal-b", {"id": [2], "right": [20]}
    )
    result = _trace(
        pl.concat([pl.scan_parquet(path_a), pl.scan_parquet(path_b)], how="diagonal")
    )
    assert result["left"] == [_source(rid_a, "left")]
    assert result["right"] == [_source(rid_b, "right")]
    assert set(result["id"]) == {_source(rid_a, "id"), _source(rid_b, "id")}


def test_horizontal_concat_preserves_distinct_inputs(tmp_path: Path):
    path_a, rid_a = _write_source(tmp_path, "rid.transform.h-a", {"left": [1, 2]})
    path_b, rid_b = _write_source(tmp_path, "rid.transform.h-b", {"right": [3, 4]})
    result = _trace(
        pl.concat(
            [pl.scan_parquet(path_a), pl.scan_parquet(path_b)],
            how="horizontal",
        )
    )
    assert result == {
        "left": [_source(rid_a, "left")],
        "right": [_source(rid_b, "right")],
    }


def test_merge_sorted_combines_each_physical_column_source(tmp_path: Path):
    path_a, rid_a = _write_source(
        tmp_path, "rid.transform.merge-a", {"id": [1], "value": [10]}
    )
    path_b, rid_b = _write_source(
        tmp_path, "rid.transform.merge-b", {"id": [2], "value": [20]}
    )
    result = _trace(
        pl.scan_parquet(path_a).merge_sorted(pl.scan_parquet(path_b), key="id")
    )
    assert set(result["value"]) == {
        _source(rid_a, "value"),
        _source(rid_b, "value"),
    }


def test_explode_preserves_list_column_source(sources):
    result = _trace(sources["left"].explode("items"))
    assert result["items"] == [_source(sources["left_rid"], "items")]


@pytest.mark.parametrize(
    "operation",
    [
        lambda lf: lf.filter(pl.col("id") > 1),
        lambda lf: lf.sort("other"),
        lambda lf: lf.slice(1, 1),
        lambda lf: lf.unique(),
        lambda lf: lf.cache(),
        lambda lf: lf.gather_every(2),
        lambda lf: lf.reverse(),
        lambda lf: lf.drop_nulls(),
    ],
    ids=[
        "filter",
        "sort",
        "slice",
        "distinct",
        "cache",
        "gather",
        "reverse",
        "drop_nulls",
    ],
)
def test_row_operations_preserve_value_lineage(sources, operation):
    result = _trace(operation(sources["left"]))
    assert result["value"] == [_source(sources["left_rid"], "value")]
    assert result["name"] == [_source(sources["left_rid"], "name")]


@pytest.mark.parametrize(
    "operation",
    [
        lambda lf: lf.sum(),
        lambda lf: lf.min(),
        lambda lf: lf.max(),
        lambda lf: lf.mean(),
        lambda lf: lf.count(),
        lambda lf: lf.null_count(),
        lambda lf: lf.quantile(0.5),
        lambda lf: lf.std(),
        lambda lf: lf.var(),
    ],
    ids=["sum", "min", "max", "mean", "count", "null_count", "quantile", "std", "var"],
)
def test_frame_statistics_preserve_per_column_lineage(sources, operation):
    result = _trace(operation(sources["left"]))
    assert result["value"] == [_source(sources["left_rid"], "value")]
    assert result["other"] == [_source(sources["left_rid"], "other")]


def test_row_index_has_no_physical_column_source(sources):
    result = _trace(sources["left"].with_row_index("row_number"))
    assert "row_number" not in result
    assert result["id"] == [_source(sources["left_rid"], "id")]


@pytest.mark.parametrize(
    "operation",
    [
        lambda lf: lf.cast({"value": pl.Float64}),
        lambda lf: lf.fill_null(0),
        lambda lf: lf.interpolate(),
    ],
    ids=["cast", "fill_null", "interpolate"],
)
def test_frame_value_transformations_preserve_per_column_lineage(sources, operation):
    result = _trace(operation(sources["left"]))
    assert result["value"] == [_source(sources["left_rid"], "value")]
    assert result["other"] == [_source(sources["left_rid"], "other")]


def test_persisted_struct_fields_trace_to_struct_column(tmp_path: Path):
    path, rid = _write_source(
        tmp_path,
        "rid.transform.struct-source",
        {"id": [1], "person": [{"first": "Ada", "age": 36}]},
    )
    result = _trace(pl.scan_parquet(path).unnest("person"))
    assert result["first"] == [_source(rid, "person")]
    assert result["age"] == [_source(rid, "person")]
    assert result["id"] == [_source(rid, "id")]


def test_computed_struct_unnest_recovers_individual_fields(sources):
    result = _trace(
        sources["left"]
        .select(pl.struct("name", "value").alias("record"))
        .unnest("record")
    )
    assert result["name"] == [_source(sources["left_rid"], "name")]
    assert result["value"] == [_source(sources["left_rid"], "value")]


def test_union_preserves_computed_struct_field_lineage(tmp_path: Path):
    path_a, rid_a = _write_source(
        tmp_path, "rid.transform.struct-a", {"first": ["Ada"], "age": [36]}
    )
    path_b, rid_b = _write_source(
        tmp_path, "rid.transform.struct-b", {"first": ["Grace"], "age": [37]}
    )
    records = [
        pl.scan_parquet(path).select(pl.struct("first", "age").alias("person"))
        for path in (path_a, path_b)
    ]

    result = _trace(pl.concat(records).unnest("person"))

    assert set(result["first"]) == {_source(rid_a, "first"), _source(rid_b, "first")}
    assert set(result["age"]) == {_source(rid_a, "age"), _source(rid_b, "age")}


def test_struct_field_expression_recovers_individual_field(sources):
    result = _trace(
        sources["left"]
        .select(pl.struct("name", "value").alias("record"))
        .select(pl.col("record").struct.field("value").alias("extracted"))
    )
    assert result["extracted"] == [_source(sources["left_rid"], "value")]


def test_unpivot_value_unions_only_value_columns(sources):
    result = _trace(
        sources["left"].unpivot(
            index=["id"],
            on=["value", "other"],
            variable_name="metric",
            value_name="metric_value",
        )
    )
    assert result["id"] == [_source(sources["left_rid"], "id")]
    assert set(result["metric_value"]) == {
        _source(sources["left_rid"], "value"),
        _source(sources["left_rid"], "other"),
    }
    assert "metric" not in result


def test_unpivot_uses_default_output_names(sources):
    result = _trace(
        sources["left"]
        .select(
            "id",
            pl.col("value").alias("amount"),
            pl.col("other").alias("quantity"),
        )
        .unpivot(index=["id"], on=["amount", "quantity"])
    )
    assert result["id"] == [_source(sources["left_rid"], "id")]
    assert set(result["value"]) == {
        _source(sources["left_rid"], "value"),
        _source(sources["left_rid"], "other"),
    }
    assert "variable" not in result


def test_plan_analysis_is_memoized_with_isolated_results(monkeypatch):
    tracer = InputSourceTracer.__new__(InputSourceTracer)
    tracer._analysis_cache = {}
    analyzed_nodes = []

    def analyze_node(_tracer, node_id):
        analyzed_nodes.append(node_id)
        return _PlanState(
            {"value": _ColumnLineage({"source.parquet|value"})}
        )

    monkeypatch.setattr(InputSourceTracer, "_analyze_node", analyze_node)

    first = tracer._analyze(7)
    first.columns["value"].sources.add("caller-mutation")
    second = tracer._analyze(7)

    assert analyzed_nodes == [7]
    assert second.columns["value"].sources == {"source.parquet|value"}


def test_asof_join_tracks_each_side_without_name_guessing(sources):
    result = _trace(
        sources["left"]
        .sort("id")
        .join_asof(sources["right"].sort("id"), on="id")
    )

    assert result["id"] == [_source(sources["left_rid"], "id")]
    assert result["name"] == [_source(sources["left_rid"], "name")]
    assert result["name_right"] == [_source(sources["right_rid"], "name")]
    assert result["score"] == [_source(sources["right_rid"], "score")]


def test_dynamic_groupby_tracks_window_index_and_aggregate(sources):
    result = _trace(
        sources["left"]
        .sort("id")
        .group_by_dynamic("id", every="2i")
        .agg(pl.col("value").sum().alias("total"))
    )

    assert result["id"] == [_source(sources["left_rid"], "id")]
    assert result["total"] == [_source(sources["left_rid"], "value")]


def test_hive_partition_column_has_no_physical_column_source(tmp_path: Path):
    rid = "rid.transform.partitioned"
    path = tmp_path / "data" / rid / "datasets" / "country=CH" / "data.parquet"
    path.parent.mkdir(parents=True)
    pl.DataFrame({"value": [1]}).write_parquet(path)

    result = _trace(pl.scan_parquet(path, hive_partitioning=True))

    assert "country" not in result


def test_unreadable_parquet_schema_does_not_invent_lineage(
    tmp_path: Path, monkeypatch
):
    path, _ = _write_source(
        tmp_path, "rid.transform.missing", {"name": ["source"]}
    )
    lazyframe = pl.scan_parquet(path)

    def fail_to_read_schema(_):
        raise OSError("schema unavailable")

    monkeypatch.setattr(pl, "read_parquet_schema", fail_to_read_schema)
    assert _trace(lazyframe) == {}


def test_expression_udf_does_not_invent_lineage(sources):
    result = _trace(
        sources["left"].select(
            pl.col("value")
            .map_elements(lambda value: value + 1, return_dtype=pl.Int64)
            .alias("mapped")
        )
    )

    assert result == {}


def test_row_count_expression_has_no_physical_column_source(sources):
    assert _trace(sources["left"].select(pl.len().alias("row_count"))) == {}


def test_tracing_after_schema_collection(sources):
    result = _trace(sources["left"].select("id", "value"), schema_first=True)
    assert set(result) == {"id", "value"}


def test_runner_writes_real_final_lineage_file(tmp_path: Path):
    source_path, rid = _write_source(
        tmp_path, "rid.transform.integration", {"name": ["wrong"], "value": [1]}
    )
    output = pl.scan_parquet(source_path).select(pl.col("value").alias("name"))

    write_column_trace_file(output, "rid.transform.output", str(tmp_path))

    trace_path = tmp_path / "data" / "rid.transform.output" / "meta" / "columns.json"
    assert json.loads(trace_path.read_text()) == {"name": [_source(rid, "value")]}


def test_live_tracer_does_not_serialize_the_lazyframe(sources, monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("optimized lineage must not serialize the LazyFrame")

    monkeypatch.setattr(pl.LazyFrame, "serialize", fail_if_called)

    result = trace_lazyframe_sources(sources["left"].select("id"))

    assert result == {"id": [_source(sources["left_rid"], "id")]}


def test_live_tracer_does_not_execute_the_lazyframe(sources, monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("lineage tracing must not execute the LazyFrame")

    monkeypatch.setattr(pl.LazyFrame, "collect", fail_if_called)
    monkeypatch.setattr(pl.LazyFrame, "sink_parquet", fail_if_called)

    result = trace_lazyframe_sources(sources["left"].select("id", "value"))

    assert set(result) == {"id", "value"}


def test_live_tracer_rejects_an_unknown_ir_version(sources, monkeypatch):
    monkeypatch.setattr(InputSourceTracer, "_SUPPORTED_IR_MAJORS", range(0))

    with pytest.raises(NotImplementedError, match="unsupported Polars IR version"):
        trace_lazyframe_sources(sources["left"].select("id"))


def test_opaque_map_does_not_fall_back_to_matching_names(sources):
    result = _trace(
        sources["left"]
        .select("name")
        .map_batches(lambda frame: frame, validate_output_schema=False)
    )
    assert result == {}


def test_materialized_lazyframe_has_no_recoverable_persisted_source():
    assert _trace(pl.DataFrame({"name": ["generated"]}).lazy()) == {}


def test_non_lazyframe_is_rejected():
    with pytest.raises(TypeError, match="polars.LazyFrame"):
        InputSourceTracer("not a LazyFrame")  # type: ignore[arg-type]


def test_source_ids_split_on_last_separator():
    tracer = InputSourceTracer.__new__(InputSourceTracer)
    tracer.column_sources = {"out": {"/tmp/a|b.parquet|source"}}
    assert tracer.get_column_sources() == {"out": ["/tmp/a|b.parquet|source"]}
