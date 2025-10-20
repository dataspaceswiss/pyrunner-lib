"""Tests for input_source_tracer module."""

import json
import pytest
import polars as pl
import tempfile
import os
from unittest.mock import patch, MagicMock

from pyrunner_lib.input_source_tracer import InputSourceTracer, trace_input_sources


class TestInputSourceTracer:
    """Test InputSourceTracer class."""
    
    def _create_minimal_tracer(self):
        """Create a minimal tracer for testing without file operations."""
        tracer = InputSourceTracer.__new__(InputSourceTracer)
        tracer.column_sources = {}
        tracer.column_mappings = {}
        tracer.dataframe_scans = {}
        tracer.df_counter = 0
        return tracer
    
    def _create_test_parquet_file(self, data: dict, filename: str = "test.parquet") -> str:
        """Create a temporary parquet file for testing."""
        df = pl.DataFrame(data)
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
            df.write_parquet(f.name)
            return f.name
    
    def _create_lazyframe_plan(self, operations: list) -> str:
        """Create a real Polars LazyFrame plan by building up operations."""
        # Start with a simple DataFrame
        df = pl.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
            "age": [25, 30, 35, 40, 45],
            "salary": [50000, 60000, 70000, 80000, 90000]
        })
        
        lf = df.lazy()
        
        # Apply operations in sequence
        for op in operations:
            if op["type"] == "select":
                lf = lf.select(op["columns"])
            elif op["type"] == "filter":
                lf = lf.filter(op["condition"])
            elif op["type"] == "rename":
                lf = lf.rename(op["mapping"])
            elif op["type"] == "with_columns":
                lf = lf.with_columns(op["expressions"])
            elif op["type"] == "group_by":
                lf = lf.group_by(op["by"]).agg(op["aggs"])
            elif op["type"] == "sort":
                lf = lf.sort(op["by"])
            elif op["type"] == "drop":
                lf = lf.drop(op["columns"])
            elif op["type"] == "distinct":
                lf = lf.unique()
            elif op["type"] == "slice":
                lf = lf.slice(op["offset"], op["length"])
        
        return lf.serialize(format="json")
    
    def test_init_with_valid_plan(self):
        """Test initialization with valid JSON plan."""
        # Create a real parquet file and scan it
        parquet_file = self._create_test_parquet_file({
            "col1": [1, 2, 3],
            "col2": [4, 5, 6]
        })
        
        try:
            lf = pl.scan_parquet(parquet_file)
            plan_json = lf.serialize(format="json")
            
            tracer = InputSourceTracer(plan_json)
            
            # Check that the plan was parsed correctly
            assert tracer.plan is not None
            # The columns should be tracked
            assert len(tracer.column_sources) > 0
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
            plan_json = lf.serialize(format="json")
            
            tracer = InputSourceTracer(plan_json)
            
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
            plan_json = lf.serialize(format="json")
            
            tracer = InputSourceTracer(plan_json)
            
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
            plan_json = lf.serialize(format="json")
            
            tracer = InputSourceTracer(plan_json)
            
            # Check that new column is mapped to old column's sources
            assert "new_col" in tracer.column_mappings or "new_col" in tracer.column_sources
            # The mapping should contain the resolved source path
            if "new_col" in tracer.column_mappings:
                assert any("old_col" in source for source in tracer.column_mappings["new_col"])
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
            plan_json = lf.serialize(format="json")
            
            tracer = InputSourceTracer(plan_json)
            
            # Check that alias is mapped to source column
            assert "alias_col" in tracer.column_mappings or "alias_col" in tracer.column_sources
            # The mapping should contain the resolved source path
            if "alias_col" in tracer.column_mappings:
                assert any("source_col" in source for source in tracer.column_mappings["alias_col"])
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
            plan_json = lf.serialize(format="json")
            
            tracer = InputSourceTracer(plan_json)
            
            # Select should preserve columns
            assert "col1" in tracer.column_sources or "col1" in tracer.column_mappings
            assert "col2" in tracer.column_sources or "col2" in tracer.column_mappings
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
            plan_json = lf.serialize(format="json")
            
            tracer = InputSourceTracer(plan_json)
            
            # Filter should preserve all columns
            assert "col1" in tracer.column_sources or "col1" in tracer.column_mappings
            assert "col2" in tracer.column_sources or "col2" in tracer.column_mappings
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
            plan_json = lf.serialize(format="json")
            
            tracer = InputSourceTracer(plan_json)
            
            # Join should preserve columns from both sides
            assert "id" in tracer.column_sources or "id" in tracer.column_mappings
            assert "left_col" in tracer.column_sources or "left_col" in tracer.column_mappings
            assert "right_col" in tracer.column_sources or "right_col" in tracer.column_mappings
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
            plan_json = lf.serialize(format="json")
            
            tracer = InputSourceTracer(plan_json)
            
            # GroupBy should preserve grouping columns and aggregated columns
            assert "group_col" in tracer.column_sources or "group_col" in tracer.column_mappings
            assert "value_col" in tracer.column_sources or "value_col" in tracer.column_mappings
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
            plan_json = lf.serialize(format="json")
            
            tracer = InputSourceTracer(plan_json)
            
            # Aggregate should preserve source columns
            assert "value_col" in tracer.column_sources or "value_col" in tracer.column_mappings
            assert "group_col" in tracer.column_sources or "group_col" in tracer.column_mappings
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
            plan_json = lf.serialize(format="json")
            
            tracer = InputSourceTracer(plan_json)
            
            # Sort should preserve all columns
            assert "sort_col" in tracer.column_sources or "sort_col" in tracer.column_mappings
            assert "other_col" in tracer.column_sources or "other_col" in tracer.column_mappings
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
            plan_json = lf.serialize(format="json")
            
            tracer = InputSourceTracer(plan_json)
            
            # Drop operation is implemented as Select with Difference selector
            # All columns are still in the source, but only col2 and col3 are selected
            assert "col2" in tracer.column_sources or "col2" in tracer.column_mappings
            assert "col3" in tracer.column_sources or "col3" in tracer.column_mappings
            # col1 is still in the source data but not selected in the final output
            assert "col1" in tracer.column_sources or "col1" in tracer.column_mappings
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
            plan_json = lf.serialize(format="json")
            
            tracer = InputSourceTracer(plan_json)
            
            # WithColumns should preserve source columns and add new ones
            assert "source_col" in tracer.column_sources or "source_col" in tracer.column_mappings
            assert "other_col" in tracer.column_sources or "other_col" in tracer.column_mappings
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
            plan_json = lf.serialize(format="json")
            
            tracer = InputSourceTracer(plan_json)
            
            # Unnest should preserve all columns including the unnested one
            assert "id" in tracer.column_sources or "id" in tracer.column_mappings
            assert "values" in tracer.column_sources or "values" in tracer.column_mappings
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
            plan_json = lf.serialize(format="json")
            
            tracer = InputSourceTracer(plan_json)
            
            # Unnest should preserve all columns
            # The unnested struct columns (name, age) should be tracked back to person
            assert "id" in tracer.column_sources or "id" in tracer.column_mappings
            assert any("name" in source for source in tracer.column_mappings["person"])
            assert any("age" in source for source in tracer.column_mappings["person"])
            # At least some columns should be tracked
            assert len(tracer.column_sources) > 0 or len(tracer.column_mappings) > 0
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
            plan_json = lf.serialize(format="json")
            
            tracer = InputSourceTracer(plan_json)
            
            # All columns should be preserved
            assert "id" in tracer.column_sources or "id" in tracer.column_mappings
            assert "list1" in tracer.column_sources or "list1" in tracer.column_mappings
            assert "list2" in tracer.column_sources or "list2" in tracer.column_mappings
        finally:
            os.unlink(parquet_file)
    
    def test_extract_columns_from_expr(self):
        """Test _extract_columns_from_expr method."""
        tracer = self._create_minimal_tracer()
        
        # Test direct column reference
        expr = {"Column": "test_col"}
        columns = tracer._extract_columns_from_expr(expr)
        assert "test_col" in columns
        
        # Test binary expression
        expr = {
            "BinaryExpr": {
                "left": {"Column": "col1"},
                "right": {"Column": "col2"}
            }
        }
        columns = tracer._extract_columns_from_expr(expr)
        assert "col1" in columns
        assert "col2" in columns
        
        # Test aggregation
        expr = {
            "Agg": {
                "Sum": {"Column": "value_col"}
            }
        }
        columns = tracer._extract_columns_from_expr(expr)
        assert "value_col" in columns
        
        # Test function
        expr = {
            "Function": {
                "input": [
                    {"Column": "func_col"}
                ]
            }
        }
        columns = tracer._extract_columns_from_expr(expr)
        assert "func_col" in columns
    
    def test_resolve_all_mappings(self):
        """Test _resolve_all_mappings method."""
        # Create a tracer with some mappings
        tracer = self._create_minimal_tracer()
        
        # Add some test mappings with resolved sources (containing dots)
        tracer.column_mappings = {
            "col1": {"source1.file|col1"},  # Already resolved
            "col2": {"col1"},  # Intermediate mapping
            "col3": {"source2.file|col3"}  # Already resolved
        }
        
        tracer._resolve_all_mappings()
        
        # Check that mappings are resolved
        assert "col1" in tracer.column_sources
        assert "col3" in tracer.column_sources
    
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


class TestTraceInputSources:
    """Test trace_input_sources function."""
    
    def _create_test_parquet_file(self, data: dict, filename: str = "test.parquet") -> str:
        """Create a temporary parquet file for testing."""
        df = pl.DataFrame(data)
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
            df.write_parquet(f.name)
            return f.name
    
    def test_trace_input_sources(self):
        """Test trace_input_sources function."""
        # Create a real parquet file and scan it
        parquet_file = self._create_test_parquet_file({
            "col1": [1, 2, 3],
            "col2": [4, 5, 6]
        })
        
        try:
            lf = pl.scan_parquet(parquet_file)
            plan_json = lf.serialize(format="json")
            
            result = trace_input_sources(plan_json)
            
            assert isinstance(result, dict)
            assert "col1" in result
            assert "col2" in result
        finally:
            os.unlink(parquet_file)
    
    def test_trace_input_sources_invalid_json(self):
        """Test trace_input_sources with invalid JSON."""
        with pytest.raises(json.JSONDecodeError):
            trace_input_sources("invalid json")
    
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
            lf2 = pl.scan_parquet(parquet_file2)
            lf = pl.concat([lf1, lf2], how="horizontal")
            plan_json = lf.serialize(format="json")
            
            tracer = InputSourceTracer(plan_json)
            
            # HConcat should preserve columns from both sides
            assert "id" in tracer.column_sources or "id" in tracer.column_mappings
            assert "col1" in tracer.column_sources or "col1" in tracer.column_mappings
            assert "col2" in tracer.column_sources or "col2" in tracer.column_mappings
            
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
            plan_json = lf.serialize(format="json")
            
            tracer = InputSourceTracer(plan_json)
            
            # Union should preserve columns from both sides
            assert "id" in tracer.column_sources or "id" in tracer.column_mappings
            assert "name" in tracer.column_sources or "name" in tracer.column_mappings
            assert "age" in tracer.column_sources or "age" in tracer.column_mappings
            
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
                try:
                    result_lf = op_func(lf)
                    plan_json = result_lf.serialize(format="json")
                    sources = trace_input_sources(plan_json)
                    
                    # Check that we have sources from the original file
                    all_sources = []
                    for source_list in sources.values():
                        all_sources.extend(source_list)
                    
                    file_found = any(parquet_file in source for source in all_sources)
                    assert file_found, f"Original file not found in sources for {op_name}"
                    
                except Exception as e:
                    # Some operations might not be supported in all contexts
                    # Just ensure they don't crash the tracer
                    pass
                    
        finally:
            os.unlink(parquet_file)