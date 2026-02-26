"""
Input Source Tracer for Polars LazyFrame Plans.

This module traces column lineage through Polars LazyFrame query plans,
mapping each output column back to its original source columns and files.
"""

import json
import re
from typing import Dict, List, Set, Any, Optional
from collections import defaultdict
import polars as pl


class InputSourceTracer:
    """
    Traces column sources through a Polars LazyFrame JSON plan.
    
    Uses bottom-up traversal: processes child nodes first, then propagates
    column source information upward through the plan tree.
    """
    
    def __init__(self, json_plan: str):
        """Initialize with a JSON string representing a Polars lazy query plan."""
        self.plan = json.loads(json_plan)
        # Map output columns to their source columns: {col_name: {source_id, ...}}
        # Source ID format: "file_path|column_name"
        self.column_sources: Dict[str, Set[str]] = defaultdict(set)
        
        # Analyze the plan
        self._analyze_plan(self.plan)
    
    def _get_source_id(self, file_path: str, column_name: str) -> str:
        """Create a source identifier for a column from a file."""
        return f"{file_path}|{column_name}"
    
    def _parse_source_id(self, source_id: str) -> tuple:
        """Parse a source ID into (file_path, column_name)."""
        parts = source_id.rsplit('|', 1)
        if len(parts) == 2:
            return parts[0], parts[1]
        return None, source_id
    
    def _analyze_plan(self, node: Dict[str, Any]) -> Dict[str, Set[str]]:
        """
        Recursively analyze the query plan to track column sources.
        
        Returns a dict mapping column names to their source IDs for this subtree.
        This enables bottom-up propagation of source information.
        """
        if not isinstance(node, dict) or not node:
            return {}
        
        op_type = next(iter(node.keys()))
        op_data = node[op_type]
        
        # Handle each operation type
        if op_type == "Scan":
            return self._handle_scan(op_data)
        elif op_type == "Select":
            return self._handle_select(op_data)
        elif op_type == "Filter":
            return self._handle_filter(op_data)
        elif op_type == "Join":
            return self._handle_join(op_data)
        elif op_type == "Union":
            return self._handle_union(op_data)
        elif op_type == "HConcat":
            return self._handle_hconcat(op_data)
        elif op_type == "MapFunction":
            return self._handle_map_function(op_data)
        elif op_type == "GroupBy":
            return self._handle_groupby(op_data)
        elif op_type == "Aggregate":
            return self._handle_aggregate(op_data)
        elif op_type == "Sort":
            return self._handle_passthrough(op_data)
        elif op_type == "Slice":
            return self._handle_passthrough(op_data)
        elif op_type == "Distinct":
            return self._handle_passthrough(op_data)
        elif op_type == "Cache":
            return self._handle_passthrough(op_data)
        elif op_type == "WithColumns":
            return self._handle_with_columns(op_data)
        elif op_type == "HStack":
            return self._handle_hstack(op_data)
        else:
            # Generic handler for unknown operations with input
            return self._handle_generic(op_data)
    
    def _handle_scan(self, op_data: Dict[str, Any]) -> Dict[str, Set[str]]:
        """Handle Scan operation - the leaf node that reads from files."""
        sources = op_data.get("sources", {})
        file_path = None
        
        # Extract file path from various source structures
        if "Paths" in sources:
            paths = sources["Paths"]
            if paths and isinstance(paths[0], dict):
                if "Local" in paths[0]:
                    file_path = paths[0]["Local"]
                elif "inner" in paths[0]:
                    file_path = paths[0]["inner"]
            elif paths and isinstance(paths[0], str):
                file_path = paths[0]
        elif "Local" in sources:
            file_path = sources["Local"]
        elif "inner" in sources:
            file_path = sources["inner"]
        else:
            file_path = str(sources)
        
        if not file_path:
            return {}
        
        # Read schema to get column names
        local_sources = {}
        try:
            schema = pl.read_parquet_schema(file_path)
            for col_name in schema:
                source_id = self._get_source_id(file_path, col_name)
                local_sources[col_name] = {source_id}
                self.column_sources[col_name].add(source_id)
        except Exception:
            # If we can't read the schema, skip
            pass
        
        return local_sources
    
    def _handle_select(self, op_data: Dict[str, Any]) -> Dict[str, Set[str]]:
        """Handle Select operation - projects and transforms columns."""
        # First process input
        input_sources = {}
        if "input" in op_data:
            input_sources = self._analyze_plan(op_data["input"])
        
        # Then process expressions
        local_sources = {}
        for expr in op_data.get("expr", []):
            col_name, col_sources = self._extract_expr_sources(expr, input_sources)
            if col_name and col_sources:
                local_sources[col_name] = col_sources
                self.column_sources[col_name].update(col_sources)
        
        return local_sources if local_sources else input_sources
    
    def _handle_filter(self, op_data: Dict[str, Any]) -> Dict[str, Set[str]]:
        """Handle Filter operation - passes through all columns."""
        input_sources = {}
        if "input" in op_data:
            input_sources = self._analyze_plan(op_data["input"])
        return input_sources
    
    def _handle_join(self, op_data: Dict[str, Any]) -> Dict[str, Set[str]]:
        """Handle Join operation - merges columns from both sides."""
        left_sources = {}
        right_sources = {}
        
        if "input_left" in op_data:
            left_sources = self._analyze_plan(op_data["input_left"])
        if "input_right" in op_data:
            right_sources = self._analyze_plan(op_data["input_right"])
        
        # Merge sources from both sides
        merged = defaultdict(set)
        for col, sources in left_sources.items():
            merged[col].update(sources)
            self.column_sources[col].update(sources)
        for col, sources in right_sources.items():
            merged[col].update(sources)
            self.column_sources[col].update(sources)
        
        return dict(merged)
    
    def _handle_union(self, op_data: Dict[str, Any]) -> Dict[str, Set[str]]:
        """Handle Union operation - merges sources from all inputs for each column."""
        merged = defaultdict(set)
        
        for input_node in op_data.get("inputs", []):
            input_sources = self._analyze_plan(input_node)
            for col, sources in input_sources.items():
                merged[col].update(sources)
                self.column_sources[col].update(sources)
        
        return dict(merged)
    
    def _handle_hconcat(self, op_data: Dict[str, Any]) -> Dict[str, Set[str]]:
        """Handle HConcat operation - columns from all inputs side by side."""
        merged = defaultdict(set)
        
        for input_node in op_data.get("inputs", []):
            input_sources = self._analyze_plan(input_node)
            for col, sources in input_sources.items():
                merged[col].update(sources)
                self.column_sources[col].update(sources)
        
        return dict(merged)
    
    def _handle_map_function(self, op_data: Dict[str, Any]) -> Dict[str, Set[str]]:
        """Handle MapFunction operation - includes Unnest, Explode, Rename, etc."""
        # First process input
        input_sources = {}
        if "input" in op_data:
            input_sources = self._analyze_plan(op_data["input"])
        
        function = op_data.get("function", {})
        
        # Handle Unnest operation
        if "Unnest" in function:
            return self._handle_unnest(function["Unnest"], input_sources)
        
        # Handle Rename operation
        if "Rename" in function:
            return self._handle_rename(function["Rename"], input_sources)
        
        # For other map functions, pass through sources
        return input_sources
    
    def _handle_rename(self, rename_data: Dict[str, Any], input_sources: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
        """Handle Rename operation - column names change but sources are preserved."""
        existing = rename_data.get("existing", [])
        new = rename_data.get("new", [])
        
        local_sources = dict(input_sources)
        
        for old_name, new_name in zip(existing, new):
            if old_name in local_sources:
                # Transfer sources from old name to new name
                local_sources[new_name] = local_sources[old_name]
                self.column_sources[new_name].update(local_sources[old_name])
                # Remove old name
                del local_sources[old_name]
        
        return local_sources
    
    def _handle_unnest(self, unnest_data: Dict[str, Any], input_sources: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
        """
        Handle Unnest operation - struct columns are expanded into their fields.
        
        The resulting field columns should trace back to the original struct column's source.
        """
        # Extract columns being unnested
        columns_to_unnest = []
        
        # Try finding Union in various places (Polars versions/plans vary)
        union_data = None
        if "Union" in unnest_data:
            union_data = unnest_data["Union"]
        elif "columns" in unnest_data and isinstance(unnest_data["columns"], dict) and "Union" in unnest_data["columns"]:
            union_data = unnest_data["columns"]["Union"]
            
        if isinstance(union_data, list):
            for item in union_data:
                if "ByName" in item and "names" in item["ByName"]:
                    columns_to_unnest.extend(item["ByName"]["names"])
        
        # For unnested struct columns, we need to find the source for the struct
        # and map it to the resulting field columns
        local_sources = dict(input_sources)
        
        for struct_col in columns_to_unnest:
            if struct_col in input_sources:
                struct_sources = input_sources[struct_col]
                for source_id in struct_sources:
                    file_path, _ = self._parse_source_id(source_id)
                    if file_path:
                        try:
                            schema = pl.read_parquet_schema(file_path)
                            if struct_col in schema:
                                struct_dtype = schema[struct_col]
                                if hasattr(struct_dtype, 'fields'):
                                    for field in struct_dtype.fields:
                                        field_name = field.name
                                        field_source = self._get_source_id(file_path, struct_col)
                                        local_sources[field_name] = {field_source}
                                        self.column_sources[field_name].add(field_source)
                        except Exception:
                            pass
                
                # Remove the struct column since it's been unnested
                if struct_col in local_sources:
                    del local_sources[struct_col]
        
        return local_sources
    
    def _handle_groupby(self, op_data: Dict[str, Any]) -> Dict[str, Set[str]]:
        """Handle GroupBy operation."""
        input_sources = {}
        if "input" in op_data:
            input_sources = self._analyze_plan(op_data["input"])
        
        local_sources = {}
        # Keys are passed through
        for key_expr in op_data.get("keys", []):
            col_name, col_sources = self._extract_expr_sources(key_expr, input_sources)
            if col_name and col_sources:
                local_sources[col_name] = col_sources
                self.column_sources[col_name].update(col_sources)
        
        # Process aggregations
        for agg_expr in op_data.get("aggs", []):
            col_name, col_sources = self._extract_expr_sources(agg_expr, input_sources)
            if col_name and col_sources:
                local_sources[col_name] = col_sources
                self.column_sources[col_name].update(col_sources)
                
        return local_sources if local_sources else input_sources
    
    def _handle_aggregate(self, op_data: Dict[str, Any]) -> Dict[str, Set[str]]:
        """Handle Aggregate operation."""
        input_sources = {}
        if "input" in op_data:
            input_sources = self._analyze_plan(op_data["input"])
        
        # Process aggregate expressions
        local_sources = dict(input_sources)
        for expr in op_data.get("expr", []):
            col_name, col_sources = self._extract_expr_sources(expr, input_sources)
            if col_name and col_sources:
                local_sources[col_name] = col_sources
                self.column_sources[col_name].update(col_sources)
        
        return local_sources
    
    def _handle_with_columns(self, op_data: Dict[str, Any]) -> Dict[str, Set[str]]:
        """Handle WithColumns operation - adds new columns."""
        input_sources = {}
        if "input" in op_data:
            input_sources = self._analyze_plan(op_data["input"])
        
        local_sources = dict(input_sources)
        for expr in op_data.get("expr", []):
            col_name, col_sources = self._extract_expr_sources(expr, input_sources)
            if col_name and col_sources:
                local_sources[col_name] = col_sources
                self.column_sources[col_name].update(col_sources)
        return local_sources

    def _handle_hstack(self, op_data: Dict[str, Any]) -> Dict[str, Set[str]]:
        """Handle HStack operation - adds new columns (Polars with_columns logic)."""
        input_sources = {}
        if "input" in op_data:
            input_sources = self._analyze_plan(op_data["input"])
        
        local_sources = dict(input_sources)
        # HStack uses exprs (plural)
        for expr in op_data.get("exprs", []):
            col_name, col_sources = self._extract_expr_sources(expr, input_sources)
            if col_name and col_sources:
                local_sources[col_name] = col_sources
                self.column_sources[col_name].update(col_sources)
        
        return local_sources
    
    def _handle_passthrough(self, op_data: Dict[str, Any]) -> Dict[str, Set[str]]:
        """Handle operations that pass through all columns unchanged."""
        if "input" in op_data:
            return self._analyze_plan(op_data["input"])
        return {}
    
    def _handle_generic(self, op_data: Dict[str, Any]) -> Dict[str, Set[str]]:
        """Generic handler for unknown operations."""
        if not isinstance(op_data, dict):
            return {}
        
        merged = defaultdict(set)
        
        # Process any nested plans in common locations
        for key in ["input", "input_left", "input_right"]:
            if key in op_data and isinstance(op_data[key], dict):
                input_sources = self._analyze_plan(op_data[key])
                for col, sources in input_sources.items():
                    merged[col].update(sources)
                    self.column_sources[col].update(sources)
        
        # Process list inputs
        if "inputs" in op_data and isinstance(op_data["inputs"], list):
            for input_node in op_data["inputs"]:
                if isinstance(input_node, dict):
                    input_sources = self._analyze_plan(input_node)
                    for col, sources in input_sources.items():
                        merged[col].update(sources)
                        self.column_sources[col].update(sources)
        
        return dict(merged) if merged else op_data.get("input", {}) if isinstance(op_data.get("input"), dict) else {}
    
    def _extract_expr_sources(self, expr: Dict[str, Any], input_sources: Dict[str, Set[str]]) -> tuple:
        """
        Extract column name and sources from an expression.
        
        Returns (column_name, set of source IDs).
        """
        if not isinstance(expr, dict):
            return None, set()
        
        # Direct column reference
        if "Column" in expr:
            col_name = expr["Column"]
            sources = input_sources.get(col_name, set())
            if sources:
                return col_name, sources
            return col_name, {col_name}  # Fallback to column name as source
        
        # Alias expression: [inner_expr, alias_name]
        if "Alias" in expr:
            alias_data = expr["Alias"]
            if isinstance(alias_data, list) and len(alias_data) == 2:
                inner_expr, alias_name = alias_data
                _, inner_sources = self._extract_expr_sources(inner_expr, input_sources)
                return alias_name, inner_sources
        
        # Binary expression
        if "BinaryExpr" in expr:
            binary = expr["BinaryExpr"]
            sources = set()
            if "left" in binary:
                _, left_sources = self._extract_expr_sources(binary["left"], input_sources)
                sources.update(left_sources)
            if "right" in binary:
                _, right_sources = self._extract_expr_sources(binary["right"], input_sources)
                sources.update(right_sources)
            return None, sources
        
        # Aggregation
        if "Agg" in expr:
            agg_data = expr["Agg"]
            for agg_type, agg_expr in agg_data.items():
                if isinstance(agg_expr, dict):
                    if "input" in agg_expr:
                        return self._extract_expr_sources(agg_expr["input"], input_sources)
                    return self._extract_expr_sources(agg_expr, input_sources)
            return None, set()
        
        # Function
        if "Function" in expr:
            func_data = expr["Function"]
            sources = set()
            for input_expr in func_data.get("input", []):
                if isinstance(input_expr, dict):
                    _, input_source = self._extract_expr_sources(input_expr, input_sources)
                    sources.update(input_source)
            return None, sources
        
        # Window function
        if "Window" in expr:
            window_data = expr["Window"]
            sources = set()
            if "function" in window_data:
                _, func_sources = self._extract_expr_sources(window_data["function"], input_sources)
                sources.update(func_sources)
            for part in window_data.get("partition_by", []):
                if isinstance(part, dict):
                    _, part_sources = self._extract_expr_sources(part, input_sources)
                    sources.update(part_sources)
            return None, sources
        
        # SortBy expression
        if "SortBy" in expr:
            sort_data = expr["SortBy"]
            if "expr" in sort_data:
                return self._extract_expr_sources(sort_data["expr"], input_sources)
        
        # Recursive fallback for other expression types
        sources = set()
        for key, value in expr.items():
            if isinstance(value, dict):
                _, nested_sources = self._extract_expr_sources(value, input_sources)
                sources.update(nested_sources)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        _, nested_sources = self._extract_expr_sources(item, input_sources)
                        sources.update(nested_sources)
        
        return None, sources
    
    def get_column_sources(self) -> Dict[str, List[str]]:
        """
        Return a dictionary mapping each output column to its original source columns.
        Format: {column_name: [transform_rid|column_name, ...]}
        """
        results = {}
        # Regex to extract RID from path like ./data/rid.transform.uuid/datasets/data.parquet
        rid_pattern = re.compile(r'rid\.transform\.[a-f0-9a-zA-Z-]+')
        
        for col, sources in self.column_sources.items():
            if not sources:
                continue
                
            formatted_sources = set()
            for source_id in sources:
                file_path, col_name = self._parse_source_id(source_id)
                if file_path:
                    match = rid_pattern.search(file_path)
                    if match:
                        rid = match.group(0)
                        formatted_sources.add(f"{rid}|{col_name}")
                    else:
                        # Use full path if no RID pattern found (important for library tests)
                        formatted_sources.add(f"{file_path}|{col_name}")
                else:
                    # Fallback for columns that couldn't be traced to a file
                    # Keep as is (usually just the column name from fallback in _extract_expr_sources)
                    formatted_sources.add(source_id)
            
            if formatted_sources:
                results[col] = sorted(list(formatted_sources))
            
        return results


def trace_input_sources(json_plan: str) -> Dict[str, List[str]]:
    """
    Trace each output column back to its original input dataframe columns.
    
    Args:
        json_plan: JSON string representing a Polars lazy query plan
        
    Returns:
        Dictionary mapping each output column to a list of its source columns with file paths
    """
    tracer = InputSourceTracer(json_plan)
    return tracer.get_column_sources()
