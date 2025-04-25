import json
import hashlib
import re
from typing import Dict, List, Set, Any, Optional
from collections import defaultdict
import polars as pl

class InputSourceTracer:
    def __init__(self, json_plan: str):
        """Initialize with a JSON string representing a Polars lazy query plan."""
        self.plan = json.loads(json_plan)
        # Map output columns to their source columns with dataframe identifiers
        self.column_sources = defaultdict(set)
        # Track dataframe scans to assign unique IDs
        self.dataframe_scans = {}
        self.df_counter = 0
        # Track intermediate column mappings
        self.column_mappings = {}
        # Start the analysis
        self._analyze_plan(self.plan)
        # Resolve all mappings to find original sources
        self._resolve_all_mappings()

    def _get_df_id(self, df_node):
        """Generate a unique identifier for a dataframe scan."""
        df_hash = hashlib.md5(json.dumps(df_node, sort_keys=True).encode()).hexdigest()
        if df_hash not in self.dataframe_scans:
            self.dataframe_scans[df_hash] = f"df_{self.df_counter}"
            self.df_counter += 1
        return df_hash

    def _analyze_plan(self, node: Dict[str, Any], parent_op: Optional[str] = None) -> None:
        """Recursively analyze the query plan to track column sources."""
        if not isinstance(node, dict):
            return
            
        op_types = list(node.keys())
        if not op_types:
            return
            
        op_type = op_types[0]
        
      
        if op_type == "Scan":
            # Record original columns from the data source
            df_id = node[op_type]["sources"]["Paths"][0]
            schema = pl.read_parquet_schema(df_id)
            for col_name in schema:
                # Extract transform id by regex \/(rid.transform.*)\/ from df_id                
                transform_id = re.search(r'\/(rid\.transform.*?)\/', df_id).group(1)
                # Map the original column to itself with the df_id
                source_id = f"{df_id}|{col_name}"
                self.column_sources[col_name].add(source_id)
                # Also keep direct mapping
                self.column_mappings[col_name] = {source_id}
        
        elif op_type == "Rename":
            # Handle explicit rename operations
            if "existing" in node[op_type] and "new" in node[op_type]:
                for old_name, new_name in zip(node[op_type]["existing"], node[op_type]["new"]):
                    if old_name in self.column_mappings:
                        self.column_mappings[new_name] = self.column_mappings[old_name].copy()
                    else:
                        # If we don't have a mapping yet, create a placeholder
                        self.column_mappings[new_name] = {old_name}
        
        elif op_type == "Alias":
            # Track column aliases - Alias is a list [expression, name]
            if isinstance(node[op_type], list) and len(node[op_type]) == 2:
                expr, alias_name = node[op_type]
                
                # Check if the expression refers directly to a column
                if isinstance(expr, dict) and "Column" in expr:
                    source_col = expr["Column"]
                    if source_col in self.column_mappings:
                        self.column_mappings[alias_name] = self.column_mappings[source_col].copy()
                    else:
                        # If we don't have a mapping yet, create a placeholder
                        self.column_mappings[alias_name] = {source_col}
                
                # Check for expressions that combine multiple columns
                elif isinstance(expr, dict):
                    # Extract all column references from the expression
                    source_cols = self._extract_columns_from_expr(expr)
                    if source_cols:
                        # Collect all sources for these columns
                        all_sources = set()
                        for col in source_cols:
                            if col in self.column_mappings:
                                all_sources.update(self.column_mappings[col])
                            else:
                                all_sources.add(col)  # Use as is if not mapped yet
                        self.column_mappings[alias_name] = all_sources
                
                # Process the expression part recursively
                self._analyze_plan(expr)
            return  # Skip the generic processing for Alias
        
        # Handle other operations with inputs
        if isinstance(node[op_type], dict):
            for key, value in node[op_type].items():
                if key in ["input", "input_left", "input_right"] and isinstance(value, dict):
                    self._analyze_plan(value, op_type)
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            self._analyze_plan(item, op_type)
                elif isinstance(value, dict):
                    self._analyze_plan(value, op_type)

    def _extract_columns_from_expr(self, expr: Dict[str, Any]) -> Set[str]:
        """Extract all column references from an expression."""
        columns = set()
        
        if not expr:
            return columns
            
        # Direct column reference
        if "Column" in expr:
            columns.add(expr["Column"])
            return columns
            
        # Handle binary expressions which might combine columns
        if "BinaryExpr" in expr:
            binary_expr = expr["BinaryExpr"]
            if "left" in binary_expr and isinstance(binary_expr["left"], dict):
                columns.update(self._extract_columns_from_expr(binary_expr["left"]))
            if "right" in binary_expr and isinstance(binary_expr["right"], dict):
                columns.update(self._extract_columns_from_expr(binary_expr["right"]))
        
        # Handle window functions and aggregations
        if "Window" in expr:
            window_expr = expr["Window"]
            if "function" in window_expr and isinstance(window_expr["function"], dict):
                columns.update(self._extract_columns_from_expr(window_expr["function"]))
            if "partition_by" in window_expr and isinstance(window_expr["partition_by"], list):
                for part in window_expr["partition_by"]:
                    if isinstance(part, dict):
                        columns.update(self._extract_columns_from_expr(part))
        
        # Handle aggregations
        if "Agg" in expr:
            agg_expr = expr["Agg"]
            for agg_type, agg_col in agg_expr.items():
                if isinstance(agg_col, dict) and "Column" in agg_col:
                    columns.add(agg_col["Column"])
        
        # Handle functions
        if "Function" in expr:
            func_expr = expr["Function"]
            if "input" in func_expr and isinstance(func_expr["input"], list):
                for input_expr in func_expr["input"]:
                    if isinstance(input_expr, dict):
                        columns.update(self._extract_columns_from_expr(input_expr))
        
        # Recursively process other dict items
        for key, value in expr.items():
            if isinstance(value, dict):
                columns.update(self._extract_columns_from_expr(value))
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        columns.update(self._extract_columns_from_expr(item))
        
        return columns

    def _resolve_all_mappings(self):
        """Resolve all column mappings to find original sources."""
        # For each column, find the ultimate sources
        resolved_sources = defaultdict(set)
        
        for col, sources in self.column_mappings.items():
            # Process each source
            for source in sources:
                if '.' in source:  # This is already a resolved source with df_id
                    resolved_sources[col].add(source)
                elif source in self.column_mappings:
                    # This is an intermediate mapping, follow it
                    for ultimate_source in self.column_mappings[source]:
                        if '.' in ultimate_source:  # This is a resolved source
                            resolved_sources[col].add(ultimate_source)
        
        # Update column_sources with resolved mappings
        for col, sources in resolved_sources.items():
            if sources:  # Only update if we found actual sources
                self.column_sources[col] = sources

    def get_column_sources(self):
        """
        Return a dictionary mapping each output column to its original source columns.
        Format: {column_name: [df_id.column_name, ...]}
        """
        return {col: list(sources) for col, sources in self.column_sources.items()}

# Main function to trace column sources
def trace_input_sources(json_plan: str) -> Dict[str, List[str]]:
    """
    Trace each output column back to its original input dataframe columns.
    
    Args:
        json_plan: JSON string representing a Polars lazy query plan
        
    Returns:
        Dictionary mapping each output column to a list of its source columns with dataframe IDs
    """
    tracer = InputSourceTracer(json_plan)
    return tracer.get_column_sources()
