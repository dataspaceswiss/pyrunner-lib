"""Recover column lineage from Polars' optimized lazy query plan."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, ClassVar

import polars as pl


@dataclass
class _ColumnLineage:
    sources: set[str] = field(default_factory=set)
    fields: dict[str, _ColumnLineage] = field(default_factory=dict)

    def clone(self) -> _ColumnLineage:
        return _ColumnLineage(
            set(self.sources),
            {name: lineage.clone() for name, lineage in self.fields.items()},
        )

    def merge(self, other: _ColumnLineage) -> None:
        self.sources.update(other.sources)
        for name, lineage in other.fields.items():
            if name in self.fields:
                self.fields[name].merge(lineage)
            else:
                self.fields[name] = lineage.clone()


@dataclass
class _PlanState:
    columns: dict[str, _ColumnLineage] = field(default_factory=dict)

    def clone(self) -> _PlanState:
        return _PlanState(
            {name: lineage.clone() for name, lineage in self.columns.items()}
        )


@dataclass
class _ExpressionItem:
    name: str | None
    lineage: _ColumnLineage


@dataclass(frozen=True)
class _OpaqueNode:
    reason: str
    inputs: tuple[int, ...]
    expression_names: tuple[str, ...]


def _format_column_sources(
    column_sources: dict[str, set[str]],
) -> dict[str, list[str]]:
    results: dict[str, list[str]] = {}
    rid_pattern = re.compile(r"rid\.transform\.[a-f0-9a-zA-Z-]+")
    for column, sources in column_sources.items():
        formatted: set[str] = set()
        for source_id in sources:
            parts = source_id.rsplit("|", 1)
            if len(parts) == 1:
                formatted.add(source_id)
                continue
            file_path, source_column = parts
            match = rid_pattern.search(file_path)
            source_name = match.group(0) if match else file_path
            formatted.add(f"{source_name}|{source_column}")
        if formatted:
            results[column] = sorted(formatted)
    return results


class InputSourceTracer:
    """Trace a LazyFrame through Polars' structured optimized IR."""

    _SUPPORTED_IR_MAJORS: ClassVar[range] = range(10, 15)
    _PASSTHROUGH_NODES: ClassVar[frozenset[str]] = frozenset(
        {"Cache", "Distinct", "Filter", "Slice", "Sort"}
    )
    _EXPRESSION_CHILDREN: ClassVar[tuple[str, ...]] = (
        "left",
        "right",
        "input",
        "expr",
        "function",
        "predicate",
        "truthy",
        "falsy",
        "partition_by",
        "order_by",
        "arguments",
        "by",
        "offset",
        "length",
        "idx",
    )

    def __init__(self, lazyframe: pl.LazyFrame):
        if not isinstance(lazyframe, pl.LazyFrame):
            raise TypeError("lazyframe must be a polars.LazyFrame")

        # NodeTraverser is Polars' versioned plugin interface for optimized IR.
        # It is not public, so reject unknown major versions instead of guessing.
        self._visitor = lazyframe._ldf.visit()  # type: ignore[attr-defined]
        version = self._visitor.version()
        if version[0] not in self._SUPPORTED_IR_MAJORS:
            raise NotImplementedError(f"unsupported Polars IR version {version}")

        self._analysis_cache: dict[int, _PlanState] = {}
        final_state = self._analyze(self._visitor.get_node())
        self.column_sources = {
            name: lineage.sources
            for name, lineage in final_state.columns.items()
            if lineage.sources
        }

    def _node(self, node_id: int) -> tuple[Any, list[str]]:
        self._visitor.set_node(node_id)
        output_names = list(self._visitor.get_schema())
        try:
            node = self._visitor.view_current_node()
        except NotImplementedError as error:
            node = _OpaqueNode(
                str(error).lower(),
                tuple(self._visitor.get_inputs()),
                tuple(
                    expression.output_name for expression in self._visitor.get_exprs()
                ),
            )
        return node, output_names

    @staticmethod
    def _state(
        output_names: Iterable[str],
        columns: dict[str, _ColumnLineage] | None = None,
    ) -> _PlanState:
        available = columns or {}
        return _PlanState(
            {name: available.get(name, _ColumnLineage()) for name in output_names}
        )

    @staticmethod
    def _items(state: _PlanState) -> list[_ExpressionItem]:
        return [
            _ExpressionItem(name, lineage.clone())
            for name, lineage in state.columns.items()
        ]

    @staticmethod
    def _by_position(
        output_names: list[str], items: Iterable[_ExpressionItem]
    ) -> dict[str, _ColumnLineage]:
        items = list(items)
        if len(items) != len(output_names):
            return {}
        return {
            output_name: item.lineage for output_name, item in zip(output_names, items)
        }

    def _analyze(self, node_id: int) -> _PlanState:
        cached = self._analysis_cache.get(node_id)
        if cached is not None:
            return cached.clone()

        state = self._analyze_node(node_id)
        self._analysis_cache[node_id] = state.clone()
        return state

    def _analyze_node(self, node_id: int) -> _PlanState:
        node, output_names = self._node(node_id)
        kind = type(node).__name__

        if kind == "Scan":
            return self._scan(node, output_names)
        if kind in {"DataFrameScan", "PythonScan", "Empty"}:
            return self._state(output_names)
        if kind == "SimpleProjection":
            return self._state(output_names, self._analyze(node.input).columns)
        if kind in {"Select", "Reduce"}:
            return self._select(node, output_names)
        if kind == "HStack":
            return self._hstack(node, output_names)
        if kind in self._PASSTHROUGH_NODES:
            return self._state(output_names, self._analyze(node.input).columns)
        if kind in {"Join", "ConditionalJoin"}:
            return self._join(node, output_names)
        if kind == "Union":
            return self._union(node, output_names)
        if kind in {"HConcat", "ExtContext"}:
            return self._hconcat(node, output_names)
        if kind == "GroupBy":
            return self._groupby(node, output_names)
        if kind == "MapFunction":
            return self._map_function(node, output_names)
        if kind == "MergeSorted":
            return self._merge_sorted(node, output_names)
        if kind == "_OpaqueNode" and node.reason == "asof join":
            return self._opaque_asof_join(node, output_names)

        return self._state(output_names)

    def _opaque_asof_join(
        self, node: _OpaqueNode, output_names: list[str]
    ) -> _PlanState:
        if len(node.inputs) != 2 or len(node.expression_names) % 2:
            return self._state(output_names)

        left = self._analyze(node.inputs[0])
        right = self._analyze(node.inputs[1])
        key_count = len(node.expression_names) // 2
        right_keys = set(node.expression_names[key_count:])
        sequence = self._items(left)
        sequence.extend(
            item for item in self._items(right) if item.name not in right_keys
        )
        return self._state(output_names, self._by_position(output_names, sequence))

    def _scan(self, node: Any, output_names: list[str]) -> _PlanState:
        columns: dict[str, _ColumnLineage] = {}
        for path in node.paths:
            file_path = path.removeprefix("file://")
            try:
                physical_columns = pl.read_parquet_schema(file_path)
            except Exception:  # noqa: BLE001, S112
                continue
            for name in output_names:
                if name in physical_columns:
                    lineage = columns.setdefault(name, _ColumnLineage())
                    lineage.sources.add(f"{file_path}|{name}")
        return self._state(output_names, columns)

    def _named_expressions(
        self, expressions: Iterable[Any], input_state: _PlanState
    ) -> dict[str, _ColumnLineage]:
        return {
            expression.output_name: self._expression_lineage(
                expression.node, input_state
            )
            for expression in expressions
        }

    def _select(self, node: Any, output_names: list[str]) -> _PlanState:
        input_state = self._analyze(node.input)
        self._visitor.set_node(node.input)
        columns = self._named_expressions(node.expr, input_state)
        return self._state(output_names, columns)

    def _hstack(self, node: Any, output_names: list[str]) -> _PlanState:
        input_state = self._analyze(node.input)
        columns = {
            name: lineage.clone() for name, lineage in input_state.columns.items()
        }
        self._visitor.set_node(node.input)
        columns.update(self._named_expressions(node.exprs, input_state))
        return self._state(output_names, columns)

    def _direct_column_name(self, expression: Any, context_node: int) -> str | None:
        self._visitor.set_node(context_node)
        node = self._visitor.view_expression(expression.node)
        return node.name if type(node).__name__ == "Column" else None

    def _join(self, node: Any, output_names: list[str]) -> _PlanState:
        left = self._analyze(node.input_left)
        right = self._analyze(node.input_right)
        left_items = self._items(left)
        right_items = self._items(right)
        left_keys = {
            name
            for expression in node.left_on
            if (name := self._direct_column_name(expression, node.input_left))
            is not None
        }
        right_keys = {
            name
            for expression in node.right_on
            if (name := self._direct_column_name(expression, node.input_right))
            is not None
        }
        how = node.options[0]
        suffix = node.options[3]
        coalesce = bool(node.options[4])

        if how in {"Semi", "Anti"}:
            sequence = left_items
        elif how in {"Full", "Cross"} and not coalesce:
            sequence = left_items + right_items
        elif how == "Right":
            sequence = [item for item in left_items if item.name not in left_keys]
            sequence.extend(right_items)
        else:
            sequence = left_items + [
                item for item in right_items if item.name not in right_keys
            ]

        if how == "Full" and coalesce:
            sequence_by_name = {item.name: item.lineage for item in sequence}
            right_by_name = {item.name: item.lineage for item in right_items}
            for left_key, right_key in zip(node.left_on, node.right_on):
                left_name = self._direct_column_name(left_key, node.input_left)
                right_name = self._direct_column_name(right_key, node.input_right)
                if left_name in sequence_by_name and right_name in right_by_name:
                    sequence_by_name[left_name].merge(right_by_name[right_name])

        columns = self._by_position(output_names, sequence)
        if columns:
            return self._state(output_names, columns)

        columns = {
            name: lineage.clone()
            for name, lineage in left.columns.items()
            if name in output_names
        }
        for name, lineage in right.columns.items():
            candidate = name if name not in left.columns else f"{name}{suffix}"
            if candidate in output_names:
                columns[candidate] = lineage.clone()
        return self._state(output_names, columns)

    def _union(self, node: Any, output_names: list[str]) -> _PlanState:
        columns = {name: _ColumnLineage() for name in output_names}
        for input_node in node.inputs:
            state = self._analyze(input_node)
            if len(state.columns) != len(output_names):
                continue
            for output_name, lineage in zip(output_names, state.columns.values()):
                columns[output_name].merge(lineage)
        return self._state(output_names, columns)

    def _hconcat(self, node: Any, output_names: list[str]) -> _PlanState:
        columns: dict[str, _ColumnLineage] = {}
        for input_node in node.inputs:
            for name, lineage in self._analyze(input_node).columns.items():
                columns.setdefault(name, lineage.clone())
        return self._state(output_names, columns)

    def _groupby(self, node: Any, output_names: list[str]) -> _PlanState:
        input_state = self._analyze(node.input)
        self._visitor.set_node(node.input)
        expressions = [*node.keys, *node.aggs]
        columns = self._named_expressions(expressions, input_state)
        window = node.options.rolling or node.options.dynamic
        if window is not None and window.index_column in input_state.columns:
            columns[window.index_column] = input_state.columns[
                window.index_column
            ].clone()
        return self._state(output_names, columns)

    def _map_function(self, node: Any, output_names: list[str]) -> _PlanState:
        input_state = self._analyze(node.input)
        function = node.function
        name = function[0].lower()

        if name in {"explode", "rechunk"}:
            return self._state(output_names, input_state.columns)
        if name == "unnest":
            separator = function[2] if len(function) > 2 else None
            return self._unnest(output_names, input_state, function[1], separator)
        if name == "unpivot":
            return self._unpivot(output_names, input_state, function)

        # Python UDFs and other opaque map functions may rewrite values while
        # keeping names, so matching schemas are not sufficient evidence.
        return self._state(output_names)

    @staticmethod
    def _unnest(
        output_names: list[str],
        input_state: _PlanState,
        struct_columns: Iterable[str],
        separator: str | None,
    ) -> _PlanState:
        struct_columns = list(struct_columns)
        columns: dict[str, _ColumnLineage] = {}
        for output_name in output_names:
            if output_name in input_state.columns and output_name not in struct_columns:
                columns[output_name] = input_state.columns[output_name].clone()
                continue
            for struct_name in struct_columns:
                struct_lineage = input_state.columns.get(struct_name)
                if struct_lineage is None:
                    continue
                field_name = output_name
                prefix = f"{struct_name}{separator}" if separator else ""
                if prefix and output_name.startswith(prefix):
                    field_name = output_name[len(prefix) :]
                field_lineage = struct_lineage.fields.get(field_name)
                columns[output_name] = (
                    field_lineage.clone()
                    if field_lineage is not None
                    else _ColumnLineage(set(struct_lineage.sources))
                )
                break
        return InputSourceTracer._state(output_names, columns)

    @staticmethod
    def _unpivot(
        output_names: list[str], input_state: _PlanState, function: tuple[Any, ...]
    ) -> _PlanState:
        _, index_names, value_names, variable_name, value_name = function
        variable_name = "variable" if variable_name is None else variable_name
        value_name = "value" if value_name is None else value_name
        columns = {
            name: input_state.columns[name].clone()
            for name in index_names
            if name in input_state.columns
        }
        value_lineage = _ColumnLineage()
        for name in value_names:
            if name in input_state.columns:
                value_lineage.merge(input_state.columns[name])
        columns[value_name] = value_lineage
        columns[variable_name] = _ColumnLineage()
        return InputSourceTracer._state(output_names, columns)

    def _merge_sorted(self, node: Any, output_names: list[str]) -> _PlanState:
        columns = {name: _ColumnLineage() for name in output_names}
        for input_node in (node.input_left, node.input_right):
            state = self._analyze(input_node)
            for output_name, lineage in zip(output_names, state.columns.values()):
                columns[output_name].merge(lineage)
        return self._state(output_names, columns)

    @staticmethod
    def _expression_ids(value: Any) -> Iterable[int]:
        if type(value) is int:
            yield value
        elif isinstance(value, (list, tuple)):
            for item in value:
                yield from InputSourceTracer._expression_ids(item)
        elif type(getattr(value, "node", None)) is int:
            yield value.node

    def _expression_lineage(
        self, expression_id: int, input_state: _PlanState
    ) -> _ColumnLineage:
        try:
            expression = self._visitor.view_expression(expression_id)
        except NotImplementedError:
            # Anonymous Python expressions are opaque to NodeTraverser.
            return _ColumnLineage()
        kind = type(expression).__name__
        if kind == "Column":
            lineage = input_state.columns.get(expression.name)
            return lineage.clone() if lineage is not None else _ColumnLineage()
        if kind in {"Literal", "Len"}:
            return _ColumnLineage()
        if kind == "Function":
            function = expression.function_data
            function_name = str(function[0]).lower()
            inputs = list(self._expression_ids(expression.input))
            if function_name == "as_struct":
                dtype = self._visitor.get_dtype(expression_id)
                field_names = [field.name for field in dtype.fields]
                fields = {
                    name: self._expression_lineage(child, input_state)
                    for name, child in zip(field_names, inputs)
                }
                lineage = _ColumnLineage(fields=fields)
                for field_lineage in fields.values():
                    lineage.merge(field_lineage)
                return lineage
            if "fieldbyname" in function_name and len(function) > 1 and inputs:
                struct_lineage = self._expression_lineage(inputs[0], input_state)
                selected_field = struct_lineage.fields.get(function[1])
                return (
                    selected_field.clone()
                    if selected_field is not None
                    else _ColumnLineage(set(struct_lineage.sources))
                )

        lineage = _ColumnLineage()
        for attribute in self._EXPRESSION_CHILDREN:
            if not hasattr(expression, attribute):
                continue
            for child in self._expression_ids(getattr(expression, attribute)):
                lineage.merge(self._expression_lineage(child, input_state))
        return lineage

    def get_column_sources(self) -> dict[str, list[str]]:
        return _format_column_sources(self.column_sources)


def trace_lazyframe_sources(lazyframe: pl.LazyFrame) -> dict[str, list[str]]:
    """Trace a live LazyFrame using Polars' optimized, structured query plan."""
    return InputSourceTracer(lazyframe).get_column_sources()
