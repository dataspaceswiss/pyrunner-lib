"""
Health checks on a transform's written output.

Checks run on the parquet file after it is written, so each check reads only the
columns it needs instead of re-running the transform. For a failed check, the
report lists the positions of the first VIOLATION_SAMPLE violating rows. They point
into the immutable output file, so the platform can show those rows without
storing any row data.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterable
from datetime import datetime, timezone
from typing import Any

import polars as pl

CheckCondition = Callable[[pl.LazyFrame, str], None]
# Marks the rows that break a check; None for checks on a whole column (sums, counts)
ViolationExpr = Callable[[str], pl.Expr]
CheckDefinition = tuple[str, str, CheckCondition, ViolationExpr | None]

# Row positions saved per failed check: enough for the distinct violating values, which DataSpace lists up to 1000
VIOLATION_SAMPLE = 1000


class HealthCheckFailure(Exception):
    """Raised when one or more fail-severity checks fail."""


def run_health_checks(
    dataset_path: str, column_builders: Iterable[Check]
) -> dict[str, Any]:
    """
    Run health checks on the written output at dataset_path.

    Writes meta/health_report.json, then raises HealthCheckFailure if a
    fail-severity check failed.
    """
    lf = pl.scan_parquet(dataset_path)

    column_checks: dict[str, list[CheckDefinition]] = {}
    for builder in column_builders:
        column, checks = builder.build()
        if checks:
            column_checks.setdefault(column, []).extend(checks)

    report_by_col: dict[str, dict[str, Any]] = {}
    overall_failures: list[tuple[str, str, str]] = []
    check_id = 0

    for col, checks in column_checks.items():
        results: list[dict[str, Any]] = []
        passed = 0

        for check_name, severity, condition, violation in checks:
            error_message = None
            rows: list[int] = []
            try:
                condition(lf, col)
                ok = True
            # A failed predicate is report data, including unexpected exceptions
            # from user-supplied custom checks.
            except Exception as e:  # noqa: BLE001
                ok = False
                error_message = str(e)
                if severity.lower() == "fail":
                    overall_failures.append((col, check_name, error_message))
                if violation is not None:
                    try:
                        rows = (
                            lf.with_row_index("row")
                            .filter(violation(col))
                            .select("row")
                            .head(VIOLATION_SAMPLE)
                            .collect()["row"]
                            .to_list()
                        )
                    # The check already failed; a broken violation expression is reported, not raised
                    except Exception as e:  # noqa: BLE001
                        error_message += f" (could not list the violating rows: {e})"

            results.append(
                {
                    "id": check_id,
                    "check": check_name,
                    "severity": severity,
                    "passed": ok,
                    "error": error_message,
                    # Positions of the first violating rows in the written dataset file
                    "rows": rows,
                }
            )
            check_id += 1

            if ok:
                passed += 1

        pass_pct = round(100 * passed / len(checks), 2)
        report_by_col[col] = {
            "pass_percentage": pass_pct,
            "checks": results,
        }

    total_checks = sum(len(v["checks"]) for v in report_by_col.values())
    total_passed = sum(
        sum(1 for c in v["checks"] if c["passed"]) for v in report_by_col.values()
    )

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "total_columns": len(report_by_col),
            "total_checks": total_checks,
            "passed": total_passed,
            "failed": total_checks - total_passed,
            "overall_pass_percentage": round(100 * total_passed / total_checks, 2)
            if total_checks > 0
            else 100.0,
        },
        "columns": report_by_col,
    }

    output_dir = os.environ["META_FOLDER"]
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "health_report.json")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # After all checks, raise if any FAIL-severity checks failed
    if overall_failures:
        summary = "\n".join(
            f"[{col}] {name}: {msg}" for col, name, msg in overall_failures
        )
        raise HealthCheckFailure(
            f"One or more 'fail'-severity checks failed:\n{summary}"
        )

    return report


class Check:
    """
    Fluent builder for column health checks.

    All checks work lazily - they only collect aggregate statistics needed
    for the specific check, not the entire dataframe.
    """

    DEFAULT_SEVERITY = "warn"

    def __init__(self, column_name: str):
        self.column = column_name
        self.checks: list[CheckDefinition] = []

    def _add(
        self,
        name: str,
        severity: str | None,
        func: CheckCondition,
        violation: ViolationExpr | None = None,
    ) -> Check:
        self.checks.append((name, severity or self.DEFAULT_SEVERITY, func, violation))
        return self

    # ---------------- Basic Checks ----------------
    def no_nulls(self, severity=None):
        """Fail if there are any nulls"""

        def condition(lf, c):
            # Only collect null count, not the whole dataframe
            result = lf.select(pl.col(c).null_count().alias("nulls")).collect()
            nulls = result["nulls"][0]
            if nulls > 0:
                raise ValueError(f"{nulls} null values found in column '{c}'")

        return self._add("no_nulls", severity, condition, lambda c: pl.col(c).is_null())

    def non_empty_strings(self, severity=None):
        """Fail if any string is empty"""

        def condition(lf, c):
            # Count empty strings lazily
            result = lf.select(
                pl.col(c).str.strip_chars().eq("").sum().alias("empty_count")
            ).collect()
            empty_count = result["empty_count"][0]
            if empty_count > 0:
                raise ValueError(f"{empty_count} empty strings found in column '{c}'")

        return self._add(
            "non_empty_strings",
            severity,
            condition,
            lambda c: pl.col(c).str.strip_chars().eq(""),
        )

    def unique(self, severity=None):
        """Warn if there are duplicates"""

        def condition(lf, c):
            # Collect only total count and unique count
            result = lf.select(
                [pl.len().alias("total"), pl.col(c).n_unique().alias("unique_count")]
            ).collect()
            total = result["total"][0]
            unique_count = result["unique_count"][0]
            if unique_count < total:
                raise ValueError(
                    f"Duplicate values found in column '{c}' ({total - unique_count} duplicates)"
                )

        return self._add(
            "unique", severity, condition, lambda c: pl.col(c).is_duplicated()
        )

    # ---------------- Range / Value Checks ----------------
    def valid_range(self, min_val=None, max_val=None, severity=None):
        """Check all values are within the specified range"""

        def out_of_range_rows(c):
            checks = []
            if min_val is not None:
                checks.append(pl.col(c) < min_val)
            if max_val is not None:
                checks.append(pl.col(c) > max_val)

            if not checks:
                return None  # No constraints

            out_of_range_expr = checks[0]
            for check in checks[1:]:
                out_of_range_expr = out_of_range_expr | check
            return out_of_range_expr

        def condition(lf, c):
            out_of_range_expr = out_of_range_rows(c)
            if out_of_range_expr is None:
                return

            result = lf.select(out_of_range_expr.sum().alias("out_of_range")).collect()
            out_of_range = result["out_of_range"][0]
            if out_of_range > 0:
                raise ValueError(f"{out_of_range} values out of range in '{c}'")

        return self._add(
            "valid_range",
            severity,
            condition,
            out_of_range_rows if min_val is not None or max_val is not None else None,
        )

    def in_values(self, allowed, severity=None, ignore_case=False):
        """Check all values are in the allowed set"""
        allowed_set = set(allowed)

        def invalid_rows(c):
            if ignore_case:
                allowed_lower = [a.lower() for a in allowed_set]
                return ~pl.col(c).str.to_lowercase().is_in(allowed_lower)
            return ~pl.col(c).is_in(list(allowed_set))

        def condition(lf, c):
            result = lf.select(invalid_rows(c).sum().alias("invalid")).collect()
            invalid = result["invalid"][0]
            if invalid > 0:
                raise ValueError(
                    f"{invalid} invalid values in '{c}' (allowed={list(allowed_set)})"
                )

        return self._add("in_values", severity, condition, invalid_rows)

    def regex_match(self, pattern, severity=None):
        """All string values must match the regex"""

        def condition(lf, c):
            # Count non-matching strings
            result = lf.select(
                (~pl.col(c).str.contains(pattern)).sum().alias("invalid_count")
            ).collect()
            invalid_count = result["invalid_count"][0]
            if invalid_count > 0:
                raise ValueError(
                    f"{invalid_count} values in '{c}' do not match regex '{pattern}'"
                )

        return self._add(
            "regex_match",
            severity,
            condition,
            lambda c: ~pl.col(c).str.contains(pattern),
        )

    def null_percentage(self, max_pct=0.0, severity=None):
        """Allow up to max_pct (0-100) nulls"""

        def condition(lf, c):
            result = lf.select(
                [pl.col(c).null_count().alias("nulls"), pl.len().alias("total")]
            ).collect()
            nulls = result["nulls"][0]
            total = result["total"][0]
            pct = (nulls / total) * 100 if total > 0 else 0
            if pct > max_pct:
                raise ValueError(f"{pct:.2f}% nulls in '{c}', exceeds {max_pct}%")

        return self._add(
            "null_percentage", severity, condition, lambda c: pl.col(c).is_null()
        )

    # ---------------- Numeric Checks ----------------
    def numeric_check(
        self,
        eq=None,
        gt=None,
        gte=None,
        lt=None,
        lte=None,
        not_eq=None,
        sum_eq=None,
        severity=None,
    ):
        """Checks for numeric columns"""

        def failing(c):
            failing_conditions = []
            if eq is not None:
                failing_conditions.append(pl.col(c) != eq)
            if gt is not None:
                failing_conditions.append(pl.col(c) <= gt)
            if gte is not None:
                failing_conditions.append(pl.col(c) < gte)
            if lt is not None:
                failing_conditions.append(pl.col(c) >= lt)
            if lte is not None:
                failing_conditions.append(pl.col(c) > lte)
            if not_eq is not None:
                failing_conditions.append(pl.col(c) == not_eq)
            if not failing_conditions:
                return None

            combined = failing_conditions[0]
            for cond in failing_conditions[1:]:
                combined = combined | cond
            return combined

        has_row_conditions = any(
            v is not None for v in (eq, gt, gte, lt, lte, not_eq)
        )

        def condition(lf, c):
            # Collect only what we need
            exprs = []
            failing_conditions = failing(c)
            if failing_conditions is not None:
                exprs.append(failing_conditions.sum().alias("failing_count"))

            if sum_eq is not None:
                exprs.append(pl.col(c).sum().alias("total_sum"))

            if not exprs:
                return  # No checks to perform

            result = lf.select(exprs).collect()

            if failing_conditions is not None:
                failing_count = result["failing_count"][0]
                if failing_count > 0:
                    raise ValueError(
                        f"{failing_count} values failed numeric check on '{c}'"
                    )

            if sum_eq is not None:
                total_sum = result["total_sum"][0]
                if total_sum != sum_eq:
                    raise ValueError(f"Sum of '{c}' is {total_sum}, expected {sum_eq}")

        return self._add(
            "numeric_check", severity, condition, failing if has_row_conditions else None
        )

    # ---------------- Aggregate / Count ----------------
    def distinct_count(self, expected_count, severity=None):
        """Check column has expected number of distinct values"""

        def condition(lf, c):
            result = lf.select(pl.col(c).n_unique().alias("unique_count")).collect()
            unique_count = result["unique_count"][0]
            if unique_count != expected_count:
                raise ValueError(
                    f"Column '{c}' has {unique_count} distinct values, expected {expected_count}"
                )

        return self._add("distinct_count", severity, condition)

    # ---------------- Custom ----------------
    def custom_check(
        self,
        name: str,
        severity: str | None,
        func: CheckCondition,
        violations: ViolationExpr | None = None,
    ) -> Check:
        """
        func(lf, col) should raise ValueError(message) if it fails.

        violations(col) optionally returns an expression that is true for the rows
        that break the check, so the platform can show them.
        """
        return self._add(name, severity, func, violations)

    def build(self) -> tuple[str, list[CheckDefinition]]:
        return self.column, self.checks
