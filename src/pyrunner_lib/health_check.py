"""
Health checks for Polars LazyFrame/DataFrame.

All checks work lazily - they only collect aggregate statistics,
not the entire dataframe.
"""

import json
import os
from datetime import datetime, timezone
import polars as pl


class HealthCheckFailure(Exception):
    """Raised when one or more fail-severity checks fail."""
    pass


def run_health_checks(lf, column_builders):
    """
    Run health checks on a LazyFrame without collecting the entire dataframe.
    
    Each check only collects the specific aggregate statistics it needs.
    """
    column_checks = dict(cb.build() for cb in column_builders)

    report_by_col = {}
    overall_failures = []

    for col, checks in column_checks.items():
        results = []
        passed = 0

        for check_name, severity, condition in checks:
            error_message = None
            try:
                condition(lf, col)
                ok = True
            except Exception as e:
                ok = False
                error_message = str(e)
                if severity.lower() == "fail":
                    overall_failures.append((col, check_name, error_message))

            results.append({
                "check": check_name,
                "severity": severity,
                "passed": ok,
                "error": error_message,
            })

            if ok:
                passed += 1

        pass_pct = round(100 * passed / len(checks), 2)
        report_by_col[col] = {
            "pass_percentage": pass_pct,
            "checks": results,
        }

    total_checks = sum(len(v["checks"]) for v in report_by_col.values())
    total_passed = sum(sum(1 for c in v["checks"] if c["passed"]) for v in report_by_col.values())

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "total_columns": len(report_by_col),
            "total_checks": total_checks,
            "passed": total_passed,
            "failed": total_checks - total_passed,
            "overall_pass_percentage": round(100 * total_passed / total_checks, 2) if total_checks > 0 else 100.0,
        },
        "columns": report_by_col,
    }

    output_dir = os.environ["META_FOLDER"]
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "health_report.json")

    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    # After all checks, raise if any FAIL-severity checks failed
    if overall_failures:
        summary = "\n".join(f"[{col}] {name}: {msg}" for col, name, msg in overall_failures)
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
        self.checks = []

    def _add(self, name, severity, func):
        self.checks.append((name, severity or self.DEFAULT_SEVERITY, func))
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
        return self._add("no_nulls", severity, condition)

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
        return self._add("non_empty_strings", severity, condition)

    def unique(self, severity=None):
        """Warn if there are duplicates"""
        def condition(lf, c):
            # Collect only total count and unique count
            result = lf.select([
                pl.len().alias("total"),
                pl.col(c).n_unique().alias("unique_count")
            ]).collect()
            total = result["total"][0]
            unique_count = result["unique_count"][0]
            if unique_count < total:
                raise ValueError(f"Duplicate values found in column '{c}' ({total - unique_count} duplicates)")
        return self._add("unique", severity, condition)

    # ---------------- Range / Value Checks ----------------
    def valid_range(self, min_val=None, max_val=None, severity=None):
        """Check all values are within the specified range"""
        def condition(lf, c):
            # Build condition for out of range
            checks = []
            if min_val is not None:
                checks.append(pl.col(c) < min_val)
            if max_val is not None:
                checks.append(pl.col(c) > max_val)
            
            if not checks:
                return  # No constraints
            
            out_of_range_expr = checks[0]
            for check in checks[1:]:
                out_of_range_expr = out_of_range_expr | check
            
            result = lf.select(out_of_range_expr.sum().alias("out_of_range")).collect()
            out_of_range = result["out_of_range"][0]
            if out_of_range > 0:
                raise ValueError(f"{out_of_range} values out of range in '{c}'")
        return self._add("valid_range", severity, condition)

    def in_values(self, allowed, severity=None, ignore_case=False):
        """Check all values are in the allowed set"""
        allowed_set = set(allowed)
        def condition(lf, c):
            if ignore_case:
                allowed_lower = list(a.lower() for a in allowed_set)
                invalid_expr = ~pl.col(c).str.to_lowercase().is_in(allowed_lower)
            else:
                invalid_expr = ~pl.col(c).is_in(list(allowed_set))
            
            result = lf.select(invalid_expr.sum().alias("invalid")).collect()
            invalid = result["invalid"][0]
            if invalid > 0:
                raise ValueError(f"{invalid} invalid values in '{c}' (allowed={list(allowed_set)})")
        return self._add("in_values", severity, condition)

    def regex_match(self, pattern, severity=None):
        """All string values must match the regex"""
        def condition(lf, c):
            # Count non-matching strings
            result = lf.select(
                (~pl.col(c).str.contains(pattern)).sum().alias("invalid_count")
            ).collect()
            invalid_count = result["invalid_count"][0]
            if invalid_count > 0:
                raise ValueError(f"{invalid_count} values in '{c}' do not match regex '{pattern}'")
        return self._add("regex_match", severity, condition)

    def null_percentage(self, max_pct=0.0, severity=None):
        """Allow up to max_pct (0-100) nulls"""
        def condition(lf, c):
            result = lf.select([
                pl.col(c).null_count().alias("nulls"),
                pl.len().alias("total")
            ]).collect()
            nulls = result["nulls"][0]
            total = result["total"][0]
            pct = (nulls / total) * 100 if total > 0 else 0
            if pct > max_pct:
                raise ValueError(f"{pct:.2f}% nulls in '{c}', exceeds {max_pct}%")
        return self._add("null_percentage", severity, condition)

    # ---------------- Numeric Checks ----------------
    def numeric_check(self, eq=None, gt=None, gte=None, lt=None, lte=None, not_eq=None, sum_eq=None, severity=None):
        """Checks for numeric columns"""
        def condition(lf, c):
            # Build failing condition
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
            
            # Collect only what we need
            exprs = []
            if failing_conditions:
                combined = failing_conditions[0]
                for cond in failing_conditions[1:]:
                    combined = combined | cond
                exprs.append(combined.sum().alias("failing_count"))
            
            if sum_eq is not None:
                exprs.append(pl.col(c).sum().alias("total_sum"))
            
            if not exprs:
                return  # No checks to perform
            
            result = lf.select(exprs).collect()
            
            if failing_conditions:
                failing_count = result["failing_count"][0]
                if failing_count > 0:
                    raise ValueError(f"{failing_count} values failed numeric check on '{c}'")
            
            if sum_eq is not None:
                total_sum = result["total_sum"][0]
                if total_sum != sum_eq:
                    raise ValueError(f"Sum of '{c}' is {total_sum}, expected {sum_eq}")
        return self._add("numeric_check", severity, condition)

    # ---------------- Aggregate / Count ----------------
    def distinct_count(self, expected_count, severity=None):
        """Check column has expected number of distinct values"""
        def condition(lf, c):
            result = lf.select(pl.col(c).n_unique().alias("unique_count")).collect()
            unique_count = result["unique_count"][0]
            if unique_count != expected_count:
                raise ValueError(f"Column '{c}' has {unique_count} distinct values, expected {expected_count}")
        return self._add("distinct_count", severity, condition)

    # ---------------- Custom ----------------
    def custom_check(self, name, severity, func):
        """func(lf, col) should raise ValueError(message) if it fails"""
        return self._add(name, severity, func)

    def build(self):
        return self.column, self.checks
