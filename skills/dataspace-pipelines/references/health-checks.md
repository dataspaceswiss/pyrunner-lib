# DataSpace Health Checks

## Runtime contract

Define declarative column checks with `pyrunner_lib.health_check.Check` and return them with the transform result:

```python
import pyrunner_lib.health_check as hc


def transform(input_df):
    checks = [
        hc.Check("id")
        .no_nulls(severity="fail")
        .unique(severity="fail"),
        hc.Check("value").valid_range(min_val=0),
        hc.Check("category").in_values(["a", "b"], ignore_case=True),
    ]
    return input_df, checks
```

The runner accepts checks alongside a Polars `LazyFrame`, Polars `DataFrame`, or pandas `DataFrame`. Checks operate through aggregate queries on a lazy frame rather than collecting the complete dataset. Preserve that behavior in custom checks.

Each check executes its aggregate against the input plan. On very large datasets, keep the check set intentional: exact uniqueness and distinct-count checks may require engine state proportional to column cardinality even though only scalar results return to Python. Never cache or collect the full frame merely to share work between checks.

Checks may be chained on one `Check` builder or split across multiple builders for the same column. The runner preserves their registration order and executes all of them.

## Severity

- `warn` is the default. Record the failed check in the report but allow the transform build to continue.
- `fail` records the result, then raises `HealthCheckFailure` after all checks execute and prevents a successful build.

Choose severity from the data contract. Use `fail` for invariants required by downstream consumers and `warn` for monitoring thresholds that should remain visible without stopping delivery.

The runner writes the report before raising for fail-severity results, so inspect `META_FOLDER/health_report.json` when diagnosing a failed build.

## Supported checks

| Check | Meaning |
| --- | --- |
| `.no_nulls(severity=None)` | Require zero null values. |
| `.non_empty_strings(severity=None)` | Require every string to remain non-empty after trimming whitespace. |
| `.unique(severity=None)` | Require the column's values to be unique. |
| `.valid_range(min_val=None, max_val=None, severity=None)` | Require values within inclusive bounds; either bound may be omitted. |
| `.in_values(allowed, severity=None, ignore_case=False)` | Require values from an allowlist; case folding is available for strings. |
| `.regex_match(pattern, severity=None)` | Require string values to satisfy the Polars regular-expression predicate. Anchor the pattern with `^` and `$` when the entire value must match. |
| `.null_percentage(max_pct=0.0, severity=None)` | Allow at most `max_pct` percent null values, using a value from 0 through 100. |
| `.numeric_check(eq=None, gt=None, gte=None, lt=None, lte=None, not_eq=None, sum_eq=None, severity=None)` | Apply per-value numeric comparisons and/or require the column sum to equal `sum_eq`. |
| `.distinct_count(expected_count, severity=None)` | Require exactly the expected number of distinct values. |
| `.custom_check(name, severity, func)` | Run a custom function receiving `(lazy_frame, column_name)`. |

Chain `.no_nulls()` or `.null_percentage()` explicitly when nulls are part of the contract; a range, regex, allowlist, or numeric predicate is not a substitute for a null policy.

Account for these current API semantics:

- `valid_range` bounds are inclusive.
- `in_values(ignore_case=True)` folds string case but does not trim whitespace.
- `regex_match` uses the supplied Polars regular expression; anchor it for a full-value match.
- `numeric_check(sum_eq=...)` uses exact equality and has no tolerance argument. For floating-point totals, prefer an integer minor unit, a suitable decimal representation, or a custom tolerance check defined by the data contract.
- A missing column, incompatible dtype, or exception inside a built-in/custom predicate is recorded as a failed check at that check's configured severity.

## Custom checks

Define a custom function as `func(lazy_frame, column_name)`. Collect only bounded aggregate results and raise `ValueError` with a useful, non-sensitive message when the condition fails. A custom check may inspect multiple columns, but its result is reported under the builder's column name.

## Health report

The runner writes `META_FOLDER/health_report.json` with:

- A UTC timestamp.
- Summary counts for columns, checks, passed checks, failed checks, and the overall pass percentage.
- Per-column pass percentages.
- One entry per check containing its name, severity, pass status, and error message.

Treat the JSON report as a build artifact for diagnostics and monitoring. Do not change its schema in a pipeline project; that schema belongs to `pyrunner-lib`.

## Validation

Test the transform result and the checks together:

1. Exercise a passing dataset and confirm the expected report summary.
2. Exercise every warning threshold and confirm the build continues.
3. Exercise every fail-severity invariant and confirm `HealthCheckFailure` occurs only after the report is written.
4. Cover empty input, all-null input, wrong data types, missing columns, boundary values, duplicates, and unexpected values where applicable.
5. Confirm custom checks collect only bounded aggregate results rather than the full dataset.
6. Run the target transform by RID and inspect `META_FOLDER/health_report.json` in the generated build tree.

## Official documentation

- [Health-check API](https://docs.dataspace.ch/api-reference/health-checks.md)
- [DataSpace documentation index](https://docs.dataspace.ch/llms.txt)
