# Health Checks Documentation

`pyrunner_lib` provides a declarative and fluent API for defining health checks on your data transformations. These checks are designed to be efficient, working lazily on Polars `LazyFrame` objects to collect only the necessary aggregate statistics without loading the entire dataset into memory.

## Overview

Health checks are defined using the `Check` class, which allows you to chain multiple validation methods for a specific column.

```python
import polars as pl
import pyrunner_lib.health_check as hc

def transform(data):
    lf = data

    # Define checks declaratively
    health_checks = [
        hc.Check("name")
            .no_nulls()
            .non_empty_strings()
            .unique(severity="warn"),

        hc.Check("age")
            .no_nulls()
            .valid_range(0, 120),

        hc.Check("city").no_nulls(),
        hc.Check("occupation").no_nulls(),
        hc.Check("country")
            .no_nulls()
            .non_empty_strings(),
    ]

    return lf, health_checks
```

## Severity Levels

Each check can have an optional `severity` level:

- **`warn`** (Default): If the check fails, it is recorded in the health report, but the transformation continues and no exception is raised.
- **`fail`**: If any check with `fail` severity fails, a `HealthCheckFailure` exception is raised after all checks have been executed, preventing the build from succeeding.

## API Reference

### `Check(column_name: str)`
Initializes a check builder for the specified column.

---

### Basic Checks

#### `.no_nulls(severity=None)`
Ensures that the column contains no null values.

#### `.non_empty_strings(severity=None)`
Ensures that all string values in the column are non-empty after stripping whitespace.

#### `.unique(severity=None)`
Ensures that all values in the column are unique. Note: This defaults to `warn` in many contexts as it can be a common occurrence.

---

### Range & Value Checks

#### `.valid_range(min_val=None, max_val=None, severity=None)`
Checks if all values are within the specified inclusive range.

#### `.in_values(allowed: list, severity=None, ignore_case=False)`
Checks if all values in the column are present in the `allowed` list.

#### `.regex_match(pattern: str, severity=None)`
Ensures that all string values match the provided regular expression pattern.

#### `.null_percentage(max_pct: float, severity=None)`
Allows up to `max_pct` (0-100) of the values in the column to be null.

---

### Numeric Checks

#### `.numeric_check(eq=None, gt=None, gte=None, lt=None, lte=None, not_eq=None, sum_eq=None, severity=None)`
Provides various numeric comparisons:
- `eq`: Equal to
- `gt`: Greater than
- `gte`: Greater than or equal to
- `lt`: Less than
- `lte`: Less than or equal to
- `not_eq`: Not equal to
- `sum_eq`: The sum of the column must equal this value.

---

### Aggregate Checks

#### `.distinct_count(expected_count: int, severity=None)`
Ensures the column has exactly the specified number of distinct values.

---

### Custom Checks

#### `.custom_check(name: str, severity: str, func: callable)`
Allows you to provide a custom validation function. The function should have the signature `func(lf: pl.LazyFrame, col: str)` and should raise a `ValueError` with a descriptive message if the check fails.

## Health Report

When health checks are run (usually handled by the runner), a `health_report.json` file is generated in the `META_FOLDER`. This report contains a summary of passed and failed checks, along with error messages for any failures.
