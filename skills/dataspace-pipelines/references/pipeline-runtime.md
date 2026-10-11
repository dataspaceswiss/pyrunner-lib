# DataSpace Pipeline Runtime Reference

## Transform structure

Place business logic in Python transform files, conventionally below `src/`. Define a callable named `transform` unless the repository uses the `pyrunner_lib.transform` decorator pattern.

The runner populates transform parameters with upstream dataframes. Return one of:

- `polars.LazyFrame` (preferred for optimization, streaming, and lineage)
- `polars.DataFrame`
- `pandas.DataFrame`

```python
import polars as pl


def transform(input_df: pl.LazyFrame) -> pl.LazyFrame:
    return input_df.filter(pl.col("active"))
```

The runner automatically persists the returned dataframe as the transform's parquet dataset. Do not write that dataset manually.

### Lazy Polars first

Write transform logic as Polars lazy expressions and return the `LazyFrame`. Two things depend on it:

- **Performance.** Polars optimizes the whole lazy query, and the runner writes a returned `LazyFrame` with a streaming sink, so the dataset does not have to fit in memory.
- **Column lineage.** The runner derives which upstream columns feed each output column from the lazy query plan. A returned `polars.DataFrame` or `pandas.DataFrame` has no plan, so DataSpace records no column lineage for that transform.

Lineage is also lost for any column that passes through Python code inside the query, such as `map_elements` or `map_batches`, because Polars cannot see into the function. Prefer native expressions, and when no expression exists, apply the function only to the columns that need it.

Avoid collecting in the middle of a transform and continuing eagerly, converting to pandas, and looping over rows in Python. When a side task needs materialized data, such as a chart, collect a bounded projection for that task and still return the lazy frame.

### Let DataSpace keep the records

Do not add a bookkeeping layer to a pipeline. DataSpace records the following for every build and shows its history, without any code in the transform:

- status, start and end time, duration, who or which schedule started it, and the commit it ran;
- the exact input snapshots it read;
- logs and resource metrics;
- the resulting dataset's row count, column count, file size, and schema;
- column lineage and health check results.

Writing run logs, manifests, row-count or schema summaries, "last run" markers, or audit trails as JSON files, artifacts, or extra datasets duplicates this, can disagree with it, and leaves more code to maintain. Use what the platform provides instead:

| Need | Use |
| --- | --- |
| Facts about an input, such as its snapshot, row count, or columns | `input_df.ds_meta` |
| What the previous run produced | `PREVIOUS_DATASET_PATH` |
| A rule the data must satisfy | Health checks |
| A file a person or a downstream transform consumes | An artifact |

Write an artifact when it is a deliverable. Do not write one to remember what the pipeline did.

### Readable formatting

Transform code is mostly long chains of `filter`, `select`, `with_columns`, joins, and aggregations, and it is read far more often than it is written. Lay it out so the pipeline reads from top to bottom:

- Wrap a chain in parentheses and give each method call its own line.
- Give each expression its own line inside `select`, `with_columns`, and `agg`.
- Name an intermediate frame when a chain has distinct stages.

```python
import polars as pl


def transform(orders: pl.LazyFrame, customers: pl.LazyFrame) -> pl.LazyFrame:
    completed_orders = (
        orders.filter(pl.col("status") == "completed")
        .filter(pl.col("ordered_at") >= pl.date(2025, 1, 1))
        .select(
            "order_id",
            "customer_id",
            (pl.col("quantity") * pl.col("unit_price")).alias("revenue"),
        )
    )

    return (
        completed_orders.join(customers, on="customer_id", how="left")
        .group_by("customer_id", "segment")
        .agg(
            pl.col("revenue").sum().alias("total_revenue"),
            pl.col("order_id").n_unique().alias("order_count"),
        )
        .sort("total_revenue", descending=True)
    )
```

A formatter or lint configuration in the repository, the project's own conventions, or an explicit request from the user takes precedence over this layout.

## Connection mapping

Use `_connections.json` to resolve transform RIDs to source files and upstream transforms.

```json
{
  "id": "rid.transform.uuid",
  "path": "src/path/to/transform.py",
  "inputs": ["rid.transform.upstream"]
}
```

Whenever a mapped Python file moves or is renamed, update its `path` in `_connections.json` in the same change. A stale path breaks execution.

DataSpace recognizes three input declaration forms.

Use a bare parameter when name-based resolution is unambiguous. This form can resolve local transforms and published transforms from a linked foreign workspace:

```python
def transform(input):
    return input
```

The runner resolves an unannotated parameter by matching its name to an upstream connection's filename stem. Renaming `src/source_clean.py` to `src/source/clean.py` changes that legacy name from `source_clean` to `clean` and can break downstream parameters even when the RID is unchanged. Search downstream transform signatures whenever a rename changes a stem.

Bind a parameter explicitly by RID with a parameter annotation:

```python
from pyrunner_lib import Transform


def transform(source_data: Transform("rid.transform.source")):
    return source_data
```

The equivalent decorator form is:

```python
from pyrunner_lib import Transform, transform


@transform(source_data=Transform("rid.transform.source"))
def build_output(source_data):
    return source_data
```

All three forms can declare an input from a linked foreign workspace. Prefer either explicit RID form when different workspaces contain transforms with the same name; the resolved connection preserves cross-workspace lineage.

`Transform("WORKSPACE.transform")`, for example `orders: Transform("INSTACART.orders")`, is a planned DataSpace syntax and is not yet supported. Until it is implemented, use the foreign transform's RID in the annotation or decorator binding.

### How DataSpace reads input declarations

DataSpace works out a transform's inputs by parsing the syntax of the source file, without executing it. This happens when the file is saved in the DataSpace editor, and the result is stored as that transform's `inputs` in `_connections.json`. A build mounts exactly those inputs. The runner, in contrast, executes the module, so a declaration that computes its RID can run locally and still be wired to the wrong input, or to none, on the platform.

| Declaration | What DataSpace reads |
| --- | --- |
| `name: Transform("rid.transform.x")`, also inside `Annotated[...]` | Input bound to that RID. |
| `@transform(name=Transform("rid.transform.x"))` | Input bound to that RID. |
| `name` | Input bound by name. |
| `name: Transform(SOME_RID)` | Depends on the DataSpace version. Newer ones use the value when `SOME_RID` is assigned exactly once in the same file, at module level, to a plain string, and reject the save otherwise. Older ones silently bind by name. |
| `name: Transform("rid." + suffix)`, an f-string, or any other computed value | Never that RID. Depending on the DataSpace version the save is rejected, or the input is silently bound by name. |
| `name: pyrunner_lib.Transform("rid.transform.x")` | Depends on the DataSpace version; older ones fall back to binding by name. |
| `name: Transform("rid.transform.x") = None` | Depends on the DataSpace version; older ones do not see the input at all. |
| `name=Transform("rid.transform.x")` as a default value | Bound by name, not by RID. |

Therefore write the RID as a plain string literal directly inside the call, import `Transform` by name with `from pyrunner_lib import Transform`, and give input parameters no default values. That form is read correctly by every DataSpace version. Use a named constant only when the repository already does, since that shows its deployment supports it.

Nothing re-derives the inputs when changes arrive over git. Whenever you add, remove, or rebind an input in code, make the same change to that transform's `inputs` in `_connections.json`, using the same RIDs.

The conventional entry point is:

```python
import pyrunner_lib

pyrunner_lib.main()
```

## Injected environment

Use the exact paths injected by DataSpace.

| Variable | Meaning |
| --- | --- |
| `TRANSFORM_ID` | Current transform RID. |
| `ARTIFACT_FOLDER` | Directory for persisted artifacts from the current build. |
| `META_FOLDER` | Directory for generated metadata such as health reports. |
| `DATASET_FOLDER` | Directory populated with the returned dataframe's parquet dataset. |
| `PREVIOUS_BUILD_FOLDER` | Previous build directory for the current transform. |
| `PREVIOUS_DATASET_PATH` | Previous generated dataset file path. |
| `PREVIOUS_ARTIFACT_FOLDER` | Previous build's artifact directory. |

Files outside the transform build directory are ephemeral and are not persisted.

## Input metadata and upstream artifacts

Input dataframes expose `ds_meta`. Available metadata includes the upstream transform ID, data snapshot ID, build ID, row and column counts, file size, creation date, columns, and artifact directory.

```python
def transform(upstream_data):
    upstream_artifacts = upstream_data.ds_meta.artifact_dir
    # Read the required upstream artifact from upstream_artifacts.
    return upstream_data
```

## Artifacts and dashboards

Write general artifacts such as JSON, CSV, or images under `ARTIFACT_FOLDER`.

Create an interactive dashboard by writing `{ARTIFACT_FOLDER}/index.html`:

```python
import os

import plotly.express as px


def transform(input_df):
    frame = (
        input_df.select("category", "value")
        .limit(10_000)
        .collect()
        .to_pandas()
    )
    figure = px.bar(frame, x="category", y="value")
    figure.write_html(os.path.join(os.environ["ARTIFACT_FOLDER"], "index.html"))
    return input_df
```

Keep any local dashboard materialization explicitly bounded and project only the required columns. For high-volume datasets, prefer server-side aggregation or an API-backed dashboard instead of collecting raw rows in the transform process.

If no `index.html` exists, DataSpace serves the artifact folder as a file listing.

For a full browser application, the transform build can compile React, Vite, or another frontend into `ARTIFACT_FOLDER`. Keep `index.html` at the artifact root and make bundled asset URLs work from the nested hosted artifact path. Read [interactive-dashboards.md](interactive-dashboards.md) for the build, hosting, and DataSpace API query pattern.

## Health checks

Return health checks alongside the dataframe as `(dataframe, checks)`. Read [health-checks.md](health-checks.md) for supported checks, severity behavior, custom checks, report interpretation, and validation.

## Incremental builds

Read `PREVIOUS_DATASET_PATH` only when it is present and points to an existing file.

```python
import os

import polars as pl


def transform(new_data: pl.LazyFrame) -> pl.LazyFrame:
    previous_path = os.environ.get("PREVIOUS_DATASET_PATH")
    if not previous_path or not os.path.isfile(previous_path):
        return new_data

    previous_data = pl.scan_parquet(previous_path)
    return pl.concat([previous_data, new_data]).unique()
```

Choose merge and deduplication keys from the domain model; do not assume whole-row uniqueness is correct for every pipeline.

## Local execution

Resolve the RID from `_connections.json`, then run:

```bash
uv run __main__.py <RID> <PROJECT_ROOT>
```

Example:

```bash
uv run __main__.py rid.transform.12345-example-uuid .
```

Local execution still requires the repository's expected configuration, upstream datasets, and relevant environment variables.

## Official documentation

- [Documentation index](https://docs.dataspace.ch/llms.txt)
- [Transform function](https://docs.dataspace.ch/api-reference/transform-function.md)
- [Plots and dashboards](https://docs.dataspace.ch/platform/transformation/plot.md)
- [Health checks](https://docs.dataspace.ch/api-reference/health-checks.md)
