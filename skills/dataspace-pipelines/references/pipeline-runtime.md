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
