---
name: dataspace-pipelines
description: Build, review, debug, or modify DataSpace pipeline projects that use pyrunner-lib, Python transform functions, `_connections.json`, transform RIDs, Polars LazyFrames, DataSpace artifacts, React or other compiled dashboards, DataSpace API-backed parquet queries, health checks, `ds_meta`, or incremental builds. Use whenever an AI coding agent works in a DataSpace transformation repository or needs to create, move, run, or validate a DataSpace transform or hosted data application.
---

# DataSpace Pipelines

Develop DataSpace transforms while preserving pipeline wiring, lineage, persisted outputs, and runner conventions.

## Compatibility

Use this skill as the compatibility baseline for `pyrunner-lib==0.5.0`. Before changing a project, compare its installed or declared `pyrunner-lib` version with this baseline. When it differs, inspect that version's public API and current DataSpace documentation before relying on signatures or runtime behavior described here. Update this baseline in the same change as every library version bump.

## Workflow

1. Inspect the repository before editing.
   - Locate `_connections.json`, `__main__.py`, transform source files, tests, and project-specific instructions.
   - Resolve the target transform RID and its upstream inputs from `_connections.json`.
   - Preserve unrelated changes in the working tree.
2. Read [references/pipeline-runtime.md](references/pipeline-runtime.md) before changing transform code, paths, artifacts, health checks, incremental behavior, or local execution.
   - Also read [references/interactive-dashboards.md](references/interactive-dashboards.md) before building a React/Vite/other compiled frontend, changing an API-backed dashboard, or querying parquet data from browser code.
   - Also read [references/health-checks.md](references/health-checks.md) before changing or interpreting health checks.
3. Implement the smallest repository-native change.
   - Prefer returning `polars.LazyFrame` to retain query optimization and column lineage.
   - Treat transform parameter names as pipeline inputs populated by the runner.
   - Use injected DataSpace paths for persisted files.
   - Return the output dataframe; do not manually write the automatic parquet dataset.
4. Preserve pipeline wiring.
   - Update the matching `path` in `_connections.json` whenever a mapped Python file moves or is renamed.
   - If a rename changes the filename stem, inspect downstream transforms that resolve unannotated parameters by that legacy stem; update the parameter or bind it explicitly to the RID.
   - Keep transform RIDs and declared upstream inputs stable unless the task explicitly changes topology.
   - Verify every mapped source path exists after editing.
5. Validate proportionally.
   - Run the target transform locally using its RID when fixtures and input data are available.
   - Run focused tests, then the relevant repository suite.
   - Confirm expected datasets, artifacts, dashboards, metadata, and health reports appear under the injected build directories.
   - For a compiled dashboard, verify the production bundle, relative asset URLs, hosted entry point, authenticated API calls, and loading/empty/error states.
   - Report any validation that cannot run because platform data or environment variables are unavailable.

## Task Guidance

- For upstream artifacts, read `input_df.ds_meta.artifact_dir`; do not reconstruct another transform's artifact path.
- Never collect an unbounded `LazyFrame` in the transform process. Project required columns and use bounded samples, scalar aggregates, or server-side queries.
- For dashboards, write `index.html` inside `ARTIFACT_FOLDER`. A transform may use its build step to compile a React, Vite, or other browser application into that directory; DataSpace then hosts the emitted application as transform artifacts.
- For API-backed dashboards, discover the exact query contract from the current DataSpace deployment or an existing project client. Query parquet-backed data server-side with projection, filtering, and row limits; never invent an endpoint or embed credentials in the bundle.
- For health checks, return `(dataframe, checks)`, register checks with `Check` builders, and choose `severity="fail"` only when a failed check should fail the build.
- For incremental transforms, treat `PREVIOUS_DATASET_PATH` as the previous generated dataset file path and guard for a missing previous build.
- DataSpace supports bare transform parameters, explicit `Transform("rid.transform...")` parameter annotations, and explicit `@transform(input=Transform("rid.transform..."))` decorator bindings for both local and linked foreign-workspace inputs. Read the connection-mapping section of [references/pipeline-runtime.md](references/pipeline-runtime.md) before declaring inputs from another workspace.
- A bare name is suitable when name-based resolution is unambiguous. Use an explicit RID annotation or decorator binding when a foreign workspace can contain a same-named transform. Workspace-qualified `Transform("WORKSPACE.transform")` references are planned but are not yet a supported replacement for the RID.
- Consult the current [DataSpace documentation index](https://docs.dataspace.ch/llms.txt) when repository behavior and bundled guidance disagree; prefer current official documentation, then verify against the installed `pyrunner-lib` version.
