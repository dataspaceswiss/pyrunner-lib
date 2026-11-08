---
name: dataspace-pipelines
description: Build, review, debug, or modify DataSpace pipeline projects that use pyrunner-lib, Python transform functions, `_connections.json`, transform RIDs, Polars LazyFrames, DataSpace artifacts, React or other compiled dashboards, DataSpace API-backed parquet queries, health checks, `ds_meta`, or incremental builds. Use whenever an AI coding agent works in a DataSpace transformation repository or needs to create, move, run, or validate a DataSpace transform or hosted data application.
---

# DataSpace Pipelines

Develop DataSpace transforms while preserving pipeline wiring, lineage, persisted outputs, and runner conventions.

## Compatibility

This skill describes `pyrunner-lib==0.6.1` running on DataSpace 1.13.1. A project declares its version as the `pyrunner-lib` entry in the `packages` of `_config.json`. When that version differs, inspect its public API and the current DataSpace documentation before relying on signatures or runtime behavior described here. Update this baseline in the same change as every library version bump.

## Workflow

1. Inspect the repository before editing.
   - Locate `_config.json`, `_connections.json`, `__main__.py`, transform source files, tests, and project-specific instructions.
   - Resolve the target transform RID and its upstream inputs from `_connections.json`.
   - Preserve unrelated changes in the working tree.
2. Read [references/pipeline-runtime.md](references/pipeline-runtime.md) before changing transform code, paths, artifacts, health checks, incremental behavior, or local execution.
   - Also read [references/interactive-dashboards.md](references/interactive-dashboards.md) before building a React/Vite/other compiled frontend, changing an API-backed dashboard, or querying parquet data from browser code.
   - Also read [references/health-checks.md](references/health-checks.md) before changing or interpreting health checks.
3. Implement the smallest repository-native change.
   - Write the transform as Polars lazy expressions from its inputs to the returned `LazyFrame`. DataSpace optimizes and streams a lazy query and derives column lineage from its plan, so eager, pandas, or Python-function code costs both speed and lineage.
   - Treat transform parameter names as pipeline inputs populated by the runner.
   - Use injected DataSpace paths for persisted files.
   - Return the output dataframe; do not manually write the automatic parquet dataset.
   - Do not build your own bookkeeping. DataSpace already records each build's status, timing, inputs, logs, row and column counts, schema, and column lineage, so run logs, manifests, or statistics written as JSON files or artifacts only duplicate it.
   - Lay the code out for a human reader. Transforms are mostly long expression chains, so give each operation and each expression its own line. The repository's formatter and conventions, or the user's stated style, take precedence.
   - Start each transform file with a docstring that says what the transform produces and from what. Keep it current when the logic changes.
4. Preserve pipeline wiring.
   - Update the matching `path` in `_connections.json` whenever a mapped Python file moves or is renamed.
   - If a rename changes the filename stem, inspect downstream transforms that resolve unannotated parameters by that legacy stem; update the parameter or bind it explicitly to the RID.
   - Keep transform RIDs and declared upstream inputs stable unless the task explicitly changes topology.
   - Write every input RID as a plain string literal directly inside `Transform("...")`. DataSpace reads input declarations from the source without running it and rejects a computed RID; a module-level constant assigned once to a plain string also works.
   - Verify every mapped source path exists after editing.
5. Validate proportionally.
   - Run the target transform locally using its RID when fixtures and input data are available.
   - Run focused tests, then the relevant repository suite.
   - Confirm expected datasets, artifacts, dashboards, metadata, and health reports appear under the injected build directories.
   - For a compiled dashboard, verify the production bundle, relative asset URLs, hosted entry point, authenticated API calls, and loading/empty/error states.
   - Report any validation that cannot run because platform data or environment variables are unavailable.

## Platform Operations

A transform or schedule exists only once DataSpace has created it: the platform issues the RID and registers it. Working on the repository over git can change transform code, move files, and adjust the inputs of existing transforms. It cannot create or delete a transform, create, change, or delete a schedule, start a build, or read build logs and datasets. Do not invent a RID or add a new entry to `_connections.json` by hand; DataSpace does not know that transform and cannot build it.

When a task needs one of those operations, say so and choose one of these instead of working around it:

- Ask the user to do it in DataSpace and give you the resulting RIDs.
- If you run in an app that can act in the user's browser session, such as the Claude or ChatGPT app, ask the user to sign in to their DataSpace instance there. With that session you can call the DataSpace API to create, modify, and delete transforms and schedules, start builds, and inspect logs and datasets. Take the calls from the official documentation or from the requests the DataSpace web app makes; do not guess an endpoint, and do not ask for the user's password.

Git access goes over HTTPS with the user's DataSpace email and password and needs the editor or owner role to push. Accounts with two-factor authentication are refused over git for now; such a user has to work in the DataSpace editor. Set the repository's `git config user.email` to the DataSpace email, so the workspace history shows the commit under that user.

## Task Guidance

- For upstream artifacts, read `input_df.ds_meta.artifact_dir`; do not reconstruct another transform's artifact path.
- Never collect an unbounded `LazyFrame` in the transform process. Project required columns and use bounded samples, scalar aggregates, or server-side queries.
- For dashboards, write `index.html` inside `ARTIFACT_FOLDER`. A transform may use its build step to compile a React, Vite, or other browser application into that directory; DataSpace then hosts the emitted application as transform artifacts.
- For API-backed dashboards, discover the exact query contract from the current DataSpace deployment or an existing project client. Query parquet-backed data server-side with projection, filtering, and row limits; never invent an endpoint or embed credentials in the bundle.
- For health checks, return `(dataframe, checks)`, register checks with `Check` builders, and choose `severity="fail"` only when a failed check should fail the build.
- For incremental transforms, treat `PREVIOUS_DATASET_PATH` as the previous generated dataset file path and guard for a missing previous build.
- Read workspace secrets from environment variables named by their key in the workspace settings. Never hardcode a secret, print it, or write it to a dataset, artifact, or log.
- A draft build runs uncommitted code on the first `draftReadLimit` rows of each input (`_config.json`, 1,000,000 by default, `-1` for all rows), and committed builds never read its output. Do not treat its row counts as final.
- DataSpace supports bare transform parameters, explicit `Transform("rid.transform...")` parameter annotations, and explicit `@transform(input=Transform("rid.transform..."))` decorator bindings for both local and linked foreign-workspace inputs. Read the connection-mapping section of [references/pipeline-runtime.md](references/pipeline-runtime.md) before declaring inputs from another workspace.
- A bare name is suitable when name-based resolution is unambiguous. Use an explicit RID annotation or decorator binding when a foreign workspace can contain a same-named transform. Workspace-qualified `Transform("WORKSPACE.transform")` references are planned but are not yet a supported replacement for the RID.
- Consult the current [DataSpace documentation index](https://docs.dataspace.ch/llms.txt) when repository behavior and bundled guidance disagree; prefer current official documentation, then verify against the installed `pyrunner-lib` version.
