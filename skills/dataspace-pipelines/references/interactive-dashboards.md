# Interactive DataSpace Dashboards

## Capability

A DataSpace transform does not have to use its build step only for dataframe transformation. It can also build or transpile a browser application and emit the production bundle into `ARTIFACT_FOLDER`. When the bundle contains `index.html` at the artifact root, DataSpace hosts it as the transform dashboard.

This supports purpose-built data applications—React, Vite, or another frontend stack—not only a single generated Plotly page. The hosted application can call the DataSpace API to query parquet-backed datasets without copying an entire dataset into the artifact bundle.

Use this pattern when the UI needs several views, rich interactions, user-driven filters, on-demand queries, or a domain-specific workflow. Prefer a generated static HTML dashboard when the data is small and the interface is simple.

## Inspect before implementing

Treat the deployed platform and an existing working client on the same deployment as the source of truth for the API contract.

Before editing:

1. Find the frontend directory, package manager lockfile, build script, bundler configuration, and transform wrapper.
2. Find the existing DataSpace API client and record its base-URL resolution, endpoint path, request payload, response schema, authentication behavior, and error handling.
3. Determine how dataset, snapshot, build, or transform identifiers reach the frontend.
4. Check whether DataSpace serves the dashboard below a nested artifact path and whether client-side route fallback is available.
5. Reuse the repository's framework, package manager, API client, and build command unless the task explicitly requests a migration.

Do not guess private endpoint paths from examples or copy tenant-specific hostnames, resource IDs, signed URLs, access tokens, cookies, or secrets into the skill or source code.

## Transform-driven frontend build

Keep the Python transform responsible for the DataSpace runtime contract: build artifacts, fail clearly when compilation fails, and still return the transform dataframe.

The exact command is repository-specific. A minimal pattern is:

```python
import os
from pathlib import Path
import subprocess


def transform(input_df):
    frontend_dir = Path(__file__).parent / "dashboard"
    artifact_dir = Path(os.environ["ARTIFACT_FOLDER"])
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # Do not expose the transform's complete secret-bearing environment to
    # frontend build plugins or package lifecycle scripts.
    build_env = {
        key: os.environ[key]
        for key in ("PATH",)
        if key in os.environ
    }
    build_env["NODE_ENV"] = "production"

    try:
        subprocess.run(
            ["npm", "run", "build", "--", "--outDir", str(artifact_dir)],
            cwd=frontend_dir,
            check=True,
            env=build_env,
            timeout=300,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("Dashboard build timed out after 300 seconds") from error

    if not (artifact_dir / "index.html").is_file():
        raise RuntimeError("Dashboard build did not emit ARTIFACT_FOLDER/index.html")

    return input_df
```

Adapt this example to the checked-in lockfile and package manager. Extend the child-environment allowlist only with values proven necessary and non-secret. Prefer a deterministic, lockfile-backed install performed by the project environment or image. Avoid downloading dependencies or running package lifecycle scripts on every production transform build, especially while DataSpace secrets are available to the parent process.

For bundlers such as Vite:

- Emit the production output directly into `ARTIFACT_FOLDER`, or build elsewhere and copy the complete output tree there.
- Configure a relative asset base such as `./` when the application is served below a nested artifact URL.
- Keep every hashed JavaScript, CSS, font, image, worker, and WebAssembly asset referenced by `index.html` inside the artifact tree.
- Check bundler `public/` assets, dynamic imports, manifest icons, service workers, and runtime-configuration fetches for the same relative-base behavior.
- Prefer hash routing or a single-page routing strategy known to work without a server-side history fallback.
- Disable production source maps unless they are intentionally required, and scan any retained maps for private source, hostnames, identifiers, or configuration.
- Do not silently serve an old bundle after a compilation failure.
- Do not write the automatic parquet output manually; return the dataframe and let the runner persist it.

## Runtime configuration

Separate public routing metadata from secrets. Dataset snapshot IDs, transform IDs, API origin selection, and feature flags may be emitted as a small runtime configuration file or an inline non-secret object during the build. Keep credentials out of compiled JavaScript.

Prefer deployment-relative or same-origin API URLs when the platform supports them. If the API is on another permitted origin, use the deployment's configured public API base and existing authentication flow. Do not bypass DataSpace authorization or fabricate bearer tokens in browser code.

A build-time secret does not become safe merely because it is compiled into JavaScript. A static hosted dashboard cannot protect a service token. If the query API requires a service credential rather than the signed-in user's authorized session, stop and use a platform-supported authenticated flow or an authorized server-side proxy.

Generated runtime configuration should be JSON-encoded rather than interpolated into executable JavaScript strings. Validate all values read from URL parameters before using them in a request.

## Querying parquet-backed data

Use the DataSpace API as the query boundary. Let the server read parquet files and push down the requested projection, filters, ordering, aggregation, pagination, and row limit. A dashboard should not fetch a whole parquet file merely to discard most rows in the browser.

Because DataSpace deployments and API versions can differ, copy the exact endpoint and payload shape from the current platform or a working project client. Keep the API call behind a small typed module so it can be changed independently of React components.

Confirm which dataset version is actually queryable. The current transform's new output snapshot may not exist until after `transform()` returns, so an upstream snapshot identifier is not automatically interchangeable with that output. Use the deployed platform's supported runtime resolution, or intentionally target an existing upstream, previous, or published snapshot.

The client should:

- Send only the dataset, snapshot, build, or transform identifier expected by the deployed API.
- Request only required columns and apply a conservative default and maximum row limit.
- Cancel superseded searches with `AbortController`.
- Distinguish authorization, validation, rate-limit, server, and network failures in the UI.
- Validate or narrow the response before rendering it.
- Preserve the platform's authenticated session using its documented same-origin or cross-origin credential behavior.
- Avoid logging sensitive query text, returned records, headers, or identifiers to the browser console.

If the API accepts SQL or another query language, treat user input as data. Prefer structured filters or parameters. The backend must enforce read-only access, authorization, timeouts, memory limits, and statement restrictions; a browser-only check is not a security boundary.

Render returned content as text by default, sanitize any intentional HTML or Markdown, and handle partial or empty results explicitly.

## Validation

Validate both halves of the feature:

1. Run the repository's locked frontend checks and production build.
2. Run the transform locally when its DataSpace fixtures and environment are available.
3. Confirm `ARTIFACT_FOLDER/index.html` and every referenced static asset exist.
4. Serve the artifact directory below a nested path locally, or use the hosted preview, and verify that no asset assumes `/` as its base.
5. Test API requests in the hosted authenticated context with a small query, no results, invalid input, expired authorization, and a server failure.
6. Confirm filters and row limits reach the API rather than being applied only after a full download.
7. Check narrow and wide layouts, keyboard access, and readable loading, empty, and error states.

When platform access is unavailable, finish all local checks and clearly list the hosted API and authorization tests that remain.

If neither a working client nor deployed API documentation reveals the endpoint, identifiers, request schema, authentication flow, and authorization model, stop the live API integration at that boundary. A static shell and mocked client may be prepared, but do not ship a guessed production request.

## Official documentation

- [Plots and dashboards](https://docs.dataspace.ch/platform/transformation/plot.md)
- [DataSpace documentation index](https://docs.dataspace.ch/llms.txt)
