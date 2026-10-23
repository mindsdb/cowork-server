# Server diagnostics

The default is `LOG_LEVEL=WARNING`, `LOG_FORMAT=json`, `ACCESS_LOG_MODE=errors`.
Dev/PR charts use errors-only access; staging/production retain ordinary successes
with `ACCESS_LOG_MODE=all`. Access has its own handler, independent of LOG_LEVEL.
Successful GET/HEAD probes at `/health`, `/api/v1/health`, `/api/v1/health/` and
`/api/v1/health/live` are quiet in both modes. Failed probes remain visible.

The HTTP boundary emits one final refusal for 4xx and a failure for 5xx or an
escaping streamed exception. `x-request-id` accepts UUID/32-hex correlation IDs;
other values are replaced. Response headers carry that correlation. Existing
server-generated turn IDs remain unchanged in responses and are retained as
`turn_request_id` when an HTTP correlation is present. Request context remains
bound until streaming ends and is then reset.

For direct Uvicorn launches, pass the installed `cowork/logging.json` with
`--log-config`; the packaged CLI, Dockerfile and chart already do this. This
protects startup diagnostics before importing the application. `LOG_LEVEL=DEBUG`
and `ACCESS_LOG_MODE=all` temporarily increase detail; restore WARNING and the
environment's access mode afterward. `LOG_FORMAT=text` retains exception privacy
for local readability. File logging uses the same schema. `anton.usage` remains
its independent plain-text INFO stream; conversation/protocol output is unchanged.

Publish the server package before building clients that copy its logging config.
Update installed Anton separately to include its own diagnostic implementation.

Diagnostics use `logged_at`, `logger`, `log_level`, and `message`. Explicit events,
reasons, source locations and correlation fields identify failures. Exception
values, source lines, locals, arbitrary extras, HTTP queries and provider bodies
are excluded. Use the structured fields when querying logs; the previous text
prefixes are no longer a parsing contract.

Enable extra detail only for an owned troubleshooting session, repeat the failing
operation, and restore the default afterward. Logging controls do not enable
framework debug responses. Roll back a release by restoring its previous image
or package version and its previous logging settings together. Before promotion,
verify collector parsing, saved searches, alerts and retained operational streams
against the emitted records. A passing unit suite does not establish deployment
or collector compatibility.
