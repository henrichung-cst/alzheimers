# S3 gzip probe

Upload this directory to the same S3/CloudFront path style used for the unified
viewer, then open `index.html` in a browser.

The page fetches `probe.json.gz` and reports:

- whether the response advertises `Content-Encoding: gzip`,
- whether the browser appears to have auto-decompressed the body,
- whether manual client-side gzip decompression with `DecompressionStream` works,
- which unified-viewer payload loading mode is appropriate.

Interpretation:

- If `Content-Encoding` is absent and manual decompression works, use the current
  raw `.json.gz` sidecar strategy.
- If `Content-Encoding: gzip` is present and the body is already JSON, let the
  browser decompress and do not manually pipe through `DecompressionStream`.
- If both fail, the `.gz` object or metadata is misconfigured.

Regenerate the gzip fixture after editing `probe.json`:

```bash
gzip -cn docs/probes/s3_gzip_probe/probe.json > docs/probes/s3_gzip_probe/probe.json.gz
```

