#!/usr/bin/env bash
set -euo pipefail
exec Rscript alz/integration/factorial.R "$@"
