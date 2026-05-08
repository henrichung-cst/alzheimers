#!/usr/bin/env bash
set -euo pipefail
exec Rscript code/integration/factorial.R "$@"
