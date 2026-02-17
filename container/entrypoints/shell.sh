#!/bin/bash
# Shell-mode entrypoint for Miles containers.
# Delegates to default.sh to keep one canonical interactive shell flow.
set -euo pipefail

exec /opt/container_entrypoints/default.sh "$@"
