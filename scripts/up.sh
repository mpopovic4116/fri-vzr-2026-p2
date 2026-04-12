#!/usr/bin/env bash

# Uploads the codebase to the configured cluster.

set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"/..

rsync -az --delete --filter '. .rsync-filter' . "$CLUSTER_HOST:$CLUSTER_DIR"
