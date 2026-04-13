#!/usr/bin/env bash

# Runs a script on the configured cluster.
# Takes positional arguments that will be passed literally to bash.
# So, you can run `scripts/on_cluster.sh bash -c 'echo test'`, and it will Just Work™

set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"/..

scripts/up.sh

# shellcheck disable=SC2029
ssh "$CLUSTER_HOST" "bash -c 'cd '$(printf '%q' "$CLUSTER_DIR")' && '$(printf '%q' "$(printf '%q ' "$@")")"
