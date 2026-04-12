#!/usr/bin/env bash

# Calls make with the given k=v args, redirecting output to stderr.
# Prints the final executable name.

set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"/..

make "$@" printexe3 3>&1 1>&2
