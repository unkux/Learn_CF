#!/bin/bash

USAGE="Usage: $0 instance [same arguments as mle_run ...]"
(( $# < 1 )) && { echo "$USAGE"; exit; }

inst=$1
[[ ! -d "$inst" ]] && { echo "$inst not exists!"; exit; }

if command -v micromamba &> /dev/null; then
    eval "$(micromamba shell hook --shell=zsh)"
    micromamba activate numba-dpex-env
fi

echo Run "$@"

base=$(dirname "$0")

"$base"/run.py "$@"
