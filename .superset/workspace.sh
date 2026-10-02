#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

mode=${1:-help}
case "$mode" in
    setup|check) ;;
    teardown)
        # Setup starts no background processes or services.
        echo 'No workspace services to stop.'
        exit 0
        ;;
    *) echo 'Usage: bash .superset/workspace.sh {setup|check|teardown}'; exit 2 ;;
esac

unset PYTHONPATH PYTHONHOME
if [[ "$mode" == setup ]]; then
    # Copy missing, untracked environment files from the root checkout.
    root_path=${SUPERSET_ROOT_PATH:-$PWD}
    [[ -d "$root_path" ]] || { echo "Root checkout not found: $root_path" >&2; exit 1; }
    shopt -s nullglob
    for source_file in "$root_path"/.env "$root_path"/.env.*; do
        [[ -f "$source_file" ]] || continue
        filename=${source_file##*/}
        if git -C "$root_path" ls-files --error-unmatch -- "$filename" >/dev/null 2>&1; then
            continue
        fi
        [[ ! -e "$filename" && ! -L "$filename" ]] || continue
        cp -p "$source_file" "$filename"
        echo "Copied $filename from root checkout."
    done
    shopt -u nullglob
    git submodule update --init
    if [[ ! -x venv/bin/python ]]; then
        make venv
    fi
fi

[[ -x venv/bin/python ]] || { echo 'Run bash .superset/workspace.sh setup first.' >&2; exit 1; }
export VIRTUAL_ENV="$PWD/venv"
export PATH="$VIRTUAL_ENV/bin:$PATH"

if [[ "$mode" == setup ]]; then
    python -m pip install -e '.[dev]'
    python -m pip check
    python -c 'import netlab, ngraph, netgraph_core, topogen; print("Workspace environment ready")'
else
    make check-ci
fi
