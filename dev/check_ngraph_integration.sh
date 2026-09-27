#!/usr/bin/env bash
# Run NetLab lint and tests against selected local project sources.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
unset PYTHONPATH PYTHONHOME NETLAB_NGRAPH_BIN NETLAB_TOPOGEN_BIN NGRAPH_ENABLE_MAXFLOW
if [[ $# -ne 3 ]]; then
    echo 'Usage: bash dev/check_ngraph_integration.sh /path/to/NetGraph /path/to/NetGraph-Core /path/to/TopoGen' >&2
    exit 2
fi
for source_dir in "$@"; do
    [[ -f "$source_dir/pyproject.toml" ]] || { echo "Missing pyproject.toml: $source_dir" >&2; exit 1; }
done
ngraph_path=$(cd "$1" && pwd -P)
core_path=$(cd "$2" && pwd -P)
topogen_path=$(cd "$3" && pwd -P)
[[ -x venv/bin/python ]] || { echo 'Run bash .superset/workspace.sh setup first.' >&2; exit 1; }
mkdir -p build/ngraph-integration
run_dir=$(mktemp -d "$PWD/build/ngraph-integration/run.XXXXXX")
echo "Integration artifacts: $run_dir"
trap 'rm -rf "$run_dir/venv" "$run_dir/core-build"' EXIT
venv/bin/python -m venv "$run_dir/venv"
export VIRTUAL_ENV="$run_dir/venv"
export PATH="$VIRTUAL_ENV/bin:$PATH"
# Record source state before building Core and compare it after testing.
source_state() {
    python - "$PWD" "$ngraph_path" "$core_path" "$topogen_path" <<'PY'
import hashlib
import json
import pathlib
import subprocess
import sys

state = {}
for root in sys.argv[1:]:
    def git(*args):
        return subprocess.check_output(['git', '-C', root, *args])

    digest = hashlib.sha256(git('diff', 'HEAD', '--binary'))
    for name in sorted(git('ls-files', '--others', '--exclude-standard', '-z').split(b'\0')):
        if not name:
            continue
        file = pathlib.Path(root) / name.decode()
        digest.update(name + b'\0')
        if file.is_symlink():
            digest.update(str(file.readlink()).encode())
        elif file.is_file():
            digest.update(file.read_bytes())
    state[root] = {'head': git('rev-parse', 'HEAD').decode().strip(),
                   'changes_sha256': digest.hexdigest()}
print(json.dumps(state, indent=2, sort_keys=True))
PY
}
source_state > "$run_dir/source-state-before.json"

export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-4}"
if [[ "$(uname -s)" == Darwin ]]; then
    export CC="$(xcrun --find clang)"
    export CXX="$(xcrun --find clang++)"
    export MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-15.0}"
fi
python -m pip install --upgrade pip
python -m pip install packaging cmake ninja
python -m pip wheel --no-deps --wheel-dir "$run_dir/wheels" \
    --config-settings="build-dir=$run_dir/core-build" "$core_path"
wheels=("$run_dir"/wheels/netgraph_core-*.whl)
[[ ${#wheels[@]} -eq 1 && -f "${wheels[0]}" ]] || { echo 'Expected one Core wheel.' >&2; exit 1; }
# Install third-party dependencies with the selected local project sources.
python - "$run_dir/requirements.txt" <<'PY'
import pathlib
import sys
import tomllib
from packaging.requirements import Requirement

project = tomllib.loads(pathlib.Path('pyproject.toml').read_text())['project']
requirements = project['dependencies'] + project['optional-dependencies']['dev']
pathlib.Path(sys.argv[1]).write_text('\n'.join(
    req for req in requirements if Requirement(req).name not in {'ngraph', 'topogen'}
) + '\n')
PY
python -m pip install -r "$run_dir/requirements.txt" \
    -e "$ngraph_path" -e "$topogen_path" "${wheels[0]}"
python -m pip install --no-deps -e .
python -m pip check
python - "$ngraph_path" "$core_path" "$topogen_path" "${wheels[0]}" <<'PY' | tee "$run_dir/provenance.txt"
import hashlib
import importlib.metadata
import pathlib
import subprocess
import sys
import _netgraph_core
import netlab
import ngraph
import topogen

ngraph_path, core_path, topogen_path, wheel = map(pathlib.Path, sys.argv[1:])
for name, root in [('NetLab', pathlib.Path.cwd()), ('NetGraph', ngraph_path),
                   ('Core', core_path), ('TopoGen', topogen_path)]:
    print(f'{name}: {root}', flush=True)
    subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'], check=True)
    subprocess.run(['git', '-C', str(root), 'status', '--short'], check=True)
for module, root in [(netlab, pathlib.Path.cwd()), (ngraph, ngraph_path), (topogen, topogen_path)]:
    print(f'{module.__name__}: {module.__file__}')
    assert pathlib.Path(module.__file__).resolve().is_relative_to(root)
print('Python:', sys.version)
print('Core version:', importlib.metadata.version('netgraph-core'))
print('Core extension:', _netgraph_core.__file__)
print('Core wheel SHA256:', hashlib.sha256(wheel.read_bytes()).hexdigest())
assert pathlib.Path(_netgraph_core.__file__).is_relative_to(sys.prefix)
PY
python -m pip freeze > "$run_dir/requirements-freeze.txt"

make lint PYTHON="$VIRTUAL_ENV/bin/python" 2>&1 | tee "$run_dir/lint.log"
# Discard inherited pytest filters so the integration gate runs the whole suite.
export PYTEST_ADDOPTS="--junitxml=$run_dir/pytest.xml"
test_status=0
# Select the explicit local-source pipeline checks as well as the normal suite.
python -m pytest tests -o 'python_files=test_*.py check_topogen_pipeline.py' \
    2>&1 | tee "$run_dir/pytest.log" || test_status=$?
source_state > "$run_dir/source-state-after.json"
if ! cmp -s "$run_dir/source-state-before.json" "$run_dir/source-state-after.json"; then
    echo 'Source checkout changed during the test run; rerun against stable sources.' >&2
    exit 1
fi
exit "$test_status"
