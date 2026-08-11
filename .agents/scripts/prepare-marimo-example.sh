#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 notebook.ipynb --name example-name [--fail-on-check]" >&2
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"

input="${1:-}"
if [[ -z "$input" || "$input" == "-h" || "$input" == "--help" ]]; then
  usage
  if [[ -z "$input" ]]; then
    exit 1
  fi
  exit 0
fi
shift

name=""
fail_on_check=0
while (($#)); do
  case "$1" in
    --name)
      shift
      (($#)) || {
        echo "missing value for --name" >&2
        exit 1
      }
      name="$1"
      ;;
    --fail-on-check)
      fail_on_check=1
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 1
      ;;
  esac
  shift
done

[[ -n "$name" ]] || {
  usage
  exit 1
}

[[ "$input" == *.ipynb ]] || {
  echo "input must be a .ipynb file: $input" >&2
  exit 1
}

slug="$name"
[[ "$slug" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] || {
  echo "--name must be a slug like 'mnist-registry' or 'mnist_registry' (no paths, dots, or spaces): $slug" >&2
  exit 1
}

module_name="${slug//-/_}"
target_dir="examples/marimo/$slug"
target_py="$target_dir/$module_name.py"
report="$target_dir/conversion-report.md"
check_output="$target_dir/marimo-check.txt"

cd "$repo_root"
mkdir -p "$target_dir"

uvx marimo convert "$input" -o "$target_py"

check_status=0
uvx marimo check "$target_py" > "$check_output" 2>&1 || check_status=$?

cat > "$report" <<EOF
# Conversion Report

Source notebook: \`$input\`
Generated notebook: \`$target_py\`

## Processing

- Created target directory: \`$target_dir\`
- Ran: \`uvx marimo convert "$input" -o "$target_py"\`
- Ran: \`uvx marimo check "$target_py"\`

## Check Result

Exit code: \`$check_status\`

See \`marimo-check.txt\`.

## Next Agent Step

Polish \`$target_py\` into an idiomatic repo-ready marimo example.
EOF

if ((check_status)); then
  echo "marimo check reported issues; see $check_output" >&2
fi

if ((fail_on_check)); then
  exit "$check_status"
fi
