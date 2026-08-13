#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 notebook.ipynb --name example-name [--force] [--fail-on-check]" >&2
  echo "       $0 path-list.txt [--force] [--fail-on-check]" >&2
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
caller_dir="$(pwd)"

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
force=0
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
    --force)
      force=1
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

cd "$repo_root"

resolve_existing_path() {
  local path="$1"
  local candidate
  local alt_path

  for candidate in "$path" "$caller_dir/$path" "${path#/}" "${path#/examples/}" "${path#examples/}"; do
    [[ -n "$candidate" ]] || continue

    if [[ -f "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi

    if [[ "$candidate" == */notebook-paths.txt ]]; then
      alt_path="${candidate%notebook-paths.txt}notebook_paths.txt"
      if [[ -f "$alt_path" ]]; then
        printf '%s\n' "$alt_path"
        return 0
      fi
    fi
  done

  return 1
}

slug_from_notebook() {
  local path="$1"
  local slug

  slug="$(basename -- "$path")"
  slug="${slug%.ipynb}"
  slug="$(printf '%s' "$slug" | tr '[:upper:]' '[:lower:]')"
  slug="$(printf '%s' "$slug" | sed -E 's/[^a-z0-9]+/-/g; s/^-+//; s/-+$//')"

  if [[ -z "$slug" ]]; then
    echo "could not derive slug from notebook path: $path" >&2
    return 1
  fi

  printf '%s\n' "$slug"
}

prepare_one_notebook() {
  local notebook="$1"
  local slug="$2"
  local input_dir module_name target_dir target_py debug_dir report check_output check_status

  notebook="$(resolve_existing_path "$notebook")" || {
    echo "input file does not exist: $notebook" >&2
    return 1
  }

  [[ "$notebook" == *.ipynb ]] || {
    echo "input must be a .ipynb file: $notebook" >&2
    return 1
  }

  input_dir="$(cd -- "$(dirname -- "$notebook")" && pwd)"
  notebook="$input_dir/$(basename -- "$notebook")"

  [[ "$slug" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] || {
    echo "--name must be a slug like 'mnist-registry' or 'mnist_registry' (no paths, dots, or spaces): $slug" >&2
    return 1
  }

  module_name="${slug//-/_}"
  target_dir="marimo/convert/$slug"
  target_py="$target_dir/$module_name.py"
  debug_dir="$target_dir/.conversion"
  report="$debug_dir/conversion-report.md"
  check_output="$debug_dir/marimo-check.txt"

  if [[ -e "$target_py" && "$force" -eq 0 ]]; then
    echo "target notebook already exists: $target_py" >&2
    echo "pass --force to overwrite it" >&2
    return 1
  fi
  mkdir -p "$target_dir" "$debug_dir"

  uvx marimo convert "$notebook" -o "$target_py"

  check_status=0
  uvx marimo check "$target_py" > "$check_output" 2>&1 || check_status=$?

  cat > "$report" <<EOF
# Conversion Report

Source notebook: \`$notebook\`
Generated notebook: \`$target_py\`

## Processing

- Created target directory: \`$target_dir\`
- Created temporary debug directory: \`$debug_dir\`
- Ran: \`uvx marimo convert "$notebook" -o "$target_py"\`
- Ran: \`uvx marimo check "$target_py"\`

## Check Result

Exit code: \`$check_status\`

See \`.conversion/marimo-check.txt\`.

## Next Agent Step

Polish \`$target_py\` into an idiomatic repo-ready marimo example.
EOF

  if ((check_status)); then
    echo "marimo check reported issues; see $check_output" >&2
  fi

  if ((fail_on_check)); then
    return "$check_status"
  fi

  return 0
}

if [[ "$input" == *.txt ]]; then
  path_list="$(resolve_existing_path "$input")" || {
    echo "path list file does not exist: $input" >&2
    exit 1
  }

  if [[ -n "$name" ]]; then
    echo "--name cannot be used with a path list; names are derived from notebook filenames" >&2
    exit 1
  fi

  failures=0
  while IFS= read -r listed_path; do
    [[ -n "$listed_path" ]] || continue

    if [[ "$listed_path" != *.ipynb ]]; then
      echo "skipping non-notebook path: $listed_path" >&2
      continue
    fi

    slug="$(slug_from_notebook "$listed_path")" || {
      failures=1
      continue
    }

    echo "preparing $listed_path as $slug"
    prepare_one_notebook "$listed_path" "$slug" || failures=1
  done < "$path_list"

  exit "$failures"
fi

[[ -n "$name" ]] || {
  usage
  exit 1
}

prepare_one_notebook "$input" "$name"
