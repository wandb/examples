"""Build and normalize marimo notebook path lists."""

import argparse
import csv
from urllib.parse import unquote


def normalize_path(path):
    """Convert a GitHub blob path from the source list into a repo-local path."""
    path = unquote(path.strip())
    prefix = "/wandb/examples/blob/"
    if not path.startswith(prefix):
        return path
    return path[len(prefix):].split("/", 1)[1]


def create_path_list_file(args):
    """Read the CSV Path column and write unique normalized paths, one per line."""
    with open(args.input_file, newline="", encoding="utf-8") as f:
        paths = {
            normalize_path(row["Path"])
            for row in csv.DictReader(f)
            if row.get("Path")
        }

    with open(args.output_file, "w") as f:
        f.write("\n".join(sorted(paths)))
        f.write("\n")


def main(args):
    if not args.input_file:
        raise SystemExit("--input_file is required")

    create_path_list_file(args)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Extract path column from CSV and return a list of unique paths.")
    argparser.add_argument("--input_file", help="Path to the input CSV file.")
    argparser.add_argument("--output_file", default="notebook_paths.txt", help="Path to the output file.")
    args = argparser.parse_args()
    main(args)
