"""Fail if anything that looks like a credential is present in tracked files.

Run from the repository root: python3 scripts/check_secrets.py
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
SKIP_DIRS = {
    ".git",
    ".venv",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "wandb",
}
SKIP_FILES = {".env", ".env.local", "uv.lock", "package-lock.json"}
TEXT_SUFFIXES = {
    ".py",
    ".ts",
    ".md",
    ".json",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
    ".example",
}

PATTERNS = (
    ("netrc-style W&B login", re.compile(r"machine\s+api\.wandb\.ai")),
    (
        "assigned WANDB_API_KEY",
        re.compile(r"WANDB_API_KEY\s*=\s*['\"]?[A-Za-z0-9]{16,}"),
    ),
    ("OpenAI-style key", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b")),
    ("AWS access key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    (
        "GitHub token",
        re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b"),
    ),
    ("Slack token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")),
    (
        "private key",
        re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    ),
    ("Google API key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    ("Stripe live key", re.compile(r"\b(?:sk|rk)_live_[0-9A-Za-z]{16,}\b")),
    ("40-char hex token", re.compile(r"\b[0-9a-f]{40}\b")),
)
ALLOW_LINE = re.compile(r"integrity|sha1-|sha512-")


def main() -> int:
    findings: list[str] = []
    for path in ROOT.rglob("*"):
        if (
            not path.is_file()
            or path.suffix not in TEXT_SUFFIXES
            or path.name in SKIP_FILES
        ):
            continue
        if SKIP_DIRS & set(path.parts):
            continue
        for number, line in enumerate(
            path.read_text(errors="ignore").splitlines(), start=1
        ):
            if ALLOW_LINE.search(line):
                continue
            for label, pattern in PATTERNS:
                if pattern.search(line):
                    findings.append(f"{path.relative_to(ROOT)}:{number}: {label}")

    gitignore = ROOT / ".gitignore"
    if not gitignore.exists() or ".env" not in gitignore.read_text():
        findings.append(".gitignore: must ignore .env")

    if findings:
        print(f"check_secrets: {len(findings)} finding(s)")
        for finding in findings:
            print(f"  {finding}")
        return 1
    print("check_secrets: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
