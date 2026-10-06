"""Check that guides reference real files and contain runnable code blocks.

Run from the repository root: python3 scripts/validate_docs.py
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
LANGUAGES = ("python", "typescript")
CODE_LANGS = {"bash", "python", "typescript", "ts", "sh"}
PATH_TOKEN = re.compile(r"\b(?:examples|guides|reference|scripts|assets)/[\w./&-]+")
PLACEHOLDER_LINE = re.compile(r"^\s*(?:#|//)?\s*\.\.\.\s*$")
MD_LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
REQUIRED_GUIDE_SECTIONS = (
    "## How it works",
    "## Run it",
    "## What you should see",
    "## Inspect it in Weave",
)
BOUNDARY_LABELS = (
    "requires a live model",
    "requires live model",
    "no model call",
    "makes no model call",
    "optional",
    "advanced",
)
STRUCTURE_EXCEPTIONS = {"07-agents"}

errors: list[str] = []


def fail(path: Path, message: str) -> None:
    errors.append(f"{path.relative_to(ROOT)}: {message}")


def code_blocks(text: str) -> list[tuple[str, str]]:
    return [
        (match.group(1).strip().lower(), match.group(2))
        for match in re.finditer(r"```(\w*)[^\n]*\n(.*?)```", text, re.DOTALL)
    ]


def check_links(md_path: Path) -> None:
    for target in MD_LINK.findall(md_path.read_text()):
        if target.startswith(("http://", "https://", "#", "mailto:")):
            continue
        resolved = (md_path.parent / target.split("#")[0]).resolve()
        if not resolved.exists():
            fail(md_path, f"broken link -> {target}")


def check_code(md_path: Path, bases: tuple[Path, ...]) -> None:
    for lang, body in code_blocks(md_path.read_text()):
        if lang not in CODE_LANGS:
            continue
        for line in body.splitlines():
            if PLACEHOLDER_LINE.match(line):
                fail(md_path, f"placeholder ellipsis in {lang} block: {line.strip()!r}")
        if lang in ("bash", "sh"):
            for token in PATH_TOKEN.findall(body):
                candidate = token.rstrip(".,")
                if not any((base / candidate).exists() for base in bases):
                    fail(md_path, f"bash block references missing path: {candidate}")


def check_guide_numbering(guides_dir: Path) -> None:
    numbered = sorted(
        p.name
        for p in guides_dir.iterdir()
        if p.is_dir() and re.match(r"\d{2}-", p.name)
    )
    numbers = [int(name[:2]) for name in numbered]
    if numbers != list(range(len(numbers))):
        fail(guides_dir, f"guide numbering has gaps or duplicates: {numbered}")
    for name in numbered:
        readme = guides_dir / name / "README.md"
        if not readme.exists():
            fail(guides_dir / name, "guide folder is missing README.md")
            continue
        check_guide_structure(readme)


def check_guide_structure(md_path: Path) -> None:
    text = md_path.read_text()
    opening = text.split("\n## ", maxsplit=1)[0].lower()
    if not any(label in opening for label in BOUNDARY_LABELS):
        fail(md_path, "opening is missing an honest boundary label")

    if md_path.parent.name in STRUCTURE_EXCEPTIONS:
        if "## Inspect it in Weave" not in text:
            fail(md_path, "missing required section: ## Inspect it in Weave")
        return

    positions = []
    for section in REQUIRED_GUIDE_SECTIONS:
        position = text.find(section)
        if position == -1:
            fail(md_path, f"missing required section: {section}")
        positions.append(position)
    present_positions = [position for position in positions if position >= 0]
    if present_positions != sorted(present_positions):
        fail(md_path, "guide sections are out of template order")
    if "assets/expected-output/" not in text:
        fail(md_path, "missing expected-output capture link")


def check_examples_have_no_placeholders(examples_dir: Path) -> None:
    for path in examples_dir.rglob("*"):
        if path.suffix not in (".py", ".ts"):
            continue
        for number, line in enumerate(path.read_text().splitlines(), start=1):
            if PLACEHOLDER_LINE.match(line):
                fail(path, f"line {number}: placeholder ellipsis in runnable code")


def check_guide_excerpts(md_path: Path, examples_dir: Path, suffix: str) -> None:
    """Every code excerpt in a guide must be a verbatim slice of an example file."""
    sources = [path.read_text() for path in sorted(examples_dir.rglob(f"*{suffix}"))]
    for lang, body in code_blocks(md_path.read_text()):
        if lang not in ("python", "typescript", "ts"):
            continue
        snippet = body.rstrip("\n")
        if not any(snippet in source for source in sources):
            first_line = snippet.splitlines()[0].strip() if snippet else ""
            fail(
                md_path,
                f"code excerpt is not verbatim from any example: starts {first_line!r}",
            )


def main() -> int:
    check_links(ROOT / "README.md")
    check_code(ROOT / "README.md", (ROOT, *(ROOT / language for language in LANGUAGES)))

    for language in LANGUAGES:
        base = ROOT / language
        suffix = ".py" if language == "python" else ".ts"
        for md_path in sorted(base.glob("**/*.md")):
            if {"node_modules", ".venv"} & set(md_path.parts):
                continue
            check_links(md_path)
            check_code(md_path, (base, ROOT))
        for md_path in sorted(base.glob("guides/**/*.md")):
            check_guide_excerpts(md_path, base / "examples", suffix)
        check_guide_numbering(base / "guides")
        for md_path in sorted(base.glob("guides/[0-9][0-9]-*/*.md")):
            if md_path.name != "README.md":
                check_guide_structure(md_path)
        check_examples_have_no_placeholders(base / "examples")

    for md_path in sorted((ROOT / "reference").glob("*.md")):
        check_links(md_path)

    if errors:
        print(f"validate_docs: {len(errors)} problem(s)")
        for error in errors:
            print(f"  {error}")
        return 1
    print("validate_docs: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
