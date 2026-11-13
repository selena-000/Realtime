"""Validate and optionally archive the exact source-release allowlist."""

import argparse
import ast
import re
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULES = ["__init__", "model", "features", "data", "metrics", "engine", "inference",
           "checkpoint", "cli", "train", "evaluate", "predict", "inspect_features"]
TESTS = ["conftest", "test_causality", "test_features", "test_checkpoint", "test_cli", "test_metrics"]
FILES = {
    ".gitignore", ".github/workflows/tests.yml", "pyproject.toml", "MANIFEST.in",
    "README.md", "LICENSE.md", "LICENSE_MiniGPT4.md", "THIRD_PARTY_NOTICES.md",
    "CITATION.cff", "docs/data.md", "docs/protocol.md", "docs/validation.md",
    "scripts/check_source.py", "scripts/verify_original.py",
} | {"src/realtime_merc/{}.py".format(name) for name in MODULES} | {
    "tests/{}.py".format(name) for name in TESTS
}
GENERATED = {".git", "__pycache__", ".pytest_cache", "build", "dist"}


def check():
    observed = set()
    for path in ROOT.rglob("*"):
        rel = path.relative_to(ROOT)
        if any(part in GENERATED or part.endswith(".egg-info") for part in rel.parts):
            continue
        if path.is_symlink():
            raise ValueError("Symlink in source release: {}".format(rel))
        if path.is_file():
            observed.add(rel.as_posix())
    if observed != FILES:
        raise ValueError("Unexpected files: {}; missing files: {}".format(
            sorted(observed - FILES), sorted(FILES - observed)))
    for name in sorted(FILES):
        path = ROOT / name
        text = path.read_text(encoding="utf-8")
        if path.stat().st_size > 200_000:
            raise ValueError("Unexpected large source file: " + name)
        if name.endswith(".py"):
            ast.parse(text, filename=name, feature_version=(3, 9))
        if re.search(r"(?:gh[pousr]_)[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,}|AKIA[A-Z0-9]{16}", text):
            raise ValueError("Possible credential in " + name)
        if re.search(r"/(?:data|home)/hxza\d+", text):
            raise ValueError("Private server path in " + name)
    return len(FILES), sum((ROOT / name).stat().st_size for name in FILES)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="Write an archive outside the source directory.")
    args = parser.parse_args()
    count, size = check()
    if args.archive:
        if ROOT == args.archive.resolve().parent or ROOT in args.archive.resolve().parents:
            parser.error("Put the archive outside the source tree.")
        with zipfile.ZipFile(args.archive, "w", zipfile.ZIP_DEFLATED) as archive:
            for name in sorted(FILES):
                archive.write(ROOT / name, "Realtime-MERC/" + name)
    print("Source release: {} allowlisted files, {} bytes; syntax and artifact checks passed.".format(count, size))


if __name__ == "__main__":
    main()
