#!/usr/bin/env python3
"""Rebuild the exact natural-text corpus used by the S55-200 baseline."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path


REVISION = "b265e226a454d9e6c29c19c22650560039cc9171"
EXPECTED_BYTES = 1_157_096
EXPECTED_SHA256 = "78493835239cb7a3b35228bf7304f72d3f293d9b99c4a7ef10e0aca206c7150b"

# Frozen en_US.UTF-8 collation order from the baseline run.  An explicit list
# avoids locale drift and excludes later untracked competition/roadmap files.
MARKDOWN_FILES = (
    "AGENT.md",
    "analysis/glm53-abliteration/PROVENANCE.md",
    "analysis/glm53-abliteration/README.md",
    "CONTRIBUTING.md",
    "cuda/mmq/VENDOR.md",
    "dir-steering/README.md",
    "docs/research/abliterated-ds4-alternative-2026-08-28.md",
    "docs/research/glm53_kda_verify2_snapshot.md",
    "docs/research/glm53-abliteration-forensics-2026-08-31.md",
    "gguf-tools/imatrix/dataset/README.md",
    "gguf-tools/imatrix/README.md",
    "gguf-tools/mixed/README.md",
    "gguf-tools/quality-testing/data/pro-0813/README.md",
    "gguf-tools/quality-testing/data/README.md",
    "gguf-tools/quality-testing/README.md",
    "gguf-tools/README.md",
    "misc/ANTHROPIC_LIVE_CONTINUATION.md",
    "misc/COMPACT.md",
    "misc/DS4PRO_AUG2026_CONVERSION.md",
    "misc/RESPONSE_API.md",
    "MODEL_CARD.md",
    "QA_BEFORE_RELEASES.md",
    "README.md",
    "speed-bench/gfx1151-prefill-results.md",
    "speed-bench/README.md",
    "STRIXHALO.md",
    "tests/test-vectors/README.md",
    "tests/vision-fixtures/glm53/README.md",
)


def git_blob(repo: Path, file_name: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(repo), "show", f"{REVISION}:{file_name}"],
        check=True,
        capture_output=True,
    ).stdout


def build_corpus(repo: Path) -> bytes:
    parts = [git_blob(repo, file_name) for file_name in MARKDOWN_FILES]
    parts.extend(
        (
            b"\n# Engine README\n\n",
            git_blob(repo, "README.md"),
            b"\n# Server source\n\n",
            git_blob(repo, "ds4_server.c"),
        )
    )
    corpus = b"".join(parts)
    digest = hashlib.sha256(corpus).hexdigest()
    if len(corpus) != EXPECTED_BYTES or digest != EXPECTED_SHA256:
        raise RuntimeError(
            f"frozen corpus mismatch: bytes={len(corpus)} sha256={digest}"
        )
    return corpus


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?", type=Path)
    parser.add_argument(
        "--verify-only", action="store_true", help="validate without writing a file"
    )
    args = parser.parse_args()
    if args.output is None and not args.verify_only:
        parser.error("OUTPUT is required unless --verify-only is used")

    repo = Path(__file__).resolve().parents[1]
    corpus = build_corpus(repo)
    if args.output is not None:
        if args.output.exists():
            existing = args.output.read_bytes()
            if existing != corpus:
                raise SystemExit(f"refusing to overwrite different file: {args.output}")
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_bytes(corpus)
    print(f"bytes={len(corpus)} sha256={EXPECTED_SHA256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
