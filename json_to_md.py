"""
Convert Docling JSON files -> Markdown.

Reads per-page .json files produced by run_ocr.py and exports them to
Markdown. Tables and all structure are preserved because the JSON
representation is fully lossless.

Usage:
    # Convert all JSON files in a directory -> combined <dir>.md + per-page .md
    python json_to_md.py out/

    # Convert a single JSON file
    python json_to_md.py out/report_p0001.json

    # Custom output path for the combined file
    python json_to_md.py out/ -o final/report.md

    # Also export per-page HTML
    python json_to_md.py out/ --save-html

    # Skip the combined file, only write per-page .md files
    python json_to_md.py out/ --no-combined
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def convert_one(json_path: Path, output_dir: Path, save_html: bool, DoclingDocument) -> str:
    """Load one Docling JSON file, write .md (and optionally .html). Returns MD text."""
    doc = DoclingDocument.load_from_json(json_path)
    md_text = doc.export_to_markdown()

    (output_dir / f"{json_path.stem}.md").write_text(md_text, encoding="utf-8")

    if save_html:
        doc.save_as_html(output_dir / f"{json_path.stem}.html")

    return md_text


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("input", type=Path, help="Directory of .json files, or a single .json file")
    ap.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output path for the combined Markdown file (default: <input_dir>/<n>.md)",
    )
    ap.add_argument("--save-html", action="store_true", help="Also export per-page HTML")
    ap.add_argument("--no-combined", action="store_true", help="Skip the combined Markdown file")
    ap.add_argument(
        "--page-separator",
        default="<!-- page: {page_id} -->",
        help="Separator inserted between pages in the combined file ({page_id} is replaced)",
    )
    args = ap.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input not found: {args.input}")

    try:
        from docling_core.types.doc import DoclingDocument
    except ImportError as e:
        print(f"Missing dependency: {e}\nInstall with: pip install docling-core", file=sys.stderr)
        return 2

    # ── Collect JSON files ────────────────────────────────────────────────────
    if args.input.is_file():
        if args.input.suffix.lower() != ".json":
            raise SystemExit(f"Expected a .json file, got: {args.input}")
        json_files = [args.input]
        output_dir = args.input.parent
        combined_name = args.input.stem
    else:
        json_files = sorted(args.input.glob("*.json"))
        if not json_files:
            raise SystemExit(f"No .json files found in {args.input}")
        output_dir = args.input
        combined_name = args.input.name

    output_dir.mkdir(parents=True, exist_ok=True)
    combined_path = None if args.no_combined else (args.output or output_dir / f"{combined_name}.md")

    print(f"[json_to_md] {len(json_files)} file(s)", flush=True)

    # ── Convert ───────────────────────────────────────────────────────────────
    t0 = time.time()
    pages: list[tuple[str, str]] = []

    for json_path in json_files:
        md_text = convert_one(json_path, output_dir, args.save_html, DoclingDocument)
        pages.append((json_path.stem, md_text))
        print(f"  [page] {json_path.stem} -> {json_path.stem}.md", flush=True)

    elapsed = time.time() - t0

    # ── Combined file ─────────────────────────────────────────────────────────
    if combined_path:
        with combined_path.open("w", encoding="utf-8") as f:
            for page_id, md_text in pages:
                separator = args.page_separator.format(page_id=page_id)
                f.write(f"\n\n{separator}\n\n")
                f.write(md_text)
        print(f"[combined] -> {combined_path}", flush=True)

    print(
        f"[done] {len(json_files)} page(s) in {elapsed:.2f}s "
        f"avg={elapsed / max(1, len(json_files)):.2f}s/page",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())