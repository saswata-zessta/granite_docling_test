"""
Synthetic end-to-end check for Granite Docling vLLM.

Generates a small test image + PDF with known text content, runs the OCR
pipeline, and asserts that the markdown output contains expected tokens.
Exits non-zero if anything fails, so it works as a CI/smoke gate.

Run from repo root:
    python synthetic_check.py                                  # uses HF id
    python synthetic_check.py --model-path models/granite-docling-258M-untied
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


EXPECTED_STRINGS = [
    "Granite Docling Synthetic Check",
    "invoice",
    "Total",
]

SYNTHETIC_TEXT = [
    ("title", "Granite Docling Synthetic Check"),
    ("body", "This document tests end-to-end OCR, layout, and table recognition."),
    ("heading", "Sample invoice"),
    ("table_header", "Item              Qty    Price"),
    ("table_row1",   "Widget             2     10.00"),
    ("table_row2",   "Gadget             1      5.50"),
    ("table_row3",   "Sprocket           4      2.25"),
    ("total",        "Total                    24.50"),
    ("footer",       "Thank you for your business."),
]


def _best_font(size: int) -> ImageFont.ImageFont:
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ):
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def build_synthetic(target_dir: Path) -> tuple[Path, Path]:
    """Render a PNG + single-page PDF with known content. Returns (png, pdf)."""
    target_dir.mkdir(parents=True, exist_ok=True)
    W, H = 1240, 1600  # ~150 DPI letter
    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)

    title_font = _best_font(42)
    heading_font = _best_font(28)
    body_font = _best_font(22)
    mono_font = _best_font(22)

    y = 80
    for kind, text in SYNTHETIC_TEXT:
        if kind == "title":
            draw.text((80, y), text, fill="black", font=title_font); y += 90
        elif kind == "heading":
            y += 30
            draw.text((80, y), text, fill="black", font=heading_font); y += 60
        elif kind.startswith("table") or kind == "total":
            draw.text((80, y), text, fill="black", font=mono_font); y += 40
        else:
            draw.text((80, y), text, fill="black", font=body_font); y += 50

    png_path = target_dir / "synthetic.png"
    pdf_path = target_dir / "synthetic.pdf"
    img.save(png_path, "PNG")
    img.save(pdf_path, "PDF", resolution=150)
    return png_path, pdf_path


def run_ocr(pdf_path: Path, out_dir: Path, model_path: str, extra: list[str]) -> None:
    cmd = [
        sys.executable, "run_ocr.py", str(pdf_path),
        "-o", str(out_dir),
        "--model", model_path,
        "--batch-size", "1",
        *extra,
    ]
    print(f"[check] $ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def assert_outputs(out_dir: Path, stem: str) -> None:
    md_path = out_dir / f"{stem}.md"
    if not md_path.exists():
        raise SystemExit(f"FAIL: expected markdown missing: {md_path}")
    text = md_path.read_text(encoding="utf-8").lower()
    missing = [s for s in EXPECTED_STRINGS if s.lower() not in text]
    if missing:
        preview = text[:600].replace("\n", " | ")
        raise SystemExit(
            f"FAIL: these expected strings were not found in {md_path}: {missing}\n"
            f"first 600 chars: {preview}"
        )
    print(f"[check] OK — {md_path} contains all expected strings")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="ibm-granite/granite-docling-258M",
                    help="HF repo id or local path (default: HF id)")
    ap.add_argument("--revision", default="untied")
    ap.add_argument("--work-dir", type=Path, default=Path("synthetic_out"))
    ap.add_argument("--keep", action="store_true", help="Don't wipe the work dir first")
    args = ap.parse_args()

    if not args.keep and args.work_dir.exists():
        shutil.rmtree(args.work_dir)
    args.work_dir.mkdir(parents=True, exist_ok=True)

    print("[check] generating synthetic PNG + PDF", flush=True)
    png_path, pdf_path = build_synthetic(args.work_dir)

    # vLLM ignores --revision for local paths, so just pass it through either way.
    extra: list[str] = ["--revision", args.revision]

    t0 = time.time()
    run_ocr(pdf_path, args.work_dir, args.model_path, extra)
    elapsed = time.time() - t0

    assert_outputs(args.work_dir, "synthetic_p0001")
    print(f"[check] PASS in {elapsed:.1f}s. Outputs in {args.work_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
