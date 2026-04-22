"""
Docling standard pipeline runner — for PDFs with native text (generated bills,
digital invoices). Uses TableFormer for table structure + native PDF text
extraction. No VLM / no OCR needed for clean digital PDFs.

For truly scanned images with no text layer, set --force-ocr.

Usage:
    python run_docling_pipeline.py input.pdf -o out/
    python run_docling_pipeline.py folder/ -o out/
    python run_docling_pipeline.py input.pdf -o out/ --force-ocr
    python run_docling_pipeline.py input.pdf -o out/ --save-json --save-html
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def _is_image_pdf(pdf_path: Path) -> bool:
    """Return True if the PDF has no extractable text (image-based / scanned)."""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(str(pdf_path))
        for page in doc:
            if page.get_text().strip():
                return False
        return True
    except Exception:
        return False  # assume text-native if we can't check


def convert_one(pdf_path: Path, output_dir: Path, args) -> None:
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.pipeline_options import (
        PdfPipelineOptions,
        TableFormerMode,
        EasyOcrOptions,
    )
    from docling.datamodel.base_models import InputFormat

    # Auto-enable OCR for image-based PDFs unless user already specified
    do_ocr = args.force_ocr
    if not do_ocr:
        do_ocr = _is_image_pdf(pdf_path)
        if do_ocr:
            print(f"[info] {pdf_path.name}: no text layer detected — enabling OCR", flush=True)

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
    pipeline_options.table_structure_options.do_cell_matching = True
    pipeline_options.do_ocr = do_ocr
    if do_ocr:
        pipeline_options.ocr_options = EasyOcrOptions(
            lang=["en"],
            use_gpu=args.use_gpu,
            force_full_page_ocr=False,  # per-region OCR → better cell matching
        )
    pipeline_options.generate_page_images = args.save_html
    pipeline_options.images_scale = args.images_scale

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    t0 = time.time()
    result = converter.convert(str(pdf_path))
    elapsed = time.time() - t0

    doc = result.document
    stem = pdf_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    md_path = output_dir / f"{stem}.md"
    doc.save_as_markdown(md_path)
    print(f"[done] {pdf_path.name} -> {md_path}  ({elapsed:.1f}s)", flush=True)

    if args.save_json:
        json_path = output_dir / f"{stem}.json"
        json_path.write_text(
            json.dumps(doc.export_to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"       -> {json_path}", flush=True)

    if args.save_html:
        html_path = output_dir / f"{stem}.html"
        doc.save_as_html(html_path)
        print(f"       -> {html_path}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("input", type=Path, help="PDF file or directory of PDFs")
    ap.add_argument("-o", "--output-dir", type=Path, default=Path("out"))
    ap.add_argument(
        "--force-ocr",
        action="store_true",
        help="Force OCR even when native text is present (for scanned PDFs)",
    )
    ap.add_argument("--save-json", action="store_true", help="Export structured JSON")
    ap.add_argument("--save-html", action="store_true", help="Export HTML")
    ap.add_argument(
        "--images-scale",
        type=float,
        default=2.0,
        help="Page image scale for HTML export (default: 2.0)",
    )
    ap.add_argument(
        "--use-gpu",
        action="store_true",
        default=True,
        help="Use GPU for EasyOCR when available (default: on)",
    )
    ap.add_argument("--no-gpu", dest="use_gpu", action="store_false", help="Disable GPU for OCR")
    args = ap.parse_args()

    try:
        from docling.document_converter import DocumentConverter
    except ImportError:
        print("Missing docling. Install: pip install docling", file=sys.stderr)
        return 2

    if args.input.is_dir():
        pdfs = sorted(args.input.glob("*.pdf"))
        if not pdfs:
            raise SystemExit(f"No PDFs found in {args.input}")
        print(f"[batch] {len(pdfs)} PDFs", flush=True)
        for pdf in pdfs:
            convert_one(pdf, args.output_dir / pdf.stem, args)
    else:
        if not args.input.exists():
            raise SystemExit(f"Not found: {args.input}")
        convert_one(args.input, args.output_dir, args)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
