"""
Granite Docling 258M vLLM OCR runner.

PDF -> rasterized pages -> batched vLLM inference -> DocTags -> Markdown.
Layout-preserving order comes from DocTags: the model emits elements in reading
order with location tags, and DoclingDocument serialization honors that order.

Usage:
    python run_ocr.py input.pdf -o out/
    python run_ocr.py input.pdf -o out/ --dpi 200 --batch-size 4
    python run_ocr.py page_dir/ -o out/   # directory of images also accepted

Requires a CUDA (or ROCm) host; vLLM does not build on Intel macOS.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Iterable

from PIL import Image


DEFAULT_MODEL = "ibm-granite/granite-docling-258M"
DEFAULT_REVISION = "untied"  # untied weights — required by vLLM >= 0.10.2
DEFAULT_PROMPT = "Convert this page to docling."
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}


def remove_highlights(img: Image.Image) -> Image.Image:
    """Replace yellow/colored highlight pixels with white.

    The yellow rows in hospital invoices cause granite-docling to hallucinate —
    it detects the table bounding box but generates empty OTSL content then
    falls into a repetition loop. Neutralising the colour before inference
    restores normal table decoding.
    """
    import numpy as np
    arr = np.array(img.convert("RGB"), dtype=np.int32)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    # Yellow highlight: R and G both high, B meaningfully lower than both
    yellow = (r > 190) & (g > 180) & (b < 210) & ((r - b) > 25) & ((g - b) > 15)
    arr[yellow] = [255, 255, 255]
    return Image.fromarray(arr.astype("uint8"))


def load_pages(source: Path, dpi: int) -> list[tuple[str, Image.Image]]:
    """Return a list of (page_id, RGB PIL.Image) for a PDF, image, or directory."""
    if source.is_dir():
        files = sorted(p for p in source.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        if not files:
            raise SystemExit(f"No images found in {source}")
        return [(p.stem, Image.open(p).convert("RGB")) for p in files]

    if source.suffix.lower() == ".pdf":
        from pdf2image import convert_from_path

        images = convert_from_path(str(source), dpi=dpi, fmt="png")
        stem = source.stem
        return [(f"{stem}_p{i + 1:04d}", img.convert("RGB")) for i, img in enumerate(images)]

    if source.suffix.lower() in IMAGE_EXTS:
        return [(source.stem, Image.open(source).convert("RGB"))]

    raise SystemExit(f"Unsupported input: {source}")


def build_inputs(pages: Iterable[tuple[str, Image.Image]], processor, prompt_text: str):
    messages = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": prompt_text}],
        }
    ]
    chat_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    batched = []
    for _pid, image in pages:
        batched.append({"prompt": chat_prompt, "multi_modal_data": {"image": image}})
    return batched


def chunked(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path, help="Path to a PDF, image, or directory of images")
    ap.add_argument("-o", "--output-dir", type=Path, default=Path("out"))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--revision", default=DEFAULT_REVISION, help="Model revision/branch (default: untied)")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--dpi", type=int, default=150, help="PDF rasterization DPI (default: 150)")
    ap.add_argument("--batch-size", type=int, default=8, help="Pages per vLLM.generate call")
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--dtype", default="auto", choices=["auto", "bfloat16", "float16", "float32"])
    ap.add_argument("--gpu-mem", type=float, default=0.90, help="vLLM gpu_memory_utilization")
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--keep-doctags", action="store_true", default=True, help="Write raw .dt files (default: on)")
    ap.add_argument("--no-doctags", dest="keep_doctags", action="store_false", help="Suppress .dt files")
    ap.add_argument("--save-html", action="store_true", help="Also export HTML")
    ap.add_argument("--save-json", action="store_true", help="Also export structured JSON")
    ap.add_argument(
        "--remove-highlights",
        action="store_true",
        default=False,
        help="Neutralise yellow/coloured highlight pixels to white before inference",
    )
    args = ap.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input not found: {args.input}")

    try:
        from vllm import LLM, SamplingParams
        from transformers import AutoProcessor
        from docling_core.types.doc import DoclingDocument
        from docling_core.types.doc.document import DocTagsDocument
    except ImportError as e:
        print(f"Missing dependency: {e}\nInstall with: pip install -r requirements.txt", file=sys.stderr)
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    pages = load_pages(args.input, dpi=args.dpi)
    if args.remove_highlights:
        pages = [(pid, remove_highlights(img)) for pid, img in pages]
        print(f"[preprocess] highlight removal applied to {len(pages)} page(s)", flush=True)
    print(f"[load] {len(pages)} page(s) in {time.time() - t0:.2f}s", flush=True)

    t0 = time.time()
    llm = LLM(
        model=args.model,
        revision=args.revision,
        dtype=args.dtype,
        limit_mm_per_prompt={"image": 1},
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=args.max_model_len,
    )
    processor = AutoProcessor.from_pretrained(args.model, revision=args.revision)
    print(f"[load-model] {time.time() - t0:.2f}s", flush=True)

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        skip_special_tokens=False,
    )

    t_infer = 0.0
    for batch in chunked(pages, args.batch_size):
        inputs = build_inputs(batch, processor, args.prompt)
        t0 = time.time()
        outputs = llm.generate(inputs, sampling_params=sampling_params)
        t_infer += time.time() - t0

        for (page_id, image), out in zip(batch, outputs):
            result = out.outputs[0]
            doctags = result.text
            finish = result.finish_reason
            print(
                f"[page] {page_id}  finish={finish}  tokens={len(result.token_ids)}  "
                f"doctags_preview={doctags[:120].replace(chr(10), ' ')!r}",
                flush=True,
            )
            if args.keep_doctags:
                (args.output_dir / f"{page_id}.dt").write_text(doctags, encoding="utf-8")

            dt_doc = DocTagsDocument.from_doctags_and_image_pairs([doctags], [image])
            doc = DoclingDocument.load_from_doctags(dt_doc, document_name=page_id)
            doc.save_as_markdown(args.output_dir / f"{page_id}.md")
            if args.save_html:
                doc.save_as_html(args.output_dir / f"{page_id}.html")
            if args.save_json:
                import json
                (args.output_dir / f"{page_id}.json").write_text(
                    json.dumps(doc.export_to_dict(), indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )

    # Combined document across all pages (preserves reading order per page).
    combined_name = args.input.stem if args.input.is_file() else args.input.name
    combined_md = args.output_dir / f"{combined_name}.md"
    with combined_md.open("w", encoding="utf-8") as f:
        for page_id, _ in pages:
            f.write(f"\n\n<!-- page: {page_id} -->\n\n")
            f.write((args.output_dir / f"{page_id}.md").read_text(encoding="utf-8"))

    print(
        f"[done] pages={len(pages)} infer={t_infer:.2f}s "
        f"avg={t_infer / max(1, len(pages)):.2f}s/page -> {combined_md}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
