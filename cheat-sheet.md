# Granite Docling 258M — vLLM Cheat Sheet

Performance, accuracy, and ops tweaks specifically for
`ibm-granite/granite-docling-258M` served through vLLM. Pair with `run_ocr.py`.

---

## 1. Model & branch

- **Always use `--revision untied`** (or `revision="untied"` in Python). vLLM
  ≤ 0.10.2 has broken support for tied embeddings, which is how the default
  branch ships. Using `main` silently corrupts generation.
- 258M params. Entire model + KV cache fits inside ~1–2 GB VRAM at bf16; you
  can run many replicas per GPU or crank batch size aggressively.

## 2. Hardware / build matrix

| Host                     | Works? | Notes                                                   |
| ------------------------ | ------ | ------------------------------------------------------- |
| Linux + CUDA (Ampere+)   | Yes    | Reference path. bf16 default, ~0.35 s/page on A100.     |
| Linux + CUDA (T4/V100)   | Yes    | No bf16 — pass `--dtype float32` or `float16`.          |
| Linux + ROCm             | Yes    | Install vLLM's ROCm wheel; same flags.                  |
| Apple Silicon (M-series) | No     | vLLM has no MPS backend. Use `transformers` + MLX path. |
| Intel macOS              | No     | vLLM pins a CUDA torch wheel. Not buildable.            |

## 3. First-run setup (Linux GPU host)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Pre-download once so the server doesn't race on first request:
hf download ibm-granite/granite-docling-258M --revision untied \
    --local-dir ./models/granite-docling-258M-untied
```

## 4. Throughput tuning

The model is tiny; the bottleneck is almost always image preprocessing or
decode-side tail latency, not matmul.

- **Batch pages, not documents.** `llm.generate(list_of_inputs)` is much faster
  than one call per page because vLLM continuously batches decodes.
- **`gpu_memory_utilization=0.90`** is fine here — there is no large weight
  footprint to leave room for. Go higher (0.95) if the only workload is Granite
  Docling.
- **`max_model_len=8192`** matches the model's typical output budget. Lowering
  it shrinks the KV cache and lets vLLM schedule more concurrent pages.
- **`max_num_seqs`** (server flag): raise to 32–64 for dense batch workloads.
- **`enforce_eager=False`** (default). Keep CUDA graphs on; they help a lot at
  this model size.
- **Pre-rasterize PDFs in a worker pool.** `pdf2image.convert_from_path` is
  single-threaded per call; for big PDFs, shard pages across processes.
- **DPI is the single biggest knob.** 150 DPI is a good default. 200 DPI gives
  a noticeable accuracy bump on small fonts; 300 DPI roughly doubles input
  tokens with little extra accuracy. Never go below 120.

## 5. Accuracy tuning

- **Temperature 0.0** — this is a structured-output task. Any sampling
  introduces DocTag corruption (unclosed tags, drifted coordinates).
- **`skip_special_tokens=False`** — DocTags *are* special tokens. Stripping
  them destroys the output.
- **`max_tokens=8192`** covers dense pages. Watch for `finish_reason="length"`
  — that means the page got truncated; re-run with a higher budget or crop
  the page.
- **Task-specific prompts** (routes the model to a specialized head):
  | Goal              | Prompt                            |
  | ----------------- | --------------------------------- |
  | Full page         | `Convert this page to docling.`   |
  | Tables            | `Convert table to OTSL.`          |
  | Formulas          | `Convert formula to LaTeX.`       |
  | Code blocks       | `Convert code to text.`           |
  | Charts            | `Convert chart to table.`         |
- **Reading order is emitted by the model.** DocTags embed `<loc_x1><loc_y1>…`
  anchors. Do not re-sort elements yourself; `DoclingDocument.load_from_doctags`
  already lays them out in natural reading order (including multi-column).
- **One image per prompt.** `limit_mm_per_prompt={"image": 1}` is not a limit
  to fight — the model was not trained on multi-image inputs.

## 6. Serving vs. offline

**Offline batch** (this repo): `LLM(...).generate(batch)`. Best for bulk PDF
ingestion.

**Online server**:
```bash
vllm serve ibm-granite/granite-docling-258M \
    --revision untied \
    --limit-mm-per-prompt image=1 \
    --max-model-len 8192 \
    --max-num-seqs 64
```
Call it with the OpenAI-compatible `/v1/chat/completions` endpoint, image as a
`image_url` content part. Keep `temperature=0`.

## 7. Post-processing

- **DocTags -> Markdown/HTML/JSON** via `docling-core`:
  ```python
  dt = DocTagsDocument.from_doctags_and_image_pairs([doctags], [image])
  doc = DoclingDocument.load_from_doctags(dt, document_name="p1")
  doc.save_as_markdown("p1.md")
  doc.save_as_html("p1.html")
  doc.export_to_dict()  # structured JSON for downstream RAG
  ```
- **Keep the raw `.dt` file** during evaluation (`--keep-doctags`). If a
  markdown render looks wrong, 90% of the time the issue is in DocTags parsing,
  not the model.
- **Per-page concat beats whole-doc prompting.** The model is page-scoped; do
  not try to feed multi-page montages.

## 8. Common failure modes

| Symptom                                         | Cause / Fix                                              |
| ----------------------------------------------- | -------------------------------------------------------- |
| Garbled tokens, never terminates                | Forgot `revision="untied"` on vLLM ≥ 0.10.                |
| `finish_reason=length`, truncated markdown      | Raise `max_tokens`; page likely very dense.              |
| Empty markdown but non-empty `.dt`              | `docling-core` version mismatch; upgrade to ≥ 2.25.      |
| Tables render as flat text                      | Use the `Convert table to OTSL.` prompt for that region. |
| Columns interleave                              | DPI too low; bump to 200 and re-run.                     |
| CUDA OOM at start                               | Lower `gpu_memory_utilization` or `max_model_len`.       |
| `Unsupported dtype bfloat16`                    | Older GPU — pass `--dtype float16` or `float32`.         |

## 9. Evaluation tips

- Build a small ground-truth set per document *class* (forms, invoices,
  scientific papers, slides). A model that wins on papers can lose on forms.
- Score text with **CER/WER** after normalizing whitespace, and score layout
  with **TEDS** on table regions — DocTags -> OTSL lets you compute TEDS
  directly.
- Always benchmark **end-to-end wall time**, not just `llm.generate`. PDF
  rasterization at high DPI often dominates on small docs.
- Warm up with one throwaway page before timing — first call pays CUDA graph
  capture cost.

## 10. Files in this repo

- `run_ocr.py` — CLI: PDF/image/dir -> `.md` (+ optional `.dt`, `.html`).
- `requirements.txt` — pinned minimums.
- `models/granite-docling-258M-untied/` — local snapshot of the untied branch.
