# Granite Docling Eval — Onboarding TODO

You're going to evaluate IBM's Granite Docling 258M (a small VLM that converts
PDFs/scans to layout-preserving markdown) on a set of test documents using a
GPU pod on RunPod. Walk through this in order. Tick each box.

---

## 0. Prereqs (one-time, on your laptop)

- [ ] Get a RunPod account, log in to https://runpod.io.
- [ ] Add a **Pay-as-you-go credit** ($10 is enough to start).
- [ ] Generate an SSH key on your laptop if you don't have one:
      `ssh-keygen -t ed25519 -C "your.email@zessta.com"` (accept defaults).
- [ ] Copy the **public** key: `cat ~/.ssh/id_ed25519.pub`.
- [ ] In RunPod: **Settings → SSH Public Keys** → paste it, save.
- [ ] Get repo access: https://github.com/zessta/granite-docking-test.
- [ ] Read [`cheat-sheet.md`](./cheat-sheet.md) end-to-end (10 min). Skim
      [`run_ocr.py`](./run_ocr.py) so you know what flags exist.

## 1. Create a persistent Network Volume (one-time)

Without this, the 256 MB model + your test data get wiped every time a pod
stops. With it, those files survive forever.

- [ ] RunPod → **Storage → Network Volumes → New Volume**.
- [ ] Region: pick one that has the GPU you want (e.g. `EU-RO-1`).
- [ ] Size: **50 GB** (plenty for the model, repo, test PDFs, outputs).
- [ ] Name it something obvious like `granite-docling-vol`.
- [ ] Note the **region** — every pod that uses this volume must be in the
      same region.

## 2. Spin up a pod (each session)

- [ ] RunPod → **Pods → Deploy**.
- [ ] **Region**: same as your volume.
- [ ] **Network Volume**: attach `granite-docling-vol` (it auto-mounts at
      `/workspace`).
- [ ] **GPU**: anything Ampere or newer is fine — `RTX 4000 Ada`, `RTX A5000`,
      or `RTX 3090` are cheap. The model is tiny (258M params), so don't pay
      for an A100.
- [ ] **Template**: pick `vllm/vllm-openai:latest` (or any community template
      with "vllm" in the name). This means torch + CUDA + vllm are
      pre-installed — saves ~10 minutes per pod.
- [ ] **Expose TCP Port 22** — needed for VS Code Remote-SSH and `scp`.
- [ ] Deploy. Wait ~30 s for the pod to be `Running`.

## 3. Connect to the pod

- [ ] Click **Connect** on the pod card. You'll see two SSH commands:
      - "SSH over exposed TCP" → `ssh root@<ip> -p <port> ...` ← **use this one**
      - "Basic SSH" → `ssh ...@ssh.runpod.io ...` ← proxy, no SFTP, fallback only
- [ ] Verify it works from your laptop terminal:
      `ssh root@<ip> -p <port> -i ~/.ssh/id_ed25519 'nvidia-smi'`
      You should see one GPU listed.
- [ ] (Optional) Add to `~/.ssh/config` for VS Code Remote-SSH:
      ```
      Host runpod-granite
          HostName <ip>
          Port <port>
          User root
          IdentityFile ~/.ssh/id_ed25519
          IdentitiesOnly yes
      ```
      Then in VS Code: Remote Explorer → SSH → `runpod-granite` → Connect.

## 4. First-time setup on the volume (one-time per volume)

Run these **on the pod** (over SSH):

- [ ] `cd /workspace`
- [ ] `git clone https://github.com/zessta/granite-docking-test.git`
- [ ] `cd granite-docking-test`
- [ ] `bash setup.sh` — installs poppler, pip deps, downloads the model,
      runs a synthetic end-to-end check. Should end with
      `[check] PASS in <N>s`.
- [ ] If the synthetic check passes, you're done with setup. The model lives
      at `/workspace/granite-docking-test/models/granite-docling-258M-untied`
      and will be there next time.

## 5. Test documents — already on the pod

After step 4's `git clone` (or any subsequent `git pull`), you already
have **20 sample PDFs** at `/workspace/granite-docking-test/test_set_new/`.
They're medical forms with realistic OCR-stress artifacts — punch holes,
JPEG noise, rubber stamps, highlighter marks, faded toner. Use these for
your first run; **no upload needed**.

- [ ] Confirm they're there:
      ```bash
      ls /workspace/granite-docking-test/test_set_new | wc -l   # should print 20
      ```

To test on **your own** documents later, scp them up from your laptop
(use the **direct-TCP** ssh details, not the proxy):
```bash
scp -O -P <port> -i ~/.ssh/id_ed25519 -r \
    ~/Downloads/your_docs \
    root@<ip>:/workspace/
```
`-O` forces the legacy SCP protocol (newer macOS scp defaults to SFTP
which trips on some RunPod proxies).

## 6. Run the OCR

Back on the **pod**:

- [ ] Single PDF (using a bundled sample):
      ```bash
      cd /workspace/granite-docking-test
      python run_ocr.py test_set_new/anson_bay_P005_v6_punch_holes.pdf \
          -o out/anson_bay/ \
          --model models/granite-docling-258M-untied
      ```
- [ ] Whole bundled test set:
      ```bash
      cd /workspace/granite-docking-test
      for f in test_set_new/*.pdf; do
        python run_ocr.py "$f" -o "out/$(basename "$f" .pdf)/" \
            --model models/granite-docling-258M-untied
      done
      ```
- [ ] Outputs in `out/<doc>/`:
      - `<doc>_p0001.md`, `<doc>_p0002.md`, … one per page
      - `<doc>.md` — concatenated full-document markdown
      - Add `--keep-doctags` for raw `.dt` files, `--save-html` for HTML.

## 7. Pull results back to your laptop

From your **laptop**:

- [ ] ```bash
      scp -O -P <port> -i ~/.ssh/id_ed25519 -r \
          root@<ip>:/workspace/granite-docking-test/out \
          ~/Downloads/granite-out
      ```

## 8. Evaluate quality

- [ ] Eyeball ~5 outputs side-by-side with the source PDFs. Check:
      - reading order (multi-column docs are the hard case),
      - tables (cells aligned? rows/columns intact?),
      - headings vs body (markdown `#` levels right?),
      - any garbled or repeating text (sign of `max_tokens` truncation).
- [ ] If a page truncates: re-run that PDF with `--max-tokens 16384`.
- [ ] If columns interleave: re-run with `--dpi 200`.
- [ ] Log findings (which doc class, what failed, what flag fixed it) in a
      shared sheet so we know where the model holds up and where it doesn't.

## 9. Stop the pod when you're done

- [ ] RunPod → Pod card → **Stop** (you keep paying GPU $/hr while it runs!).
- [ ] The volume keeps your repo + model + outputs. Next session, deploy a
      new pod with the same volume — skip steps 1 and 4.

## 10. Common gotchas (read once)

- **Anything outside `/workspace` is wiped on stop.** Save outputs there.
- **CUDA is part of the pod template** — never `apt install cuda` yourself.
- **Always use `temperature=0.0`** for this model (already set in
  `run_ocr.py`). Sampling corrupts the structured DocTags output.
- **Use the `untied` revision** of the model. The default branch breaks
  vLLM. (Already set in `run_ocr.py`.)
- **VS Code Remote-SSH needs direct TCP**, not the `ssh.runpod.io` proxy
  (the proxy doesn't support SFTP).
- **Forgot your SSH key on a new pod?** Public keys are only injected at
  pod *creation* time. Stop and start (or recreate) after adding a key.

## 11. When you're stuck

- [ ] Read [`cheat-sheet.md`](./cheat-sheet.md) — it has a failure-mode
      table (truncated output, columns interleaved, OOM, etc.) with the
      flag that fixes each one.
- [ ] Check `git log` on this repo — recent commits explain why settings
      are what they are.
- [ ] Ping me on Slack with: the exact command you ran, the full output,
      and the input PDF (or a page screenshot).
