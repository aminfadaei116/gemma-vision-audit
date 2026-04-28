# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# gemma-vision-audit

Fine-tunes Gemma 4 E2B (2B multimodal) on MagicData340K to produce natural-language
descriptions of visual artifacts in AI-generated images. The adapter is saved separately
(not merged) and loaded at inference time via PEFT. End goal: use the model as a reward
signal in a video diffusion training pipeline and publish a blog post.

---

## Hardware

| Environment | Device | dtype |
|---|---|---|
| A100 40GB (Linux/cloud VM) | `cuda` | `bfloat16` |

`scripts/utils.py::get_device()` returns `"cuda"` if available, else `"cpu"`.

---

## Pipeline Commands

Run scripts in order:

```bash
python scripts/01_download_and_sample.py        # download + stratified subset → ./data/subset_ds
python scripts/02_convert_to_vqa.py --mode rule # labels → VQA pairs → ./data/vqa_{train,val}
python scripts/03_train.py                       # LoRA fine-tune → ./outputs/...
python scripts/04_evaluate.py                   # ROUGE-L + BERTScore on val set
```

**Debug / fast pipeline validation:**
```bash
python scripts/02_convert_to_vqa.py --mode rule --debug   # 100 samples from one tar only
python scripts/03_train.py --debug                         # 100 samples, 1 epoch
python scripts/04_evaluate.py --max-samples 100
```

**LLM-based label conversion (richer descriptions, recommended for real runs):**
```bash
export OPENAI_API_KEY=sk-...
python scripts/02_convert_to_vqa.py --mode llm            # gpt-4o-mini, ~$3-5 for 20K
python scripts/02_convert_to_vqa.py --mode llm --model gpt-4o   # higher quality, ~$35
```

**Custom model:**
```bash
python scripts/03_train.py --model google/gemma-4-E2B-it --lora-rank 16 --batch-size 2
```

---

## Code Architecture

### `scripts/utils.py` — single source of truth

All shared constants live here and are imported by scripts 02–04:

| Constant | Value |
|---|---|
| `DEVICE` | `"cuda"` or `"cpu"` |
| `DTYPE` | `bfloat16` on cuda, `float32` on cpu |
| `MODEL_ID` | `"google/gemma-4-E2B-it"` |
| `ADAPTER_PATH` | `"./outputs/gemma4_e2b_artifact_assessor_lora"` (default for eval) |
| `USER_PROMPT` | Inference-time prompt asking for artifact description |
| `SYSTEM_PROMPT` | System context for LLM-based label conversion (script 02) |
| `ARTIFACT_TEMPLATES` | 10-entry dict mapping label keys → human-readable sentences |
| `LABEL_KEYS` | Ordered list of the 10 artifact label keys |

`get_model_class(model_id)` maps any HF model ID to its transformers class
(`Gemma4ForConditionalGeneration` for Gemma 3/4, `PaliGemmaForConditionalGeneration` for PaliGemma).

### Data flow

```
HF Hub (jsonl)
   ↓ script 01: load_dataset + stratified sample
data/subset_ds/            (HF arrow, ~11MB)
   ↓ script 02: _prefetch_images + rule/llm conversion
data/images/               (extracted JPGs from HF tar archives, ~5GB, image cache)
data/vqa_train/            (HF arrow with embedded images, ~4.7GB)
data/vqa_val/              (~541MB)
   ↓ script 03: SFTTrainer + LoRA
outputs/<model-slug>_artifact_assessor/      (trainer checkpoints)
outputs/<model-slug>_artifact_assessor_lora/ (final LoRA adapter, ~150MB)
   ↓ script 04: PeftModel + ROUGE-L + BERTScore
stdout metrics
```

`<model-slug>` = `model_id.replace('/', '_')`, e.g. `google_gemma-4-E2B-it` for the default model.

### Script 02 image loading

`_prefetch_images()` downloads only the HF `.tar` archives needed for the current sample subset (via `hf_hub_download`), then extracts individual images to `data/images/<part>/`. `_load_image()` reads from this cache. This means the first run downloads tars (~35 total), subsequent runs skip already-extracted images.

### Script 03 LoRA target modules

The `target_modules` regex in `LoraConfig` uses `re.fullmatch` and scopes only to the language model's linear layers:

```python
r"model\.language_model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))"
```

This avoids `Gemma4ClippableLinear` layers in the vision tower, which PEFT does not support. If you see `target_modules` warnings, run `print([n for n, _ in model.named_modules()])` to verify layer names.

### Script 03 `build_collate_fn`

The collate function formats each example into a two-turn chat (user: image + prompt, assistant: description), applies the processor's chat template, pads/truncates to `max_length=2048`, and masks pad tokens in labels (`-100`). Images are passed as `[[img], [img], ...]` (one nested list per sample) to match the processor's expected format.

---

## Dataset

**MagicData340K** (`wj-inf/MagicData340k` on HuggingFace).
Structured artifact labels across three L1 categories: Object Anatomy, Attribute, Interaction.
The HF dataset viewer shows a column mismatch error — ignore it; `load_dataset()` works fine.

Subset: ~20K samples (15K artifact / 5K clean, stratified). Script 01 derives `has_artifact`
from `response["Whether Normal"]` if the column is absent.

---

## Known Issues

| Issue | Fix |
|---|---|
| High initial loss (~13–15) | Normal Gemma 4 multimodal SFT behavior — do not stop early |
| `target_modules` PEFT warning | Run `print([n for n, _ in model.named_modules()])` to verify layer names |
| VRAM spike at batch start | Lower `max_length` in collate_fn or set `--batch-size 1` |
| Artifact column name mismatch | Inspect `dataset[0]` in script 01; pass correct name via `--artifact-col` |

---

## Model Access

```bash
huggingface-cli login
# accept license at https://huggingface.co/google/gemma-4-E2B-it
```
