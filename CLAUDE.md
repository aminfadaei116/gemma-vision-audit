# gemma-vision-audit

Fine-tunes Gemma 4 E2B (2B multimodal) on MagicData340K to produce natural-language
descriptions of visual artifacts in AI-generated images.  The adapter is saved separately
(not merged) and loaded at inference time via PEFT.  End goal: use the model as a reward
signal in a video diffusion training pipeline and publish a blog post.

---

## Hardware

| Environment | Device | dtype |
|---|---|---|
| A100 40GB (Linux/cloud VM) | `cuda` | `bfloat16` |

`scripts/utils.py::get_device()` auto-detects the environment.

---

## Model

`google/gemma-4-E2B-it` — smallest Gemma 4 vision model (~2B effective parameters, Apache 2.0).
Requires HuggingFace login and accepting the Gemma 4 license:

```bash
huggingface-cli login
# accept license at https://huggingface.co/google/gemma-4-E2B-it
```

---

## Dataset

**MagicData340K** (`wj-inf/MagicData340k` on HuggingFace).  
Structured artifact labels across three L1 categories: Object Anatomy, Attribute, Interaction.  
The HF dataset viewer shows a column mismatch error — ignore it; `load_dataset()` works fine.

Subset used for training: ~20K samples (15K artifact / 5K clean, stratified).

---

## Script Execution Order

```
python scripts/01_download_and_sample.py        # download + stratified subset → ./data/subset_ds
python scripts/02_convert_to_vqa.py --mode rule # labels → VQA pairs → ./data/vqa_{train,val}
python scripts/03_train.py                       # LoRA fine-tune → ./outputs/...
python scripts/04_evaluate.py                    # ROUGE-L + BERTScore on val set
```

**Debug / pipeline validation:**
```bash
python scripts/02_convert_to_vqa.py --mode rule
python scripts/03_train.py --debug    # 100 samples, 1 epoch, fast feedback
```

**LLM-based label conversion (recommended for the real training run):**
```bash
export OPENAI_API_KEY=sk-...
python scripts/02_convert_to_vqa.py --mode llm            # gpt-4o-mini, ~$3-5 for 20K
python scripts/02_convert_to_vqa.py --mode llm --model gpt-4o   # higher quality, ~$35
```

---

## Key Paths

| Path | Contents |
|---|---|
| `scripts/utils.py` | Shared constants: `DEVICE`, `DTYPE`, `MODEL_ID`, `ADAPTER_PATH`, `USER_PROMPT`, `ARTIFACT_TEMPLATES` |
| `data/subset_ds/` | Raw stratified subset (HF arrow format) |
| `data/vqa_train/` | Training set in VQA message format |
| `data/vqa_val/` | Validation set in VQA message format |
| `outputs/gemma4_e2b_artifact_assessor/` | Trainer checkpoints |
| `outputs/gemma4_e2b_artifact_assessor_lora/` | Final LoRA adapter (~150MB) |
| `notebooks/qualitative_analysis.ipynb` | Base vs fine-tuned side-by-side comparison |

---

## Known Issues

| Issue | Fix |
|---|---|
| High initial loss (~13–15) | Normal Gemma 4 multimodal SFT behavior — do not stop early |
| `target_modules` PEFT warning | Run `print([n for n, _ in model.named_modules()])` to verify exact layer names |
| Images as URLs in dataset | Normalize with `Image.open(BytesIO(requests.get(url).content))` before processing |
| VRAM spike at batch start | Lower `max_length` to 1024 or set `--batch-size 1` |
| Actual artifact column name | Inspect `dataset[0]` in script 01; pass correct name via `--artifact-col` |

---

## Dependencies

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets accelerate peft trl
pip install pillow tqdm pandas rouge-score bert-score
pip install openai          # only needed for --mode llm in script 02
pip install jupyter ipywidgets matplotlib   # only needed for the notebook
```
