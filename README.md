# gemma-vision-audit

Fine-tunes **Gemma 4 E2B** on MagicData340K to produce natural-language descriptions of
visual artifacts in AI-generated images (anatomy errors, texture artifacts, spatial violations).
The LoRA adapter is saved separately and loaded at inference time.

---

## Environment Setup

### Prerequisites

- Python 3.10+
- [conda](https://docs.conda.io/en/latest/miniconda.html) (recommended) or a plain virtualenv
- A HuggingFace account with the [Gemma 4 license accepted](https://huggingface.co/google/gemma-4-E2B-it)

---

### Linux / Cloud VM (CUDA)

```bash
conda create -n gemma4_finetune python=3.10 -y
conda activate gemma4_finetune

pip install -r requirements.txt
```

---

### HuggingFace Authentication

```bash
huggingface-cli login
```

Paste your HF token when prompted. The token needs read access to gated models.
Accept the Gemma 4 license at <https://huggingface.co/google/gemma-4-E2B-it> if you
haven't already — the download will fail without it.

---

### OpenAI API Key (optional)

Only required for `--mode llm` in `scripts/02_convert_to_vqa.py`:

```bash
export OPENAI_API_KEY=sk-...
```

---

## Running the Pipeline

```bash
# 1. Download MagicData340K and create a stratified subset
python scripts/01_download_and_sample.py

# 2. Convert structured labels to VQA instruction-response pairs
python scripts/02_convert_to_vqa.py --mode rule        # free, instant
python scripts/02_convert_to_vqa.py --mode llm         # GPT-4o-mini, ~$3-5 for 20K samples

# 3. Fine-tune with LoRA
python scripts/03_train.py

# 4. Evaluate on the validation set
python scripts/04_evaluate.py
```

**Debug pass (fast validation before committing to a full run):**

```bash
python scripts/02_convert_to_vqa.py --mode rule --debug
python scripts/03_train.py --debug    # 100 samples, 1 epoch
python scripts/04_evaluate.py --max-samples 100
```

---

## Key Flags

| Script | Flag | Default | Description |
|---|---|---|---|
| `01` | `--n-artifact` | 15000 | Artifact samples to include |
| `01` | `--n-clean` | 5000 | Clean samples to include |
| `02` | `--mode` | `rule` | `rule` (free) or `llm` (requires OpenAI key) |
| `02` | `--model` | `gpt-4o-mini` | OpenAI model for `--mode llm` |
| `02` | `--max-samples` | all | Cap samples processed |
| `03` | `--model` | `google/gemma-4-E2B-it` | HuggingFace model ID |
| `03` | `--lora-rank` | 16 | LoRA rank (8=debug, 32=if underfitting) |
| `03` | `--batch-size` | 2 | Per-device batch size |
| `03` | `--debug` | off | 100 samples, 1 epoch |
| `04` | `--adapter-path` | auto | Path to LoRA adapter dir |
| `04` | `--max-samples` | all | Limit eval to N samples |

---

## Project Structure

```
gemma-vision-audit/
├── scripts/
│   ├── utils.py                    shared constants and device detection
│   ├── 01_download_and_sample.py
│   ├── 02_convert_to_vqa.py
│   ├── 03_train.py
│   └── 04_evaluate.py
├── notebooks/
│   └── qualitative_analysis.ipynb  base vs fine-tuned side-by-side comparison
├── data/                           created at runtime by scripts 01-02
│   ├── images/                     image cache (downloaded by script 02)
│   ├── subset_ds/                  stratified subset in HF arrow format
│   ├── vqa_train/                  training split in VQA format
│   └── vqa_val/                    validation split in VQA format
├── outputs/                        created at runtime by script 03
│   ├── <model-slug>_artifact_assessor/       trainer checkpoints
│   └── <model-slug>_artifact_assessor_lora/  final LoRA adapter (~150MB)
├── requirements.txt
└── CLAUDE.md                       full technical reference for this project
```
