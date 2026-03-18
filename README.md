# Hate Speech Detection Pipeline

Code repository for the paper:

**"Toward Generalized Cross-Lingual Hateful Language Detection with Web-Scale Data and Ensemble LLM Annotations"**
Dang H. Dang, Jelena Mitrovic, Michael Granitzer — Passau University

---

## Overview

This repository contains Jupyter notebooks implementing the full experimental pipeline described in the paper:

1. Pre-fine-tune BERT models on large-scale unlabeled web data from OpenWebSearch (OWS)
2. Use four LLMs to annotate texts with hate speech probabilities
3. Combine LLM probabilities via ensemble methods (Vote, Mean, LightGBM) to produce synthetic labels
4. Fine-tune Llama3.2-1B and Qwen2.5-14B on the synthetic and human-labeled data
5. Evaluate all models on 16 benchmark hate speech test sets across English, German, Spanish, and Vietnamese

All datasets and fine-tuned models are hosted on Hugging Face under `danghaidang-passau`.

---

## Repository Structure

```
HateLLmOWS2-main/
    Training/
        preFineBERT.ipynb       # BERT masked language modeling on OWS unlabeled data
        LightGbm_model.ipynb    # Train LightGBM meta-models for ensemble labeling
        Lora.ipynb              # LoRA fine-tuning of Llama3.2-1B and Qwen2.5-14B
    Labelling/
        LLm_Labeling.ipynb      # Generate per-LLM token probabilities for OWS corpus
        Ensemble_Labeling.ipynb # Apply Vote/Mean/LGB ensemble to produce synthetic labels
        lgb_model_label_1.pkl   # Trained LightGBM model for Hate label
        lgb_model_label_2.pkl   # Trained LightGBM model for Neutral label
    Evaluation/
        eval_LLMs.ipynb         # Evaluate all models on 16 test sets
        Bert_probs.pkl          # Pre-computed BERT probability outputs
        Llama1B_probs.pkl       # Pre-computed Llama3.2-1B probability outputs
        Qwen14_probs.pkl        # Pre-computed Qwen2.5-14B probability outputs
    requirements.txt
    README.md
```

---

## Hugging Face Resources

All datasets and models are available at `https://huggingface.co/danghaidang-passau`.

### Datasets
- `danghaidang-passau/HateOWS-dataset-LREC2026`

### Fine-tuned Models

**BERT (OWS pre-fine-tuned):**
- `danghaidang-passau/Ows4L_16` — 4-language pretrain BERT, 
- `danghaidang-passau/OwsEng`, `OwsDeu`, `OwsSpa` — single-language variants

**Llama3.2-1B variants:**
- `danghaidang-passau/Hate-Llama3.2-1B.human.2_label`
- `danghaidang-passau/Hate-Llama3.2-1B.Lgb.2_label`
- `danghaidang-passau/Hate-Llama3.2-1B.Mean.2_label`
- `danghaidang-passau/Hate-Llama3.2-1B.Vote.2_label`
- `danghaidang-passau/Hate-Llama3.2-1B.Human_Lgb.2_label`

**Qwen2.5-14B variants:**
- `danghaidang-passau/Hate-Qwen2.5-14B.Human.2_label`
- `danghaidang-passau/Hate-Qwen2.5-14B.Lgb.2_label`
- `danghaidang-passau/Hate-Qwen2.5-14B.Mean.2_label`
- `danghaidang-passau/Hate-Qwen2.5-14B.Vote.2_label`
- `danghaidang-passau/Hate-Qwen2.5-14B.Human_Lgb.2_label`

---

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

Key dependencies: `transformers`, `datasets`, `unsloth`, `trl`, `peft`, `lightgbm`, `torch`.
A CUDA-capable GPU with at least 24 GB VRAM is required for LLM inference and LoRA fine-tuning.
An A100 (80 GB) is recommended for training Qwen2.5-14B.

---

## Workflow

The notebooks should be run in the following order:

### Step 1 — Pre-Fine-Tune BERT on OWS Data

Run `Training/preFineBERT.ipynb` to perform masked language modeling on the OWS unlabeled corpora.
This produces the `OwsEng`, `OwsDeu`, `OwsSpa`, and `Ows4L` checkpoints.

### Step 2 — Generate LLM Annotations

Run `Labelling/LLm_Labeling.ipynb` to generate per-LLM probability outputs for each text in the
synthetic corpus. This step requires loading four LLMs sequentially and takes approximately 24-40 hours
total on a single A100.

### Step 3 — Train LightGBM Meta-Model (for LGB ensemble)

Run `Training/LightGbm_model.ipynb` to train the two LightGBM classifiers using the LLM probability
outputs as features and human annotations as targets. Outputs are `lgb_model_label_1.pkl` and
`lgb_model_label_2.pkl`.

### Step 4 — Ensemble Labeling

Run `Labelling/Ensemble_Labeling.ipynb` to produce Hate/Neutral labels using LightGBM, Majority Vote,
and Mean Average ensemble strategies.

### Step 5 — Fine-Tune LLMs with LoRA

Run `Training/Lora.ipynb` to fine-tune Llama3.2-1B or Qwen2.5-14B on the synthetic or human-labeled
data using LoRA adapters via Unsloth and TRL.

### Step 6 — Evaluate

Run `Evaluation/eval_LLMs.ipynb` to evaluate all fine-tuned models on the 16 test sets. Pre-computed
Run `Evaluation/eval_Bert.ipynb` to run all fine-tuned 4 OWS Bert models on different group test sets. Pre-computed
probability outputs (`.pkl` files) are provided for quick result reproduction.

---

## Pre-computed Results

The following pre-computed probability files are included in `Evaluation/` and can be loaded directly
to reproduce the reported F1 scores without running model inference:

- `Bert_probs.pkl` — OWS-BERT model outputs
- `Llama1B_probs.pkl` — All Llama3.2-1B variant outputs
- `Qwen14_probs.pkl` — All Qwen2.5-14B variant outputs

---

## Citation

If you use this code or the associated datasets and models, please cite:

```bibtex
@inproceedings{dang2026hatelLMOWS,
  title     = {Toward Generalized Cross-Lingual Hateful Language Detection
               with Web-Scale Data and Ensemble {LLM} Annotations},
  author    = {Dang, H. Dang and Mitrovi{\'c}, Jelena and Granitzer, Michael},
  booktitle = {Proceedings of LREC 2026},
  year      = {2026}
}
```
