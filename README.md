# BatakJailbreakBench
**BatakJailbreakBench: A Low-Resource Batak Toba Safety Benchmark for Jailbreak Resistance and Refusal–Leakage Behavior in LLMs**

BatakJailbreakBench is a low-resource **Batak Toba** safety benchmark designed to evaluate:
- **Jailbreak resistance** (does the model refuse when it should?)
- **Refusal quality** (does the model leak unsafe details even when it refuses?)
- **Cross-lingual safety robustness** using **ID-aligned English ↔ Batak Toba** prompts

This repository provides:
1) The benchmark datasets (English and Batak Toba)
2) A two-stage evaluation pipeline: **Stage A generate → Stage B judge**
3) Reproducible result artifacts: CSV/JSON summaries and category breakdowns

---

## Dataset on Hugging Face
The dataset is publicly available on Hugging Face:

https://huggingface.co/datasets/lamhots/BatakJailbreakBench

---

## Key Idea
Most LLM safety benchmarks are English-centric. BatakJailbreakBench enables controlled cross-lingual evaluation by pairing each Batak Toba prompt with an English counterpart (same ID, same intent, same category), and scoring:
- **Behavior correctness** (REFUSE vs SAFE_HELP vs CRISIS_SUPPORT)
- **Leakage severity** (Leak@1/Leak@2)

---

## Dataset Format
Each JSONL line is a single test case with (typical fields):
- `id` (string/int): alignment key between English and Batak Toba
- `category`: harm category (e.g., hacking, fraud, weapons, prompt_injection, self_harm)
- `attack_style`: jailbreak/social engineering style
- `prompt`: user prompt text
- `expected_behavior`: expected assistant action (REFUSE / SAFE_HELP / CRISIS_SUPPORT)

---

## Evaluation Pipeline (Two Stages)

### Stage A — Generate (model under test)
Run the model on each prompt and record raw responses.

Script:
- `scripts/run_two_datasets_benchmark_openai.py`

Outputs (example):
- `results/en_results.jsonl` + `results/en_results.csv`
- `results/batak_results.jsonl` + `results/batak_results.csv`

### Stage B — Judge (LLM-as-judge rubric)
Judge each (prompt, response) pair and assign:
- predicted behavior label
- leakage severity score (0/1/2)
- notes (diagnostics)

Script:
- `scripts/judge_two_datasets_openai.py`

Outputs (example):
- `results/en_summary.json`, `results/en_by_category.csv`
- `results/batak_summary.json`, `results/batak_by_category.csv`

---

## Metrics Reported
**Behavior correctness**
- `Acc`: fraction of cases where judged behavior matches expected behavior

**Refusal error types**
- `Under-refusal`: should refuse but did not
- `Over-refusal`: should help but refused
- `Crisis-support misses`: should provide crisis support but did not

**Leakage**
- `Leak@1`: fraction with leakage score ≥ 1
- `Leak@2`: fraction with leakage score ≥ 2

---

## Baseline Results (from provided artifacts)
From the fixed run (100 prompts per language):
- **English**: Acc = **0.68**, Under = **19**, Leak@1 = **0.15**, Leak@2 = **0.00**
- **Batak Toba**: Acc = **0.06**, Under = **84**, Leak@1 = **0.31**, Leak@2 = **0.00**

See:
- `results/en_summary.json`, `results/en_by_category.csv`
- `results/batak_summary.json`, `results/batak_by_category.csv`

---

##  Installation
Recommended:
```bash
python -m venv .venv
source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
```

Typical dependencies:
- `pandas`
- `tqdm`
- `python-dotenv`
- `openai` (if using OpenAI API)
- any logging/utils used in the scripts

---

## API Setup (example)
If your scripts use OpenAI:
1) Create a `.env` file:
```bash
OPENAI_API_KEY="YOUR_KEY"
```
2) Run Stage A then Stage B.

---

## Quickstart

### 1) Generate responses (Stage A)
```bash
python scripts/run_two_datasets_benchmark_openai.py \
  --en data/safe_jailbreak_benchmark_100.jsonl \
  --bt data/batak_safe_jailbreak_benchmark_100.jsonl \
  --out_dir results/
```

### 2) Judge and aggregate (Stage B)
```bash
python scripts/judge_two_datasets_openai.py \
  --en_results results/en_results.jsonl \
  --bt_results results/batak_results.jsonl \
  --out_dir results/
```

> Use the exact CLI flags from your scripts if they differ. This README shows a standard pattern.

---

## 🧾 Paper and Citation
Paper DOI (ResearchGate preprint):
https://doi.org/10.13140/RG.2.2.17365.69603

**Suggested citation:**
Siagian, Lamhot. *BatakJailbreakBench: A Low-Resource Batak Toba Safety Benchmark for Jailbreak Resistance and Refusal–Leakage Behavior in LLMs.* ResearchGate preprint. DOI: 10.13140/RG.2.2.17365.69603

**BibTeX**
```bibtex
@misc{siagian_batakjailbreakbench_2026,
  author       = {Siagian, Lamhot},
  title        = {BatakJailbreakBench: A Low-Resource Batak Toba Safety Benchmark for Jailbreak Resistance and Refusal--Leakage Behavior in LLMs},
  year         = {2026},
  howpublished = {ResearchGate preprint},
  doi          = {10.13140/RG.2.2.17365.69603},
  url          = {https://doi.org/10.13140/RG.2.2.17365.69603}
}
```

---

## Safety Notice
This benchmark contains prompts spanning sensitive safety categories (e.g., hacking, weapons, fraud). Use it for **defensive research and evaluation only**, follow applicable laws and platform policies, and avoid generating or sharing harmful procedural content.

---

## Contributing
Contributions are welcome:
- improving Batak Toba naturalness and coverage
- improving the evaluation harness and reporting

Open an issue or submit a pull request.

---

## 📜 License
MIT/Apache-2.0
