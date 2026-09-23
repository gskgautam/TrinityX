# TrinityX 

(https://huggingface.co/impressive-east579/TrinityX)
---

## 📊 Datasets

AlignX uses curated datasets for each alignment axis:

| Alignment Axis | Dataset | Description |
|----------------|---------|-------------|
| **Helpfulness** | [Alpaca](https://github.com/tatsu-lab/stanford_alpaca) | 20k instruction-response pairs |
| **Harmlessness** | [BeaverTails](https://sites.google.com/view/pku-beavertails) | 30k safety-annotated QA pairs |
| **Honesty** | [TruthfulQA](https://github.com/sylinrl/TruthfulQA) | Benchmark for truthful answering |

---

## 📈 Evaluation Metrics

| Axis         | Metric                           | Description |
|--------------|----------------------------------|-------------|
| Helpfulness  | **Win Rate (↑)**                | % of samples where AlignX wins over baseline |
| Harmlessness | **Safety Score (↓)**            | % of unsafe outputs (lower is better) |
| Honesty      | **Truthful & Informative (TI ↑)** | Product of truthfulness and informativeness |
| Overall      | **Average Alignment Score (↑)** | Normalized combination of the above metrics |

**↑**: Higher is better  **↓**: Lower is better

### 📈 Evaluation (https://github.com/git-disl/h3fusion/tree/main/evaluator)

> ⚠️ Note: Make sure you have the appropriate access to the moderation models used for evaluation. These include:

- GPT-4.0 (via OpenAI API)
- beaver-dam-7b — available here: [PKU-Alignment/beaver-dam-7b](https://huggingface.co/PKU-Alignment/beaver-dam-7b)
- GPT-4.0 (via OpenAI API)

🖥️ Note on Performance Variability:
Evaluation results may vary by up to ±10% depending on your hardware configuration, especially GPU type, memory bandwidth, and compute 
