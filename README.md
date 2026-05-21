# WinoTR: Evaluating Gender Bias in Machine Translation from a Gender-Neutral Language Using Causal Inference

This repository contains the code and data for the paper **"WinoTR: Evaluating Gender Bias in Machine Translation from a Gender-Neutral Language Using Causal Inference"**, accepted at the GITT Workshop @ EAMT 2026. The project adapts the **WinoMT paradigm** (Stanovsky et al., 2019) to Turkish and applies **Double Machine Learning (DML)** to quantify causal effects of gender cues on stereotype-consistent machine translation output.

---

## Data

- `data/aggregates/`
  - `tr_neutral.txt`, `tr_pro.txt`, `tr_anti.txt`: Source sentences grouped by bias condition (neutral, pro-stereotypical, anti-stereotypical).
  - `combined_dataset.txt`: Unified dataset of all conditions.
  - `tr_rawdataset.txt`: Original raw dataset before preprocessing.

---

## Data Preparation

- `datapreparation/deepltranslate.py` – Translates sentences using DeepL API (June 2025).
- `datapreparation/google_translate_scr.py` – Translates sentences using Google Cloud Translation API (May 2025).
- `datapreparation/openai_translate_o3.py` – Translates sentences using OpenAI API (July 2025).
- `datapreparation/missing_sentence.py` – Handles incomplete translations.

---

## Alignment and Feature Extraction

- `winotrscripts/01_build_dataframe.py` – Combines all translations into a structured dataframe.
- `winotrscripts/02_gender_extraction.py` – Extracts gender labels via SimAlign and Awesome-Align; fallback heuristics and morphological rules were added to handle Turkish-specific alignment challenges.
- `manual_check.csv` – Contains a subset of human-validated alignments for quality control.

---

## Analysis (RQ1–RQ3)

- `results/rq1/` – Neutral baseline analysis (association level).
- `results/rq2/` – Pro vs. Anti signal analysis with DoubleML (intervention level).
- `results/rq3/` – Signal vs. Neutral comparison with DoubleML (second intervention level).

The analysis pipeline applies **Pearl's causal hierarchy** (Association, Intervention, Intervention) and the **potential outcomes framework** (Imbens and Rubin, 2015).

---

## Key Results

| | ATE | SE | p | Interpretation |
|---|---|---|---|---|
| RQ1 | — | — | — | Stereotype support: 62.9% (neutral baseline) |
| RQ2 | +0.245 | 0.009 | <0.001 | Cue direction has large causal effect |
| RQ3 | +0.005 | 0.007 | 0.523 | Cue presence has no significant effect |

---

## How to Run

1. Prepare raw Turkish sentences in `data/aggregates/`.
2. Run translation scripts in `datapreparation/` to obtain MT outputs from DeepL, Google, and OpenAI.
3. Execute `winotrscripts/01_build_dataframe.py` to combine all translations into a single CSV.
4. Run `winotrscripts/02_gender_extraction.py` to align professions with gendered tokens and assign gender labels.
5. Use analysis scripts in `results/rq1`, `results/rq2`, and `results/rq3` to compute descriptive statistics and causal estimates (ATE) and subgroup accuracy comparisons.
6. Final outputs (tables, logs, figures) are saved under `results/`.

---

## Citation

If you use WinoTR, please cite:

@inproceedings{albayrak2026winotr,
title={WinoTR: Evaluating Gender Bias in Machine Translation
from a Gender-Neutral Language Using Causal Inference},
author={Albayrak, Deniz},
booktitle={Proceedings of the GITT Workshop @ EAMT 2026},
year={2026}
}


---

## References

- Stanovsky, G., et al. (2019). *Evaluating Gender Bias in Machine Translation*. ACL.
- Chernozhukov, V., et al. (2018). *Double/Debiased Machine Learning for Treatment and Structural Parameters*. Econometrics Journal.
- Bach, P., et al. (2024). *DoubleML: An Object-Oriented Implementation of Double Machine Learning in Python*. JMLR.
- Pearl, J. and Mackenzie, D. (2018). *The Book of Why*. Basic Books.
- Imbens, G.W. and Rubin, D.B. (2015). *Causal Inference for Statistics, Social, and Biomedical Sciences*. Cambridge University Press.
