#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_alignment_quality.py

Diagnostic script to compare alignment quality across bias groups (pro/anti/neutral).

This script does NOT alter the main analysis workflow (RQ1–RQ3).
It only loads the Step 02 output and produces group-level summaries.

Outputs:
- CSV table with alignment statistics by bias_group
- Simple bar chart (alignment %, gender label %, avg score)
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ─────────────────────────────────────────────
# Parameters
# ─────────────────────────────────────────────
INPUT_FILE = Path("results/02_combined_gender_simalign.csv")
SUMMARY_FILE = Path("results/check_alignment_summary.csv")
PLOT_FILE = Path("results/check_alignment_summary.png")

# ─────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────
df = pd.read_csv(INPUT_FILE, encoding="utf-8")

# Check required columns exist
required_cols = {"bias_group", "translated_gender"}
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Column '{col}' is missing from input file.")

# ─────────────────────────────────────────────
# Define helper flags
# ─────────────────────────────────────────────
df["aligned_flag"] = df["translated_gender"].notna()  # aligned if gender detected
df["labeled_flag"] = df["translated_gender"].notna()  # gender label present
df["align_score"] = pd.to_numeric(df.get("align_score", pd.Series([None]*len(df))), errors="coerce")

# ─────────────────────────────────────────────
# Group-level summary
# ─────────────────────────────────────────────
summary = (
    df.groupby("bias_group")
    .agg(
        n_total=("bias_group", "size"),
        n_aligned=("aligned_flag", "sum"),
        n_labeled=("labeled_flag", "sum"),
        mean_score=("align_score", "mean")
    )
    .reset_index()
)

summary["aligned_pct"] = (summary["n_aligned"] / summary["n_total"] * 100).round(1)
summary["labeled_pct"] = (summary["n_labeled"] / summary["n_total"] * 100).round(1)

# Save summary table
SUMMARY_FILE.parent.mkdir(parents=True, exist_ok=True)
summary.to_csv(SUMMARY_FILE, index=False, encoding="utf-8")
print("✓ Summary saved:", SUMMARY_FILE)

# ─────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────
fig, ax = plt.subplots(1, 2, figsize=(10, 4))

# Bar plot for alignment & labeling %
summary.plot(
    x="bias_group", y=["aligned_pct", "labeled_pct"],
    kind="bar", ax=ax[0], rot=0, color=["steelblue", "darkorange"]
)
ax[0].set_ylabel("%")
ax[0].set_title("Alignment and Label Coverage")

# Bar plot for avg alignment score
summary.plot(
    x="bias_group", y="mean_score",
    kind="bar", ax=ax[1], rot=0, color="seagreen", legend=False
)
ax[1].set_ylabel("Average alignment score")
ax[1].set_title("Alignment Score by Group")

plt.tight_layout()
plt.savefig(PLOT_FILE, dpi=150)
print("✓ Plot saved:", PLOT_FILE)

# ─────────────────────────────────────────────
# Console preview
# ─────────────────────────────────────────────
print("\n--- Alignment quality by bias_group ---")
print(summary.to_string(index=False))
