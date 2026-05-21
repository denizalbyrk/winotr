#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 03_RQ1_neutral.py – RQ1 neutral baseline with tables + figures (weighted summaries, robust TR normalization + diagnostics)

from pathlib import Path
import re  # ← CHANGE: needed for punctuation stripping in norm_prof
import numpy as np
import pandas as pd
import unicodedata as ud
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -------- paths --------
ROOT   = Path(__file__).resolve().parents[1]
IN_CSV = ROOT / "results" / "02_combined_gender_simalign.csv"
OUTDIR = ROOT / "results" / "rq1"
OUTDIR.mkdir(parents=True, exist_ok=True)

# -------- stereotype lists (design) --------
MALE_PROFESSIONS = [
    "geliştirici","tamirci","nakliyeci","analist","satış görevlisi","avukat",
    "aşçı","doktor","çiftçi","CEO","yönetici","sürücü","güvenlik görevlisi",
    "işçi","inşaat işçisi","marangoz","kapıcı","amir","şerif","şef"
]
FEMALE_PROFESSIONS = [
    "hemşire","sekreter","öğretmen","tasarımcı","hizmetli","memur",
    "asistan","kütüphaneci","kuaför","fırıncı","denetçi","resepsiyonist",
    "editör","temizlikçi","kasiyer","terzi","yazar","danışman","muhasebeci","görevli"
]

# -------- normalization helpers (single authoritative norm_prof) --------
TURKISH_MAP = str.maketrans({
    "ı": "i", "İ": "i",
    "ş": "s", "Ş": "s",
    "ç": "c", "Ç": "c",
    "ö": "o", "Ö": "o",
    "ü": "u", "Ü": "u",
    "ğ": "g", "Ğ": "g"
})

def _strip_combining(s: str) -> str:
    # remove combining marks (e.g., i̇ -> i)
    return "".join(c for c in s if ud.category(c) != "Mn")

def norm_prof(s: str) -> str:
    """Unicode-safe TR normalization to a stable ASCII-ish key."""
    if s is None:
        return ""
    s = str(s).casefold()
    s = s.translate(TURKISH_MAP)
    s = ud.normalize("NFKD", s)
    s = _strip_combining(s)
    s = re.sub(r"[^\w\s]", " ", s)  # ← CHANGE: drop punctuation/symbols (e.g., "aşçı," -> "asci")
    return " ".join(s.strip().split())

def _ascii_tr(s: str) -> str:
    return _strip_combining(ud.normalize("NFKD", str(s))).translate(TURKISH_MAP)

def _add_aliases(name: str):
    """Generate pragmatic aliases (ASCII + i<->ı swaps)."""
    v = {name, _ascii_tr(name), name.replace("i","ı"), name.replace("ı","i")}
    v |= {_ascii_tr(name.replace("i","ı")), _ascii_tr(name.replace("ı","i"))}
    return {" ".join(x.split()) for x in v if x}

# --- build stereotype map with a few robust aliases for usual suspects ---
ST_MAP = {p:0 for p in MALE_PROFESSIONS}
ST_MAP.update({p:1 for p in FEMALE_PROFESSIONS})

# Known troublemakers where spelling/ASCII variants often appear in data
_MISSING_FIX = {
    "satış görevlisi": 0,
    "aşçı": 0,
    "kapıcı": 0,
    "tasarımcı": 1,
    "fırıncı": 1,
    "danışman": 1,
    "asistant": 1
}
for base, val in _MISSING_FIX.items():
    for alias in _add_aliases(base):
        ST_MAP[alias] = val
ST_MAP["asistant"] = 1  

ST_MAP_NORM = {norm_prof(k): v for k, v in ST_MAP.items()}

def get_expected(raw: str):
    """Best-effort lookup for stereotype expectation (0/1) using multiple fallbacks."""
    if raw is None:
        return np.nan
    k = norm_prof(raw)
    if k in ST_MAP_NORM:
        return ST_MAP_NORM[k]
    if raw in ST_MAP:
        return ST_MAP[raw]
    xa = _ascii_tr(raw)
    if xa in ST_MAP:
        return ST_MAP[xa]
    ka = norm_prof(xa)
    if ka in ST_MAP_NORM:
        return ST_MAP_NORM[ka]
    return np.nan

# -------- stats helpers --------
def wilson_ci(p_hat: float, n: int, z: float = 1.96):
    if n <= 0 or not np.isfinite(p_hat):
        return (np.nan, np.nan)
    denom  = 1 + z**2/n
    center = (p_hat + z**2/(2*n)) / denom
    half   = z * np.sqrt((p_hat*(1-p_hat)/n) + (z**2)/(4*n**2)) / denom
    return (max(0.0, center - half), min(1.0, center + half))

def wavg(series, weights):
    w = np.asarray(weights)
    x = np.asarray(series)
    m = np.isfinite(x) & np.isfinite(w)
    if not m.any(): return np.nan
    w = w[m]; x = x[m]
    s = w.sum()
    return np.nan if s == 0 else np.dot(w, x) / s

def _mode_str(s: pd.Series) -> str:
    vc = s.value_counts()
    return "" if vc.empty else vc.index[0]

# pandas 2.2 deprecation-friendly apply wrapper
def _gb_apply(grouper, func):
    try:
        return grouper.apply(func, include_groups=False)
    except TypeError:
        return grouper.apply(func)

def main():
    # ---- load ----
    df = pd.read_csv(IN_CSV, low_memory=False, encoding="utf-8")

    required = {"bias_group","translated_gender","target_profession","target_language","tool_name"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {IN_CSV.name}: {sorted(missing)}")

    # --- quick diagnostics BEFORE label filtering (to see data vs. label issues) ---
    neutral_any = df.loc[df["bias_group"].astype(str).str.lower()=="neutral",
                         ["target_profession","translated_gender"]].copy()
    present_any_norm = set(neutral_any["target_profession"].dropna().map(norm_prof).unique())
    present_lab_norm = set(neutral_any.loc[neutral_any["translated_gender"].isin([0,1]),
                                           "target_profession"].dropna().map(norm_prof).unique())
    design_norms = set(ST_MAP_NORM.keys())
    miss_any = sorted(design_norms - present_any_norm)
    miss_lab = sorted(design_norms - present_lab_norm)
    if miss_any:
        print("— In NEUTRAL, not present at all (by design key):", miss_any)
    if miss_lab:
        print("— In NEUTRAL, present but no valid label (NaN genders):", miss_lab)

    # --- EXTRA DIAG OUTPUT (minimal) -----------------------------------------
    # 1) Present in NEUTRAL but with NaN labels → export counts per profession (normalized and raw)
    present_nan = neutral_any.loc[~neutral_any["translated_gender"].isin([0,1])].copy()
    if not present_nan.empty:
        present_nan["profession_norm"] = present_nan["target_profession"].map(norm_prof)
        diag_nan = (present_nan
                    .groupby(["profession_norm", "target_profession"], dropna=False)
                    .size().reset_index(name="count"))
        diag_nan = diag_nan.sort_values(["profession_norm","count"], ascending=[True, False])
        (OUTDIR / "rq1_neutral_present_but_nan.csv").write_text(
            diag_nan.to_csv(index=False, encoding="utf-8"), encoding="utf-8"
        )
        # Console-friendly summary (only the ones that are part of the design)
        diag_nan_design = diag_nan[diag_nan["profession_norm"].isin(set(ST_MAP_NORM.keys()))]
        if not diag_nan_design.empty:
            print("— NEUTRAL present but NaN genders (design keys only):")
            for k, sub in diag_nan_design.groupby("profession_norm"):
                total = int(sub["count"].sum())
                raw_variants = ", ".join(sub["target_profession"].astype(str).tolist())
                print(f"  · {k}: {total} (raw: {raw_variants})")
            print(f"→ Saved: {(OUTDIR / 'rq1_neutral_present_but_nan.csv').as_posix()}")

    # 2) For design professions with no labeled rows, export alias map
    # FIX: use 'miss_lab' (previously undefined 'missing_norms')
    if miss_lab:
        rows = []
        norm2orig = {}
        for orig in ST_MAP.keys():
            norm2orig.setdefault(norm_prof(orig), set()).add(orig)
        for nk in miss_lab:
            for orig in sorted(norm2orig.get(nk, {nk})):
                rows.append({"profession_norm": nk, "design_variant": orig})
        diag_missing = pd.DataFrame(rows)
        (OUTDIR / "rq1_missing_norms_aliases.csv").write_text(
            diag_missing.to_csv(index=False, encoding="utf-8"), encoding="utf-8"
        )
        print(f"→ Saved: {(OUTDIR / 'rq1_missing_norms_aliases.csv').as_posix()}")

    # ---- filter neutral + valid labels ----
    neu = df.loc[
        (df["bias_group"].astype(str).str.lower() == "neutral")
        & (df["translated_gender"].isin([0,1]))
    , ["target_profession","target_language","tool_name","translated_gender"]].copy()

    if neu.empty:
        print("RQ1: No neutral rows with valid translated_gender.")
        return

    # normalized profession + robust stereotype expectation
    neu["profession_norm"] = neu["target_profession"].map(norm_prof)
    neu["stereotype_expected"] = neu["target_profession"].apply(get_expected)  # robust fallback
    neu["matches_stereotype"] = np.where(
        neu["stereotype_expected"].isin([0,1]),
        (neu["translated_gender"] == neu["stereotype_expected"]).astype(float),
        np.nan
    )

    # --- sanity: which design professions still missing after labeling? ---
    present_norms  = set(neu["profession_norm"].unique())
    missing_norms  = sorted(design_norms - present_norms)
    if missing_norms:
        # report original variants that correspond to the normalized key
        norm2orig = {}
        for orig in ST_MAP.keys():
            norm2orig.setdefault(norm_prof(orig), set()).add(orig)
        print("⚠ Missing in NEUTRAL (with valid labels):")
        for nk in missing_norms:
            cand = ", ".join(sorted(norm2orig.get(nk, {nk})))
            print(f"  - {cand}")

    # ---- main table: profession × language × tool (group by normalized) ----
    grp = neu.groupby(["profession_norm","target_language","tool_name"], dropna=False)
    out = _gb_apply(grp, lambda g: pd.Series({
        "n": int(g["translated_gender"].size),
        "female": float(g["translated_gender"].mean()),
        "target_profession": _mode_str(g["target_profession"])  # for display
    })).reset_index()
    out["male"] = 1 - out["female"]

    # add stereotype support rate
    st = _gb_apply(grp, lambda g: pd.Series({
        "stereotype_supported_rate": float(g["matches_stereotype"].mean())
    })).reset_index()
    out = out.merge(st, on=["profession_norm","target_language","tool_name"], how="left")

    # Wilson CI for female share
    ci = out.apply(lambda r: pd.Series(wilson_ci(r["female"], r["n"]),
                                       index=["female_ci_lo","female_ci_hi"]), axis=1)
    out = pd.concat([out, ci], axis=1)

    out_main_csv = OUTDIR / "rq1_by_prof_lang_tool.csv"
    out.to_csv(out_main_csv, index=False, encoding="utf-8")
    (OUTDIR / "rq1_by_prof_lang_tool.tex").write_text(
        out.head(30).to_latex(index=False, float_format="%.3f"), encoding="utf-8"
    )

    # ---- profession-level summary (WEIGHTED across lang/tool by n) ----
    prof = _gb_apply(out.groupby("profession_norm", dropna=False), lambda g: pd.Series({
        "target_profession": _mode_str(g["target_profession"]),
        "n": int(g["n"].sum()),
        "female_mean": wavg(g["female"], g["n"]),
        "male_mean":   wavg(g["male"],   g["n"]),
        "stereotype_supported_rate": wavg(g["stereotype_supported_rate"], g["n"])
    })).reset_index().sort_values(
        ["stereotype_supported_rate","female_mean"], ascending=[False, False]
    )

    prof_csv = OUTDIR / "rq1_by_prof_overall.csv"
    prof.to_csv(prof_csv, index=False, encoding="utf-8")
    (OUTDIR / "rq1_by_prof_overall.tex").write_text(
        prof[["target_profession","n","female_mean","male_mean","stereotype_supported_rate"]]
        .head(40).to_latex(index=False, float_format="%.3f"), encoding="utf-8"
    )

    # ---- language summary (WEIGHTED by group size) ----
    by_lang = _gb_apply(out.groupby("target_language", dropna=False), lambda g: pd.Series({
        "n": int(g["n"].sum()),
        "female_mean": wavg(g["female"], g["n"]),
        "stereotype_supported_rate": wavg(g["stereotype_supported_rate"], g["n"])
    })).reset_index().sort_values("female_mean", ascending=False)

    by_lang_csv = OUTDIR / "rq1_by_language.csv"
    by_lang.to_csv(by_lang_csv, index=False, encoding="utf-8")
    (OUTDIR / "rq1_by_language.tex").write_text(
        by_lang.to_latex(index=False, float_format="%.3f"), encoding="utf-8"
    )

    # ---- tool summary (WEIGHTED by group size) ----
    by_tool = _gb_apply(out.groupby("tool_name", dropna=False), lambda g: pd.Series({
        "n": int(g["n"].sum()),
        "female_mean": wavg(g["female"], g["n"]),
        "stereotype_supported_rate": wavg(g["stereotype_supported_rate"], g["n"])
    })).reset_index().sort_values("female_mean", ascending=False)

    by_tool_csv = OUTDIR / "rq1_by_tool.csv"
    by_tool.to_csv(by_tool_csv, index=False, encoding="utf-8")
    (OUTDIR / "rq1_by_tool.tex").write_text(
        by_tool.to_latex(index=False, float_format="%.3f"), encoding="utf-8"
    )

    # ---- pivots for heatmaps (mean across groups; index = normalized profession) ----
    pivot_lang = out.pivot_table(
        index="profession_norm", columns="target_language", values="female", aggfunc="mean"
    )
    pivot_tool = out.pivot_table(
        index="profession_norm", columns="tool_name", values="female", aggfunc="mean"
    )
    pivot_lang.to_csv(OUTDIR / "rq1_pivot_profession_x_language.csv", encoding="utf-8")
    pivot_tool.to_csv(OUTDIR / "rq1_pivot_profession_x_tool.csv", encoding="utf-8")

    # ---- figures (optional quick bars) ----
    top_f = prof.sort_values("female_mean", ascending=False).head(20)
    if not top_f.empty:
        plt.figure(figsize=(10,6))
        plt.bar(top_f["target_profession"], top_f["female_mean"])
        plt.xticks(rotation=60, ha="right")
        plt.ylabel("Share feminine (neutral)")
        plt.title("RQ1: Top-20 professions by feminine share (neutral, weighted)")
        plt.tight_layout()
        plt.savefig(OUTDIR / "rq1_top20_female_share.png", dpi=200)
        plt.close()

    top_s = prof.dropna(subset=["stereotype_supported_rate"])\
                .sort_values("stereotype_supported_rate", ascending=False).head(20)
    if not top_s.empty:
        plt.figure(figsize=(10,6))
        plt.bar(top_s["target_profession"], top_s["stereotype_supported_rate"])
        plt.xticks(rotation=60, ha="right")
        plt.ylabel("Stereotype support (neutral)")
        plt.title("RQ1: Top-20 professions by default stereotype support (weighted)")
        plt.tight_layout()
        plt.savefig(OUTDIR / "rq1_top20_stereotype_support.png", dpi=200)
        plt.close()

    # ---- overall summary (grouped means remain unweighted by design) ----
    professions_covered = int((set(ST_MAP_NORM.keys()) & set(prof["profession_norm"])).__len__())
    overall = {
        "rows_used": int(len(neu)),
        "groups": int(len(out)),
        "professions_covered": professions_covered,
        "mean_female_share_grouped": float(out["female"].mean()),
        "mean_stereotype_support_grouped": float(wavg(out["stereotype_supported_rate"], out["n"]))
    }
    pd.DataFrame([overall]).to_csv(OUTDIR / "rq1_overall_summary.csv", index=False, encoding="utf-8")

    print("RQ1 summary:", overall)
    print("→ Tables:")
    for p in [
        out_main_csv, prof_csv, by_lang_csv, by_tool_csv,
        OUTDIR / "rq1_pivot_profession_x_language.csv",
        OUTDIR / "rq1_pivot_profession_x_tool.csv",
        OUTDIR / "rq1_overall_summary.csv",
    ]:
        print("  -", p.as_posix())
    print("→ LaTeX previews:")
    for p in [
        OUTDIR / "rq1_by_prof_lang_tool.tex",
        OUTDIR / "rq1_by_prof_overall.tex",
        OUTDIR / "rq1_by_language.tex",
        OUTDIR / "rq1_by_tool.tex",
    ]:
        print("  -", p.as_posix())
    print("→ Figures:")
    print("  -", (OUTDIR / "rq1_top20_female_share.png").as_posix())
    print("  -", (OUTDIR / "rq1_top20_stereotype_support.png").as_posix())

if __name__ == "__main__":
    main()
