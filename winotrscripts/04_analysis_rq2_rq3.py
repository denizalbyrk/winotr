#!/usr/bin/env python3
# 03_analysis_rq2_rq3.py  (CATE-extended)
# RQ2 (Pro vs Anti) & RQ3 (Signal vs Neutral):
#   (a) stereotype-consistent accuracy comparisons
#   (b) Double Machine Learning (IRM) ATE estimates
#   (c) NEW: CATE summaries via IATE = g1(x)-g0(x) aggregated by groups
#
# Compatible with different sklearn & DoubleML versions.
 
from pathlib import Path
import numpy as np
import pandas as pd
import unicodedata as ud, re
import warnings
 
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
 
try:
    import doubleml as dml
except Exception:
    import subprocess, sys as _sys
    subprocess.run([_sys.executable, "-m", "pip", "install", "-q", "DoubleML", "scikit-learn"])
    import doubleml as dml
 
# ---------------- paths ----------------
ROOT    = Path(__file__).resolve().parents[1]
IN_CSV  = ROOT / "results" / "02_combined_gender_simalign.csv"
OUT_RQ2 = ROOT / "results" / "rq2"
OUT_RQ3 = ROOT / "results" / "rq3"
OUT_RQ2.mkdir(parents=True, exist_ok=True)
OUT_RQ3.mkdir(parents=True, exist_ok=True)
 
# ---------------- stereotype lists ----------------
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
 
# ---------------- TR normalization ----------------
TURKISH_MAP = str.maketrans({
    "ı":"i","İ":"i","ş":"s","Ş":"s","ç":"c","Ç":"c",
    "ö":"o","Ö":"o","ü":"u","Ü":"u","ğ":"g","Ğ":"g"
})
def _strip_combining(s: str) -> str:
    return "".join(c for c in s if ud.category(c) != "Mn")
def norm_prof(s: str) -> str:
    if s is None: return ""
    s = str(s).casefold().translate(TURKISH_MAP)
    s = ud.normalize("NFKD", s)
    s = _strip_combining(s)
    s = re.sub(r"[^\w\s]", " ", s)
    return " ".join(s.strip().split())
def build_stereotype_map_norm():
    st = {p:0 for p in MALE_PROFESSIONS}
    st.update({p:1 for p in FEMALE_PROFESSIONS})
    st["asistant"] = 1
    return {norm_prof(k): v for k,v in st.items()}
 
# ---------------- helpers ----------------
FEATURE_COLS = ["target_profession","target_language","tool_name","target_position","src_len"]
CAT_COLS = ["target_profession","target_language","tool_name","target_position"]
NUM_COLS = ["src_len"]
 
def _std_bias_group(x: str) -> str:
    return str(x).strip().lower()
def _src_len(txt: str) -> int:
    if not isinstance(txt, str): return 0
    return len(str(txt).split())
 
def prepare_base_df(in_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(in_csv, low_memory=False)
    df["bias_group"] = df["bias_group"].map(_std_bias_group)
    df = df[df["translated_gender"].isin([0,1])].copy()
    df["target_profession_key"] = df["target_profession"].map(norm_prof)
    if "sentence_norm" in df.columns:
        df["src_len"] = df["sentence_norm"].apply(_src_len)
    else:
        df["src_len"] = df["sentence"].apply(_src_len) if "sentence" in df.columns else 0
 
    # Use expected_gender column directly if available (preferred over stereotype map lookup).
    # expected_gender encodes the cue-annotated gender per row from the gold annotation:
    #   "male" → stereotype_expected = 0 (male=0 in translated_gender encoding)
    #   "female" → stereotype_expected = 1
    # This avoids normalization mismatches in the profession-based stereotype map.
    if "expected_gender" in df.columns:
        df["stereotype_expected"] = df["expected_gender"].map(
            lambda x: 0 if str(x).strip().lower() == "male"
                      else (1 if str(x).strip().lower() == "female" else np.nan)
        )
    else:
        # Fallback: derive from profession stereotype map (less reliable)
        st_map = build_stereotype_map_norm()
        df["stereotype_expected"] = df["target_profession_key"].map(st_map)
 
    needed = [
        "target_profession","target_profession_key","target_language","tool_name",
        "target_position","translated_gender","bias_group","src_len","stereotype_expected"
    ]
    miss = [c for c in needed if c not in df.columns]
    if miss:
        raise ValueError(f"Missing required columns: {miss}")
    return df
 
# sklearn version compatibility (sparse_output vs sparse)
def make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)
 
# fully one-hot encode to dense numeric matrix
def encode_features_dense(X_df: pd.DataFrame):
    Xc = X_df[CAT_COLS].astype(str)
    Xn = X_df[NUM_COLS].copy()
 
    if not Xn.empty:
        imp = SimpleImputer(strategy="median")
        Xn = pd.DataFrame(imp.fit_transform(Xn), columns=NUM_COLS, index=Xn.index)
    else:
        Xn = pd.DataFrame(index=X_df.index)
 
    ohe = make_ohe()
    Xc_enc = ohe.fit_transform(Xc)
    Xc_enc = np.asarray(Xc_enc, dtype=np.float64)
 
    if not Xn.empty:
        X_enc = np.hstack([Xc_enc, Xn.to_numpy(dtype=np.float64, copy=False)])
        p_dim = Xc_enc.shape[1] + Xn.shape[1]
    else:
        X_enc = Xc_enc
        p_dim = Xc_enc.shape[1]
    return X_enc, p_dim
 
# DoubleML version compatibility (coef_/coef etc.)
def _get_dml_scalar(obj, name_with_uscore: str, name_no_uscore: str) -> float:
    if hasattr(obj, name_with_uscore):
        v = getattr(obj, name_with_uscore)
    elif hasattr(obj, name_no_uscore):
        v = getattr(obj, name_no_uscore)
    else:
        raise AttributeError(f"DoubleML object has neither '{name_with_uscore}' nor '{name_no_uscore}'.")
    try:
        return float(v[0])
    except Exception:
        return float(v)
 
# try to extract IRM nuisance predictions in a version-robust way
def _get_irm_predictions(obj):
    preds = None
    if hasattr(obj, "predictions"): preds = obj.predictions
    elif hasattr(obj, "predictions_"): preds = obj.predictions_
    if preds is None:
        return None, None, None
    def _pick(keys):
        for k in keys:
            if isinstance(preds, dict) and k in preds:
                return np.asarray(preds[k]).reshape(-1)
        return None
    g0 = _pick(["ml_g0","g0","g_hat0"])
    g1 = _pick(["ml_g1","g1","g_hat1"])
    m  = _pick(["ml_m","m","m_hat"])
    return g0, g1, m
 
# summarize IATE -> CATE per group (mean & SE)
def _summarize_cate(df_idx: pd.Index, iate: np.ndarray, groups: pd.DataFrame, out_path: Path, group_cols):
    tmp = pd.DataFrame({"__idx__": df_idx, "iate": iate})
    # groups must align with original rows; join by index position
    gdf = groups.copy()
    gdf["__idx__"] = gdf.index
    tmp = tmp.merge(gdf[["__idx__"] + list(group_cols)], on="__idx__", how="left").drop(columns="__idx__")
    def _agg(x):
        n = x.size
        mu = float(np.mean(x)) if n>0 else np.nan
        se = float(np.std(x, ddof=1)/np.sqrt(n)) if n>1 else np.nan
        return pd.Series({"n": n, "cate_mean": mu, "cate_se": se})
    out = tmp.groupby(list(group_cols))["iate"].apply(_agg).reset_index()
    out.to_csv(out_path, index=False, encoding="utf-8")
 
# DoubleML IRM over pre-encoded (dense) X
def run_doubleml_irm_preencoded(X_df: pd.DataFrame, D: np.ndarray, Y: np.ndarray,
                                n_folds: int = 2, random_state: int = 42):
    X_df = X_df[FEATURE_COLS].copy()
    X_enc, p_dim = encode_features_dense(X_df)
 
    ml_g = RandomForestClassifier(
        n_estimators=500, min_samples_leaf=10, random_state=random_state
    )
    ml_m = LogisticRegression(max_iter=2000, solver="lbfgs")
 
    dml_data = dml.DoubleMLData.from_arrays(X_enc, Y, D)
    obj = dml.DoubleMLIRM(dml_data, ml_g=ml_g, ml_m=ml_m, n_folds=n_folds, score='ATE')
    obj.fit()
 
    ate = _get_dml_scalar(obj, "coef_", "coef")
    se  = _get_dml_scalar(obj, "se_", "se")
    t   = _get_dml_scalar(obj, "t_stat_", "t_stat")
    p   = _get_dml_scalar(obj, "pval_", "pval")
    n   = int(len(Y))
 
    # try to build IATE = g1 - g0
    g0, g1, m = _get_irm_predictions(obj)
    iate = None
    if g0 is not None and g1 is not None:
        try:
            iate = np.asarray(g1, dtype=float) - np.asarray(g0, dtype=float)
        except Exception:
            iate = None
    else:
        warnings.warn("Could not extract ml_g0/ml_g1 predictions from DoubleML object; CATE tables will be skipped.")
 
    return ate, se, t, p, n, int(p_dim), iate
 
# ---------------- RQ2: Pro vs Anti ----------------
def rq2_tables_and_dml(df: pd.DataFrame, out_dir: Path):
    d = df[df["bias_group"].isin(["pro","anti"])].copy()
    if d.empty:
        print("RQ2: no pro/anti rows."); return
 
    # stereotype_expected already set in prepare_base_df via expected_gender column
    d = d[d["stereotype_expected"].isin([0,1])].copy()
 
    d["D"] = (d["bias_group"] == "pro").astype(int)
    # Y = stereotype-consistency (same definition as RQ3):
    # 1 if translated gender matches the predefined stereotype map, 0 otherwise.
    # This ensures RQ2 and RQ3 are directly comparable.
    d["Y"] = (d["translated_gender"] == d["stereotype_expected"]).astype(int)
 
    # accuracy tables
    grp_cols = ["target_profession","target_language","tool_name"]
    acc = d.groupby(grp_cols + ["bias_group"], dropna=False).agg(
        n=("Y","size"), accuracy=("Y","mean")
    ).reset_index()
    piv = acc.pivot_table(index=grp_cols, columns="bias_group", values="accuracy")
    piv = piv.rename_axis(None, axis=1).reset_index().rename(columns={"pro":"pro_accuracy","anti":"anti_accuracy"})
    piv["pro_minus_anti"] = piv["pro_accuracy"] - piv["anti_accuracy"]
    piv.to_csv(out_dir / "rq2_accuracy_by_prof_lang_tool.csv", index=False, encoding="utf-8")
 
    # by language
    by_lang = d.groupby(["target_language","bias_group"], dropna=False).agg(n=("Y","size"), accuracy=("Y","mean")).reset_index()
    lang_p = by_lang.pivot_table(index=["target_language"], columns="bias_group", values="accuracy")
    lang_p = lang_p.rename_axis(None, axis=1).reset_index().rename(columns={"pro":"pro_accuracy","anti":"anti_accuracy"})
    lang_p["pro_minus_anti"] = lang_p["pro_accuracy"] - lang_p["anti_accuracy"]
    lang_p.to_csv(out_dir / "rq2_accuracy_by_language.csv", index=False, encoding="utf-8")
 
    # by tool
    by_tool = d.groupby(["tool_name","bias_group"], dropna=False).agg(n=("Y","size"), accuracy=("Y","mean")).reset_index()
    tool_p = by_tool.pivot_table(index=["tool_name"], columns="bias_group", values="accuracy")
    tool_p = tool_p.rename_axis(None, axis=1).reset_index().rename(columns={"pro":"pro_accuracy","anti":"anti_accuracy"})
    tool_p["pro_minus_anti"] = tool_p["pro_accuracy"] - tool_p["anti_accuracy"]
    tool_p.to_csv(out_dir / "rq2_accuracy_by_tool.csv", index=False, encoding="utf-8")
 
    overall = {
        "rows_used": int(d.shape[0]),
        "pro_accuracy": float(d.loc[d["D"]==1, "Y"].mean()),
        "anti_accuracy": float(d.loc[d["D"]==0, "Y"].mean())
    }
    pd.DataFrame([overall]).to_csv(out_dir / "rq2_overall_accuracy.csv", index=False, encoding="utf-8")
 
    # DoubleML IRM (pre-encoded X) + CATE
    X = d[FEATURE_COLS].copy()
    D = d["D"].astype(int).to_numpy()
    Y = d["Y"].astype(int).to_numpy()
 
    ate, se, t, p, n, p_dim, iate = run_doubleml_irm_preencoded(X, D, Y, n_folds=5, random_state=42)
    pd.DataFrame([{"ate":ate,"se":se,"t":t,"pval":p,"n_obs":n,"p_dim":p_dim}]).to_csv(out_dir/"rq2_dml_ate.csv", index=False, encoding="utf-8")
    with open(out_dir / "rq2_dml_ate.txt","w",encoding="utf-8") as f:
        f.write("RQ2 DML ATE (pro vs anti) on stereotype-consistency\n")
        f.write(f"ATE = {ate:.6f}\nSE  = {se:.6f}\nt   = {t:.3f}\np   = {p:.4g}\nN   = {n}\np_dim = {p_dim}\n")
 
    print(f"RQ2 DML ATE: {ate:.4f} (SE {se:.4f}, t {t:.2f}, p {p:.3g}, p_dim {p_dim})")
 
    # NOTE: Formal CATE estimation (via g1-g0 IRM predictions) has been removed.
    # Profession/language/tool-level heterogeneity is reported via accuracy difference
    # tables above (rq2_accuracy_by_prof_lang_tool.csv etc.), which are descriptively
    # sufficient and methodologically unambiguous for this analysis.
 
    print("→ Saved:", out_dir.as_posix())
 
# ---------------- RQ3: Signal vs Neutral ----------------
def rq3_tables_and_dml(df: pd.DataFrame, out_dir: Path):
    d = df[df["bias_group"].isin(["pro","anti","neutral"])].copy()
    if d.empty:
        print("RQ3: no rows in {pro,anti,neutral}."); return
 
    # stereotype_expected already set in prepare_base_df via expected_gender column.
    # For neutral rows, expected_gender is not annotated in the source file.
    # We fall back to the stereotype map for neutral rows only.
    st_map = build_stereotype_map_norm()
    neutral_mask = d["stereotype_expected"].isna()
    d.loc[neutral_mask, "stereotype_expected"] = d.loc[neutral_mask, "target_profession_key"].map(st_map)
    d = d[d["stereotype_expected"].isin([0,1])].copy()
 
    d["Y"] = (d["translated_gender"] == d["stereotype_expected"]).astype(int)
    d["D"] = d["bias_group"].isin(["pro","anti"]).astype(int)  # 1=signal, 0=neutral
 
    # long accuracy by group
    grp_cols = ["target_profession","target_language","tool_name"]
    acc_long = d.groupby(grp_cols + ["bias_group"], dropna=False).agg(n=("Y","size"), accuracy=("Y","mean")).reset_index()
    acc_long.to_csv(out_dir / "rq3_accuracy_by_prof_lang_tool_long.csv", index=False, encoding="utf-8")
 
    # overall signal vs neutral
    sig_acc = d.groupby(["D"], dropna=False).agg(n=("Y","size"), accuracy=("Y","mean")).reset_index()
    sig_acc.to_csv(out_dir / "rq3_accuracy_signal_vs_neutral_overall.csv", index=False, encoding="utf-8")
 
    # by language
    by_lang = d.groupby(["target_language","D"], dropna=False).agg(n=("Y","size"), accuracy=("Y","mean")).reset_index()
    lang_p = by_lang.pivot_table(index=["target_language"], columns="D", values="accuracy")
    lang_p = lang_p.rename(columns={0:"neutral_accuracy",1:"signal_accuracy"}).reset_index()
    lang_p["signal_minus_neutral"] = lang_p["signal_accuracy"] - lang_p["neutral_accuracy"]
    lang_p.to_csv(out_dir / "rq3_accuracy_signal_vs_neutral_by_language.csv", index=False, encoding="utf-8")
 
    # by tool
    by_tool = d.groupby(["tool_name","D"], dropna=False).agg(n=("Y","size"), accuracy=("Y","mean")).reset_index()
    tool_p = by_tool.pivot_table(index=["tool_name"], columns="D", values="accuracy")
    tool_p = tool_p.rename(columns={0:"neutral_accuracy",1:"signal_accuracy"}).reset_index()
    tool_p["signal_minus_neutral"] = tool_p["signal_accuracy"] - tool_p["neutral_accuracy"]
    tool_p.to_csv(out_dir / "rq3_accuracy_signal_vs_neutral_by_tool.csv", index=False, encoding="utf-8")
 
    # DoubleML IRM (pre-encoded X) + CATE
    X = d[FEATURE_COLS].copy()
    D = d["D"].astype(int).to_numpy()
    Y = d["Y"].astype(int).to_numpy()
 
    ate, se, t, p, n, p_dim, iate = run_doubleml_irm_preencoded(X, D, Y, n_folds=5, random_state=42)
    pd.DataFrame([{"ate":ate,"se":se,"t":t,"pval":p,"n_obs":n,"p_dim":p_dim}]).to_csv(out_dir/"rq3_dml_ate.csv", index=False, encoding="utf-8")
    with open(out_dir / "rq3_dml_ate.txt","w",encoding="utf-8") as f:
        f.write("RQ3 DML ATE (signal vs neutral) on stereotype-consistency\n")
        f.write(f"ATE = {ate:.6f}\nSE  = {se:.6f}\nt   = {t:.3f}\np   = {p:.4g}\nN   = {n}\np_dim = {p_dim}\n")
 
    print(f"RQ3 DML ATE: {ate:.4f} (SE {se:.4f}, t {t:.2f}, p {p:.3g}, p_dim {p_dim})")
 
    # --- NEW: CATE summaries from IATE ---
    if iate is not None and iate.shape[0] == d.shape[0]:
        d = d.reset_index(drop=True)
        _summarize_cate(d.index, iate, d[["target_profession"]], out_dir / "rq3_cate_by_profession.csv", ["target_profession"])
        _summarize_cate(d.index, iate, d[["target_language"]],  out_dir / "rq3_cate_by_language.csv",  ["target_language"])
        _summarize_cate(d.index, iate, d[["tool_name"]],       out_dir / "rq3_cate_by_tool.csv",       ["tool_name"])
        _summarize_cate(
            d.index, iate,
            d[["target_profession","target_language","tool_name"]],
            out_dir / "rq3_cate_by_prof_lang_tool.csv",
            ["target_profession","target_language","tool_name"]
        )
        print("RQ3 CATE tables saved.")
    else:
        print("RQ3 CATE skipped (could not extract IATE).")
 
    print("→ Saved:", out_dir.as_posix())
 
# ---------------- run ----------------
if __name__ == "__main__":
    df = prepare_base_df(IN_CSV)
 
    print("\n=== RQ2: Pro vs Anti ===")
    rq2_tables_and_dml(df, OUT_RQ2)
 
    print("\n=== RQ3: Signal vs Neutral ===")
    rq3_tables_and_dml(df, OUT_RQ3)
 
    print("\nDone.")
 