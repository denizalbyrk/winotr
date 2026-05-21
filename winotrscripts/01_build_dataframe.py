#!/usr/bin/env python3
#1_build_dataframe.py
#This script reads the raw MT evaluation (combined_dataset) file and produces an enriched pandas DataFrame required for analysis
# Build an enriched DataFrame from the raw WinoTR *combined_dataset.txt* file.
# normalises every sentence (accent-stripped, lower-case) → sentence_norm
# auto-detects the first profession and its position  → target_profession / target_position
# generates a stable, human-readable sentence_id
# The script is self-contained: drop it next to *combined_dataset.txt*, run it,
# and it writes *final_combined_dataframe.csv* next to the script.

from __future__ import annotations

import os, re, hashlib, shutil, datetime
import unicodedata as ud
from pathlib import Path
import pandas as pd

#Paths&parameters
ROOT        = Path("/Users/denizalbayrak/Documents/mt_gender_tr")
SRC_FILE    = ROOT / "data/aggregates/combined_dataset.txt"        # raw data set
TRANSL_DIR  = ROOT / "translations"                                # .txt translation
OUT_FILE    = ROOT / "results/01_combined_dataframe.csv"           # output CSV
SKIP_LOG    = ROOT / "results/01_skipped_lines.log"

#how many MT tools × how many target languages” — just for a quick sanity check
N_TOOLS, N_LANGS = 3, 3   # google / deepl / openai   ×   de / en / es


#Professions(Two-word critical professions such as "inşaat işçisi" are defined as LONG_PROF. The search order (PROF_ORDER) is set to “long ones first, then the rest (from long to short)”.)
LONG_PROF = ["inşaat işçisi", "güvenlik görevlisi", "satış görevlisi"]
OTHER_PROF = [
    "geliştirici", "tamirci", "nakliyeci", "analist",
    "avukat", "aşçı", "doktor", "çiftçi", "CEO", "yönetici", "sürücü", "işçi", "marangoz", "kapıcı",
    "amir", "şerif", "şef", "hemşire", "sekreter", "öğretmen", "tasarımcı", "hizmetli", "memur",
    "asistan", "kütüphaneci", "kuaför", "fırıncı", "denetçi", "resepsiyonist",
    "editör", "temizlikçi", "kasiyer", "terzi", "yazar", "danışman",
    "muhasebeci", "görevli",
]

PROF_ORDER = LONG_PROF + sorted(OTHER_PROF, key=len, reverse=True)
                                
PROF = set(PROF_ORDER)

#Professions Regexes/ Build a case-insensitive pattern that allows arbitrary spaces between words. e.g.  “satış   görevlisi”  still matches  “satış görevlisi”.
def _flex_pattern(prof: str) -> re.Pattern:
    esc   = re.escape(prof)                   # satış\ görevlisi
    flex  = re.sub(r"\s+", r"\\s+", esc)      # satış\s+görevlisi
    return re.compile(r"\b" + flex + r"\b", flags=re.I)
PATTERNS: list[tuple[str, re.Pattern]] = [(p, _flex_pattern(p)) for p in PROF_ORDER]


#Helpers

#Converts Turkish‐specific diacritics (e.g., İ → I) and all uppercase letters to lowercase, giving each sentence a single, canonical representation across every alignment.
def normalize_tr(txt: str) -> str:
  
    # Fix the notorious Turkish “İ/ı” case first
    txt = txt.replace("İ", "I").replace("ı", "i")
    
    # Decompose to NFD and drop every combining mark (kills Ç/Ö/Ü/Ğ/Ş dots, accents…)              
    txt = ''.join(ch for ch in ud.normalize("NFD", txt)
                  if ud.category(ch) != "Mn")
    
    # Unicode-aware lowercase (`casefold`)  +  tidy spacing
    return txt.casefold().strip()


#Produces a consistent, readable ID
def make_sentence_id(sentence: str, target_pos: int | float | None) -> str:
    
    sent_norm = re.sub(r"\s+", " ", sentence.lower()).strip()
    profs     = sorted(p for p in PROF if p in sent_norm)
    prof_key  = "_".join(profs) if profs else "none"
    digest    = hashlib.md5(sent_norm.encode()).hexdigest()[:6]
    return f"{prof_key}_pos{int(target_pos or 0)}_{digest}"

#1)Source data
src = (
    pd.read_csv(SRC_FILE, sep="\t", encoding="utf-8", quoting=3, keep_default_na=False)
      .assign(sentence_norm=lambda d: d["sentence"].map(normalize_tr))
)

src["sentence_id"] = src.apply(
    lambda r: make_sentence_id(r["sentence"], r.get("target_position")), axis=1
)
print(f"Source rows: {len(src):,}")

#2)Translation Files
records, skipped, errors = [], [], []
for root, _, files in os.walk(TRANSL_DIR):
    for fname in files:
        if not fname.endswith(".txt"):
            continue
        tool_m = re.search(r"(deepl|google|openai)", fname.lower())
        lang_m = re.search(r"_(de|en|es)",           fname.lower())
        if not (tool_m and lang_m):
            continue
        tool, lang = tool_m.group(1), lang_m.group(1)

        with open(Path(root) / fname, encoding="utf-8") as fh:
            for ln, line in enumerate(fh, 1):
                if line.lstrip().startswith("#") or "|||" not in line:
                    skipped.append(f"{fname}:{ln}")
                    continue

                src_s, trg_s = map(str.strip, line.split("|||", 1))

                # Catch blank line / ERROR line
                if not src_s or trg_s.upper() == "ERROR":
                    errors.append(f"{fname}:{ln}")   # for log
                    continue                         

                # Normal (valid) record
                records.append({
                    "sentence_norm"     : normalize_tr(src_s),
                    "translated_sentence": trg_s,
                    "tool_name"         : tool,
                    "target_language"   : lang,
                    "translation_pair"  : f"tr-{lang}",
                    "tool_lang"         : f"{tool}_{lang}",
                })
trans = (
    pd.DataFrame(records)
      .drop_duplicates(["sentence_norm", "tool_name", "target_language"])
)
print(f"ERROR line  : {len(errors):,}")
print(f"⏭  # comment/blank skip: {len(skipped):,}")
print(f"valid translation  : {len(records):,}")

#3)Merge and Extra Columns
df = src.merge(
        trans,
        on="sentence_norm",
        how="left",
        validate="many_to_many" 
)

df["T"]                = df["bias_group"].isin(["pro", "anti"]).astype(int)
df["translated_gender"] = pd.NA
df["is_stereotypical"]  = pd.NA
df["Y"]                = pd.NA



#4)Save

OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT_FILE, index=False, encoding="utf-8")
print(f" Written → {OUT_FILE.relative_to(ROOT)}")

expected = len(src) * N_TOOLS * N_LANGS
print(f"Sanity-check: expected ≈ {expected:,} rows | produced = {len(df):,}")

# write skipped-line log
if skipped:
    SKIP_LOG.write_text("\n".join(skipped), encoding="utf-8")
    print(f" {len(skipped):,} lines skipped → {SKIP_LOG.relative_to(ROOT)}")
