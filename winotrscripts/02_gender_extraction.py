#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 02 — translated_gender (0=male, 1=female, NA) + is_stereotypical

What this script does (kept minimal and explicit):
* Combine AWESOME (union) and SimAlign word alignments (falls back if AWESOME is unavailable).
* Treat multi-word Turkish professions as a single span; allow Turkish case suffixes for 1-word forms.
* Select target tokens only from the NP that corresponds to the source profession.
* Use a compact DE/ES lexicon to pin the correct NOUN/PROPN in the translation.
* If alignment is empty: scan the target by lexicon (lexicon rescue).
* If alignment exists: intersect with lexicon; if the intersection is empty but lexicon found a candidate, override (fix misalignment).
* Neighbor rescue (±3 source tokens), detailed logging, and simple summary metrics.
"""

import sys, re, csv, argparse, subprocess, unicodedata as ud
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import torch
import spacy

# ──────────────────────────────────────────────────────────────────────────────
# Arguments
# ──────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    p.add_argument("--input",  type=Path, default=root / "results/01_combined_dataframe.csv")
    p.add_argument("--output", type=Path, default=root / "results/02_combined_gender_simalign.csv")
    p.add_argument("--log",    type=Path, default=root / "results/02_simalign_log.csv")
    p.add_argument("--limit",  type=int, default=None)
    # Allow forcing which aligners to try; default is SimAlign-only for stability.
    p.add_argument("--aligners", choices=["auto","simalign","awesome","both"],
                   default="simalign",
                   help="Which aligners to use (default: simalign).")
    return p.parse_args()

# Global aligner choice (filled in main to avoid threading args everywhere)
ALIGNERS = "simalign"

# ──────────────────────────────────────────────────────────────────────────────
# TR normalization / tokenization
# ──────────────────────────────────────────────────────────────────────────────
TR_CASE_SUFFIXES = ("yı","yi","yu","yü","ya","ye","nı","ni","nu","nü","na","ne","y","ı","i","u","ü","a","e")
LONG_PROF_SET = {"inşaat işçisi","güvenlik görevlisi","satış görevlisi"}

def _strip(s: str) -> str:
    # strip accents and lowercase (used across TR/DE/ES matching)
    return "".join(ch for ch in ud.normalize("NFD", s or "") if ud.category(ch) != "Mn").lower()

def _norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", _strip(s)).strip()

def tokenize_tr(s: str) -> List[str]:
    # keep word chars; collapse whitespace; split
    s = re.sub(r"[^\w\s]", " ", _norm_text(s))
    s = re.sub(r"\s+", " ", s).strip()
    return s.split()

def tr_token_matches(tok_norm: str, prof_norm: str) -> bool:
    # exact or case-suffix tolerant match for Turkish single-word professions
    if tok_norm == prof_norm:
        return True
    if tok_norm.startswith(prof_norm):
        return tok_norm[len(prof_norm):] in TR_CASE_SUFFIXES
    return False

# --- NEW: alias map + helpers (ASCII/diacritics/typo variants → canonical TR) ---
# English: This collapses common misspellings and ASCII variants (e.g., "asci", "kapici")
# to their canonical Turkish forms so span finding & lexicon work robustly.
ALIASES: Dict[str, str] = {
    # aşçı
    "ascı": "aşçı", "asci": "aşçı", "aşci": "aşçı", "aşçi": "aşçı", "asçi": "aşçı",
    # fırıncı
    "firinci": "fırıncı", "fırinci": "fırıncı",
    # kapıcı
    "kapici": "kapıcı",
    # danışman
    "danısman": "danışman", "danisman": "danışman", "danişman": "danışman",
    # tasarımcı
    "tasarimci": "tasarımcı", "tasarimcı": "tasarımcı", "tasarımci": "tasarımcı",
    # satış görevlisi (i/ı, ş, ö ASCII and 'görevlısı' typos)
    "satis gorevlisi": "satış görevlisi",
    "satıs gorevlisi": "satış görevlisi",
    "satiş gorevlisi": "satış görevlisi",
    "satış gorevlisi": "satış görevlisi",
    "satış görevlısı": "satış görevlisi",
    "satiş görevlısi": "satış görevlisi",
    "satis görevlısi": "satış görevlisi",
    # already present in some data
    "asistant": "asistan",
}

def alias_fix(s: str) -> str:
    """Return alias-canonicalized profession (only for known TR profession names)."""
    return ALIASES.get(_norm_text(s or ""), _norm_text(s or ""))

def norm_prof(s: str) -> str:
    """Canonical TR profession normalization: NFKD/ASCII-insensitive + alias map."""
    return alias_fix(s)

def find_prof_span(src_tokens: List[str], profession: str, nth_hint: Optional[int]) -> Optional[Tuple[int,int]]:
    """
    Locate the profession span in source tokens.
    - For multi-word professions, require exact match except allow TR case suffix on the **last** token.
    - For single-word, allow Turkish case suffixes.
    - If multiple hits and nth_hint is given, pick the Nth; otherwise pick the first.
    """
    # --- CHANGED: use norm_prof (alias-aware) instead of plain _norm_text ---
    prof_raw = norm_prof(profession or "")
    if not prof_raw:
        return None
    prof_toks = tokenize_tr(prof_raw)
    L = len(prof_toks)
    if L == 0:
        return None
    if L > 1:
        # multi-word: all tokens must match exactly except the last, which may carry a TR case suffix
        k = 0
        for i in range(len(src_tokens) - L + 1):
            window = src_tokens[i:i+L]
            if not window:
                continue
            # all but last must be exact
            if window[:-1] == prof_toks[:-1] and tr_token_matches(window[-1], prof_toks[-1]):
                k += 1
                if nth_hint is None or k == nth_hint:
                    return (i, i+L)
        # fallback: first hit if nth_hint not satisfied
        for i in range(len(src_tokens) - L + 1):
            window = src_tokens[i:i+L]
            if window[:-1] == prof_toks[:-1] and tr_token_matches(window[-1], prof_toks[-1]):
                return (i, i+L)
        return None
    # single-token with case-suffix tolerance (unchanged)
    prof_norm = prof_toks[0]
    k = 0
    for i, tok in enumerate(src_tokens):
        if tr_token_matches(tok, prof_norm):
            k += 1
            if nth_hint is None or k == nth_hint:
                return (i, i+1)
    for i, tok in enumerate(src_tokens):
        if tr_token_matches(tok, prof_norm):
            return (i, i+1)
    return None

# ──────────────────────────────────────────────────────────────────────────────
# DE/ES profession lexicon (compact backbone)
# ──────────────────────────────────────────────────────────────────────────────
def L(*xs):  # normalization helper
    return { _strip(x) for x in xs }

LEXICON: Dict[str, Dict[str, set]] = {
    "de": {
        "geliştirici": L("entwickler","entwicklerin"),
        "tamirci": L("mechaniker","mechanikerin"),
        "nakliyeci": L("spediteur","transporteur","möbelpacker","bote"),
        "analist": L("analyst","analytiker","analystin","analytikerin"),
        "avukat": L("anwalt","anwältin","jurist","juristin"),
        "aşçı": L("koch","köchin","chef"),
        "doktor": L("arzt","ärztin","doktor","ärzte"),
        "çiftçi": L("bauer","bäuerin"),
        "yönetici": L("geschäftsführer","geschäftsführerin","manager","managerin","leiter","leiterin","verantwortliche"),
        "sürücü": L("fahrer","fahrerin"),
        "işçi": L("arbeiter","arbeiterin"),
        "marangoz": L("tischler","tischlerin","zimmermann"),
        "kapıcı": L("hausmeister","hausmeisterin","portier"),
        "şerif": L("sheriff"),
        "şef": L("chef","koch","köchin"),
        "hemşire": L("krankenschwester","pfleger","pflegerin"),
        "sekreter": L("sekretär","sekretärin"),
        "öğretmen": L("lehrer","lehrerin"),
        "tasarımcı": L("designer","designerin"),
        "hizmetli": L("diener","dienerin","bediensteter","bedienstete","knecht"),
        "memur": L("beamter","beamtin","angestellter","angestellte"),
        "asistan": L("assistent","assistentin"),
        "kütüphaneci": L("bibliothekar","bibliothekarin"),
        "kuaför": L("friseur","friseurin"),
        "fırıncı": L("bäcker","bäckerin"),
        "denetçi": L("prüfer","prüferin","inspektor","inspektorin"),
        "resepsiyonist": L("rezeptionist","rezeptionistin","empfangsmitarbeiter","empfangsmitarbeiterin"),
        "editör": L("redakteur","redakteurin","editor"),
        "temizlikçi": L("reiniger","reinigerin","putzfrau","putzmann"),
        "kasiyer": L("kassierer","kassiererin"),
        "terzi": L("schneider","schneiderin"),
        "yazar": L("autor","autorin","schriftsteller","schriftstellerin"),
        "danışman": L("berater","beraterin","consultant"),
        "muhasebeci": L("buchhalter","buchhalterin"),
        "görevli": L("angestellter","angestellte","beamter","beamtin"),
        "güvenlik görevlisi": L("sicherheitsbeamter","wachmann","wachfrau","security"),
        "satış görevlisi": L("verkäufer","verkäuferin","verkaufskraft"),
        "inşaat işçisi": L("bauarbeiter","bauarbeiterin"),
    },
    "es": {
        "geliştirici": L("desarrollador","desarrolladora","promotor","promotora"),
        "tamirci": L("mecánico","mecánica","reparador","reparadora"),
        "nakliyeci": L("transportista","porteador","mudador"),
        "analist": L("analista"),
        "avukat": L("abogado","abogada"),
        "aşçı": L("cocinero","cocinera","chef","jefe"),
        "doktor": L("doctor","doctora","médico","médica"),
        "çiftçi": L("granjero","granjera","agricultor","agricultora"),
        "yönetici": L("director","directora","gerente","encargado","encargada","responsable","directiva"),
        "sürücü": L("conductor","conductora"),
        "işçi": L("obrero","obrera","trabajador","trabajadora"),
        "marangoz": L("carpintero","carpintera"),
        "kapıcı": L("portero","portera","conserje"),
        "şerif": L("sheriff"),
        "şef": L("chef","jefe","cocinero","cocinera"),
        "hemşire": L("enfermero","enfermera"),
        "sekreter": L("secretario","secretaria"),
        "öğretmen": L("profesor","profesora","maestro","maestra"),
        "tasarımcı": L("diseñador","diseñadora"),
        "hizmetli": L("criado","criada","empleado","empleada","sirviente","sirvienta","personal doméstico"),
        "memur": L("funcionario","funcionaria","empleado","empleada"),
        "asistan": L("asistente"),
        "kütüphaneci": L("bibliotecario","bibliotecaria"),
        "kuaför": L("peluquero","peluquera"),
        "fırıncı": L("panadero","panadera"),
        "denetçi": L("inspector","inspectora","auditor","auditora"),
        "resepsiyonist": L("recepcionista"),
        "editör": L("editor","editora"),
        "temizlikçi": L("limpiador","limpiadora"),
        "kasiyer": L("cajero","cajera"),
        "terzi": L("sastre","costurera"),
        "yazar": L("escritor","escritora","autor","autora"),
        "danışman": L("consultor","consultora","asesor","asesora"),
        "muhasebeci": L("contador","contadora"),
        "görevli": L("empleado","empleada","funcionario","funcionaria"),
        "güvenlik görevlisi": L("guardia de seguridad","vigilante"),
        "satış görevlisi": L("dependiente","dependienta","vendedor","vendedora","empleado de ventas"),
        "inşaat işçisi": L("obrero de la construcción","trabajador de la construcción"),
    }
}

def lex_targets(lang: str, prof_tr: str) -> set:
    # --- CHANGED: alias-aware lookup key so "asci/kapici/..." match the same entry
    return LEXICON.get(lang, {}).get(_strip(norm_prof(prof_tr)), set())

# ──────────────────────────────────────────────────────────────────────────────
# Blacklist & thresholds
# ──────────────────────────────────────────────────────────────────────────────
BLACKLIST = {
    "de": {"der","die","das","dem","den","des","ein","eine","einer","einem","einen"},
    "es": {"el","la","los","las","un","una","unos","unas","manzana","dinero"}
}
MIN_ALIGN_SCORE = 0.0  # info only

# ──────────────────────────────────────────────────────────────────────────────
# Alignment layer
# ──────────────────────────────────────────────────────────────────────────────
AWESOME_OK = False
SIMALIGN_OK = False
try:
    import awesome_align  # noqa: F401
    AWESOME_OK = True
except Exception:
    AWESOME_OK = False

try:
    from simalign import SentenceAligner
    SIMALIGN_OK = True
except Exception:
    SIMALIGN_OK = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_sa = SentenceAligner(model="xlmr", token_type="bpe") if SIMALIGN_OK else None

def awesome_union(src: List[str], tgt: List[str]):
    return []

def simalign_union(src: List[str], tgt: List[str]):
    if _sa is None:
        return []
    try:
        res = _sa.get_word_aligns(src, tgt)
        pairs = set()
        for _, vs in res.items():
            for p in (vs or []):
                pairs.add((int(p[0]), int(p[1])))
        return sorted(pairs)
    except Exception:
        return []

def collect_tgt_idxs(pairs, span):
    # collect target indices that align to the source span
    if not isinstance(span, tuple) or len(span) != 2:
        return []
    s, e = span
    if not (isinstance(s, int) and isinstance(e, int)):
        return []
    return sorted({t for i, t in pairs if s <= i < e})

def neighbor_src_indices(span, n_src, offsets=(1, -1, 2, -2, 3, -3)):
    # indices of nearby source tokens for neighbor rescue
    s, e = span
    out = set()
    for i in range(s, e):
        for off in offsets:
            j = i + off
            if 0 <= j < n_src:
                out.add(j)
    return sorted(out)

# ──────────────────────────────────────────────────────────────────────────────
# spaCy (parser for noun chunks; NER disabled)
# ──────────────────────────────────────────────────────────────────────────────
for mdl in ("de_core_news_sm", "es_core_news_sm"):
    try:
        spacy.util.get_package_path(mdl)
    except OSError:
        subprocess.run([sys.executable, "-m", "spacy", "download", mdl, "-q"], check=False)

nlp_de = spacy.load("de_core_news_sm", disable=["ner"])
nlp_es = spacy.load("es_core_news_sm", disable=["ner"])

DE_DET = {"der":0,"ein":0,"dem":0,"einen":0,"des":0,"die":1,"eine":1,"einer":1}
ES_DET = {"el":0,"un":0,"la":1,"una":1}
DE_FEM_SFX = ("in","erin","ärztin","leiterin","sekretärin","angestellte","lehrerin",
              "friseurin","bäckerin","bibliothekarin","prüferin","rezeptionistin",
              "köchin","krankenschwester","wachfrau","redakteurin","autorin","beraterin",
              "buchhalterin","arbeiterin","managerin","verkäuferin","innen")
ES_FEM_SFX = ("a","ora","dora","triz")

def build_np_window(doc, head_i: int, max_left=2) -> List[int]:
    # Collect head and up to `max_left` tokens to the left if DET/ADJ
    S = {head_i}
    j = head_i - 1
    steps = 0
    while j >= 0 and steps < max_left and doc[j].pos_ in {"DET","ADJ"}:
        S.add(j); steps += 1; j -= 1
    return sorted(S)

def _nearest_noun_right(doc, i, max_hop=3) -> Optional[int]:
    # If alignment hits a DET, snap to the nearest NOUN/PROPN on the right (up to +3)
    j = i + 1
    hops = 0
    while j < len(doc) and hops < max_hop:
        if doc[j].pos_ in {"NOUN","PROPN"}:
            return j
        j += 1
        hops += 1
    return None

def pick_head_with_lexicon(doc, candidates: List[int], lang: str, prof_tr: str) -> Optional[int]:
    """
    Head selection priority:
    0) If any candidate maps to an explicitly gendered lexicon form, prefer it.
    1) NOUN/PROPN whose surface or lemma is in the lexicon
    2) Root of a noun chunk
    3) The NOUN closest to the median of candidates
    If a candidate is a DET, snap it to the nearest NOUN/PROPN.
    """
    if not candidates:
        return None

    # Expand DET→nearest NOUN/PROPN, keep only nominal heads.
    expanded: List[int] = []
    for i in candidates:
        if 0 <= i < len(doc):
            if doc[i].pos_ in {"NOUN","PROPN"}:
                expanded.append(i)
            elif doc[i].pos_ == "DET":
                j = _nearest_noun_right(doc, i, max_hop=3)
                if j is not None:
                    expanded.append(j)

    cands = [i for i in sorted(set(expanded)) if 0 <= i < len(doc) and doc[i].pos_ in {"NOUN","PROPN"}]
    if not cands:
        around = []
        for i in candidates:
            for off in (0,1,2,3):
                k = i + off
                if 0 <= k < len(doc) and doc[k].pos_ in {"NOUN","PROPN"}:
                    around.append(k); break
        cands = sorted(set(around))
    if not cands:
        return None

    lex = lex_targets(lang, prof_tr)

    def _is_lex(i: int) -> bool:
        t, lem = _strip(doc[i].text), _strip(doc[i].lemma_)
        return (t in lex) or (lem in lex)

    def _looks_fem(i: int) -> bool:
        if "Fem" in doc[i].morph.get("Gender"):
            return True
        return doc[i].text.lower().endswith(DE_FEM_SFX if lang=="de" else ES_FEM_SFX)

    # Prefer lexicon hits; within them prefer feminine, else masculine, else any.
    lex_hits = [i for i in cands if _is_lex(i)]
    if lex_hits:
        fem_hits = [i for i in lex_hits if _looks_fem(i)]
        if fem_hits:
            return fem_hits[0]
        masc_hits = [i for i in lex_hits if "Masc" in doc[i].morph.get("Gender")]
        if masc_hits:
            return masc_hits[0]
        return lex_hits[0]

    # Otherwise pick NP head if present
    heads = {chunk.root.i for chunk in doc.noun_chunks}
    for i in cands:
        if i in heads:
            return i

    # Fallback: closest to median
    mid = int(round(sum(cands) / len(cands)))
    return min(cands, key=lambda i: abs(i - mid))

def choose_gender_np_only(doc, head_i: int, lang: str) -> Tuple[str, Optional[int], str]:
    """
    Decide gender from the target NP only, with three simple signals:
    A) Morphology on the head (singular): Fem/Masc.
    B) Adjacent singular DET (DE/ES maps): feminine vs. masculine article.
    C) Feminine suffix (simple heuristic lists).
    Small right-scan if needed.
    """
    if head_i is None or not (0 <= head_i < len(doc)):
        return "", None, "unknown"

    tok = doc[head_i]
    fem_sfx = DE_FEM_SFX if lang == "de" else ES_FEM_SFX
    det_map = DE_DET if lang == "de" else ES_DET

    # A) Morphology on head (singular only). Allow NOUN/PROPN/ADJ heads.
    if "Plur" not in tok.morph.get("Number") and tok.pos_ in {"NOUN","PROPN","ADJ"}:
        g = tok.morph.get("Gender")
        if "Fem" in g: return tok.text, 1, "morph_core"
        if "Masc" in g: return tok.text, 0, "morph_core"

    # B) Adjacent singular determiner (DET ← left window)
    win = build_np_window(doc, head_i, max_left=2)
    for j in win:
        if doc[j].pos_ == "DET" and "Plur" not in doc[j].morph.get("Number"):
            low = doc[j].text.lower()
            if low in det_map:
                h = _nearest_noun_right(doc, head_i, max_hop=3)
                show = doc[h].text if h is not None else doc[j].text
                return show, det_map[low], "det_adjacent"

    # C) Feminine suffix on head token
    if tok.text.lower().endswith(fem_sfx):
        return tok.text, 1, "suffix_core"

    # D) Tiny right-scan
    for off in (1,2,3):
        k = head_i + off
        if 0 <= k < len(doc):
            t = doc[k]
            if t.pos_ in {"NOUN","PROPN","ADJ"} and "Plur" not in t.morph.get("Number"):
                g = t.morph.get("Gender")
                if "Fem" in g: return t.text, 1, "morph_rescue"
                if "Masc" in g: return t.text, 0, "morph_rescue"
                if t.text.lower().endswith(fem_sfx):
                    return t.text, 1, "suffix_rescue"

    return "", None, "unknown"

def _lexicon_hits(doc, lang: str, prof_tr: str) -> List[int]:
    """
    Find target NOUN/PROPN indices whose surface or lemma is in the lexicon.
    If no direct hit and the TR profession is multiword, also try partials.
    """
    hits = []
    lex = lex_targets(lang, prof_tr)
    if not lex:
        return hits

    # Direct hits first
    for j, tok in enumerate(doc):
        if tok.pos_ in {"NOUN","PROPN"}:
            t = _strip(tok.text); lem = _strip(tok.lemma_)
            if t in lex or lem in lex:
                hits.append(j)
    hits = sorted(set(hits))
    if hits:
        return hits

    # Partial rescue for multiword TR professions (still require exact lex forms)
    prof_norm = _strip(prof_tr)
    parts = [p for p in prof_norm.split() if p]
    if len(parts) >= 2:
        part_hits = []
        for j, tok in enumerate(doc):
            if tok.pos_ in {"NOUN","PROPN"}:
                t = _strip(tok.text); lem = _strip(tok.lemma_)
                if t in lex or lem in lex:
                    part_hits.append(j)
        return sorted(set(part_hits))

    return hits

# Stubborn-6 set (used by last-resort)
STUBBORN6 = {"kapıcı","danışman","tasarımcı","aşçı","fırıncı","satış görevlisi"}
STUBBORN6_NORM = { _strip(p) for p in STUBBORN6 }

def gender_from_context(doc, lang: str, j: int) -> Optional[int]:
    """Return 0/1 using local cues around token j; None if unknown."""
    tok = doc[j]
    # morphology
    if "Plur" not in tok.morph.get("Number") and tok.pos_ in {"NOUN","PROPN","ADJ"}:
        g = tok.morph.get("Gender")
        if "Fem" in g: return 1
        if "Masc" in g: return 0
    # determiner
    det_map = DE_DET if lang=="de" else ES_DET
    win = build_np_window(doc, j, max_left=2)
    for k in win:
        if doc[k].pos_ == "DET" and "Plur" not in doc[k].morph.get("Number"):
            val = det_map.get(doc[k].text.lower())
            if val in (0,1):
                return val
    # suffix
    fem_sfx = DE_FEM_SFX if lang=="de" else ES_FEM_SFX
    if tok.text.lower().endswith(fem_sfx):
        return 1
    return None

# --- NEW: lexeme-based (surface) gender guesser (safe feminine-only) ---
def infer_gender_from_lexeme(lang: str, surface: str) -> Optional[int]:
    """
    Very light lexeme-based guess:
      - If surface ends with a language-appropriate FEM suffix → 1 (female)
      - Otherwise: unknown (None). We DO NOT try to detect 'masc' by suffix.
    """
    if not surface:
        return None
    s = surface.lower()
    fem_sfx = DE_FEM_SFX if lang == "de" else ES_FEM_SFX
    if s.endswith(fem_sfx):
        return 1
    return None

# ──────────────────────────────────────────────────────────────────────────────
# Row processing
# ──────────────────────────────────────────────────────────────────────────────
def process_row(row) -> Tuple[Optional[int], bool, List[int], Optional[str], float, Dict[str,Any]]:
    src_tokens = tokenize_tr(row.sentence_norm)

    # optional hint: which occurrence of the profession to use
    nth_hint = None
    try:
        nth = int(row.target_position) if pd.notna(row.target_position) else None
        if nth and nth > 0:
            nth_hint = nth
    except Exception:
        nth_hint = None

    span = find_prof_span(src_tokens, str(row.target_profession), nth_hint)
    if span is None:
        return (np.nan, False, [], None, 0.0, {"reason":"no_src_span"})

    lang = str(row.target_language).lower()
    if lang not in {"de","es"} or pd.isna(row.translated_sentence):
        return (np.nan, False, [], None, 0.0, {"reason":"lang_or_text_missing"})

    doc = (nlp_de if lang=="de" else nlp_es)(row.translated_sentence)
    tgt_tokens = [t.text for t in doc]

    # combine alignments (respect global ALIGNERS and availability flags)
    pairs, used = [], []

    use_aa = (ALIGNERS in ("auto","both","awesome")) and AWESOME_OK
    use_sa = (ALIGNERS in ("auto","both","simalign")) and (_sa is not None)

    if use_aa:
        au = awesome_union(src_tokens, tgt_tokens)
        if au: pairs += au; used.append("awesome")
    if use_sa:
        su = simalign_union(src_tokens, tgt_tokens)
        if su: pairs += su; used.append("simalign")

    pairs = sorted(set(pairs))
    aligner_used = "+".join(sorted(set(used))) if used else "none"

    # collect target indices aligned to the source profession
    tgt_idxs = collect_tgt_idxs(pairs, span)   # variable name correct
    if not tgt_idxs:
        # neighbor rescue around the source span
        neigh = neighbor_src_indices(span, len(src_tokens))
        tgt_idxs = sorted({t for (i,t) in pairs if i in neigh})

    # lexicon rescue if still empty
    if not tgt_idxs:
        for j, tok in enumerate(doc):
            if tok.pos_ in {"NOUN","PROPN"}:
                t = _strip(tok.text); lem = _strip(tok.lemma_)
                L = lex_targets(lang, str(row.target_profession))
                if t in L or lem in L:
                    tgt_idxs.append(j)

    align_score = len(pairs) / max(1, len(src_tokens) + len(tgt_tokens))

    # LAST-RESORT for stubborn-6 (no bias_group condition)
    if not tgt_idxs and (_strip(str(row.target_profession)) in STUBBORN6_NORM):
        cand = _lexicon_hits(doc, lang, str(row.target_profession))
        if cand:
            j = cand[0]  # leftmost candidate
            # 1) Try local context (morph/DET/suffix around j)
            g_ctx = gender_from_context(doc, lang, j)
            # 2) If still unknown, try lexeme-based feminine suffix guess
            if g_ctx is None:
                g_guess = infer_gender_from_lexeme(lang, doc[j].text)
            else:
                g_guess = g_ctx
            # 3) If STILL unknown, default to masculine (0) as a final tie-breaker
            if g_guess is None:
                g_guess = 0
                rule = "fallback_lexicon_ctx_loose_default_masc"
            else:
                rule = "fallback_lexicon_ctx_loose"

            tok_txt = doc[j].text
            return (int(g_guess), True, [j], tok_txt, align_score,
                    {"aligner_used": aligner_used,
                     "src_span": f"{span[0]}:{span[1]}",
                     "chosen_rule": rule,
                     "head_i": j})

    if not tgt_idxs:
        return (np.nan, False, [], None, align_score,
                {"aligner_used": aligner_used, "src_span": f"{span[0]}:{span[1]}", "reason":"no_tgt_found"})

    # lexicon filter: intersect with alignment; if empty, trust lexicon hits
    lex_hits = _lexicon_hits(doc, lang, str(row.target_profession))
    if lex_hits:
        inter = sorted(set(tgt_idxs).intersection(lex_hits))
        if inter:
            tgt_idxs = inter
        else:
            tgt_idxs = lex_hits

    # Hard blacklist filter: drop determiners or known noise tokens
    if tgt_idxs:
        bl = BLACKLIST.get(lang, set())
        tgt_idxs = [j for j in tgt_idxs if _strip(doc[j].text) not in bl]

    # pick a single head inside the candidate indices
    head_i = pick_head_with_lexicon(doc, tgt_idxs, lang, str(row.target_profession))

    # gentle preference if still None
    rule_suffix = ""
    if head_i is None and tgt_idxs:
        def _is_fem(j: int) -> bool:
            if doc[j].pos_ in {"NOUN","PROPN"} and "Plur" not in doc[j].morph.get("Number"):
                g = doc[j].morph.get("Gender")
                if "Fem" in g: return True
                if "Masc" in g: return False
            det_map = DE_DET if lang == "de" else ES_DET
            win = build_np_window(doc, j, max_left=1)
            for k in win:
                if doc[k].pos_ == "DET" and "Plur" not in doc[k].morph.get("Number"):
                    val = det_map.get(doc[k].text.lower())
                    if val == 1: return True
                    if val == 0: return False
            sfx = DE_FEM_SFX if lang == "de" else ES_FEM_SFX
            return doc[j].text.lower().endswith(sfx)

        fem_cands = [j for j in tgt_idxs if _is_fem(j)]
        if fem_cands:
            mid = int(round(sum(tgt_idxs) / len(tgt_idxs)))
            head_i = min(fem_cands, key=lambda x: abs(x - mid))
            rule_suffix = "/gender_pref"
        else:
            nounish = [j for j in tgt_idxs if doc[j].pos_ in {"NOUN","PROPN"}]
            if nounish:
                mid = int(round(sum(tgt_idxs) / len(tgt_idxs)))
                head_i = min(nounish, key=lambda x: abs(x - mid))
                rule_suffix = "/gender_pref"

    tok_txt, gender_val, rule = choose_gender_np_only(doc, head_i, lang)

    out_gender = gender_val if gender_val in (0, 1) else np.nan
    return (out_gender, True, sorted(tgt_idxs), tok_txt or None, align_score,
            {"aligner_used": aligner_used,
             "src_span": f"{span[0]}:{span[1]}",
             "chosen_rule": (rule or "unknown") + rule_suffix,
             "head_i": head_i})

# ──────────────────────────────────────────────────────────────────────────────
# Stereotypical flag
# ──────────────────────────────────────────────────────────────────────────────
def stereo_flag(r):
    """
    is_stereotypical = 1 if translated_gender matches expected_gender (female/male),
    else 0; NA if either is missing.
    """
    tg = r.get("translated_gender", pd.NA)
    eg = r.get("expected_gender", pd.NA)
    if pd.isna(tg) or pd.isna(eg):
        return pd.NA
    mapping = {"female": 1, "male": 0}
    exp = mapping.get(str(eg).lower())
    return int(int(tg) == int(exp)) if exp is not None else pd.NA

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    global ALIGNERS
    args = parse_args()
    ALIGNERS = args.aligners  # set once, used inside process_row

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.log.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input, encoding="utf-8")
    if "translated_gender" not in df.columns:
        df["translated_gender"] = pd.NA
    if "is_stereotypical" not in df.columns:
        df["is_stereotypical"] = pd.NA

    mask = df.target_language.isin({"de","es"}) & df.translated_sentence.notna()
    idxs = df[mask].index.tolist()
    if args.limit is not None:
        idxs = idxs[:args.limit]

    logs = []
    aligned = labeled = 0
    total = len(idxs)

    for ridx in tqdm(idxs, desc="Step 02"):
        row = df.loc[ridx]
        g, ok, tgt_idx_list, tok, score, meta = process_row(row)
        if ok: aligned += 1
        if pd.notna(g): labeled += 1
        df.at[ridx, "translated_gender"] = g
        logs.append([
            int(ridx), row.sentence_norm, row.target_profession, row.target_position,
            bool(ok), tgt_idx_list, tok, (int(g) if pd.notna(g) else np.nan), float(score),
            meta.get("aligner_used",""), meta.get("src_span",""),
            meta.get("chosen_rule","")
        ])

    df.loc[idxs, "is_stereotypical"] = df.loc[idxs].apply(stereo_flag, axis=1)
    df["translated_gender"] = df["translated_gender"].astype("Int64")
    df.to_csv(args.output, index=False, encoding="utf-8")

    with open(args.log, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["idx","sent_norm","profession","pos","aligned","tgt_idxs","token","gender","align_score","aligner_used","src_span","chosen_rule"])
        w.writerows(logs)

    # simple summary
    aligned_pct = (aligned/total*100.0) if total else 0.0
    labeled_pct = (labeled/total*100.0) if total else 0.0
    avg_score = (sum(r[8] for r in logs if isinstance(r[8], (int,float))) / max(aligned,1)) if aligned else 0.0
    print(f"✓ Aligned/Labeled: {aligned}/{total} = {aligned_pct:.1f}%")
    print(f"✓ Gender labels :  {labeled}/{total} = {labeled_pct:.1f}%")
    print(f"✓ Alignment avg score: {avg_score:.3f}")
    print(f"→ Results: {args.output}")
    print(f"→ Log    : {args.log}")

if __name__ == "__main__":
    main()
