import pandas as pd, pathlib

ROOT = pathlib.Path("/Users/denizalbayrak/Documents/mt_gender_tr")
CSV  = ROOT / "results" / "01_combined_dataframe.csv"   # ← dosyanız burada


df = pd.read_csv(CSV)

# 1) Her (tool_lang) kombinasyonu için beklenen toplam
expect = df.groupby("tool_lang")["sentence_norm"].nunique().to_dict()

# 2) Gerçekte gelen satırlar
have   = df.dropna(subset=["translated_sentence"]) \
           .groupby("tool_lang")["sentence_norm"].nunique().to_dict()

report = []
for tl in sorted(expect):
    miss = expect[tl] - have.get(tl, 0)
    report.append((tl, expect[tl], have.get(tl, 0), miss))

print(f"{'tool_lang':15} expected  have  missing")
for tl, e, h, m in report:
    print(f"{tl:15} {e:8} {h:5} {m:8}")
