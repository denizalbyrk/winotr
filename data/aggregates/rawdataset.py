import pandas as pd

# 1. CSV dosyasının yolu
csv_path = "/Users/denizalbayrak/Documents/mt_gender_tr/data/aggregates/combined_dataset.txt"

# 2. TXT dosyasının hedef yolu
txt_path = "/Users/denizalbayrak/Documents/mt_gender_tr/data/aggregates/tr_rawdataset.txt"

# 3. CSV dosyasını oku
df = pd.read_csv(csv_path, sep="\t")  # Eğer tab-separated değilse sep="," yap

# 4. 'sentence' sütununu txt dosyasına yaz
with open(txt_path, "w", encoding="utf-8") as f:
    for sentence in df['sentence']:
        f.write(sentence.strip() + "\n")

print(f"Dosya başarıyla kaydedildi: {txt_path}")
