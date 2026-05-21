import deepl
import datetime
import os

# my DeepL API key
auth_key = "personalized API number"  

translator = deepl.Translator(auth_key)

# input path
input_path = "/Users/denizalbayrak/Documents/mt_gender_tr/data/aggregates/tr_rawdataset.txt"

# output folder
output_dir = "/Users/denizalbayrak/Documents/mt_gender_tr/translations/deepl/"

# date
today = datetime.date.today().isoformat()

# target languages and folders
target_languages = {
    #"DE": "deepl_de_translation",
    "EN-US": "deepl_en_translation",
    #"ES": "deepl_es_translation"
}

# read the folders all at once
with open(input_path, "r", encoding="utf-8") as infile:
    sentences = [line.strip() for line in infile if line.strip()]

for lang_code, file_prefix in target_languages.items():
    output_filename = f"{file_prefix}_{today}.txt"
    output_path = os.path.join(output_dir, output_filename)

    with open(output_path, "w", encoding="utf-8") as outfile:
        outfile.write(f"# Translated with DeepL API on {today} to {lang_code}\n")

        for sentence in sentences:
            try:
                result = translator.translate_text(sentence, source_lang="TR", target_lang=lang_code)
                translated = result.text
                outfile.write(f"{sentence} ||| {translated}\n")
            except Exception as e:
                print(f"Hata: {e} - Cümle: {sentence}")
                outfile.write(f"{sentence} ||| ERROR\n")

    print(f"{lang_code} Translation completed, file saved: {output_path}")
