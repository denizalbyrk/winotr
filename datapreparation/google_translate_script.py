# google_translate_script.py

import os
import datetime
from google.cloud import translate_v2 as translate

import os

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Downloads/mt-ClientAPI.json"



translate_client = translate.Client()

# INPUT PATH
input_path = "/mt_gender_tr/data/aggregates/tr_rawdataset.txt"

# OUTPUT PATH
output_dir = "/mt_gender_tr/translations/google/"
os.makedirs(output_dir, exist_ok=True)

# date of translation
today = datetime.date.today().isoformat()

# target languages
target_languages = {
    "de": "google_de_translation",
    "en": "google_en_translation",
    "es": "google_es_translation"
}

# read 
with open(input_path, "r", encoding="utf-8") as infile:
    sentences = [line.strip() for line in infile if line.strip()]

# translation of target languages
for lang_code, file_prefix in target_languages.items():
    output_filename = f"{file_prefix}_{today}.txt"
    output_path = os.path.join(output_dir, output_filename)

    with open(output_path, "w", encoding="utf-8") as outfile:
        outfile.write(f"# Translated with Google Translate API on {today} to {lang_code}\n")

        for sentence in sentences:
            try:
                result = translate_client.translate(sentence, source_language='tr', target_language=lang_code)
                translated = result["translatedText"]
                outfile.write(f"{sentence} ||| {translated}\n")
            except Exception as e:
                print(f"Error: {e} - Cümle: {sentence}")
                outfile.write(f"{sentence} ||| ERROR\n")

    print(f"{lang_code} translation completed: {output_path}")
