import openai
import datetime
import os

client = openai.OpenAI(api_key="personalized API number")

input_path = "/mt_gender_tr/data/aggregates/tr_rawdataset.txt"
output_dir = "/mt_gender_tr/translations/openai/"
os.makedirs(output_dir, exist_ok=True)

today = datetime.date.today().isoformat()

target_languages = {
    "German": "openai_de_translation",
    "English": "openai_en_translation",
    "Spanish": "openai_es_translation"
}

with open(input_path, "r", encoding="utf-8") as infile:
    sentences = [line.strip() for line in infile if line.strip()]

for lang_name, file_prefix in target_languages.items():
    output_file = os.path.join(output_dir, f"{file_prefix}_{today}.txt")

    with open(output_file, "w", encoding="utf-8") as outfile:
        outfile.write(f"# Translated with OpenAI GPT (new SDK) on {today} to {lang_name}\n")

        for sentence in sentences:
            try:
                prompt = f"Please translate the following Turkish sentence into {lang_name}:\n\"{sentence}\""
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",  # veya "gpt-4"
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3
                )
                translated = response.choices[0].message.content.strip()
                outfile.write(f"{sentence} ||| {translated}\n")
            except Exception as e:
                print(f"Hata oluştu: {e} - Cümle: {sentence}")
                outfile.write(f"{sentence} ||| ERROR\n")

    print(f"{lang_name} çevirisi tamamlandı: {output_file}")
