from openai import OpenAI
import traceback
import json
import os

# OpenAI API 키 설정
os.environ["OPENAI_API_KEY"] = "sk-gk9zo_Fhb1NdnI_ExiKLrqt-d4a9lqewkEtkm_DWwlT3BlbkFJAKOMtgHLDOvnFmeBuBPPFXdBiA2hM4wu9PDwPY-z4A"
client = OpenAI()

def translate_text(text, target_language="English"):
    prompt = f"Translate the following text to {target_language}, Be sure to output only the translation results: {text}"
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    translated_text = response.choices[0].message.content
    return translated_text

input_file_path = './eval.jsonl'
output_file_path = './translated_eval.jsonl'

with open(input_file_path, 'r', encoding='utf-8') as infile, open(output_file_path, 'w', encoding='utf-8') as outfile:
    for line in infile:
        data = json.loads(line)
        if 'msg' in data:
            for msg in data['msg']:
                msg['content'] = translate_text(msg['content'], target_language="English")
        json.dump(data, outfile, ensure_ascii=False)
        outfile.write('\n')

print("번역이 완료되었습니다. 결과는 translated_documents.jsonl 파일에 저장되었습니다.")