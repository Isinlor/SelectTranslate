import json

from select_translate.SelectTranslator import SelectTranslator

source = open("data/en-fi/test.src")
output = open("./outputs/en-fi-10.jsonl", mode="a")

src = 'en'  # source language
trg = 'fi'  # target language
sampling = 10

translator = SelectTranslator(src, trg)

for source_sample in source:
    raw_data = translator.select_translate(source_sample, sampling)
    print(json.dumps(raw_data), end="\n", file=output, flush=True)
    print(raw_data)