import json

from SelectTranslator import SelectTranslator

source = open("data/en-fi/newstest2017-enfi.en")
output = open("./outputs/newstest2017-en-fi-10.jsonl", mode="a")

# source = open("data/en-fi/test.src")
# output = open("./outputs/en-fi-10.jsonl", mode="a")

src = 'en'  # source language
trg = 'fi'  # target language
sampling = 10

translator = SelectTranslator(src, trg)

for source_sample in source:
    raw_data = translator.select_translate(source_sample, sampling)
    print(json.dumps(raw_data), end="\n", file=output, flush=True)
    print(raw_data)