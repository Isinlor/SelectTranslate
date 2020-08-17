import json

from SelectTranslator import SelectTranslator

input = open("./outputs/newstest2017-en-fi-10.jsonl")
output = open("./outputs/newstest2017-en-fi-10.txt", mode='a')

# input = open("./outputs/en-fi-10.jsonl")
# output = open("./outputs/en-fi-10.txt", mode='a')

src = 'en'  # source language
trg = 'fi'  # target language

translator = SelectTranslator(src, trg)

for raw_data_input in input:
    print(translator.select_best_translation(**json.loads(raw_data_input)), end="\n", file=output, flush=True)