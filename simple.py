import json

from transformers import MarianMTModel, MarianTokenizer

from SelectTranslator import SelectTranslator

source = open("data/en-fi/test.src")
output = open("./outputs/en-fi.txt", mode="a")

src = 'en'  # source language
trg = 'fi'  # target language
sampling = 10

forward_model_name = f'Helsinki-NLP/opus-mt-{src}-{trg}'
forward_model = MarianMTModel.from_pretrained(forward_model_name)
forward_tokenizer = MarianTokenizer.from_pretrained(forward_model_name)
forward_model.to('cuda')

sources = []
for source_sample in source:
    sources.append(source_sample)

batch = forward_tokenizer.prepare_translation_batch(src_texts=sources)
batch.to('cuda')
gen = forward_model.generate(**batch)
words = forward_tokenizer.batch_decode(gen, skip_special_tokens=True)
print(words)

