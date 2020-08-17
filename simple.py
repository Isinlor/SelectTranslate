import json

from transformers import MarianMTModel, MarianTokenizer

source = open("./data/en-fi/newstest2017-enfi.en")
output = open("./outputs/newstest2017-en-fi.txt", mode="a")

src = 'en'  # source language
trg = 'fi'  # target language

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

forward_model_name = f'Helsinki-NLP/opus-mt-{src}-{trg}'
forward_model = MarianMTModel.from_pretrained(forward_model_name)
forward_tokenizer = MarianTokenizer.from_pretrained(forward_model_name)
forward_model.to('cuda')

sources = []
for source_sample in source:
    sources.append(source_sample)

for chunk in chunks(sources, 10):
    batch = forward_tokenizer.prepare_translation_batch(src_texts=chunk)
    batch.to('cuda')
    gen = forward_model.generate(**batch)
    outputs = forward_tokenizer.batch_decode(gen, skip_special_tokens=True)
    for output_sample in outputs:
        print(output_sample, end="\n", file=output, flush=True)