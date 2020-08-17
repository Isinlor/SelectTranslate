import sacrebleu
from nlp import load_metric

# target = open("./data/en-fi/test.trg")
# output = open("./outputs/en-fi.txt")

target = open("./data/en-fi/newstest2017-enfi.fi")
output = open("./outputs/newstest2017-en-fi.txt")

bert_score_metric = load_metric("bertscore", device="cuda")

targets = []
outputs = []

for target_sample, output_sample in zip(target, output):
    targets.append(target_sample)
    outputs.append(output_sample)

print(sacrebleu.corpus_bleu(outputs, [targets]).score)

print(bert_score_metric.compute(
    outputs,
    targets,
    lang='fi',
    model_type="roberta-base",
    device="cuda"
)['f1'].mean())