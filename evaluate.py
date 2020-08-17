from nlp import load_metric

# target = open("./data/en-fi/test.trg")
# output = open("./outputs/en-fi.txt")

target = open("./data/en-fi/newstest2017-enfi.fi")
output = open("./outputs/newstest2017-en-fi-10.txt")

bleu_metric = load_metric("bleu")
bert_score_metric = load_metric("bertscore", device="cuda")

targets_bleu = []
targets_bert = []
outputs = []

for target_sample, output_sample in zip(target, output):
    targets_bleu.append([target_sample])
    targets_bert.append(target_sample)
    outputs.append(output_sample)

print(bleu_metric.compute(outputs, targets_bleu))

print(bert_score_metric.compute(
    outputs,
    targets_bert,
    lang='fi',
    model_type="roberta-base",
    device="cuda"
)['f1'].mean())