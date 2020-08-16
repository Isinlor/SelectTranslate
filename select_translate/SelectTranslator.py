from transformers import MarianTokenizer, MarianMTModel
from typing import List
from nlp import load_metric
import numpy as np


class SelectTranslator:

    def __init__(self, source_language: str, target_language: str):

        self.source_language = source_language
        self.target_language = target_language

        forward_model_name = f'Helsinki-NLP/opus-mt-{source_language}-{target_language}'
        self.forward_model = MarianMTModel.from_pretrained(forward_model_name)
        self.forward_tokenizer = MarianTokenizer.from_pretrained(forward_model_name)
        self.forward_model.to('cuda')

        backward_model_name = f'Helsinki-NLP/opus-mt-{target_language}-{source_language}'
        self.backward_model = MarianMTModel.from_pretrained(backward_model_name)
        self.backward_tokenizer = MarianTokenizer.from_pretrained(backward_model_name)
        self.backward_model.to('cuda')

        self.bleu_metric = load_metric("bleu")
        self.bert_score_metric = load_metric("bertscore", device="cuda")

    def translate(self, text: str, samples: int):

        raw_data = self.select_translate(text, samples)

        return self.select_best_translation(**raw_data)

    def select_best_translation(self, translations, bert_scores, bleu_scores):

        bleu_weight = 1
        bert_weight = 5
        scores = (np.array(bleu_scores) * bleu_weight + np.array(bert_scores) * bert_weight) / (bleu_weight + bert_weight)
        best_index = int(np.argmax(scores))

        return translations[best_index]

    def select_translate(self, text: str, samples: int):

        translations = self.forward_translate(text, samples)
        back_translations = self.back_translate(translations)
        bert_scores = self.bert_score_back_translations(text, back_translations)
        bleu_scores = self.bleu_score_back_translations(text, back_translations)

        return {
            "text": str(text),
            "translations": list(translations),
            "back_translations": list(back_translations),
            "bert_scores": list(bert_scores),
            "bleu_scores": list(bleu_scores)
        }

    def back_translate(self, texts: List[str]) -> List[str]:
        batch = self.backward_tokenizer.prepare_translation_batch(src_texts=texts)
        batch.to('cuda')
        gen = self.backward_model.generate(**batch)
        words: List[str] = self.backward_tokenizer.batch_decode(gen, skip_special_tokens=True)
        return words

    def forward_translate(self, text: str, samples: int) -> List[str]:
        batch = self.forward_tokenizer.prepare_translation_batch(src_texts=[text])
        batch.to('cuda')
        gen = self.forward_model.generate(**batch, num_return_sequences=samples, num_beams=samples)
        words: List[str] = self.forward_tokenizer.batch_decode(gen, skip_special_tokens=True)
        return words

    def bert_score_back_translations(self, text, back_translations):
        source_texts = [text for _ in range(0, len(back_translations))]
        bert_scores = self.normalize(self.bert_score(source_texts, back_translations, self.source_language))
        return bert_scores

    def bleu_score_back_translations(self, text, back_translations):
        source_texts = [text for _ in range(0, len(back_translations))]
        bleu_scores = self.normalize(self.bleu_score(source_texts, back_translations))
        return bleu_scores

    def bleu_score(self, predictions: List[str], references: List[str]):
        scores = []
        for prediction, reference in zip(predictions, references):
            scores.append(self.bleu_metric.compute(
                [prediction], [[reference]]
            )['bleu'])
        return scores

    def bert_score(self, predictions: List[str], references: List[str], lang: str = "en"):
        return self.bert_score_metric.compute(
            predictions,
            references,
            lang=lang, model_type="roberta-base", device="cuda"
        )['f1'].tolist()

    def normalize(self, raw_input: List[float]) -> List[float]:
        input = np.array(raw_input)
        if len(input) < 2: return raw_input
        if np.max(input) == np.min(input): return raw_input
        return (input - np.min(input)) / (np.max(input) - np.min(input))