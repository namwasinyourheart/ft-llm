import evaluate
from typing import List

def calc_bleu(predictions: List[str], references: List[str]) -> dict:
    bleu = evaluate.load('bleu')
    bleu_score = bleu.compute(predictions=predictions, references=references)['bleu']

    return {"bleu": bleu_score}


from typing import List, Dict

import evaluate
import nltk
from filelock import FileLock
from transformers.utils import is_offline_mode
 
import nltk
nltk.download('punkt_tab')
# try:
#     nltk.data.find("tokenizers/punkt")
# except (LookupError, OSError):
#     if is_offline_mode():
#         raise LookupError(
#             "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
#         )
#     with FileLock(".lock") as lock:
#         nltk.download("punkt", quiet=True)



def _postprocess_text(text: str) -> str:
    # rougeLSum expects newline after each sentence
    return "\n".join(nltk.sent_tokenize(text.strip()))


def calc_rouge(predictions: List[str], references: List[str]) -> Dict[str, float]:
    rouge = evaluate.load("rouge")
    predictions = [_postprocess_text(text) for text in predictions]
    references = [_postprocess_text(text) for text in references]

    rouge_score =  rouge.compute(predictions=predictions, references=references, use_stemmer=True)

    return rouge_score
