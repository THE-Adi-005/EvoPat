import numpy as np
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from src.embeddings import embed_texts


def cosine_similarity(text1, text2):
    v1 = embed_texts([text1])[0]
    v2 = embed_texts([text2])[0]
    return float(np.dot(v1, v2))


def rouge_similarity(text1, text2):
    scorer = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL'],
        use_stemmer=True
    )
    scores = scorer.score(text1, text2)
    return {
        "rouge1": scores['rouge1'].fmeasure,
        "rouge2": scores['rouge2'].fmeasure,
        "rougeL": scores['rougeL'].fmeasure
    }


def bert_similarity(text1, text2):
    P, R, F1 = bert_score(
        [text1],
        [text2],
        lang="en"
    )
    return float(F1[0])