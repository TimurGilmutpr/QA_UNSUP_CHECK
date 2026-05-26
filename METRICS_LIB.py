from nltk.translate.bleu_score import sentence_bleu
from typing import List

import torch
import numpy as np
import math
import time

def compute_perplexity(model, tokenizer, texts: List[str]):
    """
    Вычисляет перплексию как exp(loss)
    """

    model.eval()
    losses = []

    for text in texts:
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        losses.append(loss.item())

    mean_loss = np.mean(losses)
    perplexity = math.exp(mean_loss)

    return perplexity


def compute_bleu(references: List[str], predictions: List[str]):
    scores = []

    for ref, pred in zip(references, predictions):
        score = sentence_bleu([ref.split()], pred.split())
        scores.append(score)

    return float(np.mean(scores))


def compute_tok_per_word(tokenizer, texts: List[str]):
    ratios = []

    for text in texts:
        tokens = tokenizer.tokenize(text)
        words = text.split()
        if len(words) > 0:
            ratios.append(len(tokens) / len(words))

    return float(np.mean(ratios))



def compute_latency(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int = 128
):
    times = []

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        start = time.time()
        model.generate(**inputs, max_new_tokens=max_new_tokens)
        times.append(time.time() - start)

    return float(np.mean(times))


def compute_token_speed(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int = 128
) -> float:
    """
    Вычисляет среднюю скорость генерации токенов (токенов в секунду) на списке промптов.

    Args:
        model: языковая модель (трансформер)
        tokenizer: токенизатор
        prompts: список входных текстов
        max_new_tokens: максимальное число генерируемых токенов

    Returns:
        Средняя скорость в токенах/сек
    """
    model.eval()
    speeds = []

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_len = inputs['input_ids'].shape[1]

        start = time.time()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
        elapsed = time.time() - start

        generated_len = outputs.shape[1] - input_len
        if elapsed > 0:
            speeds.append(generated_len / elapsed)

    if not speeds:
        return 0.0

    return float(np.mean(speeds))



