from sentence_transformers import SentenceTransformer
import faiss
from typing import List
import time
def build_rag_index(texts: List[str], embedding_model="all-MiniLM-L6-v2"):
    """
    Строит FAISS индекс.
    """

    embedder = SentenceTransformer(embedding_model)
    embeddings = embedder.encode(texts, convert_to_numpy=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    return index, embedder



def rag_generate(
    query: str,
    model,
    tokenizer,
    index,
    embedder,
    corpus: List[str],
    top_k: int = 3,
    max_new_tokens: int = 4096,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
    **kwargs
):
    """
    Генерация ответа с использованием RAG.
    Возвращает (текст, время_генерации).
    """
    import time

    # Получение релевантного контекста
    query_vec = embedder.encode([query])
    distances, indices = index.search(query_vec, top_k)
    context = "\n".join([corpus[i] for i in indices[0]])

    # Формирование промпта
    prompt = f"Контекст:\n{context}\n\nВопрос:\n{query}\nОтвет:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Генерация с переданными параметрами
    start = time.time()
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample,
        **kwargs
    )
    latency = time.time() - start

    # Декодирование
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Убираем повтор промпта (опционально)
    if text.startswith(prompt):
        text = text[len(prompt):].lstrip()

    return text, latency