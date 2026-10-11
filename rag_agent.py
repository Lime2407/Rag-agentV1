"""
RAG-агент: "спроси у своих документов".

Как это работает:
1. Документы из папки documents/ разбиваются на чанки
2. Каждый чанк превращается в эмбеддинг через Ollama (модель nomic-embed-text)
3. Эмбеддинги сохраняются в локальной векторной базе ChromaDB
4. При вопросе пользователя ищутся самые похожие по смыслу чанки
5. Найденный контекст подставляется в промпт LLM, которая формирует ответ

Полностью локально, бесплатно, без API-ключей.
"""

import os
import glob
import chromadb
from ollama import chat, embed

CHAT_MODEL = "llama3.1"
EMBED_MODEL = "nomic-embed-text"
DOCS_DIR = "documents"
CHUNK_SIZE = 500        # символов на чанк
CHUNK_OVERLAP = 50      # перекрытие между чанками
TOP_K = 3                # сколько релевантных чанков подставлять в контекст

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="documents")


# ---------- ЗАГРУЗКА И РАЗБИЕНИЕ ДОКУМЕНТОВ ----------

def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Разбивает текст на пересекающиеся чанки фиксированной длины."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def load_documents() -> list[tuple[str, str]]:
    """Загружает все .txt файлы из DOCS_DIR. Возвращает список (имя_файла, текст)."""
    docs = []
    for path in glob.glob(os.path.join(DOCS_DIR, "*.txt")):
        with open(path, "r", encoding="utf-8") as f:
            docs.append((os.path.basename(path), f.read()))
    return docs


# ---------- ИНДЕКСАЦИЯ ----------

def index_documents():
    """Читает документы, режет на чанки, считает эмбеддинги и сохраняет в ChromaDB."""
    docs = load_documents()
    if not docs:
        print(f"⚠️  В папке {DOCS_DIR}/ нет .txt файлов. Добавь документы и запусти снова.")
        return

    # Очищаем коллекцию перед переиндексацией
    existing_ids = collection.get()["ids"]
    if existing_ids:
        collection.delete(ids=existing_ids)

    chunk_id = 0
    for filename, text in docs:
        chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
        for chunk in chunks:
            if not chunk.strip():
                continue
            embedding = embed(model=EMBED_MODEL, input=chunk).embeddings[0]
            collection.add(
                ids=[f"chunk_{chunk_id}"],
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[{"source": filename}],
            )
            chunk_id += 1

    print(f"✅ Проиндексировано {chunk_id} чанков из {len(docs)} документов.")


# ---------- ПОИСК + ОТВЕТ ----------

def retrieve_context(query: str, top_k: int = TOP_K) -> list[dict]:
    """Ищет top_k самых релевантных чанков для запроса."""
    query_embedding = embed(model=EMBED_MODEL, input=query).embeddings[0]
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)

    chunks = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        chunks.append({"text": doc, "source": meta["source"]})
    return chunks


def answer_question(query: str) -> str:
    """Находит релевантный контекст и генерирует ответ на его основе."""
    context_chunks = retrieve_context(query)

    if not context_chunks:
        return "В базе документов ничего не найдено. Сначала проиндексируй документы."

    context_text = "\n\n".join(
        f"[Источник: {c['source']}]\n{c['text']}" for c in context_chunks
    )

    prompt = f"""Ответь на вопрос пользователя, используя ТОЛЬКО информацию из контекста ниже.
Если в контексте нет ответа — честно скажи, что не знаешь, не выдумывай.

Контекст:
{context_text}

Вопрос: {query}

Ответ:"""

    response = chat(model=CHAT_MODEL, messages=[{"role": "user", "content": prompt}])
    return response.message.content


# ---------- ЗАПУСК ----------

if __name__ == "__main__":
    print("📚 RAG-агент: чат с твоими документами\n")

    if not collection.get()["ids"]:
        print("База пуста, индексирую документы...")
        index_documents()
    else:
        reindex = input("База уже проиндексирована. Переиндексировать? (y/n): ")
        if reindex.strip().lower() == "y":
            index_documents()

    print("\nГотово! Задавай вопросы (exit — выход, reindex — переиндексировать)\n")

    while True:
        query = input("Вы: ")
        if query.strip().lower() in ("exit", "quit"):
            break
        if query.strip().lower() == "reindex":
            index_documents()
            continue

        answer = answer_question(query)
        print(f"\nАгент: {answer}\n")
