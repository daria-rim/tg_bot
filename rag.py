import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter

with open("knowledge_base.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Настройка "умного" сплиттера
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    length_function=len,
    separators=[
        "\n\n", "\n", ".", "!", "?", ";", ":", " ", ""
    ],
)

# Разделяем
chunks = text_splitter.split_text(text)
# Для наглядности используем простые ID
ids = [str(i) for i in range(len(chunks))]

print(f"Количество чанков: {len(chunks)}")

# Создаём БД (в памяти)
client = chromadb.Client()
# Создаём русскоязычную embedding функцию
russian_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
# Создаём коллекцию в БД
collection = client.create_collection("Healthy_shoes_knowledge_base", embedding_function=russian_ef)
# Добавляем документы из нашей базы знаний в коллекцию
collection.add(documents=chunks, ids=ids)

if __name__ == "__main__":
    query = "Как понять, что обувь подходит?"
    results = collection.query(query_texts=[query], n_results=3)
    print(results)