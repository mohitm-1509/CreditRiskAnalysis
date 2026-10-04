"""ChromaDB vector store for Basel III / IFRS 9 documents."""

from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

from config.settings import PROJECT_ROOT

DOCS_DIR = PROJECT_ROOT / "docs" / "basel3"
CHROMA_DIR = PROJECT_ROOT / "data" / "chroma_db"
COLLECTION_NAME = "credit_risk_docs"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks by words."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start = end - overlap
    return chunks


def load_and_chunk_documents() -> tuple[list[str], list[dict]]:
    """Load all markdown documents from docs directory and chunk them."""
    all_chunks = []
    all_metadata = []

    for doc_path in sorted(DOCS_DIR.glob("*.md")):
        text = doc_path.read_text(encoding="utf-8")
        sections = text.split("\n## ")

        for i, section in enumerate(sections):
            if not section.strip():
                continue
            section_title = section.split("\n")[0].strip("# ")
            chunks = chunk_text(section)

            for j, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_metadata.append({
                    "source": doc_path.name,
                    "section": section_title,
                    "chunk_index": j,
                })

    print(f"Loaded {len(all_chunks)} chunks from {len(list(DOCS_DIR.glob('*.md')))} documents")
    return all_chunks, all_metadata


def create_vector_store() -> chromadb.Collection:
    """Create or load the ChromaDB vector store."""
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    chunks, metadata = load_and_chunk_documents()

    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode(chunks, show_progress_bar=True, batch_size=32)

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    ids = [f"chunk_{i}" for i in range(len(chunks))]
    collection.add(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=chunks,
        metadatas=metadata,
    )

    print(f"Vector store created with {collection.count()} chunks")
    return collection


def query_store(query: str, n_results: int = 3) -> list[dict]:
    """Query the vector store and return relevant chunks."""
    model = SentenceTransformer(EMBEDDING_MODEL)
    query_embedding = model.encode([query]).tolist()

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(COLLECTION_NAME)

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results,
    )

    retrieved = []
    for i in range(len(results["documents"][0])):
        retrieved.append({
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],
        })

    return retrieved


if __name__ == "__main__":
    create_vector_store()
