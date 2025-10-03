import torch
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore

from DataIterator import DataIterator

JSON_ROOT = "data/json/AA"
COLLECTION_NAME = "ruwiki_collection"
QDRANT_URL = "localhost"
QDRANT_PORT = "6333"
QDRANT_GRPC_PORT = "6334"
MODEL_NAME = "ai-sage/Giga-Embeddings-instruct"

TORCH_DTYPE = torch.bfloat16
CHUNK_SIZE_TOKENS = 2500
CHUNK_OVERLAP_TOKENS = 200
BATCH_SIZE = 8

# Load a sentence_transformers embedding model from the Hugging Face repository
embeddings = HuggingFaceEmbeddings(
    model_name=MODEL_NAME,
    encode_kwargs={"batch_size": BATCH_SIZE},
    model_kwargs={
        "device": "cuda",
        "trust_remote_code": True,
        "model_kwargs": {"torch_dtype": TORCH_DTYPE},
    },
)

# Recursive splitter splits the text by different separator characters ("\n\n" etc.)
text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
    embeddings._client.tokenizer,
    chunk_size=CHUNK_SIZE_TOKENS,
    chunk_overlap=CHUNK_OVERLAP_TOKENS,
)


def split_record(record: dict) -> list[Document]:
    """Splits wikiextractor package data records into text chunks - Documents."""
    text = record.get("text") or ""
    if not text.strip():
        return []
    id = record.get("id") or None
    title = record.get("title") or ""
    url = record.get("url") or None
    return text_splitter.create_documents(
        texts=[text],
        metadatas=[
            {
                "id": id,
                "title": title,
                "url": url,
            }
        ],
    )


def main_pipeline(json_root: str) -> None:
    """Split texts, vectorize, and save to Qdrant vector store."""
    vector_store: QdrantVectorStore | None = None
    buffer_texts: list[str] = []
    buffer_metas: list[dict] = []

    def commit_batch() -> None:
        """Create Qdrant vector store, or use existent one to save the data.
        Use the buffer to create batches of texts. Clear the buffer after."""
        nonlocal vector_store, buffer_texts, buffer_metas
        if not buffer_texts:
            return
        if not vector_store:
            vector_store = QdrantVectorStore.from_texts(
                texts=buffer_texts,
                metadatas=buffer_metas,
                embedding=embeddings,
                collection_name=COLLECTION_NAME,
                host=QDRANT_URL,
                port=QDRANT_PORT,
                grpc_port=QDRANT_GRPC_PORT,
                prefer_grpc=True,
            )
        else:
            vector_store.add_texts(
                texts=buffer_texts,
                metadatas=buffer_metas,
                batch_size=BATCH_SIZE,
                # Attribute "batch_size" not used unless it's smaller then our
                # BATCH_SIZE buffer. The default value for the "langchain_qdrant"
                # package is 64, but we need to make sure it is constant.
            )
        buffer_texts.clear()
        buffer_metas.clear()

    for record in DataIterator.iter_json(json_root):
        docs = split_record(record)
        for d in docs:
            buffer_texts.append(d.page_content)
            buffer_metas.append(d.metadata)
            if len(buffer_texts) >= BATCH_SIZE:
                commit_batch()
                
    commit_batch()


if __name__ == "__main__":
    main_pipeline(JSON_ROOT)
