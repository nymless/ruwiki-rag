import torch
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore

COLLECTION_NAME = "ruwiki_collection"
QDRANT_HOST = "localhost"
QDRANT_PORT = "6333"
QDRANT_GRPC_PORT = "6334"
MODEL_NAME = "ai-sage/Giga-Embeddings-instruct"
TORCH_DTYPE = torch.bfloat16
TOP_K = 4

# Load a sentence_transformers embedding model from the Hugging Face repository
embeddings = HuggingFaceEmbeddings(
    model_name=MODEL_NAME,
    encode_kwargs={},
    model_kwargs={
        "device": "cuda",
        "trust_remote_code": True,
        "model_kwargs": {"torch_dtype": TORCH_DTYPE},
    },
)

vector_store = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    collection_name=COLLECTION_NAME,
    host=QDRANT_HOST,
    port=QDRANT_PORT,
    grpc_port=QDRANT_GRPC_PORT,
    prefer_grpc=True,
)


def retrieve(query: str) -> list[Document]:
    results = vector_store.similarity_search(query=query, k=TOP_K)
    return results


if __name__ == "__main__":
    query = "Какое население у государства Литва?"
    results = retrieve(query)
    for doc in results:
        print(doc.page_content)
