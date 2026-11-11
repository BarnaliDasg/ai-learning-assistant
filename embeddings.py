
from sentence_transformers import SentenceTransformer

# Load the embedding model once
model = SentenceTransformer("all-MiniLM-L6-v2")


def generate_embeddings(texts):
    """
    Convert a list of text chunks into numerical vectors.
    """
    embeddings = model.encode(texts)

    return embeddings

if __name__ == "__main__":

    sample_chunks = [
        "RAG retrieves information from external documents.",
        "FastAPI is a Python framework for building APIs.",
        "Embeddings represent text as numerical vectors."
    ]

    result = generate_embeddings(sample_chunks)

    print("Number of chunks:", len(sample_chunks))
    print("Embedding shape:", result.shape)