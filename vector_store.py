
import chromadb

from embeddings import generate_embeddings


# Create a persistent ChromaDB client
client = chromadb.PersistentClient(path="./chroma_db")


# Create or retrieve a collection
collection = client.get_or_create_collection(
    name="learning_documents",
    metadata={"hnsw:space": "cosine"}
)


def store_chunks(chunks, filename):

    if not chunks:
        return 0

    # Generate embeddings for all chunks
    embeddings = generate_embeddings(chunks)

    # Create unique IDs
    ids = [
        f"{filename}_{i}"
        for i in range(len(chunks))
    ]

    # Store chunks, embeddings and metadata
    collection.upsert(
        ids=ids,
        documents=chunks,
        embeddings=embeddings.tolist(),
        metadatas=[
            {"source": filename}
            for _ in chunks
        ]
    )

    return len(chunks)


def search_chunks(query, n_results=3):

    # Check whether the database contains records
    total_records = collection.count()

    if total_records == 0:
        return None

    # Convert the user's question into an embedding
    query_embedding = generate_embeddings([query])

    # Search ChromaDB for similar vectors
    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=min(n_results, total_records)
    )

    return results


if __name__ == "__main__":

    # Sample document chunks
    sample_chunks = [
        "RAG retrieves relevant information from documents.",
        "FastAPI is used to build Python APIs.",
        "Embeddings represent text as numerical vectors."
    ]

    # Store sample chunks
    count = store_chunks(
        sample_chunks,
        "test_document.pdf"
    )

    print("Stored chunks:", count)
    print("Total records in database:", collection.count())

    # Test semantic search
    query = "How does AI find information in documents?"

    results = search_chunks(query)

    print("\nUser question:", query)
    print("\nRelevant results:\n")

    if results:

        for i, document in enumerate(results["documents"][0]):

            print(f"Result {i + 1}:")
            print("Document:", document)
            print("Source:", results["metadatas"][0][i])
            print("Distance:", results["distances"][0][i])
            print("-" * 50)

    else:
        print("No documents found in the database.")
