
from sentence_transformers import SentenceTransformer

# Load a pre-trained embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "RAG helps AI answer questions using external documents.",
    "Machine learning allows computers to learn from data.",
    "AI can retrieve information from uploaded PDF files."
]

# Convert sentences into numerical vectors
embeddings = model.encode(sentences)

print("Number of sentences:", len(sentences))
print("Embedding shape:", embeddings.shape)

print("\nFirst embedding:")
print(embeddings[0])
