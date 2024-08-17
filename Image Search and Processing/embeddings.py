
# embedding.py
from sentence_transformers import SentenceTransformer
from numpy import dot
from numpy.linalg import norm

# Load the model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Function to compute embeddings for a given text
def compute_embedding(text: str):
    return model.encode(text)

# Function to compute cosine similarity between two embeddings
def cosine_similarity(embedding1, embedding2):
    return dot(embedding1, embedding2) / (norm(embedding1) * norm(embedding2))
