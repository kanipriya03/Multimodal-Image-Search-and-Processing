# embedding.py
from sentence_transformers import SentenceTransformer
from numpy import dot
from numpy.linalg import norm

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def compute_embedding(text: str):
    return model.encode(text)

def cosine_similarity(embedding1, embedding2):
    return dot(embedding1, embedding2) / (norm(embedding1) * norm(embedding2))
