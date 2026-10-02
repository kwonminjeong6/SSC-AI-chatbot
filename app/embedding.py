from sentence_transformers import SentenceTransformer
from app.config import EMBEDDING_MODEL

class EmbeddingModel:
    def __init__(self):
        print(f"Loading embedding model: {EMBEDDING_MODEL}")
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        print("Embedding model loaded.")

    def encode(self, text: str):
        vector = self.model.encode(text, normalize_embeddings=True)
        return vector.tolist()

embedding_model = EmbeddingModel()