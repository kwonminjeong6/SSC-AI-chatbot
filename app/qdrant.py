from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from app.config import QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION

class QdrantManager:
    def __init__(self):
        self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        self.collection_name = QDRANT_COLLECTION

    def create_collection(self):
        collections = self.client.get_collections()
        exists = any(c.name == self.collection_name for c in collections.collections)

        if exists:
            print(f"Collection already exists: {self.collection_name}")
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
        )
        print(f"Created collection: {self.collection_name}")

    def upsert_documents(self, documents):
        points = []
        for doc in documents:
            point = PointStruct(
                id=doc["chunk_id"],  # 이미 ingest.py에서 순수 UUID로 생성된 ID 사용
                vector=doc["vector"],
                payload={
                    "document_id": doc["document_id"],
                    "document_name": doc["document_name"],
                    "document_type": doc["document_type"],
                    "page": doc["page"],
                    "chunk_index": doc["chunk_index"],
                    "text": doc["text"],
                    "department": doc.get("department", "전체"),
                    "status": doc.get("status", "current")
                }
            )
            points.append(point)

        if points:
            self.client.upsert(collection_name=self.collection_name, points=points)
            print(f"Inserted {len(points)} chunks.")

    def search(self, vector, limit=5):
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=vector,
            limit=limit,
            with_payload=True
        )
        return results.points

qdrant = QdrantManager()