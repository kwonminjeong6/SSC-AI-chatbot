import os
import uuid
import hashlib
from langchain_text_splitters import RecursiveCharacterTextSplitter
import fitz  # PyMuPDF
from app.config import QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION
from app.embedding import embedding_model
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

class DocumentIngester:
    def __init__(self):
        self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        self.collection_name = QDRANT_COLLECTION

    def create_collection(self):
        collections = self.client.get_collections()
        exists = any(c.name == self.collection_name for c in collections.collections)
        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
            )
            print(f"Created collection: {self.collection_name}")
        else:
            print(f"Collection already exists: {self.collection_name}")

    def run_ingestion(self, docs_dir="data/documents"):
        self.create_collection()
        
        # 랭체인의 정교한 텍스트 스플리터 활용 (문맥 유지력 향상)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", " ", ""]
        )

        if not os.path.exists(docs_dir):
            print(f"Directory not found: {docs_dir}")
            return

        total_chunks = 0
        for filename in os.listdir(docs_dir):
            if not filename.endswith(".pdf"):
                continue

            file_path = os.path.join(docs_dir, filename)
            print(f"\nProcessing: {file_path}")

            doc = fitz.open(file_path)
            documents_to_upsert = []

            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                if not text.strip():
                    continue

                chunks = text_splitter.split_text(text)
                for chunk_index, chunk in enumerate(chunks):
                    # 파일 내용 해시 기반 고유 문서 ID 생성
                    doc_hash = hashlib.sha256(filename.encode()).hexdigest()[:16]
                    
                    # Qdrant가 허용하는 순수 UUID 생성 (에러 방지)
                    chunk_unique_string = f"{doc_hash}-{page_num}-{chunk_index}"
                    point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_unique_string))

                    vector = embedding_model.encode(chunk)

                    documents_to_upsert.append({
                        "chunk_id": point_id,
                        "vector": vector,
                        "document_id": doc_hash,
                        "document_name": filename,
                        "document_type": "규정",
                        "page": page_num + 1,
                        "chunk_index": chunk_index,
                        "text": chunk,
                        "department": "학사지원처",
                        "status": "current"
                    })

            if documents_to_upsert:
                # 배치 단위로 Qdrant에 업서트
                points = [
                    PointStruct(
                        id=d["chunk_id"],
                        vector=d["vector"],
                        payload={k: v for k, v in d.items() if k not in ["chunk_id", "vector"]}
                    ) for d in documents_to_upsert
                ]
                self.client.upsert(collection_name=self.collection_name, points=points)
                print(f"Inserted {len(points)} chunks from {filename}.")
                total_chunks += len(points)

        print(f"\nCompleted! Total chunks inserted: {total_chunks}")

if __name__ == "__main__":
    ingester = DocumentIngester()
    ingester.run_ingestion()