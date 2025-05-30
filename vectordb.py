import os
import tiktoken
import uuid
from openai import OpenAI
from typing import List, Sequence
from qdrant_client import QdrantClient, models

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
_MAX_ITEMS   = 96
_MAX_TOKENS  = 8192 - 256      # 256-token safety cushion
_enc         = tiktoken.get_encoding("cl100k_base")   # ada-002 encoder
# Set OpenAI API key from environment variable

class VectorDB:
    def __init__(self, collection_name: str, qdrant_url=":memory:"):
        self.collection_name = collection_name
        self.client = QdrantClient(path=qdrant_url)
        self.embedding_model = "text-embedding-ada-002"
        self.vector_size = 1536

        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=self.vector_size, distance=models.Distance.COSINE
            ),
        )

    def get_embedding(self, text: str):
        response = client.embeddings.create(input=text, model=self.embedding_model)
        return response.data[0].embedding
    
    def _token_len(self, text: str) -> int:
        return len(_enc.encode(text))

    def get_embeddings(self, texts: Sequence[str]) -> List[List[float]]:
        all_vecs: list[list[float]] = []
        start = 0
        while start < len(texts):
            # Build one safe chunk
            tok_sum = 0
            end = start
            while (end < len(texts)
                   and (end - start) < _MAX_ITEMS
                   and (tok_sum + self._token_len(texts[end])) < _MAX_TOKENS):
                tok_sum += self._token_len(texts[end])
                end += 1

            batch = texts[start:end]
            resp  = client.embeddings.create(input=batch, model=self.embedding_model)
            vecs  = [d.embedding for d in sorted(resp.data, key=lambda d: d.index)]
            all_vecs.extend(vecs)

            start = end
        return all_vecs

    def add_text(self, id: int, text: str, metadata: dict = None):
        embedding = self.get_embedding(text)
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(id=id, vector=embedding, payload={"text": text, **(metadata or {})})
            ],
        )

    def add_texts( self, texts: Sequence[str], metadatas: Sequence[dict] | None = None,) -> list[int]:
        
            metadatas = metadatas or [{}] * len(texts)
            ids = [uuid.uuid4().int >> 64 for _ in texts]

            embeddings = self.get_embeddings(texts)          # existing batch helper

            points = [
                models.PointStruct(
                    id=pid,
                    vector=vec,
                    payload={"text": txt, **meta},
                )
                for pid, vec, txt, meta in zip(ids, embeddings, texts, metadatas)
            ]
            self.client.upsert(self.collection_name, points=points)
            #return ids
    
    def similarity_search(self, query_text: str, limit: int = 6):
        query_embedding = self.get_embedding(query_text)
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector= query_embedding,
            limit=limit,
        )
        return [
            {"id": result.id, "score": result.score, "text": result.payload["text"], "metadata": result.payload}
            for result in results
        ]


# Example usage (remove or comment out in production)
if __name__ == "__main__":
    vector_db = VectorDB(collection_name="my_collection")

    # Add sample texts
    vector_db.add_text(id=1, text="Hello world!", metadata={"category": "greeting"})
    vector_db.add_text(id=2, text="Vector databases are awesome!", metadata={"category": "database"})

    # Search for similar texts
    results = vector_db.similarity_search("I love vector DBs", limit=2)

    for result in results:
        print(result)
