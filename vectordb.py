import os
import tiktoken
import logging
import logging.handlers        # ← add this line
import uuid
from openai import OpenAI
from typing import List, Sequence
from qdrant_client import QdrantClient, models
from mcp_context import set_conversation_id, get_conversation_id


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
_MAX_ITEMS   = 96
_MAX_TOKENS  = 8192 - 256      # 256-token safety cushion
_enc         = tiktoken.get_encoding("cl100k_base")   # ada-002 encoder

LOG_FILE = "logs/mcp_server.log"
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.handlers.RotatingFileHandler(LOG_FILE, maxBytes=5_000_000, backupCount=5)
    ]
)
logger = logging.getLogger("vector_db")

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
        logger.debug(f"[{get_conversation_id()}] Embedding single text "
                     f"(≈{len(text)} chars)")
        response = client.embeddings.create(input=text, model=self.embedding_model)
        return response.data[0].embedding
    
    def _token_len(self, text: str) -> int:
        return len(_enc.encode(text))

    def get_embeddings(self, texts: Sequence[str]) -> List[List[float]]:
        logger.info(f"[{get_conversation_id()}] Embedding batch of {len(texts)} texts")
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
            logger.debug(f"[{get_conversation_id()}]  » Sub-batch {start}:{end} "
                         f"({len(batch)} items, {tok_sum} tokens)")
            resp  = client.embeddings.create(input=batch, model=self.embedding_model)
            vecs  = [d.embedding for d in sorted(resp.data, key=lambda d: d.index)]
            all_vecs.extend(vecs)

            start = end
            logger.info(f"[{get_conversation_id()}] Embedded {len(texts)} texts "
                    f"into {len(all_vecs)} vectors")
        return all_vecs

    def add_text(self, id: int, text: str, metadata: dict = None):
        logger.info(f"[{get_conversation_id()}] Upserting point id={id} "
                    f"into '{self.collection_name}'")
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
            logger.info(f"[{get_conversation_id()}] add_texts: {len(texts)} items → "
                f"collection '{self.collection_name}'")
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
            logger.info(f"[{get_conversation_id()}] Upserted {len(points)} points")

            #return ids
    
    def similarity_search(self, query_text: str, product: str, limit: int = 6):
        query_embedding = self.get_embedding(query_text)
        # results = self.client.search(
        #     collection_name=self.collection_name,
        #     query_vector= query_embedding,
        #     limit=limit,
        # )
        logger.info(f"[{get_conversation_id()}] similarity_search "
                    f"('{query_text[:40]}…', product={product})")

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=limit,
            query_filter=models.Filter(
                must=[models.FieldCondition(key="product",
                                            match=models.MatchValue(value=product))],
            ),
            with_payload=True
        )

        logger.info(f"[{get_conversation_id()}] Search returned {len(results)} hits")

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
