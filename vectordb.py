import openai
from qdrant_client import QdrantClient, models
import os

# Set OpenAI API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")


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
        response = openai.Embedding.create(input=text, model=self.embedding_model)
        return response['data'][0]['embedding']

    def add_text(self, id: int, text: str, metadata: dict = None):
        embedding = self.get_embedding(text)
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(id=id, vector=embedding, payload={"text": text, **(metadata or {})})
            ],
        )

    def similarity_search(self, query_text: str, limit: int = 3):
        query_embedding = self.get_embedding(query_text)
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
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
