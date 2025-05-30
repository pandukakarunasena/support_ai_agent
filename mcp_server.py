import os
import json
import aiohttp
import tiktoken
import uuid
import requests
import re
import urllib.parse
import logging
import openai
from vectordb import VectorDB    

# import onnxruntime
# from transformers import AutoTokenizer
# import numpy as np

# from sentence_transformers import SentenceTransformer
# from sentence_transformers.util import export_to_onnx
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from hashlib import sha256
from cachetools import TTLCache

from dotenv import load_dotenv

from fastmcp_http.server import FastMCPHttpServer
from mcp_context import set_conversation_id, get_conversation_id

# Logging setup
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
logger = logging.getLogger("mcp_server")

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

#chunking the JSON payload
COLLECTION_NAME = "update_json_objects"
_collection_inited = False
# model = SentenceTransformer("all-MiniLM-L6-v2")
# tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
# session = onnxruntime.InferenceSession('model.onnx')
# client = QdrantClient(path=":memory:")

COLLECTION_NAME = "update_json_objects"

# instantiate VectorDB (will create & manage the Qdrant collection for you)
vector_db = VectorDB(collection_name=COLLECTION_NAME)

API_CACHE: TTLCache = TTLCache(maxsize=100, ttl=10 * 60)

SEEN_HASHES = set()                                  
VECTOR_INIT = False 

# def encode(texts):
#     inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="np")
#     ort_inputs = {k: v for k, v in inputs.items()}
#     embeddings = session.run(None, ort_inputs)[0]
#     return embeddings

def full_response_hash(payload: list) -> str:
    """Deterministic hash of the entire JSON array (order matters)."""
    return sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

def object_hash(obj: dict) -> str:
    """Hash each item – use something stable & unique."""
    key = f"{obj.get('update-number')}-{obj.get('timestamp')}"
    return sha256(key.encode()).hexdigest()

# Prepare Qdrant collection (once)
# def ensure_vector_db(dimension: int):
#     global _collection_inited
#     if _collection_inited:
#         return

#     # If you want to start absolutely fresh each run, you can delete first:
#     try:
#         client.delete_collection(collection_name=COLLECTION_NAME)
#     except Exception:
#         pass

#     # Now create the collection
#     client.create_collection(
#         collection_name=COLLECTION_NAME,
#         vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
#     )

    # _collection_inited = True

def flatten_json(json_obj):
    return f"{json_obj.get('product-name', '')} {json_obj.get('product-version', '')} {json_obj.get('description', '')}"

def upload_new_updates(json_list: list):
   
    fresh = [o for o in json_list if object_hash(o) not in SEEN_HASHES]
    if not fresh:
        logger.info(f"[{get_conversation_id()}] o new updates – Qdrant unchanged")
        return

    # texts   = [flatten_json(o) for o in fresh]
    # vectors = encode(texts).tolist()

    # ensure_vector_db(len(vectors[0]))

    # points = [
    #     PointStruct(id=uuid.uuid4().int >> 64, vector=v, payload=o)
    #     for v, o in zip(vectors, fresh)
    # ]
    # client.upload_points(collection_name=COLLECTION_NAME, points=points)

    # for each new object, flatten & add to VectorDB
    for o in fresh:
        text = flatten_json(o)
        # use a random 64-bit id
        point_id = uuid.uuid4().int >> 64
        vector_db.add_text(id=point_id, text=text, metadata=o)

    # mark as seen
    SEEN_HASHES.update(object_hash(o) for o in fresh)
    logger.info(f"[{get_conversation_id()}] Uploaded {len(fresh)} new updates to Qdrant")

def search_similar_json(query_text, top_k=5):
    query_vec = vector_db.get_embedding([query_text]).tolist()[0]
    
    # Perform vector search
    # results = client.search(
    #     collection_name=COLLECTION_NAME,
    #     query_vector=query_vec,
    #     limit=top_k
    # )
    
    results = vector_db.similarity_search(query_text, limit=top_k)

    # logger.info fo[{get_conversation_id()}] r inspection (optional)
    logger.info(f"[{get_conversation_id()}] \n Top {top_k} results for query: '{query_text}'")
    for i, hit in enumerate(results, 1):
        logger.info(f"[{get_conversation_id()}] \nResult #{i} (Score: {hit.score:.4f})")
        logger.info(f"[{get_conversation_id()}] {hit.payload}")
    # Return the top JSON payloads
    return [hit.payload for hit in results]

async def cached_api_fetch(product: str, version: str) -> list:

    #validate the product
    # if product not in PRODUCT_REPO_MAP:
    #     raise ValueError(f"Invalid product: {product}. Valid options are: {', '.join(PRODUCT_REPO_MAP.keys())}")
    
    cache_key = f"{product}:{version}"
    entry = API_CACHE.get(cache_key)
    if entry:
        logger.info(f"[{get_conversation_id()}] sing cached API payload for {product} {version}")
        return entry["data"]

    # 4. Fetch fresh data
    token = await _fetch_token()
    payload = await _fetch_api(token, product, version)

    # 5. Compute new hash
    new_hash = full_response_hash(payload)
    if entry and entry.get("hash") == new_hash:
        logger.info(f"[{get_conversation_id()}] ached payload identical – skip refresh")
        return entry["data"]

    # 6. Store in TTLCache
    API_CACHE[cache_key] = {"hash": new_hash, "data": payload}
    logger.info(f"[{get_conversation_id()}] PI cache refreshed for {product} {version}")

    return payload


async def _fetch_token() -> str:
    token_url = os.getenv("IDP_TOKEN_URL")
    payload = {
        "grant_type": "client_credentials",
        # "username": os.getenv("IDP_USERNAME"),
        # "password": os.getenv("IDP_PASSWORD"),
        "client_id": os.getenv("IDP_CLIENT_ID"),
        "client_secret": os.getenv("IDP_CLIENT_SECRET"),
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(token_url, data=payload) as resp:
            resp.raise_for_status()
            return (await resp.json())["access_token"]

async def _fetch_api(token: str, product: str, version:str) -> dict:

    if not token:
        raise RuntimeError("Empty token passed to _fetch_api()")
    if not product:
        raise RuntimeError("Empty product passed to _fetch_api()")
    if not version:
        raise RuntimeError("Empty product level passed to _fetch_api()")
    
    token = token.strip().strip('"').strip("'")
    headers = {
        "Authorization": f"Bearer {token}".strip(),
        "Accept": "*/*",
        "User-Agent": "PostmanRuntime/7.42.0"
    }

    logger.info(f"[{get_conversation_id()}] > Authorization header: {repr(headers)}")
    

    url = os.getenv("API_BASE_URL", "").rstrip("/")
    if not url:
        raise RuntimeError("API_BASE_URL not set in environment")
    
    url = url.replace("product", product)
    url = url.replace("version", version)
    logger.info(f"[{get_conversation_id()}] > Request URL: {url}")

    async with aiohttp.ClientSession() as session:
        resp = await session.get(url, headers=headers)
        text = await resp.text()
        if resp.status == 403:
            # Dump the body so you see what the server is rejecting
            raise RuntimeError(f"403 Forbidden: {text}")
        resp.raise_for_status()
        return await resp.json()

# Create a FastMCPHttpServer instance
mcp = FastMCPHttpServer(
    name="ProtectedAPIServer",
    description=(
        "Exposes two troubleshooting tools over MCP/HTTP:\n"
        "  • u2_update_summary — fetch the official JSON summary of bug fixes, "
        "improvements, and security updates for a specific product version from our protected API;\n"
        "  • github_find_related_issues — search the WSO2 organization GitHub repos for "
        "issues matching a single technical keyword."
    )
)

@mcp.tool(
    name="u2_update_summary",
    description=(
        "Given a product version (e.g. “5.11.0”), retrieve from the protected API the "
        "latest JSON payload listing bug fixes, feature improvements, and security "
        "updates for that version."
    )
)
async def u2_update_summary(query:str, product: str, version:str, cid: str) -> str:

    # set id in thread local for correlation purposes
    cid = cid or str(uuid.uuid4())
    set_conversation_id(cid)

    logger.info(f"[{get_conversation_id()}] Received query: {query}")
    logger.info(f"[{get_conversation_id()}] Received product: {product}")
    logger.info(f"[{get_conversation_id()}] Received product level: {version}")
    
    if not query:
        raise ValueError("Query cannot be empty")
    
    if not product:   
        raise ValueError("Product level value is not valid. It should be in the format x.x.x")
    
    if not version or not re.compile(r'^\d+\.\d+\.\d+$').fullmatch(version):   
        raise ValueError("Product level value is not valid. It should be in the format x.x.x")
        
    
    json_data = await cached_api_fetch(product, version)
    upload_new_updates(json_data)
    matches = search_similar_json(query)
    
    logger.info(f"[{get_conversation_id()}] Token size of the payload from server {len(tiktoken.get_encoding('cl100k_base').encode(json.dumps(matches)))}")
    return json.dumps(matches, separators=(",", ":"))


def _build_url(term: str, repo: str | None, top_k: int) -> str:
    logger.info(f"[{get_conversation_id()}] Building GitHub search URL from desc: {term}")
    q  = f'"{term}" + in:title in:body in:comments is:issue'
    if repo:
        q += f" repo:wso2/{repo}"
    params = {
        "q":        q,
        "sort":     "updated",
        "order":    "desc",
        "per_page": str(top_k)
    }
    return "https://api.github.com/search/issues?" + urllib.parse.urlencode(params)

@mcp.tool(
    name="github_find_related_issues",
    description=(
        "Given one single-word technical term (e.g. “OAuth”, “NullPointerException”), "
        "search the WSO2 organization GitHub repositories issues and return the top_k "
        "most relevant matches."
    )
)
async def github_find_related_issues(repo: str, term: str, cid: str, top_k: int = 5) -> str:

    # set id in thread local for correlation purposes
    cid = cid or str(uuid.uuid4())
    set_conversation_id(cid)

    logger.info(f"[{get_conversation_id()}] Received repo: {repo}")
    logger.info(f"[{get_conversation_id()}] [github_find_related_issues] searching `{term}`")
    
    url = _build_url(term, repo, max(1, min(top_k, 10)))
    logger.info(f"[{get_conversation_id()}] GitHub search URL: {url}")
    headers = {
        "Accept":      "application/vnd.github+json",
        "User-Agent":  "fastmcp-github-search/1.0"  # recommended by GitHub :contentReference[oaicite:1]{index=1}
    }

    async with aiohttp.ClientSession() as s:
        async with s.get(url, headers=headers) as r:
            logger.info(f"[{get_conversation_id()}] GitHub search response: {r.status}")
            if r.status == 403:
                # GitHub responds 403 if the 60-per-hour unauth limit is exhausted
                raise RuntimeError("GitHub unauthenticated rate-limit hit (60 req/hr). "
                                   "Try later or add a token.")
            r.raise_for_status()
            items = (await r.json())["items"]
            logger.info(f"[{get_conversation_id()}] GitHub search returned {len(items)} items")

    # Trim each issue to essentials so the LLM sees a small payload
    payload = [
        {
            "title":  it["title"],
            "url":    it["html_url"],
            "number": it["number"],
            "labels": it["labels"],
            "body":   it["body"],
            "score":  round(it.get("score", 0), 2)
        }
        for it in items
    ]
    return json.dumps(payload, separators=(",",":"))

@mcp.tool(
    name="dummy_tool",
    description="This is a dummy tool which can be used to get details of the birds"
)
async def dummy_tool(query:str, version:str, cid:str) -> str:

    # set id in thread local for correlation purposes
    cid = cid or str(uuid.uuid4())
    set_conversation_id(cid)

    logger.info(f"[{get_conversation_id()}] Received query: {query}")
    logger.info(f"[{get_conversation_id()}] Received product level: {version}")

    return json.dumps("This is a dummy tool which can be used to get details of the birds", separators=(",", ":"))


# --- Run the HTTP server ------------------------------------------
if __name__ == "__main__":
    logger.info("Starting MCP HTTP server...")
    port = int(os.getenv("MCP_PORT", 9999))
    host = os.getenv("MCP_HOST", "0.0.0.0") 
    mcp.run_http(host=host, port=port, register_server=False)


# def upload_json_objects(json_list):
#     texts = [flatten_json(obj) for obj in json_list]
#     vectors = encode(texts).tolist()

#     initialize_vector_db(len(vectors[0]))

#     for vec, obj in zip(vectors, json_list):
#         point = PointStruct(
#             id=uuid.uuid4().int >> 64,
#             vector=vec,
#             payload=obj  # Store entire JSON
#         )
#         client.upload_points(collection_name=COLLECTION_NAME, points=[point])
#     logger.info(f"[{get_conversation_id()}] Uploaded {len(json_list)} JSON objects")


# chunking is not needed as the similarity search is done on the entire JSON object
#
# enc = tiktoken.get_encoding("cl100k_base")
# MAX_TOKENS = 8000
# OVERLAP = 50

# def chunk_json_list(data_list, max_tokens=MAX_TOKENS, overlap=OVERLAP):
#     chunks = []
#     current, current_tokens = [], 0

#     for item in data_list:
#         text = json.dumps(item, separators=(",", ":"))
#         size = len(enc.encode(text))

#         if size > max_tokens:
#             if current:
#                 chunks.append(current)
#                 current, current_tokens = [], 0
#             chunks.append([item])
#             continue

#         if current and (current_tokens + size > max_tokens):
#             chunks.append(current)
#             if overlap and current_tokens > overlap:
#                 overlap_list, tokens_sum = [], 0
#                 for prev in reversed(current):
#                     pts = len(enc.encode(json.dumps(prev, separators=(",", ":"))))
#                     overlap_list.insert(0, prev)
#                     tokens_sum += pts
#                     if tokens_sum >= overlap:
#                         break
#                 current, current_tokens = overlap_list, tokens_sum
#             else:
#                 current, current_tokens = [], 0

#         current.append(item)
#         current_tokens += size

#     if current:
#         chunks.append(current)
#     return chunks
