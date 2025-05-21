# fastmcp_http_server.py
# ---------------------
# An HTTP-based MCP server using fastmcp-http.
# Exposes your tools via REST endpoints:
#   GET  /tools                  → list tools
#   POST /tools/<tool_name>      → invoke tool
#
# Install dependencies:
#   pip install fastmcp-http aiohttp python-dotenv

import os
import json
import aiohttp
import tiktoken
import uuid
import requests
import re
import urllib.parse
import logging
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from hashlib import sha256
from cachetools import TTLCache

from dotenv import load_dotenv

from fastmcp_http.server import FastMCPHttpServer

load_dotenv()


#chunking the JSON payload
COLLECTION_NAME = "update_json_objects"
model = SentenceTransformer("all-MiniLM-L6-v2")
client = QdrantClient(":memory:")

API_CACHE   = TTLCache(maxsize=2, ttl=10 * 60)        
SEEN_HASHES = set()                                  
VECTOR_INIT = False 

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("mcp_server")

def full_response_hash(payload: list) -> str:
    """Deterministic hash of the entire JSON array (order matters)."""
    return sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

def object_hash(obj: dict) -> str:
    """Hash each item – use something stable & unique."""
    key = f"{obj.get('update-number')}-{obj.get('timestamp')}"
    return sha256(key.encode()).hexdigest()

# Prepare Qdrant collection (once)

def ensure_vector_db(dimension: int):
    global VECTOR_INIT
    if not VECTOR_INIT:
        client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
        )
        VECTOR_INIT = True

def flatten_json(json_obj):
    return f"{json_obj.get('product-name', '')} {json_obj.get('product-version', '')} {json_obj.get('description', '')}"

def upload_new_updates(json_list: list):
    # keep only unseen objects
    fresh = [o for o in json_list if object_hash(o) not in SEEN_HASHES]
    if not fresh:
        logging.info("No new updates – Qdrant unchanged")
        return

    texts   = [flatten_json(o) for o in fresh]
    vectors = model.encode(texts).tolist()

    ensure_vector_db(len(vectors[0]))

    points = [
        PointStruct(id=uuid.uuid4().int >> 64, vector=v, payload=o)
        for v, o in zip(vectors, fresh)
    ]
    client.upload_points(collection_name=COLLECTION_NAME, points=points)

    # mark as seen
    SEEN_HASHES.update(object_hash(o) for o in fresh)
    logging.info(f"Uploaded {len(fresh)} new updates to Qdrant")

def search_similar_json(query_text, top_k=5):
    query_vec = model.encode([query_text]).tolist()[0]
    
    # Perform vector search
    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vec,
        limit=top_k
    )
    
    # logging.info for inspection (optional)
    logging.info(f"\n Top {top_k} results for query: '{query_text}'")
    for i, hit in enumerate(results, 1):
        logging.info(f"\nResult #{i} (Score: {hit.score:.4f})")
        logging.info(hit.payload)
    
    # Return the top JSON payloads
    return [hit.payload for hit in results]

async def cached_api_fetch(product_version: str) -> list:
    cache_key = f"{product_version}"
    if cache_key in API_CACHE:
        logging.info("Using cached API payload")
        return API_CACHE[cache_key].get("data")

    token   = await _fetch_token()
    payload = await _fetch_api(token, product_version)

    # only replace cache if payload differs
    new_hash = full_response_hash(payload)
    if API_CACHE.get(cache_key, {}).get("hash") == new_hash:
        logging.info("Cached payload identical – skip refresh")
        return API_CACHE[cache_key]["data"]

    API_CACHE[cache_key] = {"hash": new_hash, "data": payload}
    logging.info("API cache refreshed")
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

async def _fetch_api(token: str, product_version:str) -> dict:

    if not token:
        raise RuntimeError("Empty token passed to _fetch_api()")
    if not product_version:
        raise RuntimeError("Empty product level passed to _fetch_api()")
    token = token.strip().strip('"').strip("'")

    headers = {
        "Authorization": f"Bearer {token}".strip(),
        "Accept": "*/*",
        "User-Agent": "PostmanRuntime/7.42.0"
    }

    logging.info(">> Authorization header:", repr(headers))

    url = os.getenv("API_BASE_URL", "").rstrip("/")
    if not url:
        raise RuntimeError("API_BASE_URL not set in environment")
    
    url = url.replace("product_version", product_version)
    logging.info(">> Request URL:", url)

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
        "  • github_find_related_issues — search the WSO2 product-is GitHub repo for "
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
async def u2_update_summary(query:str, product_version:str) -> str:

    logging.info(f"Received query: {query}")
    logging.info(f"Received product level: {product_version}")
    if not query:
        raise ValueError("Query cannot be empty")
    
    if not product_version or not re.compile(r'^\d+\.\d+\.\d+$').fullmatch(product_version):   
        # raise ValueError("Product level value is not valid. It should be in the format x.x.x")
        product_version = "5.11.0"
    
    json_data = await cached_api_fetch(product_version)
    upload_new_updates(json_data)
    matches = search_similar_json(query)
    
    logging.info(f"Token size of the payload from server {len(tiktoken.get_encoding('cl100k_base').encode(json.dumps(matches)))}")
    return json.dumps(matches, separators=(",", ":"))


def _build_url(term: str, repo: str | None, top_k: int) -> str:
    logging.info(f"Building GitHub search URL from desc: {term}")
    q  = f'"{term}" in:title,body,comments is:issue'
    if repo:
        q += f" repo:{repo}"
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
        "search the WSO2 product-is GitHub repository’s issues and return the top_k "
        "most relevant matches."
    )
)
async def github_find_related_issues(term: str, top_k: int = 5) -> str:
    repo = "wso2/product-is"
    logging.info(f"[github_find_related_issues] searching `{term}`")
    url = _build_url(term, repo, max(1, min(top_k, 10)))
    logging.info(f"GitHub search URL: {url}")
    headers = {
        "Accept":      "application/vnd.github+json",
        "User-Agent":  "fastmcp-github-search/1.0"  # recommended by GitHub :contentReference[oaicite:1]{index=1}
    }

    async with aiohttp.ClientSession() as s:
        async with s.get(url, headers=headers) as r:
            logging.info(f"GitHub search response: {r.status}")
            if r.status == 403:
                # GitHub responds 403 if the 60-per-hour unauth limit is exhausted
                raise RuntimeError("GitHub unauthenticated rate-limit hit (60 req/hr). "
                                   "Try later or add a token.")
            r.raise_for_status()
            items = (await r.json())["items"]
            logging.info(f"GitHub search returned {len(items)} items")

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
async def u2_update_summary(query:str, product_version:str) -> str:

    logging.info(f"Received query: {query}")
    logging.info(f"Received product level: {product_version}")
    return json.dumps("This is a dummy tool which can be used to get details of the birds", separators=(",", ":"))


# --- Run the HTTP server ------------------------------------------
if __name__ == "__main__":
    logging.info("Starting MCP HTTP server...")
    port = int(os.getenv("MCP_PORT", 9999))
    host = os.getenv("MCP_HOST", "0.0.0.0") 
    mcp.run_http(host=host, port=port, register_server=False)


# def upload_json_objects(json_list):
#     texts = [flatten_json(obj) for obj in json_list]
#     vectors = model.encode(texts).tolist()

#     initialize_vector_db(len(vectors[0]))

#     for vec, obj in zip(vectors, json_list):
#         point = PointStruct(
#             id=uuid.uuid4().int >> 64,
#             vector=vec,
#             payload=obj  # Store entire JSON
#         )
#         client.upload_points(collection_name=COLLECTION_NAME, points=[point])
#     logging.info(f"Uploaded {len(json_list)} JSON objects")


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
