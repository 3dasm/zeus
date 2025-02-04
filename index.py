#!/usr/bin/env python3
import os
import json
import asyncio
import datetime
from pydantic import BaseModel
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.extraction_strategy import LLMExtractionStrategy
from llama_index.embeddings.ollama import OllamaEmbedding
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv

load_dotenv()

OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL')
OLLAMA_REASONING_MODEL = os.getenv('OLLAMA_REASONING_MODEL')
OLLAMA_EMBEDDING_MODEL = os.getenv('OLLAMA_EMBEDDING_MODEL')

class AiModel(BaseModel):
    model_rank: int
    model_name: str
    arena_score: int
    votes_count: int

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection("documents")

ollama_embedding = OllamaEmbedding(
    model_name=OLLAMA_EMBEDDING_MODEL,
    base_url=OLLAMA_BASE_URL,
)

llm_strat = LLMExtractionStrategy(
    provider="ollama/" + OLLAMA_REASONING_MODEL,
    base_url=OLLAMA_BASE_URL,
    schema=AiModel.model_json_schema(),
    extraction_type="block",
    instruction="Extract the main table of models leaderboard from the content, skip empty rows, the focus on the leaderboard listing, select model name, arena score, and votes on to the respective json format.",
    chunk_token_threshold=1400,
    apply_chunking=True,
    input_format="html",
    extra_args={"temperature": 0.2, "max_tokens": 1500}
)

crawl_config = CrawlerRunConfig(
    extraction_strategy=llm_strat,
    cache_mode=CacheMode.BYPASS,
    process_iframes=False,
    remove_overlay_elements=True,
    exclude_external_links=True
)
browser_config = BrowserConfig(headless=True, verbose=True)

async def main():
    async with AsyncWebCrawler(config=browser_config) as crawler:
        url = "https://web.lmarena.ai/leaderboard"

        result = await crawler.arun(url=url, config=crawl_config)

        if result.success:
            content = result.extracted_content

            print("Extraction succeeded.")

            embeddings = ollama_embedding.get_text_embedding_batch(content, show_progress=True)
            embedding = embeddings[0]

            doc_id = "doc-1"

            collection.add(
                ids=[doc_id],
                documents=[content],
                embeddings=[embedding]
            )

            print("Document added to ChromaDB with ID:", doc_id)

        else:
            print("Crawl failed:", result.error_message)

if __name__ == "__main__":
    asyncio.run(main())
