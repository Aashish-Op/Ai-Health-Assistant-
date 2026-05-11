from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter

import aiofiles

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from config import get_settings
from models.retrieval import TextChunk
from services.chunker import TextChunker
from services.embedder import EmbeddingService
from services.pinecone_service import PineconeService
from services.pubmed_fetcher import PubMedFetcher


def parse_args() -> argparse.Namespace:
    """Parse indexing CLI arguments.

    Args:
        None.

    Returns:
        Parsed CLI namespace.

    Raises:
        SystemExit: If CLI arguments are invalid.
    """
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Index PubMed abstracts into Pinecone.")
    parser.add_argument("--max-per-query", type=int, default=settings.pubmed_max_per_query)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/pubmed/cache"))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stats", action="store_true")
    return parser.parse_args()


async def main() -> None:
    """Run the PubMed indexing workflow.

    Args:
        None.

    Returns:
        None.

    Raises:
        ValueError: If required settings are missing.
    """
    args = parse_args()
    settings = get_settings()
    checkpoint_path = Path("data/pubmed/checkpoint.json")

    pinecone = None
    if not args.dry_run or args.stats:
        pinecone = PineconeService(settings.pinecone_api_key, settings.pinecone_index_name)

    if args.stats:
        if pinecone is None:
            raise ValueError("Pinecone is required for --stats")
        print(json.dumps(await pinecone.get_index_stats(), indent=2))
        return

    print(
        json.dumps(
            {
                "index_name": settings.pinecone_index_name,
                "embedding_model": settings.embedding_model,
                "max_per_query": args.max_per_query,
                "cache_dir": str(args.cache_dir),
                "dry_run": args.dry_run,
                "resume": args.resume,
            },
            indent=2,
        )
    )

    checkpoint = await load_checkpoint(checkpoint_path) if args.resume else default_checkpoint()
    completed_pmids = set(checkpoint["completed_pmids"])
    chunker = TextChunker(settings.chunk_size_tokens, settings.chunk_overlap_tokens)
    fetcher = PubMedFetcher(settings.ncbi_email, settings.ncbi_api_key, args.cache_dir)
    embedder = None if args.dry_run else EmbeddingService(
        settings.openai_api_key,
        settings.embedding_model,
        settings.embedding_dimensions,
    )

    start = perf_counter()
    batch: list[TextChunk] = []
    total_abstracts = 0
    total_chunks = int(checkpoint["total_chunks"])
    total_tokens_estimated = 0
    tasks: set[asyncio.Task[int]] = set()
    semaphore = asyncio.Semaphore(max(args.workers, 1))

    async for abstract in fetcher.fetch_all(args.max_per_query):
        if abstract.pmid in completed_pmids:
            continue
        total_abstracts += 1
        chunks = chunker.chunk_abstract(abstract)
        if not chunks:
            completed_pmids.add(abstract.pmid)
            continue
        total_tokens_estimated += sum(chunk.token_count for chunk in chunks)
        batch.extend(chunks)
        completed_pmids.add(abstract.pmid)

        while len(batch) >= 100:
            current, batch = batch[:100], batch[100:]
            if args.dry_run:
                total_chunks += len(current)
            else:
                if embedder is None or pinecone is None:
                    raise ValueError("Embedding and Pinecone services are required outside dry-run")
                tasks.add(
                    asyncio.create_task(
                        process_batch(current, embedder, pinecone, semaphore)
                    )
                )
                if len(tasks) >= max(args.workers, 1):
                    done, tasks = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                    total_chunks += sum(task.result() for task in done)
            await write_checkpoint(checkpoint_path, completed_pmids, total_chunks)
            if total_chunks and total_chunks % 500 == 0:
                print(json.dumps({"total_abstracts": total_abstracts, "total_chunks": total_chunks}))

    if batch:
        if args.dry_run:
            total_chunks += len(batch)
        else:
            if embedder is None or pinecone is None:
                raise ValueError("Embedding and Pinecone services are required outside dry-run")
            total_chunks += await process_batch(batch, embedder, pinecone, semaphore)
        await write_checkpoint(checkpoint_path, completed_pmids, total_chunks)

    if tasks:
        done, _ = await asyncio.wait(tasks)
        total_chunks += sum(task.result() for task in done)
        await write_checkpoint(checkpoint_path, completed_pmids, total_chunks)

    final_stats = await pinecone.get_index_stats() if pinecone is not None else {}
    report = {
        "total_abstracts_fetched": total_abstracts,
        "total_chunks_indexed": total_chunks,
        "total_tokens_estimated": total_tokens_estimated,
        "duration_seconds": round(perf_counter() - start, 2),
        "pinecone_index_size": final_stats.get("total_vector_count", 0),
    }
    print(json.dumps(report, indent=2))


async def process_batch(
    chunks: list[TextChunk],
    embedder: EmbeddingService,
    pinecone: PineconeService,
    semaphore: asyncio.Semaphore,
) -> int:
    """Embed and upsert one chunk batch.

    Args:
        chunks: Text chunks to index.
        embedder: Embedding service.
        pinecone: Pinecone vector service.
        semaphore: Concurrency limiter.

    Returns:
        Number of vectors upserted.

    Raises:
        Exception: If embedding or upsert fails.
    """
    async with semaphore:
        embedded = await embedder.embed_chunks(chunks)
        return await pinecone.upsert_chunks(embedded)


def default_checkpoint() -> dict[str, object]:
    """Return an empty checkpoint.

    Args:
        None.

    Returns:
        Empty checkpoint dictionary.

    Raises:
        None.
    """
    return {"completed_pmids": [], "total_chunks": 0, "last_updated": None}


async def load_checkpoint(path: Path) -> dict[str, object]:
    """Load a checkpoint if present.

    Args:
        path: Checkpoint path.

    Returns:
        Checkpoint dictionary.

    Raises:
        json.JSONDecodeError: If checkpoint JSON is invalid.
    """
    if not path.exists():
        return default_checkpoint()
    async with aiofiles.open(path, encoding="utf-8") as file:
        return json.loads(await file.read())


async def write_checkpoint(path: Path, completed_pmids: set[str], total_chunks: int) -> None:
    """Write a resumable indexing checkpoint.

    Args:
        path: Checkpoint path.
        completed_pmids: PMIDs fully processed.
        total_chunks: Total chunks processed.

    Returns:
        None.

    Raises:
        OSError: If writing fails.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "completed_pmids": sorted(completed_pmids),
        "total_chunks": total_chunks,
        "last_updated": datetime.utcnow().isoformat(),
    }
    async with aiofiles.open(path, "w", encoding="utf-8") as file:
        await file.write(json.dumps(payload, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
