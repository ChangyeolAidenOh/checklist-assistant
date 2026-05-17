"""
ChromaDB RAG Module.

Handles document ingestion, chunking, embedding, and retrieval
for insurance product summaries.

Usage:
  # Ingest all PDFs
  rag = CoverageRAG()
  rag.ingest_directory("data/product_summaries/")

  # Query
  results = rag.search("암 진단 보장 범위", top_k=5)
"""

import logging
import hashlib
import re
import unicodedata
from pathlib import Path
from dataclasses import dataclass

import chromadb
from chromadb.config import Settings

from config import CHROMA_PERSIST_DIR, CHROMA_COLLECTION, RAG_SIMILARITY_THRESHOLD, RAG_TOP_K

logger = logging.getLogger(__name__)


@dataclass
class RAGResult:
    text: str
    source: str           # PDF filename
    section: str          # section name (보장내용, 면책사항, etc.)
    score: float          # similarity score (lower = more similar in chromadb)
    metadata: dict


class CoverageRAG:
    """RAG interface for insurance coverage documents."""

    def __init__(self, persist_dir: str | None = None):
        self.persist_dir = persist_dir or CHROMA_PERSIST_DIR
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "ChromaDB initialized: %s (%d documents)",
            self.persist_dir, self.collection.count(),
        )

    def ingest_directory(self, pdf_dir: str | Path, force: bool = False) -> dict:
        """Ingest all PDFs from a directory.

        Args:
            pdf_dir: Directory containing PDF files
            force: If True, re-ingest even if already in collection

        Returns:
            {"ingested": N, "skipped": N, "errors": N, "total_chunks": N}
        """
        pdf_dir = Path(pdf_dir)
        if not pdf_dir.exists():
            logger.error("Directory not found: %s", pdf_dir)
            return {"ingested": 0, "skipped": 0, "errors": 0, "total_chunks": 0}

        pdfs = sorted(pdf_dir.glob("*.pdf"))
        stats = {"ingested": 0, "skipped": 0, "errors": 0, "total_chunks": 0}

        for pdf_path in pdfs:
            try:
                # Normalize filename for consistent matching
                norm_name = unicodedata.normalize("NFC", pdf_path.name)

                # Check if already ingested
                if not force:
                    existing = self.collection.get(
                        where={"source": norm_name},
                        limit=1,
                    )
                    if existing and existing["ids"]:
                        stats["skipped"] += 1
                        continue

                chunks = self._extract_and_chunk(pdf_path)
                if not chunks:
                    stats["errors"] += 1
                    continue

                self._add_chunks(chunks, norm_name)
                stats["ingested"] += 1
                stats["total_chunks"] += len(chunks)
                logger.info("Ingested %s: %d chunks", norm_name, len(chunks))

            except Exception as e:
                logger.error("Failed to ingest %s: %s", pdf_path.name, e)
                stats["errors"] += 1

        logger.info("Ingestion complete: %s", stats)
        return stats

    def search(
        self,
        query: str,
        top_k: int | None = None,
        product_filter: str | None = None,
        section_filter: str | None = None,
    ) -> list[RAGResult]:
        """Search for relevant document chunks.

        Args:
            query: Search query in Korean
            top_k: Number of results (default from config)
            product_filter: Filter by product name (partial match)
            section_filter: Filter by section (보장내용, 면책사항, etc.)

        Returns:
            List of RAGResult sorted by relevance
        """
        top_k = top_k or RAG_TOP_K

        where_filter = None
        where_conditions = []
        if product_filter:
            where_conditions.append({"source": {"$contains": product_filter}})
        if section_filter:
            where_conditions.append({"section": section_filter})

        if len(where_conditions) == 1:
            where_filter = where_conditions[0]
        elif len(where_conditions) > 1:
            where_filter = {"$and": where_conditions}

        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                where=where_filter,
            )
        except Exception as e:
            logger.error("RAG search failed: %s", e)
            return []

        if not results or not results["documents"] or not results["documents"][0]:
            return []

        rag_results = []
        for i, doc in enumerate(results["documents"][0]):
            distance = results["distances"][0][i] if results["distances"] else 1.0
            metadata = results["metadatas"][0][i] if results["metadatas"] else {}

            # Convert cosine distance to similarity (1 - distance)
            similarity = 1.0 - distance

            if similarity < RAG_SIMILARITY_THRESHOLD:
                continue

            rag_results.append(RAGResult(
                text=doc,
                source=metadata.get("source", ""),
                section=metadata.get("section", ""),
                score=round(similarity, 3),
                metadata=metadata,
            ))

        return rag_results

    def search_for_coverage(self, product_name: str) -> list[RAGResult]:
        """Search specifically for coverage details of a product."""
        return self.search(
            query=f"{product_name} 보장내용 보험금 지급사유",
            product_filter=product_name if product_name else None,
            section_filter="보장내용",
            top_k=10,
        )

    def search_for_exclusions(self, product_name: str) -> list[RAGResult]:
        """Search specifically for exclusions of a product."""
        return self.search(
            query=f"{product_name} 면책사항 보험금 지급하지 않는",
            product_filter=product_name if product_name else None,
            section_filter="면책사항",
            top_k=5,
        )

    def get_stats(self) -> dict:
        """Get collection statistics."""
        count = self.collection.count()
        # Sample to get unique sources
        if count > 0:
            sample = self.collection.get(limit=min(count, 1000))
            sources = set()
            for meta in (sample.get("metadatas") or []):
                if meta and "source" in meta:
                    sources.add(meta["source"])
            return {"total_chunks": count, "unique_sources": len(sources)}
        return {"total_chunks": 0, "unique_sources": 0}

    def clear(self) -> None:
        """Clear all documents from the collection."""
        self.client.delete_collection(CHROMA_COLLECTION)
        self.collection = self.client.get_or_create_collection(
            name=CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Collection cleared")

    def _extract_and_chunk(self, pdf_path: Path) -> list[dict]:
        """Extract text from PDF and split into chunks with metadata."""
        from core_pipeline.parser.pdf_extractor import extract_sections

        sections = extract_sections(pdf_path)
        chunks = []

        section_map = {
            "특이사항": sections.특이사항,
            "가입자격": sections.가입자격,
            "보장내용": sections.보장내용,
            "면책사항": sections.면책사항,
        }

        for section_name, text in section_map.items():
            if not text or len(text) < 20:
                continue

            section_chunks = self._split_text(
                text,
                chunk_size=500,
                overlap=50,
            )

            for i, chunk in enumerate(section_chunks):
                chunk_id = hashlib.md5(
                    f"{pdf_path.name}:{section_name}:{i}".encode()
                ).hexdigest()

                chunks.append({
                    "id": chunk_id,
                    "text": chunk,
                    "metadata": {
                        "source": unicodedata.normalize("NFC", pdf_path.name),
                        "product_name": sections.product_name,
                        "section": section_name,
                        "chunk_index": i,
                        "page_count": sections.page_count,
                    },
                })

        return chunks

    def _split_text(
        self,
        text: str,
        chunk_size: int = 500,
        overlap: int = 50,
    ) -> list[str]:
        """Split text into overlapping chunks at sentence boundaries."""
        # Clean text
        text = re.sub(r"\s+", " ", text).strip()

        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size

            if end < len(text):
                # Find sentence boundary
                for sep in [".\n", ".\n", ".\n", ". ", ".\n", "다. ", "다.\n"]:
                    boundary = text.rfind(sep, start + chunk_size // 2, end + 100)
                    if boundary > start:
                        end = boundary + len(sep)
                        break

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            start = end - overlap

        return chunks

    def _add_chunks(self, chunks: list[dict], source_name: str) -> None:
        """Add chunks to ChromaDB collection."""
        # Remove existing chunks for this source
        try:
            existing = self.collection.get(where={"source": source_name})
            if existing and existing["ids"]:
                self.collection.delete(ids=existing["ids"])
        except Exception:
            pass

        if not chunks:
            return

        self.collection.add(
            ids=[c["id"] for c in chunks],
            documents=[c["text"] for c in chunks],
            metadatas=[c["metadata"] for c in chunks],
        )
