"""Compliance vector store engine utilizing ChromaDB for regulatory retrieval.

Indexes statutory Central Bank of Ireland (CBI) operational resilience guidance and EU
Digital Operational Resilience Act (DORA) frameworks. Provides semantic vector retrieval
for compliance auditors and automated supervisory rule checks.
"""

from dataclasses import dataclass
import logging
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.utils import embedding_functions

# Ensure project root is in sys.path when invoked directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import get_settings

logger = logging.getLogger("compliance_vector_store")


@dataclass
class RegulatoryChunk:
    """Representation of an indexed regulatory text chunk with statutory metadata."""

    chunk_id: str
    text: str
    metadata: Dict[str, Any]


class ComplianceVectorStore:
    """Persistent ChromaDB vector store for regulatory and supervisory frameworks."""

    def __init__(
        self,
        persist_dir: Optional[Path] = None,
        collection_name: Optional[str] = None,
    ) -> None:
        """Initialize ChromaDB client and embedding model.

        Args:
            persist_dir: Local storage folder for Chroma persistent database.
            collection_name: Unique collection name for regulatory guidelines.
        """
        app_settings = get_settings()
        self.persist_dir: Path = persist_dir or app_settings.storage.chroma_db_dir
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.collection_name: str = collection_name or app_settings.compliance.collection_name
        self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()

        # Initialize persistent Chroma client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        # Get or create collection with cosine similarity space
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "Initialized ComplianceVectorStore (Collection: '%s', Persistence: '%s')",
            self.collection_name,
            self.persist_dir,
        )

    def _chunk_regulatory_text(self, content: str, source_filename: str) -> List[RegulatoryChunk]:
        """Segment a regulatory document into section-aware semantic chunks.

        Args:
            content: Raw text content of the regulatory guidelines.
            source_filename: Name of the origin document file.

        Returns:
            List[RegulatoryChunk]: List of segmented chunks with statutory metadata.
        """
        chunks: List[RegulatoryChunk] = []

        # Split on Section headers: e.g. "SECTION 2: DORA INCIDENT CLASSIFICATION TIERS (ARTICLE 18)"
        section_pattern = re.compile(r"(SECTION\s+(\d+)\s*:[^\n]+)", re.IGNORECASE)
        splits = section_pattern.split(content)

        # Header intro section before Section 1
        if splits and len(splits) > 1:
            intro_text = splits[0].strip()
            if intro_text:
                chunks.append(
                    RegulatoryChunk(
                        chunk_id=f"{Path(source_filename).stem}_preamble_0",
                        text=intro_text,
                        metadata={
                            "source_file": source_filename,
                            "section": "PREAMBLE & JURISDICTION",
                            "section_number": 0,
                            "regulatory_body": "Central Bank of Ireland / European Union",
                            "framework": "DORA & Operational Resilience",
                        },
                    )
                )

            # Process matched sections: pattern.split returns (pre_text, header, section_num, body, header, section_num, body...)
            i = 1
            while i < len(splits):
                header = splits[i].strip()
                sec_num = int(splits[i + 1])
                body = splits[i + 2].strip() if i + 2 < len(splits) else ""
                i += 3

                full_section_text = f"{header}\n\n{body}"

                # Further split subsections if section is extensive
                subsections = [sub.strip() for sub in full_section_text.split("\n\n") if sub.strip()]
                current_chunk_parts: List[str] = [header]
                current_length = len(header)
                sub_idx = 0

                for part in subsections:
                    if part == header:
                        continue
                    if current_length + len(part) > 1000 and len(current_chunk_parts) > 1:
                        chunk_text = "\n\n".join(current_chunk_parts)
                        chunk_id = f"{Path(source_filename).stem}_sec{sec_num}_part{sub_idx}"
                        chunks.append(
                            RegulatoryChunk(
                                chunk_id=chunk_id,
                                text=chunk_text,
                                metadata={
                                    "source_file": source_filename,
                                    "section": header,
                                    "section_number": sec_num,
                                    "regulatory_body": "Central Bank of Ireland / European Union",
                                    "framework": "DORA & Operational Resilience",
                                },
                            )
                        )
                        sub_idx += 1
                        current_chunk_parts = [header, part]
                        current_length = len(header) + len(part)
                    else:
                        current_chunk_parts.append(part)
                        current_length += len(part)

                if current_chunk_parts:
                    chunk_text = "\n\n".join(current_chunk_parts)
                    chunk_id = f"{Path(source_filename).stem}_sec{sec_num}_part{sub_idx}"
                    chunks.append(
                        RegulatoryChunk(
                            chunk_id=chunk_id,
                            text=chunk_text,
                            metadata={
                                "source_file": source_filename,
                                "section": header,
                                "section_number": sec_num,
                                "regulatory_body": "Central Bank of Ireland / European Union",
                                "framework": "DORA & Operational Resilience",
                            },
                        )
                    )

        else:
            # Fallback if text does not match section regex
            chunks.append(
                RegulatoryChunk(
                    chunk_id=f"{Path(source_filename).stem}_full_0",
                    text=content.strip(),
                    metadata={
                        "source_file": source_filename,
                        "section": "GENERAL GUIDELINES",
                        "section_number": 1,
                        "regulatory_body": "Central Bank of Ireland / European Union",
                        "framework": "DORA & Operational Resilience",
                    },
                )
            )

        return chunks

    def index_corpus(
        self,
        corpus_dir: Optional[Path] = None,
        force_reindex: bool = False,
    ) -> int:
        """Index all regulatory guideline documents from corpus directory into ChromaDB.

        Args:
            corpus_dir: Directory containing statutory guideline text files.
            force_reindex: When True, clears existing collection documents before indexing.

        Returns:
            int: Number of regulatory chunks indexed.
        """
        app_settings = get_settings()
        target_dir: Path = corpus_dir or app_settings.storage.regulatory_corpus_dir

        if not target_dir.exists():
            logger.warning("Regulatory corpus directory does not exist: %s", target_dir)
            return 0

        existing_count = self.collection.count()
        if existing_count > 0 and not force_reindex:
            logger.info("Collection '%s' already contains %d chunks. Skipping indexing.", self.collection_name, existing_count)
            return existing_count

        if force_reindex and existing_count > 0:
            logger.info("Force reindex requested. Clearing collection '%s'...", self.collection_name)
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_fn,
                metadata={"hnsw:space": "cosine"},
            )

        files = list(target_dir.glob("*.txt"))
        if not files:
            logger.warning("No regulatory .txt files found in: %s", target_dir)
            return 0

        all_chunks: List[RegulatoryChunk] = []
        for file_path in files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    text_content = f.read()
                file_chunks = self._chunk_regulatory_text(text_content, file_path.name)
                all_chunks.extend(file_chunks)
                logger.info("Parsed %d chunks from %s", len(file_chunks), file_path.name)
            except Exception as exc:
                logger.error("Failed processing regulatory file '%s': %s", file_path.name, exc)

        if not all_chunks:
            return 0

        # Batch upsert into ChromaDB
        self.collection.upsert(
            ids=[c.chunk_id for c in all_chunks],
            documents=[c.text for c in all_chunks],
            metadatas=[c.metadata for c in all_chunks],
        )

        total_indexed = self.collection.count()
        logger.info("Successfully indexed %d chunks into collection '%s'", total_indexed, self.collection_name)
        return total_indexed

    def search_guidelines(
        self,
        query: str,
        top_k: int = 3,
        section_number: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve the most relevant statutory guidelines for a given query.

        Args:
            query: Semantic search query or compliance evaluation question.
            top_k: Maximum number of guideline excerpts to retrieve.
            section_number: Optional filter to restrict search to a specific regulatory section.

        Returns:
            List[Dict[str, Any]]: List of matching guideline hits with text, metadata, and similarity.
        """
        where_filter: Optional[Dict[str, Any]] = None
        if section_number is not None:
            where_filter = {"section_number": section_number}

        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        hits: List[Dict[str, Any]] = []
        if results and results["documents"] and results["documents"][0]:
            docs = results["documents"][0]
            metas = results["metadatas"][0] if results["metadatas"] else [{}] * len(docs)
            distances = results["distances"][0] if results["distances"] else [0.0] * len(docs)
            ids = results["ids"][0] if results["ids"] else [""] * len(docs)

            for i in range(len(docs)):
                # Cosine distance to similarity: similarity = 1.0 - (distance / 2.0)
                cos_dist = float(distances[i])
                similarity = round(max(0.0, 1.0 - cos_dist), 4)

                hits.append(
                    {
                        "chunk_id": ids[i],
                        "text": docs[i],
                        "metadata": metas[i],
                        "distance": cos_dist,
                        "similarity_score": similarity,
                    }
                )

        return hits


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
    print("=" * 80)
    print("Central Bank of Ireland & DORA Regulatory Vector Store Test")
    print("=" * 80)

    store = ComplianceVectorStore()
    count = store.index_corpus(force_reindex=True)
    print(f"Total Guidelines Indexed in ChromaDB: {count}")

    # Query 1: Notification Latency
    q1 = "What is the statutory deadline and SLA for submitting an insurance claim notification?"
    results1 = store.search_guidelines(q1, top_k=2)
    print(f"\nQuery: '{q1}'")
    for r in results1:
        print(f"-> Section: {r['metadata'].get('section')} (Similarity: {r['similarity_score']:.2%})")
        print(f"   Excerpt: {r['text'][:180]}...\n")

    # Query 2: Missing Policy Number
    q2 = "Is a policy number mandatory for claim liquidation and what happens if missing?"
    results2 = store.search_guidelines(q2, top_k=1)
    print(f"Query: '{q2}'")
    for r in results2:
        print(f"-> Section: {r['metadata'].get('section')} (Similarity: {r['similarity_score']:.2%})")
        print(f"   Excerpt: {r['text'][:180]}...\n")
    print("=" * 80)
