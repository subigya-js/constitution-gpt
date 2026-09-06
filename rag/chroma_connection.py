"""Shared Chroma Cloud configuration for ingestion and retrieval."""

from __future__ import annotations

import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from langchain_chroma import Chroma


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_CLOUD_VARIABLES = (
    "CHROMA_API_KEY",
    "CHROMA_TENANT",
    "CHROMA_DATABASE",
)

load_dotenv(PROJECT_ROOT / ".env")


def _environment_value(name: str) -> str | None:
    value = os.getenv(name, "").strip()
    return value or None


def get_collection_name() -> str:
    return _environment_value("CHROMA_COLLECTION") or "constitution_english"


def create_chroma_client():
    """Create a Chroma Cloud client or fail on incomplete configuration."""
    missing = [
        name for name in REQUIRED_CLOUD_VARIABLES if not _environment_value(name)
    ]
    if missing:
        raise RuntimeError(
            "Chroma Cloud configuration is required. Missing: " + ", ".join(missing)
        )

    options = {
        "tenant": _environment_value("CHROMA_TENANT"),
        "database": _environment_value("CHROMA_DATABASE"),
        "api_key": _environment_value("CHROMA_API_KEY"),
    }
    cloud_host = _environment_value("CHROMA_HOST")
    if cloud_host:
        options["cloud_host"] = cloud_host
    return chromadb.CloudClient(**options)


def create_langchain_chroma(embedding_function) -> Chroma:
    """Create the LangChain adapter around the Chroma Cloud client."""
    return Chroma(
        client=create_chroma_client(),
        collection_name=get_collection_name(),
        embedding_function=embedding_function,
        collection_metadata={"hnsw:space": "cosine"},
    )
