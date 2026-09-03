"""Shared Chroma connection configuration for ingestion and retrieval."""

from __future__ import annotations

import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from langchain_chroma import Chroma


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCAL_PATH = PROJECT_ROOT / "db" / "chroma_db"
REQUIRED_CLOUD_VARIABLES = (
    "CHROMA_API_KEY",
    "CHROMA_TENANT",
    "CHROMA_DATABASE",
)

load_dotenv(PROJECT_ROOT / ".env")


def _environment_value(name: str) -> str | None:
    value = os.getenv(name, "").strip()
    return value or None


def cloud_is_configured() -> bool:
    return all(_environment_value(name) for name in REQUIRED_CLOUD_VARIABLES)


def cloud_is_partially_configured() -> bool:
    return any(_environment_value(name) for name in REQUIRED_CLOUD_VARIABLES)


def get_collection_name() -> str:
    configured_name = _environment_value("CHROMA_COLLECTION")
    if configured_name:
        return configured_name
    return "constitution_english" if cloud_is_configured() else "langchain"


def create_chroma_client(local_path: str | Path | None = None):
    """Create a cloud client when configured, otherwise a local client."""
    if cloud_is_partially_configured() and not cloud_is_configured():
        missing = [
            name
            for name in REQUIRED_CLOUD_VARIABLES
            if not _environment_value(name)
        ]
        raise RuntimeError(
            "Incomplete Chroma Cloud configuration. Missing: " + ", ".join(missing)
        )

    if cloud_is_configured():
        options = {
            "tenant": _environment_value("CHROMA_TENANT"),
            "database": _environment_value("CHROMA_DATABASE"),
            "api_key": _environment_value("CHROMA_API_KEY"),
        }
        cloud_host = _environment_value("CHROMA_HOST")
        if cloud_host:
            options["cloud_host"] = cloud_host
        return chromadb.CloudClient(**options)

    path = Path(local_path) if local_path else DEFAULT_LOCAL_PATH
    return chromadb.PersistentClient(path=str(path.resolve()))


def create_langchain_chroma(embedding_function, local_path=None) -> Chroma:
    """Create the LangChain adapter around the configured Chroma client."""
    return Chroma(
        client=create_chroma_client(local_path),
        collection_name=get_collection_name(),
        embedding_function=embedding_function,
        collection_metadata={"hnsw:space": "cosine"},
    )
