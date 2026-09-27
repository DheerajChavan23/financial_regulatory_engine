"""Enterprise Regulatory & Operational Ingestion Engine - Core Configuration.

This module provides enterprise-grade, strongly typed, and validated configuration
management utilizing Pydantic v2 and pydantic-settings. It supports environment variable
overrides, tiered lakehouse paths, compliance RAG parameters, and data governance policies.
"""

from functools import lru_cache
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Literal

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Base Project Root Resolution
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent


class StorageSettings(BaseModel):
    """Storage directory configurations for raw ingestion and tiered lakehouse architecture."""

    base_dir: Path = Field(
        default=PROJECT_ROOT,
        description="Root directory of the ingestion engine project.",
    )
    raw_inbound_dir: Path = Field(
        default_factory=lambda: PROJECT_ROOT / "data" / "01_raw_inbound",
        description="Landing zone for raw inbound documents and payloads.",
    )
    bronze_lakehouse_dir: Path = Field(
        default_factory=lambda: PROJECT_ROOT / "data" / "02_bronze_lakehouse",
        description="Bronze layer: Raw immutable event stream and untransformed extracts.",
    )
    silver_lakehouse_dir: Path = Field(
        default_factory=lambda: PROJECT_ROOT / "data" / "03_silver_lakehouse",
        description="Silver layer: Cleansed, PII-governed, quality-validated conformant datasets.",
    )
    gold_lakehouse_dir: Path = Field(
        default_factory=lambda: PROJECT_ROOT / "data" / "04_gold_lakehouse",
        description="Gold layer: Aggregated, audit-ready operational and regulatory metrics.",
    )
    regulatory_corpus_dir: Path = Field(
        default_factory=lambda: PROJECT_ROOT / "regulatory_corpus",
        description="Directory housing official regulatory guidelines and framework documents.",
    )
    chroma_db_dir: Path = Field(
        default_factory=lambda: PROJECT_ROOT / "data" / "chroma_db",
        description="Persistent local storage for vector embeddings used in compliance RAG.",
    )
    config_dir: Path = Field(
        default_factory=lambda: PROJECT_ROOT / "config",
        description="Directory containing configuration specifications and rule definitions.",
    )
    regulatory_rules_file: Path = Field(
        default_factory=lambda: PROJECT_ROOT / "config" / "regulatory_rules.yaml",
        description="Path to regulatory rules YAML specification file.",
    )


class GovernanceSettings(BaseModel):
    """Settings governing PII redaction, anonymization, and data quality thresholds."""

    mask_char: str = Field(
        default="*",
        min_length=1,
        max_length=1,
        description="Character used for masking sensitive PII values.",
    )
    salt: str = Field(
        default="enterprise_audit_secret_salt_2026",
        description="Cryptographic salt used for deterministic pseudo-anonymization hashing.",
    )
    min_quality_score: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Minimum composite quality score required for silver lakehouse promotion.",
    )
    completeness_threshold: float = Field(
        default=0.90,
        ge=0.0,
        le=1.0,
        description="Minimum mandatory attribute completeness ratio threshold.",
    )
    allowed_currencies: List[str] = Field(
        default_factory=lambda: ["EUR", "USD", "GBP", "CHF", "JPY", "CAD", "AUD"],
        description="ISO 4217 compliant permitted financial transaction currencies.",
    )
    strict_quarantine: bool = Field(
        default=True,
        description="When True, records failing quality or governance checks are diverted to quarantine.",
    )


class ComplianceRAGSettings(BaseModel):
    """Configuration for retrieval-augmented generation compliance auditing."""

    collection_name: str = Field(
        default="cbi_dora_regulatory_corpus",
        description="ChromaDB collection identifier for compliance guidelines.",
    )
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence-transformer embedding model identifier for regulatory retrieval.",
    )
    chunk_size: int = Field(
        default=512,
        gt=0,
        description="Maximum token chunk size for segmenting regulatory documents.",
    )
    chunk_overlap: int = Field(
        default=64,
        ge=0,
        description="Token overlap count between adjacent document chunks.",
    )
    top_k: int = Field(
        default=4,
        gt=0,
        description="Number of relevant regulatory excerpts retrieved per compliance audit evaluation.",
    )
    similarity_threshold: float = Field(
        default=0.65,
        ge=0.0,
        le=1.0,
        description="Minimum cosine similarity cutoff for compliance context retrieval.",
    )


class LakehouseSettings(BaseModel):
    """Lakehouse physical storage, partitioning, and format controls."""

    file_format: Literal["parquet", "feather"] = Field(
        default="parquet",
        description="Default columnar file format used across Lakehouse layers.",
    )
    compression_codec: Literal["snappy", "gzip", "zstd", "none"] = Field(
        default="snappy",
        description="Compression codec applied to columnar Parquet files.",
    )
    partition_cols: List[str] = Field(
        default_factory=lambda: ["ingestion_date", "document_type"],
        description="Default hierarchical partition keys for Lakehouse storage.",
    )


class AppSettings(BaseSettings):
    """Application-level operational environment variables and server parameters."""

    model_config = SettingsConfigDict(
        env_prefix="ENGINE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = Field(
        default="Financial Ingestion & Audit Engine",
        description="Human-readable title of the ingestion engine.",
    )
    environment: Literal["development", "staging", "production"] = Field(
        default="development",
        description="Runtime deployment environment.",
    )
    debug: bool = Field(
        default=False,
        description="Flag enabling verbose debugging diagnostics.",
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Standard logging severity output level.",
    )
    server_host: str = Field(
        default="0.0.0.0",
        description="Bind host address for operational APIs and dashboards.",
    )
    server_port: int = Field(
        default=8501,
        ge=1024,
        le=65535,
        description="Bind port address for operational APIs and dashboards.",
    )


class Settings(BaseSettings):
    """Unified master settings container for the entire ingestion engine."""

    model_config = SettingsConfigDict(
        env_prefix="ENGINE_",
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    app: AppSettings = Field(default_factory=AppSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    governance: GovernanceSettings = Field(default_factory=GovernanceSettings)
    compliance: ComplianceRAGSettings = Field(default_factory=ComplianceRAGSettings)
    lakehouse: LakehouseSettings = Field(default_factory=LakehouseSettings)

    def ensure_directories(self) -> None:
        """Create all required Lakehouse, data, and regulatory directories if they do not exist.

        Raises:
            OSError: If any directory cannot be created due to permissions or filesystem failure.
        """
        directories: List[Path] = [
            self.storage.raw_inbound_dir,
            self.storage.bronze_lakehouse_dir,
            self.storage.silver_lakehouse_dir,
            self.storage.gold_lakehouse_dir,
            self.storage.regulatory_corpus_dir,
            self.storage.chroma_db_dir,
            self.storage.config_dir,
        ]

        logger = logging.getLogger(self.app.app_name)
        for directory in directories:
            try:
                directory.mkdir(parents=True, exist_ok=True)
                logger.debug("Verified directory exists: %s", directory)
            except OSError as err:
                logger.error("Failed creating directory '%s': %s", directory, err)
                raise OSError(f"Unable to initialize directory: {directory}") from err


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Retrieve the cached singleton application settings instance.

    Returns:
        Settings: Validated, immutable application settings instance.
    """
    settings = Settings()
    # Configure root logging based on loaded settings
    numeric_level = getattr(logging, settings.app.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return settings


if __name__ == "__main__":
    # Self-validation execution test
    try:
        active_settings = get_settings()
        active_settings.ensure_directories()
        print(f"Configuration successfully loaded for: '{active_settings.app.app_name}'")
        print(f"Environment: {active_settings.app.environment}")
        print(f"Project root: {active_settings.storage.base_dir}")
        print("All system Lakehouse directories initialized successfully.")
    except Exception as exc:
        print(f"Configuration Initialization Error: {exc}", file=sys.stderr)
        sys.exit(1)
