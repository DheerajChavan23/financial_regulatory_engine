"""Pydantic V2 schema definitions for financial claim document extraction and Bronze Lakehouse ingestion.

Provides strongly typed models for raw unstructured extracts, itemized financial ledgers,
audit metadata envelopes, and validation rules compliant with European regulatory standards.
"""

from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ExtractedLineItem(BaseModel):
    """Line item representing an individual financial loss entry in a claim dossier."""

    model_config = ConfigDict(str_strip_whitespace=True)

    category: str = Field(
        ...,
        description="Category classification of the operational or property loss.",
    )
    description: str = Field(
        ...,
        description="Detailed narrative explaining the specific loss expense.",
    )
    amount: float = Field(
        ...,
        description="Gross loss expense amount before deductible.",
    )
    deductible: float = Field(
        default=0.0,
        description="Policyholder deductible subtracted from the gross amount.",
    )
    net_claimed: float = Field(
        ...,
        description="Net requested compensation amount for this line item.",
    )


class RawClaimExtract(BaseModel):
    """Raw structured data extracted from an insurance claim document before governance cleansing."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="ignore")

    claim_id: str = Field(
        ...,
        description="Unique claim tracking identifier (e.g., CLM-IE-2026-0001).",
    )
    policy_number: Optional[str] = Field(
        default=None,
        description="Underlying insurance policy number, or None if omitted/missing.",
    )
    status_banner: Optional[str] = Field(
        default=None,
        description="Status flag present on document header (e.g. CLEAN INGESTION, FLAGGED ANOMALY).",
    )
    filing_date: str = Field(
        ...,
        description="Date when the formal claim dossier was submitted (YYYY-MM-DD).",
    )
    incident_date: str = Field(
        ...,
        description="Date when the underlying operational or property incident occurred (YYYY-MM-DD).",
    )
    days_to_notify: Optional[int] = Field(
        default=None,
        description="Elapsed days between the incident date and formal submission notice.",
    )
    dora_classification: Optional[str] = Field(
        default="Standard Operational Claim (Non-ICT)",
        description="DORA or CBI regulatory incident classification tier.",
    )
    claimant_name: Optional[str] = Field(
        default=None,
        description="Full legal name of the insured claimant or authorized representative.",
    )
    claimant_ppsn: Optional[str] = Field(
        default=None,
        description="Personal Public Service Number (PPSN) or tax identification of the claimant.",
    )
    claimant_email: Optional[str] = Field(
        default=None,
        description="Official contact email address of the claimant.",
    )
    claimant_phone: Optional[str] = Field(
        default=None,
        description="Primary telephone number of the claimant.",
    )
    loss_location: Optional[str] = Field(
        default=None,
        description="Full street address and county where the physical/operational loss occurred.",
    )
    eircode: Optional[str] = Field(
        default=None,
        description="Irish national routing key and postal code identifier.",
    )
    incident_type: Optional[str] = Field(
        default=None,
        description="Nature of the peril or operational failure.",
    )
    incident_description: Optional[str] = Field(
        default=None,
        description="Narrative detailing root causes, timeline, and mitigation actions taken.",
    )
    currency: str = Field(
        default="EUR",
        description="Currency code for financial items in this claim (ISO 4217).",
    )
    line_items: List[ExtractedLineItem] = Field(
        default_factory=list,
        description="Itemized schedule of losses and remediation expenses.",
    )
    total_loss_amount: float = Field(
        default=0.0,
        description="Cumulative gross loss total reported on the claim form.",
    )
    total_deductible: float = Field(
        default=0.0,
        description="Cumulative deductible amount applied against gross loss.",
    )
    total_net_claimed: float = Field(
        ...,
        description="Total net indemnity requested by the policyholder.",
    )
    statutory_attestation: Optional[str] = Field(
        default=None,
        description="Regulatory or statutory legal attestation statement extracted from document.",
    )

    @field_validator("policy_number", mode="before")
    @classmethod
    def sanitize_policy_number(cls, v: Any) -> Optional[str]:
        """Convert placeholder missing indicators into None."""
        if v is None:
            return None
        v_str = str(v).strip()
        if v_str in ("", "N/A", "None", "[MISSING]", "[MISSING / NOT PROVIDED]", "null"):
            return None
        return v_str


class BronzeLakehouseEnvelope(BaseModel):
    """Bronze lakehouse audit envelope encapsulating raw immutable payload and provenance metadata."""

    model_config = ConfigDict(str_strip_whitespace=True)

    ingestion_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Globally unique ingestion event tracking identifier.",
    )
    source_filename: str = Field(
        ...,
        description="Original name of the ingested source file.",
    )
    source_file_path: str = Field(
        ...,
        description="Absolute or normalized path of the raw source file.",
    )
    source_file_size_bytes: int = Field(
        ...,
        ge=0,
        description="Size of the raw inbound file in bytes.",
    )
    source_file_sha256: str = Field(
        ...,
        description="Cryptographic SHA-256 digest of the ingested raw document for immutability auditing.",
    )
    ingestion_timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 UTC timestamp of ingestion into Bronze Lakehouse.",
    )
    extraction_method: str = Field(
        ...,
        description="Name of the extraction pipeline or model used (e.g. google_genai_gemini_2_5_flash).",
    )
    model_version: Optional[str] = Field(
        default=None,
        description="Exact model name/version used for document parsing.",
    )
    extracted_payload: RawClaimExtract = Field(
        ...,
        description="Extracted and structured claim record.",
    )
    raw_text_length: int = Field(
        default=0,
        description="Character count of raw text extracted from the document.",
    )
    lakehouse_layer: str = Field(
        default="bronze",
        description="Data lakehouse tier designation (bronze = raw immutable extract).",
    )

    @classmethod
    def create_envelope(
        cls,
        file_path: Path,
        raw_extract: RawClaimExtract,
        extraction_method: str,
        model_version: Optional[str] = None,
        raw_text_length: int = 0,
    ) -> "BronzeLakehouseEnvelope":
        """Factory method to construct a Bronze Lakehouse envelope from a source file and extract.

        Args:
            file_path: Path to the raw inbound document.
            raw_extract: Parsed RawClaimExtract model.
            extraction_method: Description of the parser mechanism.
            model_version: Optional model identifier.
            raw_text_length: Character length of the parsed text.

        Returns:
            BronzeLakehouseEnvelope: Auditable bronze record envelope.
        """
        # Calculate SHA-256 digest of original raw file
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(65536), b""):
                sha256_hash.update(byte_block)
        file_digest = sha256_hash.hexdigest()

        return cls(
            source_filename=file_path.name,
            source_file_path=str(file_path.resolve()),
            source_file_size_bytes=file_path.stat().st_size,
            source_file_sha256=file_digest,
            extraction_method=extraction_method,
            model_version=model_version,
            extracted_payload=raw_extract,
            raw_text_length=raw_text_length,
        )
