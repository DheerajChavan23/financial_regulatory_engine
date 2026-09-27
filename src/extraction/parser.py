"""Document extraction engine for financial claims ingestion into Bronze Lakehouse.

Supports extraction using the Google GenAI SDK (Gemini 2.5 Flash) with structured
Pydantic V2 schema output, coupled with an enterprise-grade deterministic fallback
parser (pypdf + regex) to guarantee 100% operational resilience across offline, local CI,
and live cloud environments.
"""

from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

from pypdf import PdfReader

# Ensure project root is in sys.path when invoked directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import get_settings
from src.extraction.schemas import (
    BronzeLakehouseEnvelope,
    ExtractedLineItem,
    RawClaimExtract,
)

# Optional import of Google GenAI SDK
try:
    from google import genai
    from google.genai import types
    from google.genai.errors import APIError

    GENAI_SDK_AVAILABLE = True
except ImportError:
    GENAI_SDK_AVAILABLE = False
    genai = None  # type: ignore
    types = None  # type: ignore
    APIError = Exception  # type: ignore

logger = logging.getLogger("claim_pdf_parser")


class ClaimPDFParser:
    """Enterprise parser ingesting unstructured claim PDFs into Bronze Lakehouse envelopes."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.5-flash",
    ) -> None:
        """Initialize parser with GenAI client or local fallback engine.

        Args:
            api_key: Optional Google GenAI API key. If omitted, checks environment variables.
            model_name: Gemini model name for multimodal structured extraction.
        """
        self.settings = get_settings()
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self.genai_client = None

        if GENAI_SDK_AVAILABLE and self.api_key:
            try:
                self.genai_client = genai.Client(api_key=self.api_key)
                logger.info("Initialized Google GenAI client with model '%s'", self.model_name)
            except Exception as exc:
                logger.warning("Failed to initialize Google GenAI Client: %s. Fallback parser active.", exc)
        else:
            if not GENAI_SDK_AVAILABLE:
                logger.info("google-genai SDK not installed. Running in deterministic extraction mode.")
            else:
                logger.info("GEMINI_API_KEY not detected. Running in deterministic extraction mode.")

    def parse_pdf(self, pdf_path: Path) -> BronzeLakehouseEnvelope:
        """Parse an inbound PDF claim file into a validated Bronze Lakehouse envelope.

        Attempts multimodal extraction via Google GenAI SDK first. If the API key is
        missing or an API error occurs, switches to the deterministic extractor.

        Args:
            pdf_path: File system path to the target PDF document.

        Returns:
            BronzeLakehouseEnvelope: Immutable envelope containing extraction and metadata.

        Raises:
            FileNotFoundError: If the source PDF does not exist.
            ValueError: If parsing fails to yield a valid schema.
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"Inbound PDF not found at: {pdf_path}")

        raw_text = self._extract_raw_text(pdf_path)

        # 1. Attempt Google GenAI SDK extraction if client is configured
        if self.genai_client is not None:
            try:
                logger.debug("Executing Google GenAI multimodal extraction on %s", pdf_path.name)
                return self._parse_with_genai(pdf_path, raw_text)
            except Exception as exc:
                logger.warning(
                    "Google GenAI extraction failed for %s (%s). Falling back to deterministic parser.",
                    pdf_path.name,
                    exc,
                )

        # 2. Deterministic high-fidelity fallback extraction
        logger.debug("Executing deterministic extraction on %s", pdf_path.name)
        return self._parse_with_fallback(pdf_path, raw_text)

    def _extract_raw_text(self, pdf_path: Path) -> str:
        """Extract plain text from all pages of the target PDF via pypdf."""
        try:
            reader = PdfReader(str(pdf_path))
            pages_text = [page.extract_text() or "" for page in reader.pages]
            return "\n".join(pages_text).strip()
        except Exception as exc:
            logger.error("Failed reading text from PDF '%s': %s", pdf_path, exc)
            return ""

    def _parse_with_genai(self, pdf_path: Path, raw_text: str) -> BronzeLakehouseEnvelope:
        """Execute structured content extraction using Google GenAI SDK."""
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()

        system_instruction = (
            "You are a regulatory insurance and operational loss claims auditor. "
            "Extract all metadata, claimant details, loss categories, financial line items, "
            "and statutory declarations strictly conforming to the requested schema. "
            "If a field like policy_number is missing, blank, or contains '[MISSING / NOT PROVIDED]', "
            "set it to null. Ensure negative amounts are accurately captured if present."
        )

        response = self.genai_client.models.generate_content(
            model=self.model_name,
            contents=[
                types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"),
                system_instruction,
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=RawClaimExtract,
                temperature=0.0,
            ),
        )

        raw_extract = RawClaimExtract.model_validate_json(response.text)
        return BronzeLakehouseEnvelope.create_envelope(
            file_path=pdf_path,
            raw_extract=raw_extract,
            extraction_method=f"google_genai_{self.model_name}",
            model_version=self.model_name,
            raw_text_length=len(raw_text),
        )

    def _parse_with_fallback(self, pdf_path: Path, raw_text: str) -> BronzeLakehouseEnvelope:
        """Deterministic parser mapping textual claim documents into RawClaimExtract models."""
        claim_id_match = re.search(r"(?:Claim Reference:|DOSSIER ID:)\s*([A-Z0-9\-]+)", raw_text)
        claim_id = claim_id_match.group(1).strip() if claim_id_match else f"CLM-UNKNOWN-{pdf_path.stem}"

        policy_match = re.search(r"Policy Number:\s*([^\n\r]+)", raw_text)
        policy_raw = policy_match.group(1).strip() if policy_match else None
        if policy_raw and any(k in policy_raw.upper() for k in ["MISSING", "NOT PROVIDED", "NONE", "N/A"]):
            policy_number = None
        else:
            policy_number = policy_raw

        status_match = re.search(r"STATUS:\s*([^\n\r]+)", raw_text)
        status_banner = status_match.group(1).strip() if status_match else None

        filing_match = re.search(r"Filing Date:\s*(\d{4}-\d{2}-\d{2})", raw_text)
        filing_date = filing_match.group(1).strip() if filing_match else "2026-02-01"

        incident_match = re.search(r"Incident Date:\s*(\d{4}-\d{2}-\d{2})", raw_text)
        incident_date = incident_match.group(1).strip() if incident_match else filing_date

        latency_match = re.search(r"Notice Latency:\s*(\d+)\s*days", raw_text)
        days_to_notify = int(latency_match.group(1)) if latency_match else 0

        dora_match = re.search(r"DORA Tier:\s*([^\n\r]+)", raw_text)
        dora_classification = dora_match.group(1).strip() if dora_match else "Standard Operational Claim (Non-ICT)"

        claimant_match = re.search(r"Claimant Name:\s*([^\n\r]+)", raw_text)
        claimant_name = claimant_match.group(1).strip() if claimant_match else None

        ppsn_match = re.search(r"Claimant PPSN:\s*([^\n\r]+)", raw_text)
        claimant_ppsn = ppsn_match.group(1).strip() if ppsn_match else None

        email_match = re.search(r"Contact Email:\s*([^\n\r]+)", raw_text)
        claimant_email = email_match.group(1).strip() if email_match else None

        phone_match = re.search(r"Contact Phone:\s*([^\n\r]+)", raw_text)
        claimant_phone = phone_match.group(1).strip() if phone_match else None

        location_match = re.search(r"Loss Location:\s*([^\n\r]+(?:\n[^\n\r]+)?)", raw_text)
        loss_location = location_match.group(1).replace("\n", " ").strip() if location_match else None

        eircode_match = re.search(r"\[([A-Z0-9]{3}\s+[A-Z0-9]{4})\]", raw_text)
        eircode = eircode_match.group(1).strip() if eircode_match else None

        type_match = re.search(r"Incident Type:\s*([^\n\r]+)", raw_text)
        incident_type = type_match.group(1).strip() if type_match else None

        synopsis_match = re.search(
            r"INCIDENT SYNOPSIS & REGULATORY CLASSIFICATION\s*(.*?)\s*ITEMIZED FINANCIAL LOSS ASSESSMENT",
            raw_text,
            re.DOTALL,
        )
        incident_description = synopsis_match.group(1).strip() if synopsis_match else None

        # Financial totals
        totals_match = re.search(
            r"TOTALS\s+Aggregated Claim Liquidation Value\s+(-?\d[\d,]*\.\d{2})\s+(-?\d[\d,]*\.\d{2})\s+(-?\d[\d,]*\.\d{2})",
            raw_text,
        )
        if totals_match:
            tot_loss = float(totals_match.group(1).replace(",", ""))
            tot_deduct = float(totals_match.group(2).replace(",", ""))
            tot_net = float(totals_match.group(3).replace(",", ""))
        else:
            tot_loss = 0.0
            tot_deduct = 0.0
            tot_net = 0.0

        # Line items extraction
        line_items: List[ExtractedLineItem] = []
        table_section_match = re.search(
            r"ITEMIZED FINANCIAL LOSS ASSESSMENT.*?Net Claim.*?\n(.*?)\s+TOTALS",
            raw_text,
            re.DOTALL,
        )
        if table_section_match:
            table_text = table_section_match.group(1)
            # Find all triple-number lines denoting (Gross, Deductible, Net)
            num_pattern = re.compile(r"(-?\d[\d,]*\.\d{2})\s+(-?\d[\d,]*\.\d{2})\s+(-?\d[\d,]*\.\d{2})")
            matches = list(num_pattern.finditer(table_text))
            for idx, m in enumerate(matches):
                g_amt = float(m.group(1).replace(",", ""))
                d_amt = float(m.group(2).replace(",", ""))
                n_amt = float(m.group(3).replace(",", ""))
                line_items.append(
                    ExtractedLineItem(
                        category=f"Operational Loss Item {idx + 1}",
                        description=f"Incurred expense during event mitigation for {incident_type or 'claim'}.",
                        amount=g_amt,
                        deductible=d_amt,
                        net_claimed=n_amt,
                    )
                )

        # Attestation
        attest_match = re.search(r"STATUTORY DECLARATION:\s*(.*?)\s*Claimant / Authorized", raw_text, re.DOTALL)
        statutory_attestation = attest_match.group(1).strip() if attest_match else None

        raw_extract = RawClaimExtract(
            claim_id=claim_id,
            policy_number=policy_number,
            status_banner=status_banner,
            filing_date=filing_date,
            incident_date=incident_date,
            days_to_notify=days_to_notify,
            dora_classification=dora_classification,
            claimant_name=claimant_name,
            claimant_ppsn=claimant_ppsn,
            claimant_email=claimant_email,
            claimant_phone=claimant_phone,
            loss_location=loss_location,
            eircode=eircode,
            incident_type=incident_type,
            incident_description=incident_description,
            currency="EUR",
            line_items=line_items,
            total_loss_amount=tot_loss,
            total_deductible=tot_deduct,
            total_net_claimed=tot_net,
            statutory_attestation=statutory_attestation,
        )

        return BronzeLakehouseEnvelope.create_envelope(
            file_path=pdf_path,
            raw_extract=raw_extract,
            extraction_method="deterministic_pypdf_regex",
            model_version="parser_engine_v1",
            raw_text_length=len(raw_text),
        )


def ingest_bronze_lakehouse(
    inbound_dir: Optional[Path] = None,
    bronze_dir: Optional[Path] = None,
) -> Tuple[List[Path], Dict[str, Any]]:
    """Scan raw inbound directory, parse PDFs, and write Bronze Lakehouse JSON files.

    Args:
        inbound_dir: Directory containing raw claim PDFs (defaults to 01_raw_inbound).
        bronze_dir: Target directory for Bronze Lakehouse JSON files (defaults to 02_bronze_lakehouse).

    Returns:
        Tuple[List[Path], Dict[str, Any]]: List of Bronze JSON file paths and ingestion statistics.
    """
    settings = get_settings()
    src_dir = inbound_dir or settings.storage.raw_inbound_dir
    dst_dir = bronze_dir or settings.storage.bronze_lakehouse_dir
    dst_dir.mkdir(parents=True, exist_ok=True)

    parser = ClaimPDFParser()
    pdf_files = sorted(src_dir.glob("*.pdf"))

    if not pdf_files:
        logger.warning("No PDF files found in inbound directory: %s", src_dir)
        return [], {"total_processed": 0, "successful": 0, "failed": 0}

    bronze_files: List[Path] = []
    stats: Dict[str, Any] = {
        "total_processed": 0,
        "successful": 0,
        "failed": 0,
        "anomalies_detected": 0,
        "clean_detected": 0,
        "errors": [],
    }

    for pdf_path in pdf_files:
        stats["total_processed"] += 1
        try:
            envelope = parser.parse_pdf(pdf_path)
            claim_id = envelope.extracted_payload.claim_id

            # Save immutable Bronze Lakehouse JSON file
            output_file = dst_dir / f"bronze_{claim_id}.json"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(envelope.model_dump_json(indent=2))

            bronze_files.append(output_file)
            stats["successful"] += 1

            if envelope.extracted_payload.status_banner == "FLAGGED ANOMALY":
                stats["anomalies_detected"] += 1
            else:
                stats["clean_detected"] += 1

            logger.info("Ingested %s -> %s (Method: %s)", pdf_path.name, output_file.name, envelope.extraction_method)
        except Exception as exc:
            stats["failed"] += 1
            stats["errors"].append({"file": pdf_path.name, "error": str(exc)})
            logger.error("Failed ingesting %s: %s", pdf_path.name, exc)

    return bronze_files, stats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
    print("=" * 80)
    print("Bronze Lakehouse Ingestion Engine (Google GenAI SDK & Fallback)")
    print("=" * 80)

    bronze_records, report = ingest_bronze_lakehouse()

    print(f"\nIngestion Complete: {report['successful']} successful, {report['failed']} failed")
    print(f"Bronze Lakehouse Output Directory: {get_settings().storage.bronze_lakehouse_dir}")
    print(f"Identified Clean Claims: {report['clean_detected']} | Identified Anomalous Claims: {report['anomalies_detected']}")
    print("-" * 80)

    # Preview sample record
    if bronze_records:
        sample_path = bronze_records[0]
        print(f"Sample Ingested Record: {sample_path.name}")
        with open(sample_path, "r", encoding="utf-8") as sf:
            preview_data = json.load(sf)
        print(f"Ingestion ID: {preview_data['ingestion_id']}")
        print(f"Source SHA256: {preview_data['source_file_sha256']}")
        print(f"Extraction Method: {preview_data['extraction_method']}")
        print(f"Claim ID: {preview_data['extracted_payload']['claim_id']}")
        print(f"Claimant: {preview_data['extracted_payload']['claimant_name']}")
        print(f"Net Amount: EUR {preview_data['extracted_payload']['total_net_claimed']:,.2f}")
        print("=" * 80)
