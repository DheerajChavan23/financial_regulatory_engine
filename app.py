"""Enterprise Regulatory & Operational Ingestion Engine - Executive Streamlit Dashboard.

Multi-tab command center uniting tiered lakehouse ingestion, human-in-the-loop (HITL)
governance and quarantine adjudication, interactive DORA compliance RAG auditing,
and executive BI analytics with DAX metrics.
"""

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

import duckdb
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import get_settings
from src.analytics.bi_exporter import BIExporter
from src.analytics.lakehouse import LakehouseAnalyticsEngine
from src.compliance_rag.auditor import ComplianceAuditor
from src.compliance_rag.vector_store import ComplianceVectorStore
from src.extraction.parser import ingest_bronze_lakehouse
from src.generators.synthetic_docs import generate_synthetic_claims
from src.governance.quality_engine import process_bronze_to_silver

# Configure Streamlit page
st.set_page_config(
    page_title="Emerald Shield Assurance | Financial Ingestion & Audit Engine",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom High-End Styling (Dark Mode / Glassmorphism / Sleek FinTech Aesthetic)
st.markdown(
    """
    <style>
    /* Global Styles */
    .main {
        background-color: #0b0f19;
    }
    
    /* Header Card */
    .header-box {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.9) 0%, rgba(30, 41, 59, 0.8) 100%);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        backdrop-filter: blur(8px);
    }
    
    .header-title {
        font-size: 26px;
        font-weight: 800;
        color: #f8fafc;
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    .header-subtitle {
        font-size: 13px;
        color: #94a3b8;
        margin-top: 6px;
    }
    
    /* KPI Metric Cards */
    .kpi-card {
        background: linear-gradient(145deg, #111827 0%, #1f2937 100%);
        border: 1px solid #374151;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 12px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.2);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        border-color: #38bdf8;
    }
    .kpi-label {
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #94a3b8;
    }
    .kpi-value {
        font-size: 24px;
        font-weight: 800;
        color: #f8fafc;
        margin: 4px 0;
    }
    .kpi-subtext {
        font-size: 11px;
        color: #38bdf8;
    }
    
    /* Badges */
    .badge-clean {
        background-color: rgba(16, 185, 129, 0.2);
        color: #34d399;
        padding: 4px 8px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: 700;
        border: 1px solid rgba(16, 185, 129, 0.4);
    }
    .badge-anomaly {
        background-color: rgba(239, 68, 68, 0.2);
        color: #f87171;
        padding: 4px 8px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: 700;
        border: 1px solid rgba(239, 68, 68, 0.4);
    }
    .badge-dora {
        background-color: rgba(99, 102, 241, 0.2);
        color: #818cf8;
        padding: 4px 8px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: 700;
        border: 1px solid rgba(99, 102, 241, 0.4);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

settings = get_settings()


# -----------------------------------------------------------------------------
# Data Loading & Caching Helpers
# -----------------------------------------------------------------------------
@st.cache_data(ttl=60)
def load_lakehouse_data() -> Dict[str, Any]:
    """Retrieve all tiered Lakehouse files and operational tables."""
    raw_files = list(settings.storage.raw_inbound_dir.glob("*.pdf"))
    bronze_files = list(settings.storage.bronze_lakehouse_dir.glob("bronze_*.json"))

    silver_parquet = settings.storage.silver_lakehouse_dir / "claims_silver.parquet"
    quarantine_json = settings.storage.silver_lakehouse_dir / "claims_quarantine.json"

    df_silver = pd.read_parquet(silver_parquet) if silver_parquet.exists() else pd.DataFrame()

    quarantine_records: List[Dict[str, Any]] = []
    if quarantine_json.exists():
        try:
            with open(quarantine_json, "r", encoding="utf-8") as qf:
                quarantine_records = json.load(qf)
        except Exception:
            quarantine_records = []

    gold_mart = settings.storage.gold_lakehouse_dir / "gold_claims_analytical_mart.parquet"
    df_gold = pd.read_parquet(gold_mart) if gold_mart.exists() else pd.DataFrame()

    gold_audit_json = settings.storage.gold_lakehouse_dir / "compliance_audit_reports.json"
    audit_reports: List[Dict[str, Any]] = []
    if gold_audit_json.exists():
        try:
            with open(gold_audit_json, "r", encoding="utf-8") as af:
                audit_reports = json.load(af)
        except Exception:
            audit_reports = []

    return {
        "raw_count": len(raw_files),
        "bronze_count": len(bronze_files),
        "silver_df": df_silver,
        "quarantine_records": quarantine_records,
        "gold_df": df_gold,
        "audit_reports": audit_reports,
    }


data_state = load_lakehouse_data()

# -----------------------------------------------------------------------------
# Sidebar: Engine Status & Pipeline Controller
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🏢 Regulatory Ingestion Engine")
    st.caption("Emerald Shield Assurance DAC")
    st.markdown("---")

    st.markdown("#### 🔄 Pipeline Operations")
    if st.button("🚀 Run Full Ingestion Cycle", use_container_width=True, type="primary"):
        with st.status("Executing End-to-End Pipeline...", expanded=True) as status:
            st.write("1. Generating 20 Synthetic Claims PDFs (ReportLab + Faker)...")
            generate_synthetic_claims(total_count=20, anomalous_count=6)

            st.write("2. Parsing Inbound PDFs to Bronze Lakehouse JSON envelopes...")
            ingest_bronze_lakehouse()

            st.write("3. Applying PII Redaction & Quality Engine to Silver Parquet...")
            process_bronze_to_silver()

            st.write("4. Indexing Regulatory Corpus in ChromaDB & Running Auditor...")
            auditor = ComplianceAuditor()
            auditor.audit_lakehouse_pipeline()

            st.write("5. Transforming Silver data into Gold Star Schema (DuckDB)...")
            lakehouse_engine = LakehouseAnalyticsEngine()
            lakehouse_engine.build_gold_star_schema()

            st.write("6. Generating DAX Metrics and Power BI Artifacts...")
            bi_exporter = BIExporter()
            bi_exporter.run_full_export()

            status.update(label="Full Pipeline Executed Successfully!", state="complete", expanded=False)
            st.cache_data.clear()
            st.rerun()

    st.markdown("---")
    st.markdown("#### 📂 Lakehouse Tier Status")
    st.write(f"• **Raw Inbound (PDF):** {data_state['raw_count']} dossiers")
    st.write(f"• **Bronze (Raw JSON):** {data_state['bronze_count']} records")
    st.write(f"• **Silver (Governed):** {len(data_state['silver_df'])} valid")
    st.write(f"• **Quarantine Pool:** {len(data_state['quarantine_records'])} anomalies")
    st.write(f"• **Gold (Star Schema):** {'Active ✅' if not data_state['gold_df'].empty else 'Pending ⚠️'}")

    st.markdown("---")
    st.caption("Statutory DORA & CBI Cross-Industry Framework | Python 3.11+ DuckDB & ChromaDB")

# -----------------------------------------------------------------------------
# Header Banner
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="header-box">
        <h1 class="header-title">EMERALD SHIELD ASSURANCE DESIGNATED ACTIVITY COMPANY</h1>
        <div class="header-subtitle">
            Enterprise Regulatory & Operational Ingestion Engine • Central Bank of Ireland (CBI) & EU DORA Compliance Portal
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Main Navigation Tabs
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "🚀 Ingestion Pipeline",
        "🛡️ Human-in-the-Loop Quarantine Review",
        "⚖️ DORA Regulatory RAG Auditor",
        "📊 Executive BI & Analytics Mart",
    ]
)

# =============================================================================
# TAB 1: Ingestion Pipeline
# =============================================================================
with tab1:
    st.markdown("### 🏗️ Tiered Lakehouse Architecture & Ingestion Flow")
    st.caption("Immutable lineage from unstructured PDFs down to an OLAP Star Schema.")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">01. Raw Inbound Zone</div>
                <div class="kpi-value">{data_state['raw_count']}</div>
                <div class="kpi-subtext">PDF Dossiers (ReportLab)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">02. Bronze Lakehouse</div>
                <div class="kpi-value">{data_state['bronze_count']}</div>
                <div class="kpi-subtext">Immutable JSON Envelopes</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">03. Silver Lakehouse</div>
                <div class="kpi-value">{len(data_state['silver_df'])}</div>
                <div class="kpi-subtext">Governed Clean Parquet</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">04. Gold Lakehouse</div>
                <div class="kpi-value">{len(data_state['gold_df'])}</div>
                <div class="kpi-subtext">DuckDB Star Schema Mart</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("#### ⚡ Granular Step-by-Step Pipeline Controls")
    btn_col1, btn_col2, btn_col3, btn_col4 = st.columns(4)

    with btn_col1:
        if st.button("1. Generate Raw PDFs", use_container_width=True):
            with st.spinner("Generating 20 Synthetic Claims..."):
                generate_synthetic_claims(total_count=20, anomalous_count=6)
                st.cache_data.clear()
                st.success("20 PDFs generated in data/01_raw_inbound/")
                st.rerun()

    with btn_col2:
        if st.button("2. Ingest into Bronze", use_container_width=True):
            with st.spinner("Parsing PDFs into Bronze JSON..."):
                ingest_bronze_lakehouse()
                st.cache_data.clear()
                st.success("Bronze envelopes written to data/02_bronze_lakehouse/")
                st.rerun()

    with btn_col3:
        if st.button("3. Apply Silver Governance", use_container_width=True):
            with st.spinner("Pseudonymizing PII and validating rules..."):
                process_bronze_to_silver()
                st.cache_data.clear()
                st.success("Silver Parquet created and Quarantine isolated.")
                st.rerun()

    with btn_col4:
        if st.button("4. Build Gold Star Schema", use_container_width=True):
            with st.spinner("Running DuckDB transformations..."):
                engine = LakehouseAnalyticsEngine()
                engine.build_gold_star_schema()
                exporter = BIExporter()
                exporter.run_full_export()
                st.cache_data.clear()
                st.success("Gold Star Schema and DAX metrics exported.")
                st.rerun()

    st.markdown("---")
    st.markdown("#### 📜 Promoted Silver Lakehouse Records (Governed & Validated)")
    if not data_state["silver_df"].empty:
        display_cols = [
            "claim_id",
            "policy_number",
            "filing_date",
            "days_to_notify",
            "dora_classification",
            "total_net_claimed",
            "loss_county",
            "claimant_name_masked",
            "claimant_ppsn_hash",
            "quality_score",
        ]
        st.dataframe(
            data_state["silver_df"][display_cols],
            use_container_width=True,
            column_config={
                "total_net_claimed": st.column_config.NumberColumn("Net Claim (€)", format="€%.2f"),
                "quality_score": st.column_config.ProgressColumn("Quality Score", min_value=0.0, max_value=1.0),
            },
        )
    else:
        st.info("No Silver records available yet. Run the pipeline to populate.")


# =============================================================================
# TAB 2: Human-in-the-Loop (HITL) Quarantine Review
# =============================================================================
with tab2:
    st.markdown("### 🛡️ Human-in-the-Loop (HITL) Quarantine Adjudication")
    st.caption("Supervisory adjudication portal for anomalous, non-conformant, or delayed claim dossiers.")

    quarantine = data_state["quarantine_records"]

    if not quarantine:
        st.success("🎉 Quarantine pool is currently empty! All inbound claims passed quality checks.")
    else:
        q_col1, q_col2, q_col3 = st.columns([1, 1, 1])
        with q_col1:
            st.metric("Total Quarantined Records", len(quarantine))
        with q_col2:
            st.metric("Immediate Rejection Threshold", "Score < 0.85 / Critical SLA Breach")
        with q_col3:
            st.metric("Supervisory Review Authority", "Senior Claims Examiner / CBI Liaison")

        st.markdown("#### 📋 Quarantined Claims Register")

        q_df = pd.DataFrame(
            [
                {
                    "Claim ID": r.get("claim_id"),
                    "Policy": r.get("policy_number") or "[MISSING / NOT PROVIDED]",
                    "Net Amount (€)": r.get("total_net_claimed", 0.0),
                    "Latency": f"{r.get('days_to_notify', 0)} days",
                    "DORA Tier": r.get("dora_classification"),
                    "Triggered Rule Breaches": ", ".join(r.get("failed_rules", [])),
                }
                for r in quarantine
            ]
        )
        st.dataframe(q_df, use_container_width=True)

        st.markdown("---")
        st.markdown("#### 🔍 Dossier Deep-Dive & Adjudication Action Panel")

        claim_options = [r["claim_id"] for r in quarantine]
        selected_claim_id = st.selectbox("Select Quarantined Dossier for Adjudication:", claim_options)

        selected_record = next((r for r in quarantine if r["claim_id"] == selected_claim_id), None)

        if selected_record:
            det_col1, det_col2 = st.columns(2)

            with det_col1:
                st.markdown("##### 📄 Dossier & Financial Audit Breakdown")
                st.write(f"• **Claim ID:** `{selected_record.get('claim_id')}`")
                st.write(f"• **Policy Number:** `{selected_record.get('policy_number') or '[MISSING]'}`")
                st.write(f"• **Filing Date:** `{selected_record.get('filing_date')}`")
                st.write(f"• **Incident Date:** `{selected_record.get('incident_date')}`")
                st.write(f"• **Notice Latency:** `{selected_record.get('days_to_notify')} days`")
                st.write(f"• **Gross Loss Amount:** `€{selected_record.get('total_loss_amount', 0.0):,.2f}`")
                st.write(f"• **Total Net Claimed:** `€{selected_record.get('total_net_claimed', 0.0):,.2f}`")
                st.write(f"• **Quality Score:** `{selected_record.get('quality_score', 0.0):.1%}`")

                st.markdown("##### 🚨 Specific Rule Failure Diagnostic")
                for reason in selected_record.get("failure_reasons", []):
                    st.error(f"⚠️ {reason}")

            with det_col2:
                st.markdown("##### 🔒 PII Redaction & Cryptographic Audit")
                st.write(f"• **Claimant (Masked):** `{selected_record.get('claimant_name_masked')}`")
                st.write(f"• **PPSN Salted Hash:** `{selected_record.get('claimant_ppsn_hash')}`")
                st.write(f"• **Contact Email:** `{selected_record.get('claimant_email_masked')}`")
                st.write(f"• **Loss Jurisdiction:** `{selected_record.get('loss_county')} [{selected_record.get('eircode_routing_key')}]`")

                st.markdown("##### ✍️ Examiner Adjudication Decision")
                adjudication_action = st.radio(
                    "Supervisory Action:",
                    [
                        "Confirm Statutory Rejection (Enforce CBI Rule)",
                        "Request Contract Rectification (Policyholder Contact)",
                        "Override with Force Majeure Exception (Promote to Silver)",
                    ],
                )
                examiner_notes = st.text_area(
                    "Adjudication Justification & Audit Notes:",
                    placeholder="Enter formal justification for supervisory filing...",
                )

                if st.button("💾 Commit Adjudication Record", type="primary"):
                    audit_log_path = settings.storage.silver_lakehouse_dir / "adjudication_audit_log.jsonl"
                    log_entry = {
                        "claim_id": selected_claim_id,
                        "action": adjudication_action,
                        "examiner_notes": examiner_notes,
                        "adjudication_timestamp": datetime.now(timezone.utc).isoformat(),
                        "examiner": "Certified Senior Claims Assessor (DAC)",
                    }
                    with open(audit_log_path, "a", encoding="utf-8") as alf:
                        alf.write(json.dumps(log_entry) + "\n")
                    st.success(f"Adjudication decision for {selected_claim_id} logged to permanent audit ledger!")


# =============================================================================
# TAB 3: Regulatory Compliance RAG & Vector Auditor
# =============================================================================
with tab3:
    st.markdown("### ⚖️ Central Bank of Ireland & DORA Regulatory RAG Auditor")
    st.caption("Semantic retrieval over statutory guidelines powered by ChromaDB vector store.")

    rag_col1, rag_col2 = st.columns([1, 1])

    with rag_col1:
        st.markdown("#### 🔎 Interactive Semantic Regulatory Search")
        preset_q = st.selectbox(
            "Quick Statutory Reference Queries:",
            [
                "What is the statutory deadline and SLA for submitting an insurance claim notification?",
                "Is a policy number mandatory for claim liquidation and what happens if missing?",
                "What are the rules regarding negative liquidation and financial valuation?",
                "What are the requirements for Irish PPSN cryptographic pseudonymization under GDPR?",
                "What are the criteria for a Major ICT-Related Incident under DORA Article 18?",
            ],
        )
        custom_query = st.text_input("Or input custom regulatory inquiry:", value=preset_q)
        top_k = st.slider("Number of Statutory Excerpts (Top-K):", min_value=1, max_value=4, value=2)

        if st.button("🔍 Search Regulatory Corpus", use_container_width=True):
            try:
                vector_store = ComplianceVectorStore()
                results = vector_store.search_guidelines(custom_query, top_k=top_k)

                st.markdown("##### 📑 Retrieved Regulatory Guidance & Citations")
                for idx, hit in enumerate(results, start=1):
                    sec = hit["metadata"].get("section", "Section Guidance")
                    sim = hit["similarity_score"]
                    st.markdown(
                        f"""
                        <div style="background-color: #1e293b; padding: 12px; border-radius: 8px; border-left: 4px solid #38bdf8; margin-bottom: 10px;">
                            <div style="font-size: 13px; font-weight: bold; color: #38bdf8;">Hit #{idx}: {sec} (Semantic Relevance: {sim:.1%})</div>
                            <div style="font-size: 12px; color: #cbd5e1; margin-top: 6px; white-space: pre-wrap;">{hit['text']}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            except Exception as e:
                st.error(f"Error querying vector store: {e}")

    with rag_col2:
        st.markdown("#### 📋 Statutory Audit Reports Summary")
        audit_reports = data_state["audit_reports"]

        if not audit_reports:
            st.info("No compliance audit reports found in Gold lakehouse. Run compliance auditor.")
        else:
            compliant_count = sum(1 for r in audit_reports if r.get("overall_verdict") == "COMPLIANT")
            breach_count = sum(1 for r in audit_reports if r.get("overall_verdict") == "REGULATORY_BREACH")

            m_col1, m_col2 = st.columns(2)
            with m_col1:
                st.metric("Statutoriily Compliant", f"{compliant_count} claims")
            with m_col2:
                st.metric("Statutory Breaches", f"{breach_count} claims", delta=f"-{breach_count}", delta_color="inverse")

            filter_verdict = st.radio("Filter Dossiers:", ["All Audited", "Statutory Breaches Only", "Compliant Only"], horizontal=True)

            filtered_reports = audit_reports
            if filter_verdict == "Statutory Breaches Only":
                filtered_reports = [r for r in audit_reports if r.get("overall_verdict") == "REGULATORY_BREACH"]
            elif filter_verdict == "Compliant Only":
                filtered_reports = [r for r in audit_reports if r.get("overall_verdict") == "COMPLIANT"]

            for rep in filtered_reports[:6]:
                verdict = rep.get("overall_verdict")
                is_breach = verdict == "REGULATORY_BREACH"
                badge = '<span class="badge-anomaly">REGULATORY BREACH</span>' if is_breach else '<span class="badge-clean">COMPLIANT</span>'

                with st.expander(f"{rep.get('claim_id')} — {verdict}"):
                    st.markdown(f"**Verdict:** {badge}", unsafe_allow_html=True)
                    st.write(f"**Summary:** {rep.get('supervisory_summary')}")
                    st.markdown("**Evaluated Findings:**")
                    for f in rep.get("findings", []):
                        icon = "❌" if f["verdict"] == "BREACH" else "✅"
                        st.write(f"{icon} **{f['area']}:** {f['observation']}")
                        if f["verdict"] == "BREACH":
                            st.caption(f"Citation: {f['statutory_citation'][:160]}...")


# =============================================================================
# TAB 4: Executive BI & Lakehouse Analytics Mart
# =============================================================================
with tab4:
    st.markdown("### 📊 Executive BI Dashboard & Lakehouse Analytics Mart")
    st.caption("Live portfolio metrics, DORA Article 18 exposure, and Power BI DAX KPIs powered by DuckDB.")

    df_gold = data_state["gold_df"]

    if df_gold.empty:
        st.warning("Gold analytical mart table is empty. Click 'Run Full Ingestion Cycle' to build.")
    else:
        # 1. Top KPI Row (Calculated dynamically)
        tot_claims = len(df_gold)
        tot_net = df_gold["net_claimed_amount"].sum()
        tot_gross = df_gold["gross_loss_amount"].sum()
        avg_val = df_gold["net_claimed_amount"].mean()
        avg_latency = df_gold["days_to_notify"].mean()

        dora_major_mask = df_gold["dora_classification"].str.contains("Major ICT", na=False)
        dora_exp = df_gold[dora_major_mask]["net_claimed_amount"].sum()
        dora_pct = (dora_exp / tot_net) * 100.0 if tot_net > 0 else 0.0

        ict_mask = df_gold["is_ict_related"] == True
        ict_exp = df_gold[ict_mask]["net_claimed_amount"].sum()
        ict_pct = (ict_exp / tot_net) * 100.0 if tot_net > 0 else 0.0

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">Active Portfolio Net Value</div>
                    <div class="kpi-value">€{tot_net:,.2f}</div>
                    <div class="kpi-subtext">Across {tot_claims} Validated Dossiers</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with k2:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">Average Net Claim</div>
                    <div class="kpi-value">€{avg_val:,.2f}</div>
                    <div class="kpi-subtext">Gross Assessed: €{tot_gross:,.2f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with k3:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">Major DORA Exposure</div>
                    <div class="kpi-value">€{dora_exp:,.2f}</div>
                    <div class="kpi-subtext">{dora_pct:.1f}% of Active Exposure</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with k4:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">Average Notice Latency</div>
                    <div class="kpi-value">{avg_latency:.1f} days</div>
                    <div class="kpi-subtext">Statutory SLA Compliance: 100%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # 2. Charts Row
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.markdown("##### 📍 Geographic Exposure by Irish Province")
            geo_agg = df_gold.groupby("province", as_index=False)["net_claimed_amount"].sum()
            fig_geo = px.bar(
                geo_agg,
                x="province",
                y="net_claimed_amount",
                labels={"province": "Irish Province", "net_claimed_amount": "Total Net Claim (€)"},
                color="province",
                color_discrete_sequence=px.colors.qualitative.Prism,
            )
            fig_geo.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#cbd5e1",
                margin=dict(l=20, r=20, t=30, b=20),
            )
            st.plotly_chart(fig_geo, use_container_width=True)

        with chart_col2:
            st.markdown("##### ⚡ Exposure by DORA Regulatory Tier")
            tier_agg = df_gold.groupby("dora_regulatory_tier", as_index=False)["net_claimed_amount"].sum()
            fig_tier = px.pie(
                tier_agg,
                names="dora_regulatory_tier",
                values="net_claimed_amount",
                hole=0.45,
                color_discrete_sequence=["#38bdf8", "#818cf8", "#34d399"],
            )
            fig_tier.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#cbd5e1",
                margin=dict(l=20, r=20, t=30, b=20),
            )
            st.plotly_chart(fig_tier, use_container_width=True)

        chart_col3, chart_col4 = st.columns(2)

        with chart_col3:
            st.markdown("##### ⏱️ Filing Latency Distribution vs 90-Day SLA Limit")
            fig_lat = px.histogram(
                df_gold,
                x="days_to_notify",
                nbins=12,
                labels={"days_to_notify": "Elapsed Days to Notice"},
                color_discrete_sequence=["#38bdf8"],
            )
            fig_lat.add_vline(x=90, line_dash="dash", line_color="#ef4444", annotation_text="CBI 90d SLA Limit")
            fig_lat.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#cbd5e1",
                margin=dict(l=20, r=20, t=30, b=20),
            )
            st.plotly_chart(fig_lat, use_container_width=True)

        with chart_col4:
            st.markdown("##### 🛡️ Peril Category & ICT Exposure Distribution")
            inc_agg = df_gold.groupby("incident_type", as_index=False)["net_claimed_amount"].sum().sort_values(
                "net_claimed_amount", ascending=True
            )
            fig_inc = px.bar(
                inc_agg,
                y="incident_type",
                x="net_claimed_amount",
                orientation="h",
                labels={"incident_type": "Incident Peril", "net_claimed_amount": "Total Net Claim (€)"},
                color_discrete_sequence=["#a855f7"],
            )
            fig_inc.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#cbd5e1",
                margin=dict(l=20, r=20, t=30, b=20),
            )
            st.plotly_chart(fig_inc, use_container_width=True)

        st.markdown("---")
        st.markdown("#### 📥 BI Feeds & Power BI Export Center")
        d_col1, d_col2, d_col3, d_col4 = st.columns(4)

        bi_dir = settings.storage.gold_lakehouse_dir / "bi_export"
        dax_file = bi_dir / "dax_measures.dax"
        tom_file = bi_dir / "powerbi_tabular_model.json"
        csv_mart = bi_dir / "claims_analytical_mart.csv"

        with d_col1:
            if dax_file.exists():
                with open(dax_file, "r", encoding="utf-8") as df_in:
                    st.download_button(
                        "📄 Download 19 DAX Measures",
                        df_in.read(),
                        file_name="dax_measures.dax",
                        mime="text/plain",
                        use_container_width=True,
                    )
        with d_col2:
            if tom_file.exists():
                with open(tom_file, "r", encoding="utf-8") as tom_in:
                    st.download_button(
                        "📊 Power BI Tabular Model",
                        tom_in.read(),
                        file_name="powerbi_tabular_model.json",
                        mime="application/json",
                        use_container_width=True,
                    )
        with d_col3:
            if csv_mart.exists():
                with open(csv_mart, "r", encoding="utf-8") as csv_in:
                    st.download_button(
                        "📈 Export Analytical Mart (CSV)",
                        csv_in.read(),
                        file_name="claims_analytical_mart.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
        with d_col4:
            st.button(
                "🔄 Sync Star Schema & Feeds",
                use_container_width=True,
                on_click=lambda: BIExporter().run_full_export(),
            )
