"""
Project-wide configuration.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

# Paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
PRODUCT_SUMMARIES_DIR = DATA_DIR / "product_summaries"
NHIS_STATS_DIR = DATA_DIR / "nhis_stats"
SAMPLE_PROFILES_DIR = DATA_DIR / "sample_profiles"
FIGURES_DIR = PROJECT_ROOT / "figures"

# LLM tiers (overridable via env vars)
PARSER_MODEL = os.getenv("PARSER_MODEL", "claude-haiku-4-5-20251001")
GENERATOR_MODEL = os.getenv("GENERATOR_MODEL", "claude-sonnet-4-6")

# ChromaDB
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", str(PROJECT_ROOT / ".chroma_db"))
CHROMA_COLLECTION = "coverage_docs"

# RAG
RAG_SIMILARITY_THRESHOLD = float(os.getenv("RAG_SIMILARITY_THRESHOLD", "0.65"))
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "5"))

# Guardrail
RECOMMENDATION_BLOCKLIST = [
    "추천합니다",
    "가입하세요",
    "이 상품이 좋습니다",
    "반드시 가입",
    "꼭 필요합니다",
    "가입을 권합니다",
    "이 보험을 드세요",
]
DISCLAIMER = (
    "이 분석은 약관 이해를 돕기 위한 참고 자료입니다. "
    "최종 판단은 반드시 보험 설계사 또는 전문 상담을 통해 이루어져야 합니다."
)

# Streamlit
STREAMLIT_PAGE_TITLE = "Coverage Checklist Assistant"
STREAMLIT_PAGE_ICON = "🛡️"
STREAMLIT_LAYOUT = "wide"
