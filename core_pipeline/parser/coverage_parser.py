"""
Node 1: Coverage Parser.

Parses insurance product summaries into structured JSON.
Converts coverage items to plain language with source references.

Model tier: Lightweight LLM (Haiku) - structured extraction task.
"""

import logging
from dataclasses import dataclass, field

from core_pipeline.llm.backend import get_backend

logger = logging.getLogger(__name__)

PARSER_SYSTEM_PROMPT = """\
You are an insurance document parser specializing in Korean insurance products.

Your task: Parse the given product summary and extract structured coverage information.

Output ONLY valid JSON with this schema:
{
  "product_name": "상품명",
  "insurer": "보험사명",
  "product_type": "종신보험 | 건강보험 | 실손보험 | ...",
  "coverage_items": [
    {
      "category": "사망 | 입원 | 수술 | 통원 | 진단 | 치과 | ...",
      "item_name": "보장 항목 원문",
      "plain_description": "일상 언어로 변환한 설명",
      "benefit_amount": "보장 금액 (원문 그대로)",
      "conditions": "조건/제한 사항",
      "waiting_period": "대기기간 (있는 경우)",
      "source_reference": "원문에서 해당 내용이 있는 위치/섹션"
    }
  ],
  "exclusions": [
    {
      "item": "면책 사항",
      "plain_description": "일상 언어 설명",
      "source_reference": "원문 위치"
    }
  ],
  "premium_info": {
    "monthly_premium": "월 보험료",
    "payment_period": "납입 기간",
    "coverage_period": "보장 기간"
  },
  "parse_confidence": "high | medium | low",
  "unparsed_sections": ["파싱하지 못한 섹션 목록"]
}

Rules:
- Extract ONLY information explicitly stated in the document
- If information is unclear, set parse_confidence to "low" and note in unparsed_sections
- plain_description must be understandable by a non-expert Korean speaker
- Always include source_reference pointing to the original text location
- Do NOT infer or assume coverage details not in the document
- CRITICAL: Keep ALL string values under 80 characters. No newlines in values.
- Extract at most 10 coverage_items (the most important ones)
- Extract at most 5 exclusions
"""


@dataclass
class CoverageItem:
    category: str
    item_name: str
    plain_description: str
    benefit_amount: str
    conditions: str = ""
    waiting_period: str = ""
    source_reference: str = ""


@dataclass
class ParsedCoverage:
    product_name: str
    insurer: str
    product_type: str
    coverage_items: list[CoverageItem] = field(default_factory=list)
    exclusions: list[dict] = field(default_factory=list)
    premium_info: dict = field(default_factory=dict)
    parse_confidence: str = "low"
    unparsed_sections: list[str] = field(default_factory=list)
    raw_json: dict = field(default_factory=dict)


class CoverageParser:
    """Parses insurance product summaries into structured data."""

    def __init__(self):
        self.backend = get_backend("parser")

    def parse(self, document_text: str) -> ParsedCoverage:
        """Parse a product summary document.

        Args:
            document_text: Raw text of the product summary

        Returns:
            ParsedCoverage with structured coverage data
        """
        logger.info("Parsing product summary (%d chars)", len(document_text))

        prompt = (
            "다음 보험 상품 요약서를 분석하고 구조화된 JSON으로 변환하세요.\n\n"
            f"--- 상품 요약서 ---\n{document_text}\n--- 끝 ---"
        )

        try:
            result = self.backend.generate_structured(
                prompt=prompt,
                system=PARSER_SYSTEM_PROMPT,
            )
        except Exception as e:
            logger.error("Parsing failed: %s", e)
            return ParsedCoverage(
                product_name="파싱 실패",
                insurer="",
                product_type="",
                parse_confidence="low",
                unparsed_sections=["전체 문서 - 자동 분석 범위 밖"],
            )

        return self._to_parsed_coverage(result)

    def _to_parsed_coverage(self, raw: dict) -> ParsedCoverage:
        """Convert raw JSON to ParsedCoverage dataclass."""
        items = []
        for item in raw.get("coverage_items", []):
            items.append(CoverageItem(
                category=item.get("category", ""),
                item_name=item.get("item_name", ""),
                plain_description=item.get("plain_description", ""),
                benefit_amount=item.get("benefit_amount", ""),
                conditions=item.get("conditions", ""),
                waiting_period=item.get("waiting_period", ""),
                source_reference=item.get("source_reference", ""),
            ))

        return ParsedCoverage(
            product_name=raw.get("product_name", ""),
            insurer=raw.get("insurer", ""),
            product_type=raw.get("product_type", ""),
            coverage_items=items,
            exclusions=raw.get("exclusions", []),
            premium_info=raw.get("premium_info", {}),
            parse_confidence=raw.get("parse_confidence", "low"),
            unparsed_sections=raw.get("unparsed_sections", []),
            raw_json=raw,
        )
