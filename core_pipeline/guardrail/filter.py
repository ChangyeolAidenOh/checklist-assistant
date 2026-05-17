"""
Guardrail Filter.

Post-processing filter that:
1. Blocks product recommendation language
2. Ensures disclaimer is always present
3. Sanitizes any specific product promotion

Applied to all Node 4 outputs before display.
"""

import logging
import re

from config import RECOMMENDATION_BLOCKLIST, DISCLAIMER

logger = logging.getLogger(__name__)


class GuardrailFilter:
    """Filters LLM output to ensure regulatory safety."""

    def __init__(self, blocklist: list[str] | None = None):
        self.blocklist = blocklist or RECOMMENDATION_BLOCKLIST
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Compile regex patterns for blocked phrases."""
        self.patterns = [re.compile(re.escape(phrase)) for phrase in self.blocklist]

        # Additional patterns for product promotion
        self.promo_patterns = [
            re.compile(r"(이|이\s*)?상품[을를]?\s*(추천|권유|권합|드립)"),
            re.compile(r"가입[을를]?\s*(추천|권유|권합|드립)"),
            re.compile(r"꼭\s*(가입|들어|넣어)"),
            re.compile(r"보험[을를]?\s*(들|가입하|넣)"),
        ]

    def filter_text(self, text: str) -> str:
        """Filter a single text string.

        Replaces blocked phrases with safe alternatives.
        """
        filtered = text

        for pattern in self.patterns:
            filtered = pattern.sub("[설계사와 상의가 필요합니다]", filtered)

        for pattern in self.promo_patterns:
            filtered = pattern.sub("설계사와 상의해보세요", filtered)

        return filtered

    def filter_output(self, output: dict) -> dict:
        """Filter entire output dictionary recursively."""
        return self._filter_recursive(output)

    def _filter_recursive(self, obj):
        """Recursively filter all string values in a dict/list."""
        if isinstance(obj, str):
            return self.filter_text(obj)
        elif isinstance(obj, dict):
            return {k: self._filter_recursive(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._filter_recursive(item) for item in obj]
        return obj

    def ensure_disclaimer(self, output: dict) -> dict:
        """Ensure disclaimer is present in output."""
        if "disclaimer" not in output or not output["disclaimer"]:
            output["disclaimer"] = DISCLAIMER
        return output

    def validate(self, text: str) -> tuple[bool, list[str]]:
        """Check if text contains any blocked phrases.

        Returns:
            (is_safe, list_of_violations)
        """
        violations = []

        for phrase in self.blocklist:
            if phrase in text:
                violations.append(f"Blocked phrase detected: '{phrase}'")

        for pattern in self.promo_patterns:
            match = pattern.search(text)
            if match:
                violations.append(f"Promo pattern detected: '{match.group()}'")

        return len(violations) == 0, violations
