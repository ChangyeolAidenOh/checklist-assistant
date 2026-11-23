"""Helpers for rendering LLM / parsed text in Streamlit.

Streamlit renders st.write / st.info / st.markdown strings as Markdown, so
insurance text like "20~100%: 가입금액 × 지급률(20~100%)" turns the span between
two tildes into strikethrough, and "$" pairs into LaTeX. Escape both.
"""


def md(text) -> str:
    """Escape Markdown-active characters that appear in insurance amounts/ranges."""
    if text is None:
        return ""
    return str(text).replace("~", r"\~").replace("$", r"\$")
