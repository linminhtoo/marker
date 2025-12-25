import re

NO_CORRECTION_PHRASES = (
    "no_corrections",
    "no corrections",
    "no corrections needed",
    "no correction needed",
    "no correction required",
    "no corrections required",
    "no errors detected",
    "no errors found",
    "no changes needed",
    "no change needed",
    "looks good",
)


def strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text.strip()


def string_indicates_no_corrections(text: str) -> bool:
    lowered = text.lower()
    return any(phrase in lowered for phrase in NO_CORRECTION_PHRASES)
