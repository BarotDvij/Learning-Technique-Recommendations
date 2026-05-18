"""
Syllabus parsing — extract text from uploaded files (PDF, txt, markdown).

Architecture:
    * Plain text / markdown → decoded directly, no LLM call
    * Text-based PDF        → pypdf for native text extraction
    * Image-based / scanned → Gemini 2.5 Flash multimodal fallback

The fast path covers ~95% of real syllabi (universities publish them as text-based
PDFs). The vision fallback handles the long tail (scanned handouts, screenshots).
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass
from typing import Optional, Union

try:
    import pypdf  # type: ignore

    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False

try:
    from google import genai  # type: ignore
    from google.genai import types as genai_types  # type: ignore

    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False


# ──────────────────────────────────────────────────────────────────────────────
# Public result type
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class ExtractedSyllabus:
    """Result of parsing an uploaded file."""

    text: str
    source: str  # 'text' | 'pdf-native' | 'pdf-vision' | 'unknown'
    filename: str
    char_count: int
    used_llm: bool = False
    warnings: list[str] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


# ──────────────────────────────────────────────────────────────────────────────
# Native (no-LLM) extractors
# ──────────────────────────────────────────────────────────────────────────────


def _extract_pdf_native(file_bytes: bytes) -> str:
    """Extract text from a text-based PDF using pypdf. Returns empty string on failure."""
    if not PYPDF_AVAILABLE:
        return ""
    try:
        reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        chunks = []
        for page in reader.pages:
            try:
                chunks.append(page.extract_text() or "")
            except Exception:
                continue
        return "\n\n".join(c.strip() for c in chunks if c.strip())
    except Exception:
        return ""


def _extract_text_native(file_bytes: bytes) -> str:
    """Decode raw bytes as UTF-8 with a permissive fallback."""
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return file_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue
    return file_bytes.decode("utf-8", errors="replace")


# ──────────────────────────────────────────────────────────────────────────────
# Vision fallback (Gemini)
# ──────────────────────────────────────────────────────────────────────────────


def _get_genai_client(api_key: Optional[str]):
    if not GENAI_AVAILABLE:
        return None
    key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        return None
    try:
        return genai.Client(api_key=key)
    except Exception:
        return None


def _extract_pdf_vision(file_bytes: bytes, api_key: Optional[str]) -> str:
    """
    Use Gemini 2.5 Flash multimodal to extract text from an image-based PDF.

    Returns empty string if Gemini is unavailable or the call fails.
    """
    client = _get_genai_client(api_key)
    if client is None:
        return ""
    try:
        pdf_part = genai_types.Part.from_bytes(data=file_bytes, mime_type="application/pdf")
        prompt = (
            "This is a course syllabus that may be scanned or image-based. "
            "Extract every word of readable text in reading order. "
            "Preserve section headings, lists, and the original structure. "
            "Return only the extracted text — no commentary, no markdown formatting."
        )
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt, pdf_part],
        )
        return (response.text or "").strip()
    except Exception:
        return ""


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────


def extract_syllabus(
    file_bytes: bytes,
    filename: str,
    api_key: Optional[str] = None,
    min_native_chars: int = 200,
) -> ExtractedSyllabus:
    """
    Extract syllabus text from an uploaded file's bytes.

    The extractor picks a strategy based on file extension and content:

    * ``.txt``, ``.md``       → decode as UTF-8
    * ``.pdf`` (text-based)   → pypdf native extraction
    * ``.pdf`` (image-based)  → Gemini multimodal vision fallback when native
                                extraction returns less than ``min_native_chars``

    Args:
        file_bytes: Raw bytes of the uploaded file.
        filename:   Original filename, used for extension detection + reporting.
        api_key:    Optional Gemini API key. Falls back to ``GEMINI_API_KEY`` /
                    ``GOOGLE_API_KEY`` env vars. Required only for the vision path.
        min_native_chars: If native PDF extraction returns fewer characters than
                    this, the vision fallback is attempted.

    Returns:
        ExtractedSyllabus with text, source label, char count, and warnings.
    """
    lower = filename.lower()
    warnings: list[str] = []

    if lower.endswith((".txt", ".md", ".markdown")):
        text = _extract_text_native(file_bytes)
        return ExtractedSyllabus(
            text=text.strip(),
            source="text",
            filename=filename,
            char_count=len(text),
            used_llm=False,
            warnings=warnings,
        )

    if lower.endswith(".pdf"):
        if not PYPDF_AVAILABLE:
            warnings.append("pypdf not installed — cannot parse PDFs natively.")

        native_text = _extract_pdf_native(file_bytes)
        if len(native_text) >= min_native_chars:
            return ExtractedSyllabus(
                text=native_text,
                source="pdf-native",
                filename=filename,
                char_count=len(native_text),
                used_llm=False,
                warnings=warnings,
            )

        if len(native_text) > 0:
            warnings.append(
                f"Only {len(native_text)} chars extracted natively — attempting vision fallback."
            )
        else:
            warnings.append("No native text — likely a scanned PDF.")

        vision_text = _extract_pdf_vision(file_bytes, api_key=api_key)
        if vision_text:
            return ExtractedSyllabus(
                text=vision_text,
                source="pdf-vision",
                filename=filename,
                char_count=len(vision_text),
                used_llm=True,
                warnings=warnings,
            )

        warnings.append(
            "Vision fallback unavailable (no API key or call failed). "
            "Returning whatever native text was extracted."
        )
        return ExtractedSyllabus(
            text=native_text,
            source="pdf-native",
            filename=filename,
            char_count=len(native_text),
            used_llm=False,
            warnings=warnings,
        )

    # Unknown file type — try as text
    text = _extract_text_native(file_bytes)
    warnings.append("Unknown file extension; treated as plain text.")
    return ExtractedSyllabus(
        text=text.strip(),
        source="unknown",
        filename=filename,
        char_count=len(text),
        used_llm=False,
        warnings=warnings,
    )
