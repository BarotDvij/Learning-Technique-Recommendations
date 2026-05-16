"""Unit tests for src/parsing.py — no network calls required."""

from __future__ import annotations

import io

import pytest

from src.parsing import ExtractedSyllabus, extract_syllabus


SAMPLE_TEXT = (
    "PHIL 101: Introduction to Philosophy\n\n"
    "This course explores the foundational questions of philosophy "
    "through close reading of primary texts and structured discussion."
)


def test_extract_plain_text() -> None:
    result = extract_syllabus(SAMPLE_TEXT.encode("utf-8"), "intro.txt")
    assert isinstance(result, ExtractedSyllabus)
    assert result.source == "text"
    assert result.char_count > 0
    assert "Philosophy" in result.text
    assert result.used_llm is False
    assert result.warnings == []


def test_extract_markdown_treated_as_text() -> None:
    result = extract_syllabus(b"# heading\n\nbody text", "course.md")
    assert result.source == "text"
    assert "heading" in result.text


def test_extract_unknown_extension_falls_back_to_text() -> None:
    result = extract_syllabus(b"random body", "weird.rtf")
    assert result.source == "unknown"
    assert "Unknown file extension" in " ".join(result.warnings)


def test_extract_handles_non_utf8_bytes() -> None:
    payload = "naïve café".encode("latin-1")
    result = extract_syllabus(payload, "fancy.txt")
    assert result.char_count > 0
    # Either decoded cleanly or replaced gracefully — must not raise.


def test_extract_real_pdf_native_path() -> None:
    reportlab = pytest.importorskip("reportlab.pdfgen")
    from reportlab.pdfgen import canvas  # type: ignore
    from reportlab.lib.pagesizes import letter  # type: ignore

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    c.setFont("Helvetica", 11)
    y = 750
    for line in [
        "BIO 240: Cell Biology",
        "This course covers cellular structure, function, and division.",
        "Topics include membrane transport, the cell cycle, and signaling.",
    ]:
        c.drawString(72, y, line)
        y -= 14
    c.save()

    result = extract_syllabus(buf.getvalue(), "cell_bio.pdf")
    assert result.source == "pdf-native"
    assert "Cell Biology" in result.text
    assert result.used_llm is False


def test_extract_pdf_with_no_text_returns_warning() -> None:
    # Garbage bytes labeled as a PDF — pypdf will fail or return nothing,
    # and without an API key the vision fallback should also be skipped.
    result = extract_syllabus(b"%PDF-1.4 garbage", "broken.pdf")
    assert result.source in ("pdf-native", "pdf-vision")
    assert any(
        "scanned" in w.lower() or "chars extracted" in w.lower() or "vision" in w.lower()
        for w in result.warnings
    )
