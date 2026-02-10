"""
Clause-based document chunker for legal contracts.

WHY CLAUSE-BASED CHUNKING?
- Legal documents have natural semantic boundaries: numbered sections (1., 2., 3...).
- Each section covers one legal concept (e.g., "Liability", "Termination").
- Splitting by section preserves the complete meaning of each clause.
- Fixed-size chunking (e.g., every 200 tokens) would blindly split mid-clause,
  causing the retriever to return fragments that miss critical details.

WHAT THIS MODULE DOES:
1. Reads each .txt file from the data/ directory
2. Extracts the document title (first line, e.g., "NON-DISCLOSURE AGREEMENT (NDA)")
3. Splits the text at section boundaries (lines starting with "1.", "2.", etc.)
4. Returns a list of chunks, each with:
   - text: the full section content
   - metadata: document name, source file, section number, section title
"""

import re
from pathlib import Path
from dataclasses import dataclass, field

from src.config import DATA_DIR


@dataclass
class DocumentChunk:
    """A single chunk extracted from a legal document."""
    text: str
    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        doc = self.metadata.get("document", "?")
        section = self.metadata.get("section_number", "?")
        title = self.metadata.get("section_title", "?")
        return f"Chunk({doc} §{section}: {title}, {len(self.text)} chars)"


def _extract_document_title(first_line: str) -> str:
    """
    Extract a clean document title from the first line of the file.

    Example: "NON-DISCLOSURE AGREEMENT (NDA)" -> "NDA"
             "VENDOR SERVICES AGREEMENT" -> "Vendor Services Agreement"

    We look for a parenthetical abbreviation first (e.g., "(NDA)", "(DPA)", "(SLA)").
    If none found, we title-case the full name.
    """
    # Check for abbreviation in parentheses like (NDA), (DPA), (SLA)
    abbrev_match = re.search(r'\(([A-Z]{2,5})\)', first_line)
    if abbrev_match:
        return abbrev_match.group(1)

    # Otherwise, use the full title cleaned up
    return first_line.strip().title()


def _split_into_sections(text: str) -> list[tuple[str, str, str]]:
    """
    Split document text into sections based on numbered headings.

    Looks for patterns like:
        "1. Definition of Confidential Information"
        "2. Confidentiality Obligations"

    Returns list of (section_number, section_title, section_body) tuples.
    """
    # Pattern: start of line, digit(s), dot, space, then the title text
    # Example: "3. Term and Termination"
    section_pattern = re.compile(r'^(\d+)\.\s+(.+)', re.MULTILINE)

    matches = list(section_pattern.finditer(text))

    if not matches:
        # No numbered sections found -- treat entire document as one chunk
        return [("0", "Full Document", text.strip())]

    sections = []
    for i, match in enumerate(matches):
        section_number = match.group(1)
        section_title = match.group(2).strip()

        # Section body: from this match to the next match (or end of text)
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_body = text[start:end].strip()

        sections.append((section_number, section_title, section_body))

    return sections


def chunk_document(file_path: Path) -> list[DocumentChunk]:
    """
    Parse a single legal document into clause-based chunks.

    Each numbered section becomes one chunk with rich metadata
    that enables precise citation in answers.
    """
    text = file_path.read_text(encoding="utf-8")
    lines = text.strip().split("\n")

    # First non-empty line is the document title
    doc_title_line = next(line for line in lines if line.strip())
    doc_short_name = _extract_document_title(doc_title_line)

    # Split into sections
    sections = _split_into_sections(text)

    chunks = []
    for section_number, section_title, section_body in sections:
        chunk = DocumentChunk(
            text=section_body,
            metadata={
                "document": doc_short_name,
                "document_full_title": doc_title_line.strip(),
                "source_file": file_path.name,
                "section_number": section_number,
                "section_title": section_title,
            }
        )
        chunks.append(chunk)

    return chunks


def chunk_all_documents() -> list[DocumentChunk]:
    """
    Process all .txt files in the data directory.

    Returns a flat list of all chunks across all documents,
    ready to be embedded and stored in the vector database.
    """
    all_chunks = []

    txt_files = sorted(DATA_DIR.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in {DATA_DIR}")

    for file_path in txt_files:
        chunks = chunk_document(file_path)
        all_chunks.extend(chunks)
        print(f"  Chunked {file_path.name}: {len(chunks)} sections")

    print(f"\nTotal chunks: {len(all_chunks)}")
    return all_chunks


# ──────────────────────────────────────────────
# Quick test: run `python -m src.chunker` to see the chunks
# ──────────────────────────────────────────────
if __name__ == "__main__":
    chunks = chunk_all_documents()
    print("\n" + "=" * 60)
    for chunk in chunks:
        print(f"\n{chunk}")
        print(f"  Text preview: {chunk.text[:100]}...")
