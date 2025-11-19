# app/pdf_parser.py
import base64
import io
from typing import List, Dict, Any
from uuid import uuid4

import pdfplumber

from schemas import ParsedChunk


def _pdf_from_bytes(pdf_bytes: bytes):
    return pdfplumber.open(io.BytesIO(pdf_bytes))


def load_pdf_from_url(url: str) -> bytes:
    import requests

    resp = requests.get(url)
    resp.raise_for_status()
    return resp.content


def load_pdf_from_base64(b64: str) -> bytes:
    return base64.b64decode(b64)


def parse_pdf(
    pdf_bytes: bytes,
    document_id: str,
    language: str = "fa"
) -> List[ParsedChunk]:
    """
    Parses PDF and returns a list of ParsedChunk.
    Relationships:
      - page number stored in metadata["page"]
      - all text/table chunks on a page reference images on that page via metadata["related_images"]
    """
    chunks: List[ParsedChunk] = []

    with _pdf_from_bytes(pdf_bytes) as pdf:
        for page_index, page in enumerate(pdf.pages):
            page_number = page_index + 1

            # --- 1. Images (create image chunks first, to reference them from text/table chunks) ---
            page_image_ids: List[str] = []
            for img_idx, img in enumerate(page.images):
                img_chunk_id = str(uuid4())
                page_image_ids.append(img_chunk_id)

                # Minimal textual representation for embedding.
                img_text = (
                    f"Image from document {document_id}, page {page_number}. "
                    f"Position approx: x0={img.get('x0')}, top={img.get('top')}."
                )

                chunks.append(
                    ParsedChunk(
                        id=img_chunk_id,
                        content_type="image",
                        text=img_text,
                        metadata={
                            "document_id": document_id,
                            "page": page_number,
                            "language": language,
                            "bbox": {
                                "x0": img.get("x0"),
                                "top": img.get("top"),
                                "x1": img.get("x1"),
                                "bottom": img.get("bottom"),
                            },
                            "object_index_on_page": img_idx,
                        },
                    )
                )

            # --- 2. Text ---
            full_text = page.extract_text() or ""
            # Naive paragraph splitting on double newlines
            raw_paragraphs = [p.strip() for p in full_text.split("\n\n") if p.strip()]

            for para_idx, para in enumerate(raw_paragraphs):
                # Skip trivial small text chunks
                if len(para) < 10:
                    continue

                chunks.append(
                    ParsedChunk(
                        content_type="text",
                        text=para,
                        metadata={
                            "document_id": document_id,
                            "page": page_number,
                            "language": language,
                            "paragraph_index_on_page": para_idx,
                            # Maintain relationship with images on same page
                            "related_images": page_image_ids,
                            # Very naive section detection: heading-like text
                            "section": _guess_section_from_text(para),
                        },
                    )
                )

            # --- 3. Tables ---
            try:
                tables = page.extract_tables()
            except Exception:
                tables = []

            for tbl_idx, tbl in enumerate(tables):
                # Convert table (list of rows) to a text representation
                # Row elements may be None; handle gracefully
                normalized_rows = [
                    [cell if cell is not None else "" for cell in row]
                    for row in tbl
                ]
                lines = [" | ".join(row) for row in normalized_rows]
                table_text = "\n".join(lines)

                if not table_text.strip():
                    continue

                chunks.append(
                    ParsedChunk(
                        content_type="table",
                        text=table_text,
                        metadata={
                            "document_id": document_id,
                            "page": page_number,
                            "language": language,
                            "table_index_on_page": tbl_idx,
                            "related_images": page_image_ids,
                        },
                    )
                )

    return chunks


def _guess_section_from_text(text: str) -> str:
    """
    Very simple heuristic:
      - If the text is short and all-caps-ish, consider it a section heading.
    This is just for demo; real-world use would be more advanced.
    """
    snippet = text.strip().split("\n")[0]
    if len(snippet) <= 80 and snippet.isupper():
        return snippet
    return ""