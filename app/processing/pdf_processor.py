import asyncio
import hashlib
import logging
import tempfile
from collections import defaultdict
from typing import List, Tuple, Dict, Any

from langchain_community.document_loaders import UnstructuredPDFLoader
from unstructured.documents.elements import Element, Text, Image, Table

from app.core.config import settings
from app.models.schemas import ProcessedContent, DocumentMetadata

logger = logging.getLogger(__name__)

def is_potential_section_title(text: str) -> bool:
    """
    A more advanced heuristic to detect if a line of text is a section title.
    - It's not too long.
    - It's in title case, uppercase, or starts with common section words.
    - It doesn't end with a period.
    """
    text = text.strip()
    if not text or len(text) > 100 or text.endswith('.'):
        return False
    if text.isupper() or text.istitle():
        return True
    common_starters = ("introduction", "abstract", "conclusion", "references", "appendix", "section")
    if text.lower().startswith(common_starters):
        return True
    return False

class PDFProcessorService:
    """A service to process PDF files using LangChain and Unstructured."""

    def __init__(self, ocr_language: str = settings.OCR_LANGUAGE):
        self.ocr_language = ocr_language

    async def process_pdf(self, file_content: bytes) -> Tuple[str, List[ProcessedContent]]:
        """
        Processes a PDF file's content asynchronously.

        Args:
            file_content: The raw byte content of the PDF file.

        Returns:
            A tuple containing the file hash and a list of processed content chunks.
        """
        file_hash = self._calculate_hash(file_content)

        # UnstructuredPDFLoader requires a file path, so we use a temporary file.
        with tempfile.NamedTemporaryFile(delete=True, suffix=".pdf") as tmp:
            tmp.write(file_content)
            tmp.flush()

            # The loader is synchronous, so we run it in a thread pool.
            elements = await asyncio.to_thread(self._load_and_partition_pdf, tmp.name)

        if not elements:
            raise ValueError("No content could be extracted from the PDF.")

        return file_hash, self._structure_elements(elements, file_hash)

    def _load_and_partition_pdf(self, file_path: str) -> List[Element]:
        """Loads and partitions the PDF using UnstructuredPDFLoader."""
        try:
            loader = UnstructuredPDFLoader(
                file_path,
                mode="elements",
                strategy="hi_res",
                infer_table_structure=True,
                # Pass OCR language to unstructured
                languages=self.ocr_language.split('+')
            )
            return loader.load()
        except Exception as e:
            logger.exception("PDF parsing failed for file path %s: %s", file_path, e)
            raise ValueError(f"Failed to parse PDF: {e}") from e

    def _calculate_hash(self, file_content: bytes) -> str:
        """Calculates the SHA256 hash of the file content."""
        return hashlib.sha256(file_content).hexdigest()

    def _structure_elements(self, elements: List[Element], file_hash: str) -> List[ProcessedContent]:
        """
        Groups elements by page and structures them into ProcessedContent objects.
        """
        page_elements: Dict[int, List[Element]] = defaultdict(list)
        for el in elements:
            page_num = el.metadata.page_number or 0
            page_elements[page_num].append(el)

        processed_contents: List[ProcessedContent] = []
        for page_num, elems in sorted(page_elements.items()):
            image_ids_on_page = [
                f"img_{page_num}_{i}" for i, el in enumerate(elems) if isinstance(el, Image)
            ]

            current_section = "Unknown"
            image_counter = 0

            for el in elems:
                content_type = "unknown"
                text_content = ""

                if isinstance(el, Text):
                    text = el.text.strip()
                    if is_potential_section_title(text):
                        current_section = text
                        continue # Skip adding the title as a separate content chunk
                    content_type = "text"
                    text_content = text
                elif isinstance(el, Table):
                    content_type = "table"
                    text_content = el.metadata.text_as_html or ""
                elif isinstance(el, Image):
                    content_type = "image"
                    # For images, the "text" is a descriptive caption.
                    # The actual OCR'd text will be in the embedding if desired.
                    text_content = f"Image on page {page_num}: A visual element."
                    image_id = f"img_{page_num}_{image_counter}"
                    image_counter += 1

                if not text_content:
                    continue

                metadata = DocumentMetadata(
                    page=page_num,
                    section=current_section,
                    related_images=image_ids_on_page,
                    file_hash=file_hash,
                )

                processed_contents.append(
                    ProcessedContent(
                        content_type=content_type,
                        text_content=text_content,
                        metadata=metadata,
                    )
                )
        return processed_contents
