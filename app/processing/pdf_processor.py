import asyncio
import hashlib
import logging
from typing import List, Tuple, Optional, Dict
from collections import defaultdict

# We use the direct unstructured library for finer control over OCR and element types
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import (
    Element, 
    Text, 
    Title, 
    Table, 
    Image, 
    NarrativeText, 
    ListItem
)

from app.core.config import settings
from app.models.schemas import ProcessedContent, DocumentMetadata

logger = logging.getLogger(__name__)

class PDFProcessorService:
    """
    Service to process PDF files locally using Unstructured.
    Designed to work with file paths to optimize memory usage (avoiding loading full binary into RAM).
    """

    def __init__(self):
        # "fas" is the Tesseract code for Persian. "eng" is English.
        # Ensure 'tesseract-ocr-fas' is installed in the Dockerfile.
        self.ocr_languages = ["fas", "eng"] 

    async def process_pdf(self, file_path: str) -> Tuple[str, List[ProcessedContent]]:
        """
        Main entry point.
        1. Calculates file hash (efficiently).
        2. Offloads CPU-intensive PDF parsing to a thread.
        3. Structures the extracted elements into semantic chunks.
        """
        try:
            # 1. Calculate Hash
            file_hash = await asyncio.to_thread(self._calculate_file_hash, file_path)
            
            # 2. Partition PDF (Heavy CPU bound operation)
            logger.info(f"Starting PDF partition for {file_path} with languages {self.ocr_languages}")
            elements = await asyncio.to_thread(self._partition_file, file_path)
            
            if not elements:
                logger.warning(f"No elements found in PDF: {file_path}")
                return file_hash, []

            # 3. Structure Data
            structured_content = self._structure_elements(elements, file_hash)
            
            return file_hash, structured_content

        except Exception as e:
            logger.exception(f"Failed to process PDF {file_path}: {e}")
            raise RuntimeError(f"PDF Processing failed: {e}") from e

    def _calculate_file_hash(self, file_path: str) -> str:
        """
        Calculates SHA256 hash of a file reading in chunks to save RAM.
        """
        sha256_hash = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                # Read in 8k chunks
                for byte_block in iter(lambda: f.read(8192), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
        except IOError as e:
            logger.error(f"IO Error calculating hash for {file_path}: {e}")
            raise

    def _partition_file(self, file_path: str) -> List[Element]:
        """
        Wraps unstructured.partition_pdf.
        Strategy 'hi_res' is required for Table extraction and Image detection.
        """
        try:
            return partition_pdf(
                filename=file_path,
                strategy="hi_res",           # Essential for tables/images
                infer_table_structure=True,  # Extract HTML for tables
                extract_images_in_pdf=False, # We detect them, but don't dump to disk here
                languages=self.ocr_languages, # Handle Persian
                include_page_breaks=False,
            )
        except Exception as e:
            # Fallback logic: sometimes strict language checks fail if tesseract data is missing
            logger.error(f"Primary partitioning failed: {e}. Retrying with default settings.")
            return partition_pdf(
                filename=file_path,
                strategy="fast", # Fallback to text-only if hi_res fails
            )

    def _structure_elements(self, elements: List[Element], file_hash: str) -> List[ProcessedContent]:
        """
        Converts raw Unstructured elements into our ProcessedContent schema.
        - Maintains Section context.
        - Links Images to the Text on the same page.
        """
        processed_contents: List[ProcessedContent] = []
        
        # Group elements by page number first
        pages: Dict[int, List[Element]] = defaultdict(list)
        for el in elements:
            page_num = getattr(el.metadata, "page_number", 1) or 1
            pages[page_num].append(el)

        current_section = "Introduction" # Default section

        # Iterate through pages to preserve flow
        for page_num in sorted(pages.keys()):
            page_elements = pages[page_num]
            
            # Pass 1: Identify all images on this page to create context
            # We create IDs like: "img_p1_0", "img_p1_1"
            page_images = [el for el in page_elements if isinstance(el, Image)]
            related_image_ids = [f"img_p{page_num}_{i}" for i, _ in enumerate(page_images)]

            # Pass 2: Process text and tables
            image_counter = 0
            
            for el in page_elements:
                content_type = "text"
                text_content = ""
                
                # --- Handle Section Titles ---
                if isinstance(el, Title):
                    text = el.text.strip()
                    if len(text) > 2: # Filter noise
                        current_section = text
                    # We also treat titles as embeddable text
                    text_content = text

                # --- Handle Tables ---
                elif isinstance(el, Table):
                    content_type = "table"
                    # Unstructured puts HTML in metadata.text_as_html
                    if hasattr(el.metadata, "text_as_html") and el.metadata.text_as_html:
                        text_content = el.metadata.text_as_html
                    else:
                        text_content = el.text # Fallback to raw text

                # --- Handle Images ---
                elif isinstance(el, Image):
                    content_type = "image"
                    # Placeholder for Vision AI captioning
                    # In a real scenario, you would pass the image bytes to GPT-4o here
                    text_content = (
                        f"Image detected on page {page_num}. "
                        f"Reference ID: img_p{page_num}_{image_counter}. "
                        f"This image is visually located in section '{current_section}'."
                    )
                    image_counter += 1

                # --- Handle Standard Text ---
                elif isinstance(el, (NarrativeText, ListItem, Text)):
                    text_content = el.text.strip()

                # --- Validate and Build Object ---
                if text_content and len(text_content) > 10: # Filter very short noise
                    
                    meta = DocumentMetadata(
                        file_hash=file_hash,
                        page=page_num,
                        section=current_section,
                        related_images=related_image_ids # Context injection
                    )

                    processed_contents.append(
                        ProcessedContent(
                            id=el.id, # Unstructured generates UUIDs
                            content_type=content_type,
                            text_content=text_content,
                            metadata=meta
                        )
                    )

        return processed_contents