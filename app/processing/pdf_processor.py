import asyncio
from collections import defaultdict
import hashlib
import logging
import base64
from typing import List, Tuple, Dict
from concurrent.futures import ProcessPoolExecutor

# Unstructured imports
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title
from unstructured.documents.elements import (
    Element,
    Table,
    Image,
)

from models.schemas import ProcessedContent, DocumentMetadata

logger = logging.getLogger(__name__)

class PDFProcessorService:
    """
    Service to process PDF files locally using Unstructured.
    Optimized for RAG (Retrieval Augmented Generation) pipelines.
    """

    def __init__(self):
        self.ocr_languages = ["fas", "eng"]
        # Configuration for chunking
        self.max_chunk_characters = 1500  # Target size for a text chunk
        self.new_after_n_chars = 1200     # Soft limit to start breaking
        self.overlap = 150                # Overlap for better context continuity

    async def process_pdf(self, file_path: str) -> Tuple[str, List[ProcessedContent]]:
        """
        Main entry point.
        1. Calculates hash.
        2. Offloads CPU-intensive OCR/Partitioning to a separate PROCESS.
        3. Chunks and structures data.
        """
        try:
            # 1. Calculate Hash (IO Bound - usually fast enough to run in thread)
            file_hash = await asyncio.to_thread(self._calculate_file_hash, file_path)

            # 2. Partition PDF (CPU Bound - Heavy)
            # We use a ProcessPoolExecutor to avoid blocking the Main Event Loop (GIL)
            loop = asyncio.get_running_loop()
            with ProcessPoolExecutor() as pool:
                logger.info(f"Starting PDF partition for {file_path} in separate process...")
                elements = await loop.run_in_executor(
                    pool, 
                    self._partition_file_sync, 
                    file_path
                )

            if not elements:
                logger.warning(f"No elements found in PDF: {file_path}")
                return file_hash, []

            # 3. Chunking & Structuring (CPU Bound - Light/Medium)
            # Merges small elements into semantic chunks
            structured_content = await asyncio.to_thread(
                self._structure_and_chunk_elements, elements, file_hash
            )

            logger.info(f"Successfully processed {file_path}: {len(structured_content)} chunks generated.")
            return file_hash, structured_content

        except Exception as e:
            logger.exception(f"Failed to process PDF {file_path}")
            raise RuntimeError(f"PDF Processing failed: {str(e)}") from e

    def _calculate_file_hash(self, file_path: str) -> str:
        sha256_hash = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                # 64k chunks are generally optimal for modern OS file I/O
                for byte_block in iter(lambda: f.read(65536), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
        except IOError as e:
            logger.error(f"IO Error calculating hash: {e}")
            raise

    def _partition_file_sync(self, file_path: str) -> List[Element]:
        """
        Synchronous wrapper for partition_pdf to be run in ProcessPool.
        """
        try:
            # 'hi_res' is required for Table extraction and OCR
            # 'extract_images_in_pdf=True' extracts image objects for processing
            return partition_pdf(
                filename=file_path,
                strategy="hi_res",
                infer_table_structure=True,
                languages=self.ocr_languages,
                extract_images_in_pdf=True,  # We need the actual image data
                extract_image_block_types=["Image", "Table"], # Get crops of tables too if needed
                include_page_breaks=False,
            )
        except Exception as e:
            logger.error(f"Primary partitioning failed: {e}. Retrying with 'fast' strategy.")
            # Fallback: Fast is much faster but loses Tables and OCR capability
            return partition_pdf(
                filename=file_path,
                strategy="fast", 
            )

    def _structure_and_chunk_elements(self, elements: List[Element], file_hash: str) -> List[ProcessedContent]:
        """
        1. Aggregates raw elements into semantic chunks (Text).
        2. Handles Tables and Images separately.
        """
        processed_contents: List[ProcessedContent] = []
        
        # Separate Images/Tables from Text for specialized handling
        text_elements = []
        special_elements = [] # Tables and Images

        image_map: Dict[int, List[str]] = defaultdict(list) # Map page -> List[ImageDescriptions]

        for el in elements:
            if isinstance(el, (Table, Image)):
                special_elements.append(el)
            else:
                text_elements.append(el)

        # --- Step A: Smart Chunking for Text ---
        # unstructured.chunking.title combines paragraphs based on titles/headers
        chunked_text = chunk_by_title(
            text_elements,
            max_characters=self.max_chunk_characters,
            new_after_n_chars=self.new_after_n_chars,
            overlap=self.overlap
        )

        # --- Step B: Process Images & Tables first to build Context ---
        for el in special_elements:
            page_num = getattr(el.metadata, "page_number", 1) or 1
            
            if isinstance(el, Table):
                html_content = getattr(el.metadata, "text_as_html", None)
                text_content = html_content if html_content else el.text
                
                processed_contents.append(ProcessedContent(
                    id=el.id,
                    content_type="table",
                    text_content=text_content,
                    metadata=DocumentMetadata(
                        file_hash=file_hash,
                        page=page_num,
                        section="Table Data" # Metadata often misses section for tables
                    )
                ))

            elif isinstance(el, Image):
                # Extract Base64 logic
                image_b64 = None
                if hasattr(el.metadata, "image_base64"):
                    image_b64 = el.metadata.image_base64
                elif hasattr(el, "path") and el.path:
                    # If unstructured saved to temp disk, read it
                    try:
                        with open(el.path, "rb") as img_f:
                            image_b64 = base64.b64encode(img_f.read()).decode('utf-8')
                    except Exception:
                        pass
                
                # Store reference for text chunks to use
                ref_id = f"img_{el.id[:8]}"
                image_desc = f"Image Reference [{ref_id}] on page {page_num}"
                image_map[page_num].append(image_desc)

                # Create the Image Content Object
                # Note: For production, you might upload image_b64 to S3 and store the URL here
                processed_contents.append(ProcessedContent(
                    id=el.id,
                    content_type="image",
                    text_content=image_desc, # Placeholder until Vision AI generates caption
                    image_data=image_b64,    # Add this field to your Schema if possible
                    metadata=DocumentMetadata(
                        file_hash=file_hash,
                        page=page_num,
                        section="Visual Content"
                    )
                ))

        # --- Step C: Process Text Chunks with Context ---
        for chunk in chunked_text:
            # CompositeElement represents a merged chunk
            content_text = chunk.text
            
            # metadata.page_number might be a list in CompositeElement if it spans pages
            # We take the first page for simplicity
            page_num = 1
            if hasattr(chunk.metadata, "page_number"):
                pn = chunk.metadata.page_number
                page_num = pn[0] if isinstance(pn, list) else pn

            # Identify Section
            section = "General"
            # Try to find section in metadata or hierarchy
            if hasattr(chunk.metadata, "section"):
                section = chunk.metadata.section
            
            # Inject "See Image" context if images exist on this page
            related_imgs = image_map.get(page_num, [])
            
            processed_contents.append(ProcessedContent(
                id=chunk.id,
                content_type="text",
                text_content=content_text,
                metadata=DocumentMetadata(
                    file_hash=file_hash,
                    page=page_num,
                    section=section,
                    related_images=related_imgs # Pass IDs/Descriptions for context
                )
            ))

        return processed_contents