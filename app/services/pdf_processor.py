import hashlib
from typing import IO, List, Tuple
import uuid
from unstructured.partition.pdf import partition_pdf
import logging

from models.schemas import ProcessedContent

class PDFProcessor:
    """
    A service class to handle the processing of PDF documents from a file-like object.
    This includes hashing and extracting content from the PDF stream.
    """
    def __init__(self):
        pass

    @staticmethod
    def _calculate_hash_from_stream(file_obj: IO) -> str:
        """
        Calculates the SHA256 hash of a file from a stream.
        """
        sha256_hash = hashlib.sha256()
        for byte_block in iter(lambda: file_obj.read(4096), b""):
            sha256_hash.update(byte_block)
        file_obj.seek(0) # Reset stream position after reading
        return sha256_hash.hexdigest()

    @staticmethod
    def _generate_image_caption(page_num: int) -> str:
        """
        Generates a placeholder caption for an image.
        """
        return f"Image detected on page {page_num}"

    async def process_pdf(self, file_obj: IO) -> Tuple[str, List[ProcessedContent]]:
        """
        Orchestrates the PDF processing workflow from a file-like object.
        """
        try:
            file_hash = self._calculate_hash_from_stream(file_obj)

            try:
                # unstructured may raise various exceptions on malformed PDFs
                elements = partition_pdf(
                    file=file_obj,
                    strategy="hi_res",
                )
            except Exception as e:
                logging.exception("PDF parsing failed for file hash %s: %s", file_hash, e)
                raise ValueError(f"Failed to parse PDF: {e}") from e

            if not elements:
                logging.warning("No elements extracted from PDF (hash=%s).", file_hash)
                # treat as parsing failure for API consumers
                raise ValueError("No content could be extracted from the PDF.")

            from collections import defaultdict
            page_elements = defaultdict(list)
            for element in elements:
                page_number = element.metadata.page_number
                page_elements[page_number].append(element)

            processed_contents: List[ProcessedContent] = []
            for page_num, elems in page_elements.items():
                prev_type = None
                prev_id = None
                section = None
                image_ids = []

                for idx, element in enumerate(elems):
                    if "unstructured.documents.elements.Image" in str(type(element)):
                        img_id = str(uuid.uuid4())
                        image_ids.append(img_id)
                        processed_contents.append(ProcessedContent(
                            id=img_id,
                            content_type="image",
                            text_content=self._generate_image_caption(page_num),
                            metadata={
                                "page": page_num,
                                "file_hash": file_hash,
                                "section": None,
                                "related_images": []
                            }
                        ))

                # Now process text and tables, linking related images
                for idx, element in enumerate(elems):
                    if "unstructured.documents.elements.Image" in str(type(element)):
                        continue

                    # Heuristic for section: if text is all caps or bold, treat as section
                    if hasattr(element, "text") and element.text:
                        txt = element.text.strip()
                        if len(txt) > 0 and (txt.isupper() or txt.startswith("Section")):
                            section = txt

                    content_type = "unknown"
                    text_content = ""
                    if "unstructured.documents.elements.Table" in str(type(element)):
                        content_type = "table"
                        text_content = element.metadata.text_as_html
                        text_content = text_content.replace("<table>", "") \
                                                  .replace("</table>", "") \
                                                  .replace("<tr>", "\n| ") \
                                                  .replace("</tr>", " |") \
                                                  .replace("<td>", " ") \
                                                  .replace("</td>", " |")
                    else:
                        content_type = "text"
                        text_content = getattr(element, "text", "")

                    if text_content:
                        # Related images: images that appear before/after this chunk on the same page
                        related_images = image_ids.copy()
                        processed_contents.append(ProcessedContent(
                            content_type=content_type,
                            text_content=text_content,
                            metadata={
                                "page": page_num,
                                "file_hash": file_hash,
                                "section": section,
                                "related_images": related_images
                            }
                        ))

            return file_hash, processed_contents
        finally:
            try:
                file_obj.close()
            except Exception:
                logging.exception("Failed to close file object for PDF processor.")
