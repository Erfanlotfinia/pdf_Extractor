import hashlib
import tempfile
from typing import List, Tuple
import uuid
import httpx
from unstructured.partition.pdf import partition_pdf

from models.schemas import ProcessedContent

class PDFProcessor:
    """
    A service class to handle the processing of PDF documents.
    This includes downloading, hashing, and extracting content from the PDF.
    """

    def __init__(self, client: httpx.AsyncClient):
        self.client = client

    async def _download_pdf(self, url: str) -> str:
        """
        Downloads a PDF from a given URL to a temporary file.
        
        Args:
            url: The URL of the PDF to download.
        
        Returns:
            The file path of the downloaded temporary file.
            
        Raises:
            httpx.HTTPStatusError: If the download fails.
            ValueError: If the content is not a PDF.
        """
        async with self.client.stream("GET", url, follow_redirects=True) as response:
            response.raise_for_status()
            
            content_type = response.headers.get("Content-Type", "")
            if "application/pdf" not in content_type:
                raise ValueError("The provided URL does not point to a PDF file.")

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                async for chunk in response.aiter_bytes():
                    tmp_file.write(chunk)
                return tmp_file.name

    @staticmethod
    def _calculate_hash(file_path: str) -> str:
        """
        Calculates the SHA256 hash of a file.
        """
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    @staticmethod
    def _generate_image_caption(page_num: int) -> str:
        """
        Generates a placeholder caption for an image.
        
        NOTE: In a production environment, this function would call a multimodal model
        like OpenAI's GPT-4o (Vision) to generate a descriptive text caption for the
        image, which would then be embedded.
        """
        return f"Image detected on page {page_num}"

    async def process_pdf(self, url: str) -> Tuple[str, List[ProcessedContent]]:
        """
        Orchestrates the PDF processing workflow.
        
        Args:
            url: The URL of the PDF to process.
            
        Returns:
            A tuple containing the file hash and a list of ProcessedContent objects.
        """
        temp_pdf_path = await self._download_pdf(url)
        try:
            file_hash = self._calculate_hash(temp_pdf_path)
            elements = partition_pdf(
                filename=temp_pdf_path,
                strategy="hi_res",
            )

            # --- Enhanced relationship preservation ---
            # Group elements by page
            from collections import defaultdict
            page_elements = defaultdict(list)
            for element in elements:
                page_number = element.metadata.page_number
                page_elements[page_number].append(element)

            processed_contents: List[ProcessedContent] = []
            for page_num, elems in page_elements.items():
                # Track previous element for context
                prev_type = None
                prev_id = None
                section = None
                image_ids = []

                # First, collect image ids for this page
                for idx, element in enumerate(elems):
                    if "unstructured.documents.elements.Image" in str(type(element)):
                        img_id = str(uuid.uuid4())
                        image_ids.append(img_id)
                        # Add image chunk
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
                        continue  # Already processed above

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
            import os
            os.remove(temp_pdf_path)
