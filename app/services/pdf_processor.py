import hashlib
import io
import logging
import uuid
from typing import IO, List, Tuple, Optional, Dict, Any
from collections import defaultdict

import asyncio
import numpy as np
from PIL import Image

from unstructured.partition.pdf import partition_pdf

from models.schemas import ProcessedContent


class PDFProcessor:
    """
    PDF processing service that:
      - computes a SHA256 hash of the incoming stream,
      - extracts elements via `unstructured.partition.pdf` (hi_res strategy),
      - extracts/normalizes text and tables,
      - detects images and runs OCR on them using PaddleOCR (if available),
      - returns a list of ProcessedContent objects.

    Requirements:
      - `paddleocr` (and `paddlepaddle`) for OCR features. The class degrades gracefully if absent.
    """

    def __init__(self, ocr_lang: str = "en", ocr_use_angle_cls: bool = True, min_image_area: int = 1024):
        """
        Args:
            ocr_lang: PaddleOCR language (e.g., "en", "ch", "en|ch" depending on version).
            ocr_use_angle_cls: whether to use angle classification.
            min_image_area: skip OCR for images with area smaller than this threshold (pixels).
        """
        self._ocr = None
        self._ocr_lang = ocr_lang
        self._ocr_use_angle_cls = ocr_use_angle_cls
        self._min_image_area = min_image_area
        self._ocr_init_error = None

    @staticmethod
    def _calculate_hash_from_stream(file_obj: IO) -> str:
        """Calculates SHA256 hash of file-like stream and resets position to start."""
        sha256_hash = hashlib.sha256()
        for byte_block in iter(lambda: file_obj.read(4096), b""):
            sha256_hash.update(byte_block)
        file_obj.seek(0)
        return sha256_hash.hexdigest()

    @staticmethod
    def _generate_image_caption(page_num: int) -> str:
        return f"Image detected on page {page_num}"

    def _init_ocr(self) -> None:
        """Lazy initialize PaddleOCR; safe to call multiple times."""
        if self._ocr is not None or self._ocr_init_error is not None:
            return

        try:
            # import inside method so module import is optional at process start
            from paddleocr import PaddleOCR  # type: ignore
            # create PaddleOCR instance (this can be heavy)
            self._ocr = PaddleOCR(use_angle_cls=self._ocr_use_angle_cls, lang=self._ocr_lang)
            logging.info("PaddleOCR initialized (lang=%s).", self._ocr_lang)
        except Exception as e:
            # Keep the exception so we do not repeatedly try to import on every PDF
            self._ocr_init_error = e
            self._ocr = None
            logging.warning("PaddleOCR is not available or failed to init: %s. OCR will be skipped.", e)

    @staticmethod
    def _element_to_pil_image(element) -> Optional[Image.Image]:
        """
        Attempt multiple common ways to obtain a PIL.Image from an unstructured Image element.
        Returns a PIL.Image or None if extraction fails.
        """
        # 1) If element already has a PIL Image-like attribute
        for attr in ("image", "element", "img", "_element"):
            img_obj = getattr(element, attr, None)
            if img_obj is None:
                continue
            # If it's already a PIL Image
            if isinstance(img_obj, Image.Image):
                return img_obj
            # If it's raw bytes
            if isinstance(img_obj, (bytes, bytearray)):
                try:
                    return Image.open(io.BytesIO(img_obj)).convert("RGB")
                except Exception:
                    continue
            # If it's a numpy array
            try:
                import numpy as _np
                if isinstance(img_obj, _np.ndarray):
                    return Image.fromarray(img_obj).convert("RGB")
            except Exception:
                pass

        # 2) Sometimes the element contains image bytes in metadata or a src field
        metadata = getattr(element, "metadata", None)
        if isinstance(metadata, dict):
            for key in ("image", "image_bytes", "raw_bytes", "src", "data"):
                val = metadata.get(key)
                if val:
                    try:
                        if isinstance(val, (bytes, bytearray)):
                            return Image.open(io.BytesIO(val)).convert("RGB")
                        # If it's base64 string (data URI), try decode
                        if isinstance(val, str) and val.startswith("data:"):
                            import base64
                            header, encoded = val.split(",", 1)
                            return Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")
                    except Exception:
                        continue

        # 3) As a last resort, try to access a raw attribute that might be bytes
        for attr in dir(element):
            try:
                val = getattr(element, attr)
            except Exception:
                continue
            if isinstance(val, (bytes, bytearray)):
                try:
                    return Image.open(io.BytesIO(val)).convert("RGB")
                except Exception:
                    continue

        return None

    def _ocr_image_sync(self, pil_image: Image.Image) -> Dict[str, Any]:
        """
        Blocking OCR call using PaddleOCR. Returns a dictionary containing:
            - text: recognized text (joined lines)
            - avg_score: average confidence (0..1) or None if unavailable
            - details: raw OCR result (for debugging or more structured processing)
        This method is intended to run in a thread via run_in_executor.
        """
        # Ensure OCR available
        self._init_ocr()
        if self._ocr is None:
            return {"text": "", "avg_score": None, "details": None, "error": self._ocr_init_error}

        try:
            # convert PIL to numpy array (BGR / RGB handling is internal to PaddleOCR)
            np_img = np.array(pil_image)
            result = self._ocr.ocr(np_img, cls=self._ocr_use_angle_cls)  # blocking call
            # result is list[ list[ [box], (text, score) ] ]
            lines = []
            scores = []
            for line in result:
                # line may be a nested list or tuple
                if not line:
                    continue
                # If expected format [ [box], (text, score) ]
                # some versions return different nesting; handle nested list
                for item in line:
                    try:
                        if isinstance(item, (list, tuple)) and len(item) >= 2:
                            maybe_text_score = item[-1]
                            if isinstance(maybe_text_score, (list, tuple)) and len(maybe_text_score) >= 2:
                                txt, score = maybe_text_score[0], maybe_text_score[1]
                                lines.append(str(txt))
                                try:
                                    scores.append(float(score))
                                except Exception:
                                    pass
                            else:
                                # fallback: item could be (bbox, text) depending on version
                                pass
                    except Exception:
                        continue

            avg_score = float(np.mean(scores)) if scores else None
            joined = "\n".join([ln for ln in lines if ln])
            return {"text": joined, "avg_score": avg_score, "details": result}
        except Exception as e:
            logging.exception("PaddleOCR failed during OCR: %s", e)
            return {"text": "", "avg_score": None, "details": None, "error": e}

    async def process_pdf(self, file_obj: IO) -> Tuple[str, List[ProcessedContent]]:
        """
        Process PDF:
          - compute hash,
          - extract elements via unstructured (hi_res),
          - for each page produce ProcessedContent for images (with OCR text) and text/tables.

        Returns:
            (file_hash, list_of_processed_contents)
        """
        loop = asyncio.get_running_loop()
        try:
            file_hash = self._calculate_hash_from_stream(file_obj)

            try:
                elements = partition_pdf(file=file_obj, strategy="hi_res")
            except Exception as e:
                logging.exception("PDF parsing failed for file hash %s: %s", file_hash, e)
                raise ValueError(f"Failed to parse PDF: {e}") from e

            if not elements:
                logging.warning("No elements extracted from PDF (hash=%s).", file_hash)
                raise ValueError("No content could be extracted from the PDF.")

            # group elements by page
            page_elements = defaultdict(list)
            for element in elements:
                page_number = getattr(element, "metadata", None)
                # element.metadata may be object or dict with page_number; handle multiple shapes
                if isinstance(page_number, dict) and "page_number" in page_number:
                    pn = page_number["page_number"]
                else:
                    pn = getattr(getattr(element, "metadata", None), "page_number", None)
                # fallback: some elements have element.metadata.page_number
                if pn is None:
                    pn = getattr(element, "page_number", None)
                if pn is None:
                    pn = 0
                page_elements[int(pn)].append(element)

            processed_contents: List[ProcessedContent] = []

            # collect OCR tasks to run concurrently
            ocr_tasks = []
            ocr_task_meta = []  # metadata to reconstruct ProcessedContent after OCR completes

            for page_num, elems in page_elements.items():
                # collect image ids for page (used to relate images to text)
                image_ids = []

                # First pass: detect images and schedule OCR tasks
                for element in elems:
                    if "unstructured.documents.elements.Image" in str(type(element)):
                        # try to extract PIL image
                        pil_img = self._element_to_pil_image(element)
                        img_id = str(uuid.uuid4())
                        image_ids.append(img_id)

                        # save a placeholder ProcessedContent for the image; we'll fill OCR result once done
                        placeholder = ProcessedContent(
                            id=img_id,
                            content_type="image",
                            text_content=self._generate_image_caption(page_num),
                            metadata={
                                "page": page_num,
                                "file_hash": file_hash,
                                "section": None,
                                "related_images": [],
                                "ocr_text": None,
                                "ocr_confidence": None,
                            }
                        )
                        processed_contents.append(placeholder)

                        if pil_img is None:
                            logging.debug("Image found on page %s but could not extract PIL image.", page_num)
                            continue

                        # skip tiny images (likely icons)
                        w, h = pil_img.size
                        if (w * h) < self._min_image_area:
                            logging.debug("Skipping OCR for tiny image (area=%s) on page %s.", w*h, page_num)
                            continue

                        # schedule OCR in executor (blocking)
                        task = loop.run_in_executor(None, self._ocr_image_sync, pil_img)
                        ocr_tasks.append(task)
                        ocr_task_meta.append({"img_id": img_id, "page": page_num, "placeholder_idx": len(processed_contents)-1})

                # Second pass: process text and tables on the same page
                section = None
                for element in elems:
                    if "unstructured.documents.elements.Image" in str(type(element)):
                        continue

                    # simple section heuristic
                    if hasattr(element, "text") and element.text:
                        txt = element.text.strip()
                        if len(txt) > 0 and (txt.isupper() or txt.startswith("Section")):
                            section = txt

                    content_type = "unknown"
                    text_content = ""
                    # handle table
                    if "unstructured.documents.elements.Table" in str(type(element)):
                        content_type = "table"

                        html = getattr(element.metadata, "text_as_html", None)

                        if isinstance(html, str):
                            text_content = (
                                html.replace("<table>", "")
                                    .replace("</table>", "")
                                    .replace("<tr>", "\n| ")
                                    .replace("</tr>", " |")
                                    .replace("<td>", " ")
                                    .replace("</td>", " |")
                            )
                        else:
                            # fallback: extract plain text
                            text_content = getattr(element, "text", "") or ""

                    if text_content:
                        processed_contents.append(ProcessedContent(
                            content_type=content_type,
                            text_content=text_content,
                            metadata={
                                "page": page_num,
                                "file_hash": file_hash,
                                "section": section,
                                "related_images": image_ids.copy()
                            }
                        ))

            # Wait for all OCR tasks and populate placeholders with OCR results
            if ocr_tasks:
                try:
                    ocr_results = await asyncio.gather(*ocr_tasks, return_exceptions=True)
                except Exception as e:
                    logging.exception("OCR tasks failed with exception: %s", e)
                    ocr_results = [e] * len(ocr_tasks)

                # attach OCR results back to the processed_contents placeholders
                for meta, result in zip(ocr_task_meta, ocr_results):
                    img_id = meta["img_id"]
                    idx = meta["placeholder_idx"]
                    # defensive: make sure index is still valid
                    if idx < 0 or idx >= len(processed_contents):
                        logging.warning("Invalid placeholder index for OCR result (img_id=%s).", img_id)
                        continue
                    if isinstance(result, Exception):
                        logging.warning("OCR for image %s failed: %s", img_id, result)
                        processed_contents[idx].metadata["ocr_text"] = ""
                        processed_contents[idx].metadata["ocr_confidence"] = None
                    else:
                        processed_contents[idx].metadata["ocr_text"] = result.get("text") or ""
                        processed_contents[idx].metadata["ocr_confidence"] = result.get("avg_score")

            return file_hash, processed_contents

        finally:
            try:
                file_obj.close()
            except Exception:
                logging.exception("Failed to close file object for PDF processor.")
