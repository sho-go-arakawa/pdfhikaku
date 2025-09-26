"""PDF text extraction with multiple engine support and OCR fallback."""

import os
import re
import fitz
import pdfminer.high_level
import pytesseract
from PIL import Image
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import tempfile
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from .config import config

logger = logging.getLogger(__name__)

class PDFExtractor:
    """PDF text extraction with PyMuPDF, pdfminer, and OCR fallback."""

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        self.cfg = cfg or config.get_extract_config()
        self.engine_priority = self.cfg['engine_priority']
        self.ocr_lang = self.cfg['ocr_lang']
        self.max_pages = self.cfg['max_pages']
        self.remove_patterns = [re.compile(pattern) for pattern in self.cfg['remove_patterns']]

        # Set tesseract command if specified
        if 'tesseract_cmd' in self.cfg and self.cfg['tesseract_cmd']:
            pytesseract.pytesseract.tesseract_cmd = self.cfg['tesseract_cmd']

    def extract_text(self, pdf_path: str, progress_callback=None) -> List[Dict[str, Any]]:
        """Extract text from PDF with fallback mechanism and progress tracking."""
        logger.info(f"Starting text extraction from {pdf_path}")

        for engine in self.engine_priority:
            try:
                if engine == 'pymupdf':
                    return self._extract_with_fitz(pdf_path, progress_callback)
                elif engine == 'pdfminer':
                    return self._extract_with_pdfminer(pdf_path, progress_callback)
                elif engine == 'ocr':
                    return self._extract_with_ocr(pdf_path, progress_callback)
            except Exception as e:
                logger.warning(f"Extraction failed with {engine} for {pdf_path}: {e}")
                continue

        logger.error(f"All extraction methods failed for {pdf_path}")
        return []

    def _extract_with_fitz(self, pdf_path: str, progress_callback=None) -> List[Dict[str, Any]]:
        """Extract using PyMuPDF (fitz) with enhanced metadata."""
        doc = fitz.open(pdf_path)
        pages = []
        total_pages = min(len(doc), self.max_pages) if self.max_pages else len(doc)

        logger.info(f"Extracting {total_pages} pages using PyMuPDF")

        for page_num in range(total_pages):
            if progress_callback:
                progress_callback(page_num + 1, total_pages, f"Extracting page {page_num + 1}")

            page = doc.load_page(page_num)

            # Extract text
            text = page.get_text()

            # Extract text blocks with formatting info
            blocks = page.get_text("dict")["blocks"]

            # Extract spans with detailed formatting
            spans = self._extract_spans_with_formatting(blocks)

            # Extract headings based on font analysis
            headings = self._extract_headings_from_blocks(blocks)

            # Extract main body text (excluding headers/footers/sidebars)
            body_text = self._extract_body_text(blocks, page.rect)

            # Clean text using patterns
            text = self._clean_text(text)
            body_text = self._clean_text(body_text)

            # OCR for image-heavy pages if text is sparse
            if len(text.strip()) < 50 and 'ocr' in self.engine_priority:
                try:
                    pix = page.get_pixmap()
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    ocr_text = pytesseract.image_to_string(img, lang=self.ocr_lang)
                    if len(ocr_text.strip()) > len(text.strip()):
                        text = self._clean_text(ocr_text)
                        body_text = self._clean_text(ocr_text)  # For OCR, use same text
                        logger.info(f"Used OCR for page {page_num + 1}")
                except Exception as e:
                    logger.warning(f"OCR failed for page {page_num + 1}: {e}")

            pages.append({
                "page": page_num + 1,
                "text": text,
                "body_text": body_text,
                "blocks": blocks,
                "spans": spans,
                "headings": headings,
                "char_count": len(text),
                "body_char_count": len(body_text),
                "extraction_method": "pymupdf"
            })

        doc.close()
        logger.info(f"Successfully extracted text from {total_pages} pages using PyMuPDF")
        return pages

    def _extract_with_pdfminer(self, pdf_path: str, progress_callback=None) -> List[Dict[str, Any]]:
        """Extract using pdfminer as fallback."""
        logger.info("Extracting text using pdfminer")

        text = pdfminer.high_level.extract_text(pdf_path)
        text = self._clean_text(text)

        # Split by page breaks (approximate)
        page_texts = text.split('\f')  # Form feed character

        pages = []
        total_pages = min(len(page_texts), self.max_pages) if self.max_pages else len(page_texts)

        for i, page_text in enumerate(page_texts[:total_pages]):
            if progress_callback:
                progress_callback(i + 1, total_pages, f"Processing page {i + 1}")

            if page_text.strip():
                headings = self._extract_headings_from_text(page_text)
                # For pdfminer, approximate body text by filtering obvious headers/footers
                body_text = self._filter_body_text_from_plain_text(page_text)

                pages.append({
                    "page": i + 1,
                    "text": page_text,
                    "body_text": body_text,
                    "blocks": [],
                    "spans": [],
                    "headings": headings,
                    "char_count": len(page_text),
                    "body_char_count": len(body_text),
                    "extraction_method": "pdfminer"
                })

        logger.info(f"Successfully extracted text from {len(pages)} pages using pdfminer")
        return pages

    def _extract_with_ocr(self, pdf_path: str, progress_callback=None) -> List[Dict[str, Any]]:
        """Extract using OCR as last resort."""
        logger.info("Extracting text using OCR")

        doc = fitz.open(pdf_path)
        pages = []
        total_pages = min(len(doc), self.max_pages) if self.max_pages else len(doc)

        for page_num in range(total_pages):
            if progress_callback:
                progress_callback(page_num + 1, total_pages, f"OCR processing page {page_num + 1}")

            page = doc.load_page(page_num)

            try:
                # Convert page to image
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # Higher resolution
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

                # OCR the image
                text = pytesseract.image_to_string(img, lang=self.ocr_lang)
                text = self._clean_text(text)

                headings = self._extract_headings_from_text(text)
                # For OCR, approximate body text by filtering obvious headers/footers
                body_text = self._filter_body_text_from_plain_text(text)

                pages.append({
                    "page": page_num + 1,
                    "text": text,
                    "body_text": body_text,
                    "blocks": [],
                    "spans": [],
                    "headings": headings,
                    "char_count": len(text),
                    "body_char_count": len(body_text),
                    "extraction_method": "ocr"
                })

            except Exception as e:
                logger.warning(f"OCR failed for page {page_num + 1}: {e}")
                pages.append({
                    "page": page_num + 1,
                    "text": "",
                    "body_text": "",
                    "blocks": [],
                    "spans": [],
                    "headings": [],
                    "char_count": 0,
                    "body_char_count": 0,
                    "extraction_method": "ocr_failed"
                })

        doc.close()
        logger.info(f"Successfully processed {len(pages)} pages using OCR")
        return pages

    def _extract_spans_with_formatting(self, blocks: List[Dict]) -> List[Dict[str, Any]]:
        """Extract text spans with detailed formatting information."""
        spans = []

        for block in blocks:
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        spans.append({
                            "text": span["text"],
                            "bbox": span["bbox"],
                            "font": span["font"],
                            "size": span["size"],
                            "flags": span["flags"],
                            "color": span.get("color", 0)
                        })

        return spans

    def _extract_headings_from_blocks(self, blocks: List[Dict]) -> List[Dict[str, Any]]:
        """Extract headings from text blocks with improved font analysis."""
        headings = []
        font_sizes = []
        font_weights = []

        # Collect all font sizes and weights
        for block in blocks:
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        font_sizes.append(span["size"])
                        font_weights.append(span["flags"])

        if not font_sizes:
            return headings

        # Statistical analysis of fonts
        font_sizes = np.array(font_sizes)
        mean_size = np.mean(font_sizes)
        std_size = np.std(font_sizes)

        # Dynamic threshold based on font size distribution
        heading_threshold = mean_size + (0.5 * std_size)

        for block in blocks:
            if "lines" in block:
                for line in block["lines"]:
                    line_text = ""
                    max_font_size = 0
                    is_bold = False

                    for span in line["spans"]:
                        line_text += span["text"]
                        max_font_size = max(max_font_size, span["size"])
                        # Check if text is bold (flag & 16)
                        if span["flags"] & 16:
                            is_bold = True

                    line_text = line_text.strip()

                    # Enhanced heading detection criteria
                    if (line_text and
                        len(line_text) < 150 and  # Not too long
                        len(line_text) > 3 and    # Not too short
                        (max_font_size >= heading_threshold or is_bold) and
                        not line_text.endswith('.') and  # Unlikely to be a sentence
                        self._is_heading_pattern(line_text)):

                        level = self._determine_heading_level(max_font_size, font_sizes, is_bold)
                        headings.append({
                            "text": line_text,
                            "level": level,
                            "font_size": max_font_size,
                            "is_bold": is_bold,
                            "bbox": block.get("bbox", [0, 0, 0, 0])
                        })

        return sorted(headings, key=lambda x: x.get("bbox", [0])[1])  # Sort by y-position

    def _is_heading_pattern(self, text: str) -> bool:
        """Check if text matches common heading patterns."""
        # Japanese and English heading patterns
        patterns = [
            r'^\d+[\.\s]',  # 1. or 1
            r'^[A-Z][A-Z\s]+$',  # ALL CAPS
            r'^第\d+章',  # Japanese chapters
            r'^[第\d\s章節項目]+',  # Japanese sections
            r'^\d+\.\d+',  # 1.1, 1.2
            r'^[IVXLCDM]+\.',  # Roman numerals
            r'^[A-Z]\.',  # A., B., C.
            r'まとめ$|概要$|はじめに$|終わりに$'  # Japanese conclusion words
        ]

        for pattern in patterns:
            if re.match(pattern, text):
                return True

        # Check for title case in English
        if re.match(r'^[A-Z][a-z]+(\s[A-Z][a-z]+)*$', text) and len(text.split()) > 1:
            return True

        return False

    def _extract_headings_from_text(self, text: str) -> List[Dict[str, Any]]:
        """Extract headings from plain text using enhanced pattern matching."""
        headings = []
        lines = text.split('\n')

        # Enhanced patterns for headings
        patterns = [
            (r'^(\d+\.?\d*\.?\d*)\s+(.+)$', 1),  # 1. Title, 1.1 Title
            (r'^([A-Z]+\.)\s+(.+)$', 2),         # A. Title
            (r'^([IVXLCDM]+\.)\s+(.+)$', 2),     # I. Title (Roman numerals)
            (r'^(第\d+章)\s*(.*)$', 1),          # 第1章
            (r'^(第\d+節)\s*(.*)$', 2),          # 第1節
            (r'^(\d+\.\d+)\s+(.+)$', 2),         # 1.1 Title
            (r'^([A-Z][A-Z\s]+)$', 2),           # ALL CAPS titles
        ]

        for line_num, line in enumerate(lines):
            line = line.strip()
            if not line or len(line) > 150:
                continue

            for pattern, level in patterns:
                match = re.match(pattern, line)
                if match:
                    headings.append({
                        "text": line,
                        "level": level,
                        "font_size": None,
                        "is_bold": None,
                        "line_number": line_num + 1
                    })
                    break

        return headings

    def _determine_heading_level(self, font_size: float, all_font_sizes: np.ndarray, is_bold: bool) -> int:
        """Determine heading level based on font size and formatting."""
        unique_sizes = sorted(set(all_font_sizes), reverse=True)

        # Boost level if bold
        size_rank = next((i for i, size in enumerate(unique_sizes) if font_size >= size), len(unique_sizes) - 1)

        if is_bold and size_rank > 0:
            size_rank -= 1

        return min(max(size_rank + 1, 1), 6)  # Levels 1-6

    def _clean_text(self, text: str) -> str:
        """Clean extracted text using configured patterns."""
        if not text:
            return text

        lines = text.split('\n')
        cleaned_lines = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check if line matches any remove pattern
            should_remove = False
            for pattern in self.remove_patterns:
                if pattern.match(line):
                    should_remove = True
                    break

            if not should_remove:
                # Basic text normalization
                line = re.sub(r'\s+', ' ', line)  # Normalize whitespace
                line = re.sub(r'[^\w\s\u3040-\u309f\u30a0-\u30ff\u4e00-\u9faf\u3400-\u4dbf.,!?;:()[\]{}"\'-]', '', line)  # Keep basic punctuation and Japanese
                cleaned_lines.append(line)

        return '\n'.join(cleaned_lines)

    def _extract_body_text(self, blocks: List[Dict], page_rect) -> str:
        """Extract main body text excluding headers, footers, and sidebars."""
        if not blocks:
            return ""

        body_blocks = []
        page_height = page_rect.height
        page_width = page_rect.width

        # Define regions (as percentages of page dimensions)
        header_threshold = page_height * 0.12  # Top 12%
        footer_threshold = page_height * 0.88  # Bottom 12%
        left_sidebar_threshold = page_width * 0.15  # Left 15%
        right_sidebar_threshold = page_width * 0.85  # Right 15%

        # Analyze all text blocks to determine layout
        text_blocks = []
        for block in blocks:
            if block.get("type") == 0 and "lines" in block:  # Text block
                bbox = block.get("bbox", [0, 0, 0, 0])
                if len(bbox) >= 4:
                    block_text = ""
                    for line in block["lines"]:
                        for span in line["spans"]:
                            block_text += span["text"]

                    if block_text.strip():
                        text_blocks.append({
                            "bbox": bbox,
                            "text": block_text.strip(),
                            "block": block
                        })

        if not text_blocks:
            return ""

        # Filter out headers and footers based on position
        main_content_blocks = []
        for text_block in text_blocks:
            bbox = text_block["bbox"]
            top, bottom = bbox[1], bbox[3]
            left, right = bbox[0], bbox[2]

            # Skip headers (top region)
            if top < header_threshold:
                # Allow if it's a substantial block that might be a title
                if len(text_block["text"]) > 50 and not self._is_likely_header_footer(text_block["text"]):
                    main_content_blocks.append(text_block)
                continue

            # Skip footers (bottom region)
            if bottom > footer_threshold:
                # Allow if it's substantial content
                if len(text_block["text"]) > 50 and not self._is_likely_header_footer(text_block["text"]):
                    main_content_blocks.append(text_block)
                continue

            # Skip sidebars (left and right margins)
            if (right < left_sidebar_threshold or left > right_sidebar_threshold):
                # Allow if it's substantial content (might be wide margins with real content)
                if len(text_block["text"]) > 100:
                    main_content_blocks.append(text_block)
                continue

            # This block is in the main content area
            main_content_blocks.append(text_block)

        # Additional filtering based on content patterns
        filtered_blocks = []
        for text_block in main_content_blocks:
            text = text_block["text"]

            # Skip likely headers/footers based on content
            if self._is_likely_header_footer(text):
                continue

            # Skip very short texts unless they're likely important
            if len(text.strip()) < 10 and not self._is_likely_important_short_text(text):
                continue

            filtered_blocks.append(text_block)

        # Sort blocks by reading order (top to bottom, left to right)
        filtered_blocks.sort(key=lambda x: (x["bbox"][1], x["bbox"][0]))

        # Extract text from filtered blocks
        body_text_parts = []
        for text_block in filtered_blocks:
            body_text_parts.append(text_block["text"])

        return '\n'.join(body_text_parts)

    def _is_likely_header_footer(self, text: str) -> bool:
        """Check if text is likely a header or footer."""
        text_lower = text.lower().strip()

        # Common header/footer patterns
        header_footer_patterns = [
            r'^\d+$',  # Just page numbers
            r'^page\s+\d+',  # "Page X"
            r'^第?\s*\d+\s*[ページ頁]',  # Page numbers in Japanese
            r'^\d+\s*/\s*\d+$',  # "1/10" format
            r'^-\s*\d+\s*-$',  # "-1-" format
            r'^copyright|©',  # Copyright notices
            r'^confidential|機密',  # Confidential markings
            r'^\d{4}[/-]\d{1,2}[/-]\d{1,2}',  # Dates
            r'^[月火水木金土日]\s*曜日',  # Day of week in Japanese
            r'^draft|下書き|草案',  # Draft markings
            r'^header|footer|ヘッダー|フッター',  # Explicit header/footer
            r'^chapter\s+\d+|第\d+章$',  # Chapter headers (if very short)
        ]

        for pattern in header_footer_patterns:
            if re.match(pattern, text_lower):
                return True

        # Very short repetitive text
        if len(text.strip()) < 3:
            return True

        # All caps short text (often titles in headers)
        if len(text) < 20 and text.isupper() and not any(c.isdigit() for c in text):
            return True

        return False

    def _is_likely_important_short_text(self, text: str) -> bool:
        """Check if short text is likely important content."""
        text_stripped = text.strip()

        # Patterns that indicate important short text
        important_patterns = [
            r'^第\d+[章節項条]',  # Chapter/section markers
            r'^\d+[\.)]',  # Numbered items
            r'^[A-Z]\.',  # Letter items
            r'^\([a-z]\)',  # Lettered subitems
            r'^[◆◇■□●○▲△]',  # Bullet points
            r'^注[意記]?[:：]',  # Notes
            r'^重要[:：]',  # Important notices
            r'^警告[:：]',  # Warnings
            r'^[※＊]',  # Asterisk notes
        ]

        for pattern in important_patterns:
            if re.match(pattern, text_stripped):
                return True

        return False

    def _filter_body_text_from_plain_text(self, text: str) -> str:
        """Filter body text from plain text (for pdfminer/OCR without block information)."""
        if not text.strip():
            return ""

        lines = text.split('\n')
        body_lines = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Skip likely headers/footers
            if self._is_likely_header_footer(line):
                continue

            # Skip very short lines unless they're important
            if len(line) < 10 and not self._is_likely_important_short_text(line):
                continue

            body_lines.append(line)

        return '\n'.join(body_lines)