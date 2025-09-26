import sys
import os

# Add user site-packages to Python path
user_site = '/home/vscode/.local/lib/python3.11/site-packages'
if user_site not in sys.path:
    sys.path.insert(0, user_site)

import streamlit as st
import fitz
import pdfminer.high_level
import pytesseract
from PIL import Image
import pandas as pd
import numpy as np
import json
import difflib
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import tempfile
import os
from datetime import datetime
import rapidfuzz
from Levenshtein import distance as levenshtein_distance
import matplotlib.pyplot as plt
import openpyxl
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import concurrent.futures
import io

# Configure logging
logging.basicConfig(
    filename='logs/app.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class PDFExtractor:
    """PDF text extraction with PyMuPDF and pdfminer fallback"""

    def __init__(self, use_ocr: bool = False):
        self.use_ocr = use_ocr

    def extract_text(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract text from PDF with fallback mechanism"""
        try:
            return self._extract_with_fitz(pdf_path)
        except Exception as e:
            logging.warning(f"PyMuPDF extraction failed for {pdf_path}: {e}")
            try:
                return self._extract_with_pdfminer(pdf_path)
            except Exception as e:
                logging.error(f"Both extraction methods failed for {pdf_path}: {e}")
                return []

    def _extract_with_fitz(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract using PyMuPDF (fitz)"""
        doc = fitz.open(pdf_path)
        pages = []

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)

            # Extract text
            text = page.get_text()

            # Extract text blocks with formatting info
            blocks = page.get_text("dict")["blocks"]

            # Extract headings based on font size
            headings = self._extract_headings_from_blocks(blocks)

            # OCR for image pages if enabled
            if not text.strip() and self.use_ocr:
                try:
                    pix = page.get_pixmap()
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    text = pytesseract.image_to_string(img, lang='jpn+eng')
                except Exception as e:
                    logging.warning(f"OCR failed for page {page_num + 1}: {e}")

            pages.append({
                "page": page_num + 1,
                "text": text,
                "blocks": blocks,
                "headings": headings
            })

        doc.close()
        return pages

    def _extract_with_pdfminer(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract using pdfminer as fallback"""
        text = pdfminer.high_level.extract_text(pdf_path)
        # Split by page breaks (approximate)
        page_texts = text.split('\f')  # Form feed character

        pages = []
        for i, page_text in enumerate(page_texts):
            if page_text.strip():
                pages.append({
                    "page": i + 1,
                    "text": page_text,
                    "blocks": [],
                    "headings": self._extract_headings_from_text(page_text)
                })

        return pages

    def _extract_headings_from_blocks(self, blocks: List[Dict]) -> List[Dict[str, Any]]:
        """Extract headings from text blocks based on font size"""
        headings = []
        font_sizes = []

        # Collect all font sizes
        for block in blocks:
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        font_sizes.append(span["size"])

        if not font_sizes:
            return headings

        # Determine heading threshold (larger fonts)
        avg_font_size = np.mean(font_sizes)
        heading_threshold = avg_font_size * 1.2

        for block in blocks:
            if "lines" in block:
                for line in block["lines"]:
                    line_text = ""
                    max_font_size = 0

                    for span in line["spans"]:
                        line_text += span["text"]
                        max_font_size = max(max_font_size, span["size"])

                    # Check if it's a heading
                    if (max_font_size >= heading_threshold and
                        line_text.strip() and
                        len(line_text.strip()) < 100):

                        level = self._determine_heading_level(max_font_size, font_sizes)
                        headings.append({
                            "text": line_text.strip(),
                            "level": level,
                            "font_size": max_font_size
                        })

        return headings

    def _extract_headings_from_text(self, text: str) -> List[Dict[str, Any]]:
        """Extract headings from plain text using patterns"""
        headings = []
        lines = text.split('\n')

        # Patterns for headings
        patterns = [
            r'^(\d+\.?\d*\.?\d*)\s+(.+)$',  # 1. Title, 1.1 Title
            r'^([A-Z]+\.)\s+(.+)$',         # A. Title
            r'^([IVXLCDM]+\.)\s+(.+)$',     # I. Title (Roman numerals)
            r'^([第\d+章節項].*?)\s+(.+)$', # Japanese chapter/section
        ]

        for line in lines:
            line = line.strip()
            if not line or len(line) > 100:
                continue

            for pattern in patterns:
                match = re.match(pattern, line)
                if match:
                    headings.append({
                        "text": line,
                        "level": 1,  # Default level
                        "font_size": None
                    })
                    break

        return headings

    def _determine_heading_level(self, font_size: float, all_font_sizes: List[float]) -> int:
        """Determine heading level based on font size"""
        sorted_sizes = sorted(set(all_font_sizes), reverse=True)

        for level, size in enumerate(sorted_sizes[:6], 1):
            if font_size >= size:
                return level

        return 6

class OutlineDetector:
    """Detect document outline and structure"""

    def extract_outline(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract existing outline from PDF"""
        try:
            doc = fitz.open(pdf_path)
            toc = doc.get_toc()
            doc.close()

            outline = []
            for level, title, page in toc:
                outline.append({
                    "id": f"{len(outline) + 1}",
                    "level": level,
                    "title": title.strip(),
                    "page": page,
                    "children": []
                })

            return self._build_tree(outline)
        except Exception as e:
            logging.warning(f"Failed to extract outline from {pdf_path}: {e}")
            return []

    def create_outline_from_headings(self, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create outline from detected headings"""
        outline = []
        current_id = 1

        for page in pages:
            for heading in page["headings"]:
                outline.append({
                    "id": str(current_id),
                    "level": heading["level"],
                    "title": heading["text"],
                    "page": page["page"],
                    "children": []
                })
                current_id += 1

        return self._build_tree(outline)

    def _build_tree(self, flat_outline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Build hierarchical tree from flat outline"""
        if not flat_outline:
            return []

        # Simple tree building - group by level
        tree = []
        stack = []

        for item in flat_outline:
            level = item["level"]

            # Pop items from stack that are at same or deeper level
            while stack and stack[-1]["level"] >= level:
                stack.pop()

            if stack:
                # Add as child to parent
                if "children" not in stack[-1]:
                    stack[-1]["children"] = []
                stack[-1]["children"].append(item)
            else:
                # Add as root item
                tree.append(item)

            stack.append(item)

        return tree

class SectionAligner:
    """Align sections between two PDFs using similarity matching"""

    def __init__(self, similarity_threshold: float = 0.6):
        self.similarity_threshold = similarity_threshold

    def align_sections(self, outline_a: List[Dict], outline_b: List[Dict]) -> List[Dict[str, Any]]:
        """Create section mapping between two outlines"""
        flat_a = self._flatten_outline(outline_a)
        flat_b = self._flatten_outline(outline_b)

        mappings = []
        used_b = set()

        for item_a in flat_a:
            best_match = None
            best_score = 0

            for item_b in flat_b:
                if item_b["id"] in used_b:
                    continue

                score = self._calculate_similarity(item_a["title"], item_b["title"])

                if score > best_score and score >= self.similarity_threshold:
                    best_score = score
                    best_match = item_b

            if best_match:
                mappings.append({
                    "A_id": item_a["id"],
                    "B_id": best_match["id"],
                    "A_title": item_a["title"],
                    "B_title": best_match["title"],
                    "score": best_score
                })
                used_b.add(best_match["id"])

        return mappings

    def _flatten_outline(self, outline: List[Dict]) -> List[Dict]:
        """Flatten hierarchical outline to list"""
        result = []

        def traverse(items):
            for item in items:
                result.append(item)
                if "children" in item and item["children"]:
                    traverse(item["children"])

        traverse(outline)
        return result

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity using multiple methods"""
        if not text1 or not text2:
            return 0.0

        # Normalize texts
        text1 = re.sub(r'[^\w\s]', '', text1.lower().strip())
        text2 = re.sub(r'[^\w\s]', '', text2.lower().strip())

        if text1 == text2:
            return 1.0

        # Use rapidfuzz for fuzzy matching
        ratio = rapidfuzz.fuzz.ratio(text1, text2) / 100.0

        # Use token-based similarity
        token_ratio = rapidfuzz.fuzz.token_sort_ratio(text1, text2) / 100.0

        # Return maximum similarity
        return max(ratio, token_ratio)

class TextDiffer:
    """Generate text differences with color highlighting"""

    def __init__(self, granularity: str = "sentence"):
        self.granularity = granularity

    def generate_diff(self, text_a: str, text_b: str) -> Tuple[str, str, Dict[str, int]]:
        """Generate HTML diff with statistics"""
        tokens_a = self._tokenize(text_a)
        tokens_b = self._tokenize(text_b)

        # Generate diff using difflib
        differ = difflib.SequenceMatcher(None, tokens_a, tokens_b)
        opcodes = differ.get_opcodes()

        html_a = []
        html_b = []
        stats = {"match_tokens": 0, "total_tokens": len(tokens_a) + len(tokens_b)}

        for op, a1, a2, b1, b2 in opcodes:
            if op == 'equal':
                # Matching text - white background
                content = ' '.join(tokens_a[a1:a2])
                html_a.append(f'<span class="match">{content}</span>')
                html_b.append(f'<span class="match">{content}</span>')
                stats["match_tokens"] += (a2 - a1) + (b2 - b1)

            elif op == 'delete':
                # Deleted text - red background
                content = ' '.join(tokens_a[a1:a2])
                html_a.append(f'<span class="delete">{content}</span>')

            elif op == 'insert':
                # Inserted text - green background
                content = ' '.join(tokens_b[b1:b2])
                html_b.append(f'<span class="insert">{content}</span>')

            elif op == 'replace':
                # Changed text - yellow background
                content_a = ' '.join(tokens_a[a1:a2])
                content_b = ' '.join(tokens_b[b1:b2])
                html_a.append(f'<span class="replace">{content_a}</span>')
                html_b.append(f'<span class="replace">{content_b}</span>')

        return ' '.join(html_a), ' '.join(html_b), stats

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text based on granularity"""
        if self.granularity == "character":
            return list(text)
        elif self.granularity == "word":
            return text.split()
        elif self.granularity == "sentence":
            sentences = re.split(r'[.!?]+', text)
            return [s.strip() for s in sentences if s.strip()]
        elif self.granularity == "paragraph":
            paragraphs = text.split('\n\n')
            return [p.strip() for p in paragraphs if p.strip()]
        else:
            return text.split()

class SimilarityCalculator:
    """Calculate similarity scores and statistics"""

    def calculate_overall_similarity(self, all_stats: List[Dict[str, int]]) -> float:
        """Calculate overall similarity percentage"""
        total_match = sum(stat["match_tokens"] for stat in all_stats)
        total_tokens = sum(stat["total_tokens"] for stat in all_stats)

        if total_tokens == 0:
            return 0.0

        return (total_match / total_tokens) * 100

    def calculate_section_similarities(self, section_stats: Dict[str, Dict[str, int]]) -> pd.DataFrame:
        """Calculate per-section similarity scores"""
        data = []

        for section_id, stats in section_stats.items():
            if stats["total_tokens"] > 0:
                similarity = (stats["match_tokens"] / stats["total_tokens"]) * 100
            else:
                similarity = 0.0

            data.append({
                "Section": section_id,
                "Match Tokens": stats["match_tokens"],
                "Total Tokens": stats["total_tokens"],
                "Similarity %": similarity
            })

        return pd.DataFrame(data)

class BookmarkGenerator:
    """Generate bookmarks for PDFs based on detected structure"""

    def add_bookmarks_to_pdf(self, input_pdf_path: str, outline: List[Dict], output_path: str):
        """Add bookmarks to PDF based on outline structure"""
        try:
            doc = fitz.open(input_pdf_path)

            # Clear existing bookmarks
            doc.set_toc([])

            # Convert outline to TOC format
            toc = self._outline_to_toc(outline)

            # Set new bookmarks
            if toc:
                doc.set_toc(toc)

            # Save with bookmarks
            doc.save(output_path)
            doc.close()

            logging.info(f"Added {len(toc)} bookmarks to {output_path}")
            return True

        except Exception as e:
            logging.error(f"Failed to add bookmarks to {input_pdf_path}: {e}")
            return False

    def _outline_to_toc(self, outline: List[Dict]) -> List[List]:
        """Convert outline structure to PyMuPDF TOC format"""
        toc = []

        def process_items(items, level=1):
            for item in items:
                # Add current item
                toc.append([
                    level,
                    item.get("title", ""),
                    item.get("page", 1)
                ])

                # Process children
                if "children" in item and item["children"]:
                    process_items(item["children"], level + 1)

        process_items(outline)
        return toc

class ReportExporter:
    """Export comparison results to various formats"""

    def export_pdf_report(self, summary: Dict, output_path: str):
        """Export summary report as PDF"""
        c = canvas.Canvas(output_path, pagesize=letter)
        width, height = letter

        # Title
        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, height - 50, "PDF Comparison Report")

        # Summary
        c.setFont("Helvetica", 12)
        y = height - 100

        c.drawString(50, y, f"Overall Similarity: {summary['overall_similarity']:.2f}%")
        y -= 30

        c.drawString(50, y, f"Total Sections Compared: {summary['total_sections']}")
        y -= 20

        c.drawString(50, y, f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        c.save()

    def export_excel_report(self, section_df: pd.DataFrame, output_path: str):
        """Export detailed report as Excel"""
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            section_df.to_excel(writer, sheet_name='Section Similarities', index=False)

    def export_json_report(self, data: Dict, output_path: str):
        """Export full data structure as JSON"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

def create_css():
    """Create CSS for diff highlighting"""
    return """
    <style>
    .match {
        background-color: white;
        color: black;
    }
    .delete {
        background-color: #ffcccc;
        color: #660000;
        text-decoration: line-through;
    }
    .insert {
        background-color: #ccffcc;
        color: #006600;
    }
    .replace {
        background-color: #ffffcc;
        color: #666600;
    }
    .diff-container {
        font-family: monospace;
        white-space: pre-wrap;
        line-height: 1.5;
        padding: 10px;
        border: 1px solid #ccc;
        border-radius: 5px;
        max-height: 400px;
        overflow-y: auto;
    }
    </style>
    """

def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="PDF比較ツール (PDFHikaku)",
        page_icon="📄",
        layout="wide"
    )

    st.title("📄 PDF差分比較アプリ (PDFHikaku)")
    st.markdown(create_css(), unsafe_allow_html=True)

    # Sidebar for settings
    with st.sidebar:
        st.header("⚙️ 設定")

        use_ocr = st.checkbox("OCR使用 (画像PDF対応)", value=False)
        granularity = st.selectbox(
            "差分粒度",
            ["character", "word", "sentence", "paragraph"],
            index=2
        )
        similarity_threshold = st.slider(
            "類似度閾値",
            min_value=0.0,
            max_value=1.0,
            value=0.6,
            step=0.1
        )
        max_workers = st.slider(
            "並列処理数",
            min_value=1,
            max_value=8,
            value=4
        )

    # File upload
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📄 PDF A")
        pdf_a = st.file_uploader("PDF Aをアップロード", type="pdf", key="pdf_a")

    with col2:
        st.subheader("📄 PDF B")
        pdf_b = st.file_uploader("PDF Bをアップロード", type="pdf", key="pdf_b")

    if pdf_a and pdf_b:
        if st.button("🔍 比較開始", type="primary"):

            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                # Save uploaded files temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_a:
                    tmp_a.write(pdf_a.read())
                    pdf_a_path = tmp_a.name

                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_b:
                    tmp_b.write(pdf_b.read())
                    pdf_b_path = tmp_b.name

                # Initialize components
                extractor = PDFExtractor(use_ocr=use_ocr)
                outline_detector = OutlineDetector()
                aligner = SectionAligner(similarity_threshold=similarity_threshold)
                differ = TextDiffer(granularity=granularity)
                calculator = SimilarityCalculator()
                exporter = ReportExporter()
                bookmark_generator = BookmarkGenerator()

                # Extract text
                status_text.text("PDF Aからテキスト抽出中...")
                progress_bar.progress(10)
                pages_a = extractor.extract_text(pdf_a_path)

                status_text.text("PDF Bからテキスト抽出中...")
                progress_bar.progress(30)
                pages_b = extractor.extract_text(pdf_b_path)

                # Detect outlines
                status_text.text("目次構造を解析中...")
                progress_bar.progress(50)

                outline_a = outline_detector.extract_outline(pdf_a_path)
                if not outline_a:
                    outline_a = outline_detector.create_outline_from_headings(pages_a)

                outline_b = outline_detector.extract_outline(pdf_b_path)
                if not outline_b:
                    outline_b = outline_detector.create_outline_from_headings(pages_b)

                # Align sections
                status_text.text("セクションマッピング中...")
                progress_bar.progress(70)
                mappings = aligner.align_sections(outline_a, outline_b)

                # Generate diffs
                status_text.text("差分生成中...")
                progress_bar.progress(90)

                diff_results = []
                section_stats = {}

                for mapping in mappings:
                    # Get text for sections (simplified)
                    text_a = f"Section {mapping['A_title']}"  # In real implementation, extract actual section text
                    text_b = f"Section {mapping['B_title']}"

                    html_a, html_b, stats = differ.generate_diff(text_a, text_b)

                    diff_results.append({
                        "mapping": mapping,
                        "html_a": html_a,
                        "html_b": html_b,
                        "stats": stats
                    })

                    section_stats[mapping["A_id"]] = stats

                # Calculate similarities
                overall_similarity = calculator.calculate_overall_similarity([r["stats"] for r in diff_results])
                section_df = calculator.calculate_section_similarities(section_stats)

                progress_bar.progress(100)
                status_text.text("完了!")

                # Display results
                st.success(f"🎉 比較完了! 全体一致率: {overall_similarity:.2f}%")

                # Summary metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("全体一致率", f"{overall_similarity:.2f}%")
                with col2:
                    st.metric("比較セクション数", len(mappings))
                with col3:
                    st.metric("処理ページ数", len(pages_a) + len(pages_b))

                # Section similarities chart
                st.subheader("📊 セクション別一致率")
                if not section_df.empty:
                    fig, ax = plt.subplots()
                    ax.bar(section_df["Section"], section_df["Similarity %"])
                    ax.set_xlabel("セクション")
                    ax.set_ylabel("一致率 (%)")
                    ax.set_title("セクション別一致率")
                    plt.xticks(rotation=45)
                    st.pyplot(fig)

                    st.dataframe(section_df)

                # Diff display
                st.subheader("🔍 差分表示")
                for i, result in enumerate(diff_results):
                    with st.expander(f"セクション: {result['mapping']['A_title']} ↔ {result['mapping']['B_title']}"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**PDF A**")
                            st.markdown(f'<div class="diff-container">{result["html_a"]}</div>', unsafe_allow_html=True)
                        with col2:
                            st.markdown("**PDF B**")
                            st.markdown(f'<div class="diff-container">{result["html_b"]}</div>', unsafe_allow_html=True)

                # Export options
                st.subheader("📤 エクスポート")
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    if st.button("📄 PDFレポート出力"):
                        summary = {
                            "overall_similarity": overall_similarity,
                            "total_sections": len(mappings)
                        }
                        exporter.export_pdf_report(summary, "reports/summary.pdf")
                        st.success("reports/summary.pdf に保存しました")

                with col2:
                    if st.button("📊 Excelレポート出力"):
                        exporter.export_excel_report(section_df, "reports/summary.xlsx")
                        st.success("reports/summary.xlsx に保存しました")

                with col3:
                    if st.button("🗂️ JSONデータ出力"):
                        data = {
                            "mappings": mappings,
                            "section_stats": section_stats,
                            "overall_similarity": overall_similarity
                        }
                        exporter.export_json_report(data, "reports/diff.json")
                        st.success("reports/diff.json に保存しました")

                with col4:
                    if st.button("🔖 しおり付きPDF出力"):
                        # Generate bookmarked PDFs
                        output_a = f"outputs/with_bookmarks_{pdf_a.name}"
                        output_b = f"outputs/with_bookmarks_{pdf_b.name}"

                        success_a = bookmark_generator.add_bookmarks_to_pdf(pdf_a_path, outline_a, output_a)
                        success_b = bookmark_generator.add_bookmarks_to_pdf(pdf_b_path, outline_b, output_b)

                        if success_a and success_b:
                            st.success(f"しおり付きPDFを出力しました:\n- {output_a}\n- {output_b}")
                        else:
                            st.warning("しおり生成に一部失敗しました")

                # Cleanup temp files
                os.unlink(pdf_a_path)
                os.unlink(pdf_b_path)

            except Exception as e:
                st.error(f"エラーが発生しました: {str(e)}")
                logging.error(f"Application error: {e}")

if __name__ == "__main__":
    # Import and run the new modular application
    from app.main import main as app_main
    app_main()