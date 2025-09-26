"""Main Streamlit application with enhanced UI and modular architecture."""

import streamlit as st
import tempfile
import os
import time
import logging
import traceback
import re
from pathlib import Path
from typing import Dict, Any, Optional, List

# Import our modules
from .config import config
from .extractors import PDFExtractor
from .headings import HeadingDetector, TOCExtractor
from .chunker import DocumentChunker
from .align import DocumentAligner
from .diffing import DiffVisualizer, create_diff_css
from .scoring import SimilarityCalculator
from .report import MultiFormatReporter
from .bookmarks import BookmarkGenerator
from .utils import ProgressTracker, MemoryManager

logger = logging.getLogger(__name__)

class PDFHikakuApp:
    """Main application class that coordinates all components."""

    def __init__(self):
        self.components = self._initialize_components()
        self.progress_tracker = ProgressTracker()
        self.memory_manager = MemoryManager()

    def _initialize_components(self) -> Dict[str, Any]:
        """Initialize all application components."""
        return {
            'extractor': PDFExtractor(),
            'heading_detector': HeadingDetector(),
            'toc_extractor': TOCExtractor(),
            'chunker': DocumentChunker(),
            'aligner': DocumentAligner(),
            'diff_visualizer': DiffVisualizer(),
            'similarity_calculator': SimilarityCalculator(),
            'reporter': MultiFormatReporter(),
            'bookmark_generator': BookmarkGenerator()
        }

    def run(self):
        """Run the main Streamlit application."""
        st.set_page_config(
            page_title="PDF比較ツール (PDFHikaku)",
            page_icon="📄",
            layout=config.get_ui_config()['layout']
        )

        st.title("📄 PDF差分比較アプリ (PDFHikaku)")
        st.info("✨ **本文テキスト抽出機能有効** - ヘッダー・フッターを除いた本文のみを高精度比較")
        st.markdown(create_diff_css(), unsafe_allow_html=True)

        # Sidebar configuration
        self._render_sidebar()

        # Main content area
        uploaded_files = self._render_file_upload()

        if uploaded_files['pdf_a'] and uploaded_files['pdf_b']:
            if st.button("🔍 比較開始", type="primary"):
                self._run_comparison(uploaded_files)

        # Display results if available
        if 'comparison_results' in st.session_state:
            self._render_results()

    def _render_sidebar(self):
        """Render configuration sidebar."""
        with st.sidebar:
            st.header("⚙️ 設定")

            # Extraction settings
            st.subheader("📄 テキスト抽出")
            use_ocr = st.checkbox("OCR使用 (画像PDF対応)", value=False)
            ocr_lang = st.selectbox("OCR言語", ["jpn+eng", "jpn", "eng"], index=0)

            # Alignment settings
            st.subheader("🔗 整列設定")
            embed_weight = st.slider("埋め込み類似度の重み", 0.0, 1.0, 0.6, 0.1)
            string_weight = st.slider("文字列類似度の重み", 0.0, 1.0, 0.4, 0.1)
            exact_threshold = st.slider("完全一致閾値", 0.0, 1.0, 0.92, 0.01)
            partial_threshold = st.slider("部分一致閾値", 0.0, 1.0, 0.75, 0.01)

            # Diff settings
            st.subheader("🎨 差分表示")
            granularity = st.selectbox(
                "差分粒度",
                ["word", "sentence", "character", "line"],
                index=0
            )

            # Performance settings
            st.subheader("⚡ パフォーマンス")
            max_workers = st.slider("並列処理数", 1, 8, 4)

            # Report settings
            st.subheader("📊 レポート出力")
            export_formats = st.multiselect(
                "出力形式",
                ["html", "csv", "json", "pdf"],
                default=["html", "csv"]
            )

            # Store settings in session state
            st.session_state['settings'] = {
                'use_ocr': use_ocr,
                'ocr_lang': ocr_lang,
                'embed_weight': embed_weight,
                'string_weight': string_weight,
                'exact_threshold': exact_threshold,
                'partial_threshold': partial_threshold,
                'granularity': granularity,
                'max_workers': max_workers,
                'export_formats': export_formats
            }

    def _render_file_upload(self) -> Dict[str, Any]:
        """Render file upload area."""
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📄 旧スタイル")
            pdf_a = st.file_uploader("旧スタイルをアップロード", type="pdf", key="pdf_a")

        with col2:
            st.subheader("📄 新スタイル")
            pdf_b = st.file_uploader("新スタイルをアップロード", type="pdf", key="pdf_b")

        return {'pdf_a': pdf_a, 'pdf_b': pdf_b}

    def _run_comparison(self, uploaded_files: Dict[str, Any]):
        """Run the complete comparison process."""
        try:
            start_time = time.time()

            # Initialize progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            self.progress_tracker.start(status_text, progress_bar)

            # Save uploaded files
            with st.spinner("ファイルを準備中..."):
                pdf_paths = self._save_uploaded_files(uploaded_files)

            # Extract text from both PDFs
            self.progress_tracker.update(10, "旧スタイル からテキスト抽出中...")
            pages_a = self._extract_pdf_text(pdf_paths['pdf_a'], 'A')

            self.progress_tracker.update(25, "新スタイル からテキスト抽出中...")
            pages_b = self._extract_pdf_text(pdf_paths['pdf_b'], 'B')

            # Detect document structure
            self.progress_tracker.update(40, "文書構造を解析中...")
            structure_a = self._detect_structure(pages_a, pdf_paths['pdf_a'])
            structure_b = self._detect_structure(pages_b, pdf_paths['pdf_b'])

            # Create chunks
            self.progress_tracker.update(55, "テキストをチャンク化中...")
            chunks_a = self._create_chunks(pages_a, structure_a, 'doc_a')
            chunks_b = self._create_chunks(pages_b, structure_b, 'doc_b')

            # Align documents
            self.progress_tracker.update(70, "文書を整列中...")
            alignment_result = self._align_documents({
                'sections': structure_a['sections'],
                'chunks': chunks_a
            }, {
                'sections': structure_b['sections'],
                'chunks': chunks_b
            })

            # Calculate similarities
            self.progress_tracker.update(85, "類似度を計算中...")
            similarity_result = self.components['similarity_calculator'].calculate_document_similarity(alignment_result)

            # Generate visualizations
            self.progress_tracker.update(95, "可視化を生成中...")
            diff_visualization = self._create_diff_visualization(alignment_result)

            # Complete
            processing_time = time.time() - start_time
            self.progress_tracker.update(100, f"完了! ({processing_time:.1f}秒)")

            # Store results
            st.session_state['comparison_results'] = {
                'pdf_paths': pdf_paths,
                'pages_a': pages_a,
                'pages_b': pages_b,
                'structure_a': structure_a,
                'structure_b': structure_b,
                'chunks_a': chunks_a,
                'chunks_b': chunks_b,
                'alignment_result': alignment_result,
                'similarity_result': similarity_result,
                'diff_visualization': diff_visualization,
                'processing_info': {
                    'total_time_seconds': processing_time,
                    'settings': st.session_state.get('settings', {})
                }
            }

            # Cleanup temporary files
            self._cleanup_temp_files(pdf_paths)

            st.success(f"🎉 比較完了! 総合類似度: {similarity_result['overall_similarity']:.2f}%")

        except Exception as e:
            logger.error(f"Comparison failed: {e}")
            logger.error(traceback.format_exc())
            st.error(f"比較処理中にエラーが発生しました: {str(e)}")

    def _save_uploaded_files(self, uploaded_files: Dict[str, Any]) -> Dict[str, str]:
        """Save uploaded files to temporary locations."""
        pdf_paths = {}

        for key, uploaded_file in uploaded_files.items():
            if uploaded_file is not None:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    pdf_paths[key] = tmp_file.name

        return pdf_paths

    def _extract_pdf_text(self, pdf_path: str, doc_id: str) -> list:
        """Extract text from PDF with progress tracking."""
        def progress_callback(current, total, message):
            progress = 10 + (current / total) * 15  # Allocate 15% for each PDF
            if doc_id == 'B':
                progress += 15
            self.progress_tracker.update(progress, message)

        return self.components['extractor'].extract_text(pdf_path, progress_callback)

    def _detect_structure(self, pages: list, pdf_path: str) -> Dict[str, Any]:
        """Detect document structure prioritizing PDF bookmarks over text detection."""
        # Detect content start page (after TOC)
        content_start_page = self.components['heading_detector'].detect_content_start_page(pages)
        logger.info(f"Content starts at page {content_start_page}")

        # Detect numbered chapter structure (X-Y.Title format)
        numbered_chapters = self.components['heading_detector'].detect_numbered_chapter_structure(
            pages, content_start_page
        )

        # Always extract comprehensive TOC for display purposes
        comprehensive_toc = self.components['toc_extractor'].extract_comprehensive_toc(pdf_path, pages)

        if numbered_chapters:
            logger.info(f"Using numbered chapter structure with {len(numbered_chapters)} chapters")
            return {
                'headings': numbered_chapters,
                'sections': self._create_sections_from_numbered_chapters(numbered_chapters),
                'toc_source': 'numbered_chapters',
                'toc_entries': comprehensive_toc,  # Store TOC for display
                'primary_structure': 'numbered_chapters',
                'content_start_page': content_start_page,
                'numbered_chapters': numbered_chapters
            }


        if comprehensive_toc:
            # Use bookmark-based structure as primary headings
            primary_headings = []
            sections = []

            for i, toc_entry in enumerate(comprehensive_toc):
                # Convert TOC entries to heading format for compatibility
                heading = {
                    'text': toc_entry['title'],
                    'level': toc_entry['level'],
                    'page': toc_entry['page'],
                    'detection_method': toc_entry.get('detection_method', 'toc'),
                    'confidence': toc_entry.get('confidence', 1.0),
                    'source': toc_entry.get('source', 'bookmark'),
                    'id': toc_entry.get('id', f'toc_heading_{i+1}')
                }

                # Add enhanced text if available
                if 'heading_text' in toc_entry:
                    heading['heading_text'] = toc_entry['heading_text']

                primary_headings.append(heading)

                # Create section from TOC entry
                sections.append({
                    'id': f"section_{i + 1}",
                    'title': toc_entry['title'],
                    'level': toc_entry['level'],
                    'page_start': toc_entry['page'],
                    'page_end': toc_entry['page'],
                    'source': toc_entry.get('source', 'bookmark'),
                    'heading_info': heading,
                    'toc_entry': toc_entry
                })

            # Supplement with detected headings if needed (for content between bookmarks)
            detected_headings = self.components['heading_detector'].detect_headings_from_pages(pages)
            supplementary_headings = self._merge_headings_with_toc(primary_headings, detected_headings)

            return {
                'headings': primary_headings + supplementary_headings,
                'sections': sections,
                'toc_source': 'comprehensive',
                'toc_entries': comprehensive_toc,
                'primary_structure': 'bookmark'
            }
        else:
            # Fallback to traditional heading detection
            detected_headings = self.components['heading_detector'].detect_headings_from_pages(pages)
            sections = []

            for i, heading in enumerate(detected_headings):
                sections.append({
                    'id': f"section_{i + 1}",
                    'title': heading['text'],
                    'level': heading['level'],
                    'page_start': heading['page'],
                    'page_end': heading['page'],
                    'heading_info': heading,
                    'source': 'detected'
                })

            return {
                'headings': detected_headings,
                'sections': sections,
                'toc_source': 'detected',
                'toc_entries': [],
                'primary_structure': 'heading_detection',
                'content_start_page': content_start_page
            }

    def _create_sections_from_numbered_chapters(self, numbered_chapters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create sections from numbered chapters for compatibility."""
        sections = []

        for i, chapter in enumerate(numbered_chapters):
            sections.append({
                'id': f"chapter_{chapter['numbering']}",
                'title': f"{chapter['numbering']} {chapter['title']}",
                'level': chapter['level'],
                'page_start': chapter['page'],
                'page_end': chapter['page'],  # Will be updated during processing
                'heading_info': chapter,
                'source': 'numbered_chapters'
            })

        return sections

    def _merge_headings_with_toc(self, toc_headings: List[Dict[str, Any]], detected_headings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge detected headings with TOC headings to fill gaps."""
        supplementary = []

        # Create page ranges from TOC headings
        toc_pages = set(h['page'] for h in toc_headings)

        for heading in detected_headings:
            # Only add detected headings if they're not on pages already covered by TOC
            # or if they provide additional structure detail
            if heading['page'] not in toc_pages:
                # Check if this heading adds value (different level or significant text)
                is_valuable = True
                for toc_heading in toc_headings:
                    if (abs(heading['page'] - toc_heading['page']) <= 1 and
                        self._headings_are_similar(heading['text'], toc_heading['text'])):
                        is_valuable = False
                        break

                if is_valuable:
                    heading['source'] = 'detected_supplementary'
                    heading['confidence'] = heading.get('confidence', 0.5) * 0.8  # Lower confidence for supplementary
                    supplementary.append(heading)

        return supplementary

    def _headings_are_similar(self, text1: str, text2: str) -> bool:
        """Check if two headings are similar enough to be considered duplicates."""
        # Simple similarity check
        words1 = set(re.sub(r'[^\w\s]', '', text1.lower()).split())
        words2 = set(re.sub(r'[^\w\s]', '', text2.lower()).split())

        if not words1 or not words2:
            return False

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return (intersection / union) > 0.7 if union > 0 else False

    def _create_chunks(self, pages: list, structure: dict, doc_id: str) -> list:
        """Create hierarchical chunks from pages and structure."""
        # Check if we have numbered chapters
        if structure.get('primary_structure') == 'numbered_chapters':
            numbered_chapters = structure.get('numbered_chapters', [])
            content_start_page = structure.get('content_start_page', 1)

            logger.info(f"Creating numbered chapter chunks starting from page {content_start_page}")
            return self.components['chunker'].create_numbered_chapter_chunks(
                pages, numbered_chapters, content_start_page, doc_id
            )
        else:
            # Use traditional heading-based chunking
            headings = structure.get('headings', [])
            return self.components['chunker'].create_chunks(pages, headings, doc_id)

    def _align_documents(self, doc_a: Dict[str, Any], doc_b: Dict[str, Any]) -> Dict[str, Any]:
        """Align two documents."""
        return self.components['aligner'].align_documents(doc_a, doc_b)

    def _create_diff_visualization(self, alignment_result: Dict[str, Any]) -> Dict[str, Any]:
        """Create diff visualization."""
        settings = st.session_state.get('settings', {})
        granularity = settings.get('granularity', 'word')

        paragraph_alignments = alignment_result.get('paragraph_alignments', [])
        return self.components['diff_visualizer'].create_side_by_side_diff(
            paragraph_alignments, granularity
        )

    def _render_results(self):
        """Render comparison results with tabbed interface."""
        results = st.session_state['comparison_results']

        # Create tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 概要", "🔍 差分表示", "📋 目次", "📄 差分リスト", "📤 レポート"
        ])

        with tab1:
            self._render_summary_tab(results)

        with tab2:
            self._render_diff_tab(results)

        with tab3:
            self._render_toc_tab(results)

        with tab4:
            self._render_diff_list_tab(results)

        with tab5:
            self._render_report_tab(results)

    def _render_summary_tab(self, results: Dict[str, Any]):
        """Render summary tab."""
        st.header("📊 比較概要")

        similarity_result = results['similarity_result']
        processing_info = results['processing_info']

        # Key metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "総合類似度",
                f"{similarity_result['overall_similarity']:.2f}%",
                delta=None
            )

        with col2:
            section_count = len(similarity_result.get('section_similarities', []))
            st.metric("比較セクション数", section_count)

        with col3:
            alignment_stats = similarity_result.get('alignment_statistics', {})
            st.metric("比較段落数", alignment_stats.get('total_alignments', 0))

        with col4:
            st.metric("処理時間", f"{processing_info['total_time_seconds']:.1f}秒")

        # Section similarities chart
        st.subheader("セクション別類似度")
        section_similarities = similarity_result.get('section_similarities', [])

        if section_similarities:
            import pandas as pd

            df = pd.DataFrame(section_similarities)
            st.bar_chart(df.set_index('section_a_title')['similarity_score'])
            st.dataframe(df[['section_a_title', 'similarity_score', 'match_type', 'paragraph_count']])

        # Summary statistics
        st.subheader("統計情報")
        summary = similarity_result.get('summary', {})
        col1, col2, col3 = st.columns(3)

        with col1:
            st.write("**マッチ統計:**")
            st.write(f"- 一致率: {summary.get('match_rate', 0):.1f}%")
            st.write(f"- 高類似度率: {summary.get('high_similarity_rate', 0):.1f}%")
            st.write(f"- 評価: {summary.get('grade', 'N/A')}")

        with col2:
            st.write("**処理統計:**")
            st.write(f"- セクション数: {summary.get('total_sections', 0)}")
            st.write(f"- 整列数: {summary.get('total_alignments', 0)}")

        with col3:
            st.write("**テキスト抽出:**")
            st.write("- ✅ 本文テキスト使用")
            st.write("- ✅ ヘッダー/フッター除去")
            st.write("- ✅ 高精度比較モード")

            # Show structure type
            structure_a = results.get('structure_a', {})
            structure_type = structure_a.get('primary_structure', 'unknown')
            toc_entries = structure_a.get('toc_entries', [])

            if toc_entries:
                bookmark_count = len([e for e in toc_entries if e.get('source') == 'bookmark'])
                if bookmark_count > 0:
                    st.write(f"- 📖 総目次 (ブックマーク): {bookmark_count}項目")
                else:
                    st.write(f"- 📝 総目次 (テキスト): {len(toc_entries)}項目")

            if structure_type == 'numbered_chapters':
                st.write("- 📚 番号付き章構造検出")
                content_start = structure_a.get('content_start_page', 1)
                st.write(f"- 📄 内容開始: {content_start}ページ〜")

    def _render_diff_tab(self, results: Dict[str, Any]):
        """Render side-by-side diff tab."""
        st.header("🔍 側面差分表示")

        diff_visualization = results['diff_visualization']
        diff_blocks = diff_visualization.get('diff_blocks', [])

        if not diff_blocks:
            st.warning("表示する差分がありません。")
            return

        # Chapter comparison controls
        st.subheader("🔍 比較設定")

        # Mode selection
        comparison_mode = st.selectbox(
            "比較モード",
            ["全章表示", "章別比較"],
            help="全章を表示するか、特定の章同士を比較するかを選択"
        )

        # Get structure data for chapter selection
        results = st.session_state.get('comparison_results', {})
        structure_a = results.get('structure_a', {})
        structure_b = results.get('structure_b', {})

        # Initialize variables
        selected_sections = []
        selected_a_section = None
        selected_b_section = None
        min_similarity = 0
        max_blocks = 20
        sort_by = "類似度順（降順）"

        if comparison_mode == "章別比較":
            # Chapter comparison mode
            col_a, col_b, col_sim, col_max = st.columns(4)

            with col_a:
                # 旧スタイル chapters
                pdf_a_sections = []
                if structure_a.get('sections'):
                    pdf_a_sections = [s.get('title', f"Section {i}") for i, s in enumerate(structure_a['sections'])]
                if not pdf_a_sections:
                    pdf_a_sections = ["章情報なし"]

                selected_a_section = st.selectbox(
                    "📄 旧スタイル の章",
                    pdf_a_sections,
                    help="比較したい旧スタイルの章を選択"
                )

            with col_b:
                # 新スタイル chapters
                pdf_b_sections = []
                if structure_b.get('sections'):
                    pdf_b_sections = [s.get('title', f"Section {i}") for i, s in enumerate(structure_b['sections'])]
                if not pdf_b_sections:
                    pdf_b_sections = ["章情報なし"]

                selected_b_section = st.selectbox(
                    "📄 新スタイル の章",
                    pdf_b_sections,
                    help="比較したい新スタイルの章を選択"
                )

            with col_sim:
                min_similarity = st.slider(
                    "📊 最小類似度(%)",
                    0, 100, 0,
                    help="指定した類似度以上のブロックのみ表示"
                )

            with col_max:
                max_blocks = st.slider("📄 表示ブロック数", 1, min(len(diff_blocks), 100), 20)

            # Filter for chapter comparison
            section_titles = list(set(block.get('section_title', 'Unknown Section') for block in diff_blocks))
            for block in diff_blocks:
                section_title = block.get('section_title', '')
                if (selected_a_section in section_title or
                    selected_b_section in section_title or
                    f"{selected_a_section} ↔ {selected_b_section}" == section_title):
                    if section_title not in selected_sections:
                        selected_sections.append(section_title)

            if not selected_sections:
                selected_sections = section_titles  # Fallback

            st.info(f"📋 **章別比較**: 旧スタイル「{selected_a_section}」↔ 新スタイル「{selected_b_section}」")

        else:
            # All chapters mode
            col_filter, col_sim, col_max, col_sort = st.columns(4)

            with col_filter:
                section_titles = list(set(block.get('section_title', 'Unknown Section') for block in diff_blocks))
                section_titles.sort()
                selected_sections = st.multiselect(
                    "📚 章でフィルタ",
                    section_titles,
                    default=section_titles,
                    help="表示したい章を選択してください"
                )

            with col_sim:
                min_similarity = st.slider(
                    "📊 最小類似度(%)",
                    0, 100, 0,
                    help="指定した類似度以上のブロックのみ表示"
                )

            with col_max:
                max_blocks = st.slider("📄 表示ブロック数", 1, min(len(diff_blocks), 100), 20)

            with col_sort:
                sort_by = st.selectbox(
                    "📋 ソート順",
                    ["類似度順（降順）", "類似度順（昇順）", "章順"],
                    help="表示順序を選択してください"
                )

        # 差分タイプは全て表示（フィルタなし）
        alignment_types = list(set(block['alignment_type'] for block in diff_blocks))
        selected_types = alignment_types  # 全ての差分タイプを選択状態にする

        # 差分タイプのラベル定義（統計情報表示で使用）
        alignment_type_labels = {
            'match': '✅ 一致',
            'partial': '🟡 部分一致',
            'replace': '🔄 置換',
            'delete': '❌ 削除',
            'insert': '➕ 追加'
        }

        # 検索バー（全幅）
        search_term = st.text_input(
            "🔍 テキスト検索",
            "",
            placeholder="章タイトルや内容で検索...",
            help="章タイトルまたは内容に含まれる文字で検索できます"
        )

        # Filter diff blocks
        filtered_blocks = []
        for block in diff_blocks:
            # 章フィルタ
            if block.get('section_title', 'Unknown Section') not in selected_sections:
                continue

            # 差分タイプフィルタ
            if block['alignment_type'] not in selected_types:
                continue

            # 類似度フィルタ
            if block['similarity'] < min_similarity:
                continue

            # 検索フィルタ
            if search_term:
                search_text = f"{block.get('section_title', '')} {block.get('html_a', '')} {block.get('html_b', '')}".lower()
                if search_term.lower() not in search_text:
                    continue

            filtered_blocks.append(block)

        # ソート処理
        if sort_by == "類似度順（降順）":
            filtered_blocks.sort(key=lambda x: x['similarity'], reverse=True)
        elif sort_by == "類似度順（昇順）":
            filtered_blocks.sort(key=lambda x: x['similarity'])
        elif sort_by == "章順":
            filtered_blocks.sort(key=lambda x: x.get('section_title', ''))

        filtered_blocks = filtered_blocks[:max_blocks]

        # フィルタ結果の統計情報表示
        st.write(f"📊 **フィルタ結果**: {len(filtered_blocks)} ブロック (全体: {len(diff_blocks)} ブロック)")

        if filtered_blocks:
            # 統計情報を表示
            stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)

            # タイプ別統計
            type_counts = {}
            total_similarity = 0
            for block in filtered_blocks:
                block_type = block['alignment_type']
                type_counts[block_type] = type_counts.get(block_type, 0) + 1
                total_similarity += block['similarity']

            avg_similarity = total_similarity / len(filtered_blocks) if filtered_blocks else 0

            with stats_col1:
                st.metric("平均類似度", f"{avg_similarity:.1f}%")

            with stats_col2:
                st.metric("選択章数", len(selected_sections))

            with stats_col3:
                most_common_type = max(type_counts.items(), key=lambda x: x[1]) if type_counts else ('N/A', 0)
                st.metric("最多差分タイプ", alignment_type_labels.get(most_common_type[0], most_common_type[0]))

            with stats_col4:
                high_similarity_count = sum(1 for block in filtered_blocks if block['similarity'] >= 80)
                st.metric("高類似度ブロック", f"{high_similarity_count}/{len(filtered_blocks)}")

        if not filtered_blocks:
            st.warning("フィルタ条件に一致するブロックがありません。フィルタ設定を確認してください。")
            return

        # 色分けの凡例を表示
        st.subheader("🎨 差分タイプ別の色分け")
        legend_col1, legend_col2, legend_col3, legend_col4, legend_col5, legend_col6 = st.columns(6)

        with legend_col1:
            st.markdown(
                '<div style="display: flex; align-items: center; margin-bottom: 5px;">'
                '<span style="background-color: #90EE90; padding: 2px 8px; border-radius: 4px; margin-right: 8px;">✅</span>'
                '<span><strong>一致</strong></span>'
                '</div>',
                unsafe_allow_html=True
            )

        with legend_col2:
            st.markdown(
                '<div style="display: flex; align-items: center; margin-bottom: 5px;">'
                '<span style="background-color: #FFD700; padding: 2px 8px; border-radius: 4px; margin-right: 8px;">🟡</span>'
                '<span><strong>部分一致</strong></span>'
                '</div>',
                unsafe_allow_html=True
            )

        with legend_col3:
            st.markdown(
                '<div style="display: flex; align-items: center; margin-bottom: 5px;">'
                '<span style="background-color: #FFA500; padding: 2px 8px; border-radius: 4px; margin-right: 8px;">🔄</span>'
                '<span><strong>置換</strong></span>'
                '</div>',
                unsafe_allow_html=True
            )

        with legend_col4:
            st.markdown(
                '<div style="display: flex; align-items: center; margin-bottom: 5px;">'
                '<span style="background-color: #FFB6C1; padding: 2px 8px; border-radius: 4px; margin-right: 8px;">❌</span>'
                '<span><strong>削除</strong></span>'
                '</div>',
                unsafe_allow_html=True
            )

        with legend_col5:
            st.markdown(
                '<div style="display: flex; align-items: center; margin-bottom: 5px;">'
                '<span style="background-color: #87CEEB; padding: 2px 8px; border-radius: 4px; margin-right: 8px;">➕</span>'
                '<span><strong>追加</strong></span>'
                '</div>',
                unsafe_allow_html=True
            )

        with legend_col6:
            st.markdown(
                '<div style="display: flex; align-items: center; margin-bottom: 5px;">'
                '<span style="background-color: #f3e5f5; padding: 2px 8px; border-radius: 4px; margin-right: 8px;">🔍</span>'
                '<span><strong>詳細差分</strong></span>'
                '</div>',
                unsafe_allow_html=True
            )

        st.caption("💡 **詳細差分（紫）**: 部分一致内で文字レベルの詳細な違いを表示")
        st.markdown("---")

        # Display diff blocks with enhanced information
        for i, block in enumerate(filtered_blocks, 1):
            # 差分タイプに応じたアイコンとカラー
            type_icons = {
                'match': '✅',
                'partial': '🟡',
                'replace': '🔄',
                'delete': '❌',
                'insert': '➕'
            }

            type_colors = {
                'match': '#90EE90',    # Light green
                'partial': '#FFD700',   # Gold
                'replace': '#FFA500',   # Orange
                'delete': '#FFB6C1',    # Light pink
                'insert': '#87CEEB'     # Sky blue
            }

            icon = type_icons.get(block['alignment_type'], '❓')
            color = type_colors.get(block['alignment_type'], '#FFFFFF')

            with st.expander(
                f"{icon} **[{i:02d}]** {block['section_title']} (類似度: {block['similarity']:.1f}%)",
                expanded=False
            ):
                # ブロック詳細情報の表示
                info_col1, info_col2, info_col3, info_col4 = st.columns(4)
                with info_col1:
                    st.caption(f"🏷️ **タイプ**: {alignment_type_labels.get(block['alignment_type'], block['alignment_type'])}")
                with info_col2:
                    st.caption(f"📊 **類似度**: {block['similarity']:.1f}%")
                with info_col3:
                    chunk_a_id = block.get('chunk_a_id', 'N/A')
                    chunk_b_id = block.get('chunk_b_id', 'N/A')
                    st.caption(f"🔗 **チャンクID**: A:{chunk_a_id} | B:{chunk_b_id}")
                with info_col4:
                    st.caption("📝 **本文テキスト使用中**")

                st.markdown("---")

                # 差分内容の表示
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**📄 旧スタイル**")
                    if block["html_a"]:
                        st.markdown(
                            f'<div class="diff-container" style="background-color: {color}20;">{block["html_a"]}</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown("*（この章にはコンテンツがありません）*")

                with col2:
                    st.markdown("**📄 新スタイル**")
                    if block["html_b"]:
                        st.markdown(
                            f'<div class="diff-container" style="background-color: {color}20;">{block["html_b"]}</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown("*（この章にはコンテンツがありません）*")

    def _render_toc_tab(self, results: Dict[str, Any]):
        """Render table of contents tab."""
        st.header("📋 目次構造")

        structure_a = results['structure_a']
        structure_b = results['structure_b']

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("旧スタイル の目次")

            # Prioritize TOC entries from bookmarks/comprehensive TOC
            toc_entries_a = structure_a.get('toc_entries', [])
            if toc_entries_a:
                st.info("📖 総目次から抽出")
                self._display_comprehensive_toc_tree(toc_entries_a, key_suffix="a")
            elif structure_a.get('primary_structure') == 'numbered_chapters':
                st.info("📚 番号付き章構造で解析")
                self._display_numbered_chapters_tree(structure_a.get('numbered_chapters', []), key_suffix="a")
            else:
                self._display_toc_tree(structure_a.get('headings', []), key_suffix="a")

        with col2:
            st.subheader("新スタイル の目次")

            # Prioritize TOC entries from bookmarks/comprehensive TOC
            toc_entries_b = structure_b.get('toc_entries', [])
            if toc_entries_b:
                st.info("📖 総目次から抽出")
                self._display_comprehensive_toc_tree(toc_entries_b, key_suffix="b")
            elif structure_b.get('primary_structure') == 'numbered_chapters':
                st.info("📚 番号付き章構造で解析")
                self._display_numbered_chapters_tree(structure_b.get('numbered_chapters', []), key_suffix="b")
            else:
                self._display_toc_tree(structure_b.get('headings', []), key_suffix="b")

    def _display_toc_tree(self, headings: list, key_suffix: str = ""):
        """Display TOC as a hierarchical tree showing comprehensive structure including sub-levels."""
        if not headings:
            st.write("見出しが検出されませんでした。")
            return

        # Separate headings by source
        bookmark_headings = [h for h in headings if h.get('source') == 'bookmark']
        detected_headings = [h for h in headings if h.get('source') in ['detected', 'detected_supplementary']]

        # Display mode selector with unique key
        display_mode = st.radio(
            "表示モード",
            ["総目次（ブックマーク）", "章構造", "全見出し（検出含む）", "階層別表示"],
            horizontal=True,
            key=f"toc_display_mode_{key_suffix}"
        )

        if display_mode == "総目次（ブックマーク）":
            self._display_bookmark_headings(bookmark_headings, detected_headings)
        elif display_mode == "章構造":
            self._display_chapter_structure(bookmark_headings)
        elif display_mode == "全見出し（検出含む）":
            self._display_all_headings(headings)
        else:  # 階層別表示
            self._display_hierarchical_headings(headings)

    def _display_bookmark_headings(self, bookmark_headings: list, detected_headings: list):
        """Display only bookmark headings."""
        if not bookmark_headings:
            st.warning("📑 このPDFには総目次（ブックマーク）が含まれていません。")
            if detected_headings:
                st.info(f"💡 代わりにテキスト検出による {len(detected_headings)} 個の見出しが見つかりました。")
            return

        st.info(f"📑 PDF総目次から {len(bookmark_headings)} 項目を表示しています")
        self._render_heading_list(bookmark_headings, show_source=False)

    def _display_chapter_structure(self, bookmark_headings: list):
        """Display bookmarks organized by chapter structure."""
        if not bookmark_headings:
            st.warning("📑 このPDFには総目次（ブックマーク）が含まれていません。")
            return

        # Debug information
        if st.checkbox("デバッグ情報を表示", key="debug_chapter_structure"):
            st.subheader("🔧 デバッグ情報")

            # Show detected chapter numbers
            all_items = []
            for heading in bookmark_headings:
                numbering = heading.get('numbering', '')
                level = self._determine_chapter_level_from_numbering(numbering)
                chapter_num = self._extract_chapter_number_from_numbering(numbering)
                all_items.append({
                    "numbering": numbering,
                    "level": level,
                    "chapter_number": chapter_num,
                    "title": heading.get('title', '')
                })

            chapter_numbers = self._identify_all_chapter_numbers(all_items)

            with st.expander("検出された章番号"):
                st.write(f"章番号: {sorted(chapter_numbers, key=lambda x: int(x) if x.isdigit() else 999)}")

            with st.expander("見出し一覧（レベル判定付き）"):
                for i, heading in enumerate(bookmark_headings):
                    level = self._determine_chapter_level_from_numbering(heading.get('numbering', ''))
                    chapter_num = self._extract_chapter_number_from_numbering(heading.get('numbering', ''))
                    virtual_mark = " (仮想章作成対象)" if level > 1 and chapter_num not in [item["chapter_number"] for item in all_items if item["level"] == 1] else ""
                    st.write(f"{i+1}. `{heading.get('numbering', '')}` - {heading.get('title', '')} (レベル: {level}, 章: {chapter_num}){virtual_mark}")

        # Organize headings by chapter
        chapters = self._organize_headings_by_chapter(bookmark_headings)

        if not chapters:
            st.warning("章構造を認識できませんでした。")
            if st.checkbox("詳細診断", key="detailed_diagnosis"):
                st.write("**利用可能な見出し:**")
                for heading in bookmark_headings:
                    st.write(f"- 番号: `{heading.get('numbering', 'なし')}`, タイトル: `{heading.get('title', '')}`")
            return

        # Count sections and subsections with virtual chapter info
        total_sections = sum(len(chapter['sections']) for chapter in chapters.values())
        total_subsections = sum(
            len(section.get('subsections', []))
            for chapter in chapters.values()
            for section in chapter['sections']
        )

        # Count virtual vs explicit chapters
        virtual_chapters = sum(1 for chapter in chapters.values() if chapter.get('is_virtual', False))
        explicit_chapters = len(chapters) - virtual_chapters

        info_text = f"📖 {len(chapters)} 章、{total_sections} 節、{total_subsections} 項の構造を表示しています"
        if virtual_chapters > 0:
            info_text += f" (うち推定章: {virtual_chapters}章)"

        st.info(info_text)

        for chapter_key, chapter_data in chapters.items():
            # Use display_number for consistent chapter numbering
            display_chapter = chapter_data.get('display_number', chapter_data.get('numbering', ''))

            # Distinguish between explicit and virtual chapters
            is_virtual = chapter_data.get('is_virtual', False)
            chapter_icon = "📚" if is_virtual else "📖"
            virtual_note = " (推定)" if is_virtual else ""

            with st.expander(f"{chapter_icon} {display_chapter}章 {chapter_data['title']}{virtual_note} (p.{chapter_data['page']})", expanded=True):

                # Show information about virtual chapters
                if is_virtual:
                    st.caption("💡 この章は子セクション（節・項）から自動的に推定されました")

                # Display chapter sections
                if chapter_data['sections']:
                    st.subheader("📝 節")
                    for section in chapter_data['sections']:
                        with st.container():
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                # Use display_number for consistent section numbering
                                display_section = section.get('display_number', section.get('numbering', ''))
                                st.write(f"　📝 {display_section} {section['title']}")
                            with col2:
                                st.caption(f"p.{section['page']}")

                            # Display subsections
                            if section.get('subsections'):
                                for subsection in section['subsections']:
                                    sub_col1, sub_col2 = st.columns([3, 1])
                                    with sub_col1:
                                        # Use display_number for consistent subsection numbering
                                        display_subsection = subsection.get('display_number', subsection.get('numbering', ''))
                                        st.write(f"　　📄 {display_subsection} {subsection['title']}")
                                    with sub_col2:
                                        st.caption(f"p.{subsection['page']}")
                else:
                    # Check if there are any potential sections that might have been missed
                    display_chapter_num = chapter_data.get('display_number', '')
                    potential_sections = self._find_potential_missed_sections(bookmark_headings, display_chapter_num)

                    if potential_sections:
                        st.subheader("📝 検出された関連項目")
                        st.caption("⚠️ 以下の項目がこの章に関連している可能性があります")
                        for pot_section in potential_sections:
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.write(f"　📝 {pot_section.get('numbering', '')} {pot_section.get('title', '')}")
                            with col2:
                                st.caption(f"p.{pot_section.get('page', '?')}")
                    else:
                        st.write("　*この章には下位セクションがありません*")

    def _find_potential_missed_sections(self, all_headings: list, chapter_number: str) -> list:
        """Find potential sections that might have been missed due to numbering issues."""
        if not chapter_number:
            return []

        potential_sections = []

        for heading in all_headings:
            numbering = heading.get('numbering', '')
            title = heading.get('title', heading.get('text', ''))

            # Look for headings that might be related but weren't caught
            # e.g., if chapter is "1", look for "1.1", "1-1", or similar patterns
            if (numbering.startswith(f"{chapter_number}.") or
                numbering.startswith(f"{chapter_number}-") or
                re.match(rf'^{chapter_number}[^\d]', numbering)):

                level = self._determine_chapter_level_from_numbering(numbering)
                if level > 1:  # Not a main chapter
                    potential_sections.append(heading)

        return potential_sections

    def _organize_headings_by_chapter(self, headings: list) -> dict:
        """Organize headings into chapter structure with automatic parent chapter identification."""
        # First pass: Create all items as flat list with hierarchy info
        all_items = []
        for heading in headings:
            numbering = heading.get('numbering', '')
            title = heading.get('title', heading.get('text', ''))
            clean_title = heading.get('clean_title', title)

            item = {
                "title": clean_title,
                "numbering": numbering,
                "page": heading.get("page", 1),
                "original_heading": heading,
                "level": self._determine_chapter_level_from_numbering(numbering),
                "chapter_number": self._extract_chapter_number_from_numbering(numbering)
            }
            all_items.append(item)

        # Identify all chapter numbers from sections and subsections
        chapter_numbers = self._identify_all_chapter_numbers(all_items)

        # Second pass: Build hierarchical structure including virtual chapters
        chapters = {}

        # Create chapters (both explicit and virtual)
        for chapter_num in sorted(chapter_numbers, key=lambda x: int(x) if x.isdigit() else 999):
            # Find explicit chapter heading
            explicit_chapter = self._find_explicit_chapter(all_items, chapter_num)

            # Find all sections belonging to this chapter
            chapter_sections = self._find_sections_for_chapter(all_items, chapter_num)

            if explicit_chapter or chapter_sections:  # Create chapter if it has content
                chapters[chapter_num] = {
                    "title": explicit_chapter["title"] if explicit_chapter else self._generate_virtual_chapter_title(chapter_num, chapter_sections),
                    "numbering": explicit_chapter["numbering"] if explicit_chapter else chapter_num,
                    "display_number": chapter_num,
                    "page": explicit_chapter["page"] if explicit_chapter else (chapter_sections[0]["page"] if chapter_sections else 1),
                    "sections": chapter_sections,
                    "original_heading": explicit_chapter["original_heading"] if explicit_chapter else None,
                    "is_virtual": not explicit_chapter  # Mark virtual chapters
                }

        return chapters

    def _identify_all_chapter_numbers(self, all_items: list) -> set:
        """Identify all chapter numbers from sections and subsections."""
        chapter_numbers = set()

        for item in all_items:
            if item["level"] == 1:  # Explicit chapters
                if item["chapter_number"]:
                    chapter_numbers.add(item["chapter_number"])
            elif item["level"] in [2, 3]:  # Sections and subsections
                parent_chapter = self._extract_chapter_number_from_numbering(item["numbering"])
                if parent_chapter:
                    chapter_numbers.add(parent_chapter)

        return chapter_numbers

    def _find_explicit_chapter(self, all_items: list, chapter_number: str) -> dict:
        """Find explicit chapter heading for the given chapter number."""
        for item in all_items:
            if item["level"] == 1 and item["chapter_number"] == chapter_number:
                return item
        return None

    def _generate_virtual_chapter_title(self, chapter_number: str, sections: list) -> str:
        """Generate a virtual chapter title based on its sections."""
        if not sections:
            return f"第{chapter_number}章"

        # Try to infer chapter theme from section titles
        section_titles = [section["title"] for section in sections[:3]]  # Use first 3 sections

        # Common patterns for chapter title inference
        common_themes = {
            "安全": "安全管理",
            "操作": "操作方法",
            "運転": "運転操作",
            "保守": "保守・点検",
            "メンテナンス": "メンテナンス",
            "設定": "設定・調整",
            "トラブル": "トラブルシューティング",
            "故障": "故障対応",
            "仕様": "仕様・性能",
            "構造": "構造・機能"
        }

        # Look for common themes in section titles
        for keyword, theme in common_themes.items():
            if any(keyword in title for title in section_titles):
                return f"第{chapter_number}章 {theme}"

        # Fallback: use first section title as hint
        if sections:
            first_section_title = sections[0]["title"]
            # Remove numbering from title
            clean_title = re.sub(r'^[\d\-\.]+\s*', '', first_section_title)
            if clean_title:
                return f"第{chapter_number}章 {clean_title.split('・')[0].split('、')[0][:10]}..."

        return f"第{chapter_number}章"

    def _find_sections_for_chapter(self, all_items: list, chapter_number: str) -> list:
        """Find all sections and subsections belonging to a chapter."""
        sections = []

        for item in all_items:
            if item["level"] == 2:  # Section level
                if self._item_belongs_to_chapter(item, chapter_number):
                    # Find all subsections for this section
                    subsections = self._find_subsections_for_section(all_items, item["numbering"])

                    section = {
                        "title": item["title"],
                        "numbering": item["numbering"],
                        "display_number": self._format_section_number(item["numbering"], chapter_number),
                        "page": item["page"],
                        "subsections": subsections,
                        "original_heading": item["original_heading"]
                    }
                    sections.append(section)

        return sections

    def _find_subsections_for_section(self, all_items: list, section_numbering: str) -> list:
        """Find all subsections belonging to a section."""
        subsections = []

        for item in all_items:
            if item["level"] == 3:  # Subsection level
                if self._item_belongs_to_section(item, section_numbering):
                    subsection = {
                        "title": item["title"],
                        "numbering": item["numbering"],
                        "display_number": self._format_subsection_number(item["numbering"], ""),
                        "page": item["page"],
                        "original_heading": item["original_heading"]
                    }
                    subsections.append(subsection)

        return subsections

    def _item_belongs_to_chapter(self, item: dict, chapter_number: str) -> bool:
        """Check if an item belongs to the specified chapter."""
        item_chapter = self._extract_chapter_number_from_numbering(item["numbering"])
        return item_chapter == chapter_number

    def _item_belongs_to_section(self, item: dict, section_numbering: str) -> bool:
        """Check if an item belongs to the specified section."""
        if not item["numbering"] or not section_numbering:
            return False

        # For dash-separated numbering (1-1-1 belongs to 1-1)
        if '-' in item["numbering"] and '-' in section_numbering:
            item_parts = item["numbering"].split('-')
            section_parts = section_numbering.split('-')

            if len(item_parts) >= 3 and len(section_parts) >= 2:
                return (item_parts[0] == section_parts[0] and
                        item_parts[1] == section_parts[1])

        # For dot-separated numbering (1.1.1 belongs to 1.1)
        elif '.' in item["numbering"] and '.' in section_numbering:
            item_parts = item["numbering"].split('.')
            section_parts = section_numbering.split('.')

            if len(item_parts) >= 3 and len(section_parts) >= 2:
                return (item_parts[0] == section_parts[0] and
                        item_parts[1] == section_parts[1])

        return False

    def _determine_chapter_level_from_numbering(self, numbering: str) -> int:
        """Determine hierarchical level from numbering pattern with enhanced recognition."""
        if not numbering:
            return 1

        # Remove trailing periods or spaces for cleaner analysis
        clean_numbering = numbering.strip().rstrip('.')

        # Count separators to determine depth
        dash_count = clean_numbering.count('-')
        dot_count = clean_numbering.count('.')

        # Enhanced level determination
        if dash_count >= 2:
            return 3  # Subsection (1-1-1, 1-2-3)
        elif dash_count == 1:
            return 2  # Section (1-1, 1-2, 5-1)
        elif dot_count >= 2:
            return 3  # Subsection (1.1.1, 1.2.3)
        elif dot_count == 1:
            return 2  # Section (1.1, 1.2, 5.1)
        else:
            # Check for special patterns that might indicate sections even without separators
            # This catches cases where levels aren't properly detected from PDF bookmarks
            if re.match(r'^\d+$', clean_numbering):
                # Simple numbers could be chapters, but check if they're actually sections
                # by looking at their context or range
                number = int(clean_numbering)
                if number > 20:  # Heuristic: numbers > 20 are likely not main chapters
                    return 2
                return 1  # Chapter (1, 2, 3)
            else:
                return 1  # Default to chapter level

    def _extract_chapter_number_from_numbering(self, numbering: str) -> str:
        """Extract main chapter number from numbering."""
        if not numbering:
            return ""

        if '-' in numbering:
            return numbering.split('-')[0]
        elif '.' in numbering:
            return numbering.split('.')[0]
        else:
            return numbering

    def _section_belongs_to_chapter(self, section_numbering: str, chapter_number: str) -> bool:
        """Check if section belongs to the current chapter."""
        if not section_numbering or not chapter_number:
            return True

        section_chapter = self._extract_chapter_number_from_numbering(section_numbering)
        return section_chapter == chapter_number

    def _subsection_belongs_to_section(self, subsection_numbering: str, section_numbering: str) -> bool:
        """Check if subsection belongs to the current section."""
        if not subsection_numbering or not section_numbering:
            return True

        # Extract chapter-section parts for comparison
        if '-' in subsection_numbering and '-' in section_numbering:
            sub_parts = subsection_numbering.split('-')
            sec_parts = section_numbering.split('-')
            return (len(sub_parts) >= 2 and len(sec_parts) >= 2 and
                    sub_parts[0] == sec_parts[0] and sub_parts[1] == sec_parts[1])
        elif '.' in subsection_numbering and '.' in section_numbering:
            sub_parts = subsection_numbering.split('.')
            sec_parts = section_numbering.split('.')
            return (len(sub_parts) >= 2 and len(sec_parts) >= 2 and
                    sub_parts[0] == sec_parts[0] and sub_parts[1] == sec_parts[1])

        return True

    def _format_section_number(self, numbering: str, chapter_number: str) -> str:
        """Format section number consistently as chapter-section (e.g., 1-1, 5-2)."""
        if not numbering:
            return ""

        if '-' in numbering:
            return numbering  # Already in correct format
        elif '.' in numbering:
            # Convert 1.2 to 1-2
            parts = numbering.split('.')
            if len(parts) >= 2:
                return f"{parts[0]}-{parts[1]}"

        return numbering

    def _format_subsection_number(self, numbering: str, section_display_number: str) -> str:
        """Format subsection number consistently as chapter-section-subsection (e.g., 1-1-1)."""
        if not numbering:
            return ""

        if '-' in numbering and numbering.count('-') >= 2:
            return numbering  # Already in correct format
        elif '.' in numbering:
            # Convert 1.2.3 to 1-2-3
            parts = numbering.split('.')
            if len(parts) >= 3:
                return f"{parts[0]}-{parts[1]}-{parts[2]}"

        return numbering

    def _display_all_headings(self, headings: list):
        """Display all headings including detected ones."""
        st.info(f"📋 全見出し {len(headings)} 項目を表示しています")
        self._render_heading_list(headings, show_source=True)

    def _display_hierarchical_headings(self, headings: list):
        """Display headings organized by level with enhanced hierarchy."""
        # Group by level
        level_groups = {}
        for heading in headings:
            level = heading.get('level', 1)
            if level not in level_groups:
                level_groups[level] = []
            level_groups[level].append(heading)

        st.info(f"📊 {len(level_groups)} 階層レベル、{len(headings)} 項目を表示")

        for level in sorted(level_groups.keys()):
            headings_at_level = level_groups[level]

            # Level indicators
            level_icon = "📖" if level == 1 else "📝" if level == 2 else "📄" if level == 3 else "🔹"
            level_name = "章" if level == 1 else "節" if level == 2 else "項" if level == 3 else f"レベル{level}"

            st.subheader(f"{level_icon} {level_name}（レベル{level}）- {len(headings_at_level)}項目")

            # Display chapter/section numbering patterns
            numbering_patterns = set()
            for heading in headings_at_level:
                numbering = heading.get('numbering', '')
                if numbering:
                    numbering_patterns.add(numbering.split('-')[0] if '-' in numbering else numbering.split('.')[0] if '.' in numbering else numbering)

            if numbering_patterns:
                # Extract numbers for sorting
                def get_sort_key(x):
                    numbers = re.findall(r'\d+', x)
                    return int(numbers[0]) if numbers else 0

                sorted_patterns = sorted(numbering_patterns, key=get_sort_key)
                st.caption(f"番号体系: {', '.join(sorted_patterns)}")

            self._render_heading_list(headings_at_level, show_source=True, compact=True)

    def _render_heading_list(self, headings: list, show_source: bool = False, compact: bool = False):
        """Render a list of headings with proper indentation and details."""
        for heading in headings:
            level = heading.get('level', 1)
            indent = "　" * (level - 1)  # Use full-width space for better visual hierarchy

            # Level-specific icons
            level_icons = {1: "📖", 2: "📝", 3: "📄", 4: "🔹", 5: "▫️"}
            icon = level_icons.get(level, "▪️")

            # Extract numbering and title information
            text = heading.get('text', 'Unknown')
            numbering = heading.get('numbering', '')
            title = heading.get('title', heading.get('text', 'Unknown'))

            # Create display text showing hierarchy
            if numbering:
                display_text = f"{numbering} {title}"
            else:
                display_text = title

            # Source indicator
            source = heading.get('source', 'unknown')
            source_indicators = {
                'bookmark': '📑',
                'detected': '🔍',
                'detected_supplementary': '🔍+'
            }
            source_icon = source_indicators.get(source, '❓')

            confidence = heading.get('confidence', 1.0)
            confidence_text = f" (信頼度: {confidence:.1f})" if confidence > 0 and confidence < 1 else ""

            page_info = heading.get('page', '?')

            if compact:
                # Compact display for hierarchical view
                label = f"{indent}{icon} {display_text}"
                if show_source:
                    label += f" {source_icon}"
                label += f" (p.{page_info}){confidence_text}"

                st.write(label)
            else:
                # Detailed display with expanders
                with st.expander(f"{indent}{icon} {display_text} (ページ {page_info}){confidence_text}", expanded=False):
                    if show_source:
                        source_names = {
                            'bookmark': 'PDF総目次（ブックマーク）',
                            'detected': 'テキスト検出',
                            'detected_supplementary': 'テキスト検出（補完）'
                        }
                        st.caption(f"📍 **ソース**: {source_names.get(source, source)}")

                    st.caption(f"📊 **階層レベル**: {level}")
                    st.caption(f"📄 **ページ**: {page_info}")

                    if numbering:
                        st.caption(f"🔢 **番号**: {numbering}")
                    if title and title != text:
                        st.caption(f"📝 **タイトル**: {title}")

                    if 'detection_method' in heading:
                        st.caption(f"🔧 **検出方法**: {heading['detection_method']}")

                    if 'font_size' in heading and heading.get('font_size'):
                        st.caption(f"📐 **フォントサイズ**: {heading['font_size']:.1f}pt")

                    if 'heading_text' in heading and heading['heading_text'] != text:
                        st.caption(f"📄 **検出テキスト**: {heading['heading_text']}")

                    if 'toc_entry' in heading and heading['toc_entry']:
                        toc_entry = heading['toc_entry']
                        if 'id' in toc_entry:
                            st.caption(f"🆔 **TOCエントリID**: {toc_entry['id']}")

    def _render_diff_list_tab(self, results: Dict[str, Any]):
        """Render difference list tab."""
        st.header("📄 差分一覧")

        alignment_result = results['alignment_result']
        paragraph_alignments = alignment_result.get('paragraph_alignments', [])

        if not paragraph_alignments:
            st.warning("差分データがありません。")
            return

        # Create DataFrame for display
        import pandas as pd

        diff_data = []
        for alignment in paragraph_alignments:
            diff_data.append({
                'タイプ': alignment.get('alignment_type', 'unknown'),
                '類似度': f"{alignment.get('similarity_score', 0) * 100:.1f}%",
                'テキストA': (alignment.get('chunk_a_text', '') or '')[:100] + '...',
                'テキストB': (alignment.get('chunk_b_text', '') or '')[:100] + '...'
            })

        df = pd.DataFrame(diff_data)

        # Display with filtering
        alignment_types = df['タイプ'].unique()
        selected_types = st.multiselect(
            "表示する差分タイプ",
            alignment_types,
            default=alignment_types
        )

        filtered_df = df[df['タイプ'].isin(selected_types)]
        st.dataframe(filtered_df, use_container_width=True)

        # Export CSV
        if st.button("CSVとしてダウンロード"):
            csv = filtered_df.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="差分リスト.csvをダウンロード",
                data=csv,
                file_name="diff_list.csv",
                mime="text/csv"
            )

    def _render_report_tab(self, results: Dict[str, Any]):
        """Render report generation tab."""
        st.header("📤 レポート生成")

        settings = st.session_state.get('settings', {})
        export_formats = settings.get('export_formats', ['html'])

        if st.button("レポート生成", type="primary"):
            with st.spinner("レポートを生成中..."):
                try:
                    # Generate reports
                    report_results = self.components['reporter'].generate_all_reports(
                        results, "reports/comparison_report", export_formats
                    )

                    # Display results
                    st.success("レポート生成完了!")

                    for format_type, success in report_results.items():
                        if success:
                            st.success(f"✅ {format_type.upper()} レポート生成成功")
                            # Here you could add download buttons for the generated files
                        else:
                            st.error(f"❌ {format_type.upper()} レポート生成失敗")

                except Exception as e:
                    st.error(f"レポート生成中にエラーが発生しました: {str(e)}")

        # Bookmark generation
        st.subheader("🔖 しおり付きPDF生成")

        if st.button("しおり付きPDFを生成"):
            with st.spinner("しおり付きPDFを生成中..."):
                try:
                    pdf_paths = results['pdf_paths']
                    structure_a = results['structure_a']
                    structure_b = results['structure_b']

                    # Generate bookmarked PDFs
                    success_a = self.components['bookmark_generator'].add_bookmarks_to_pdf(
                        pdf_paths['pdf_a'],
                        structure_a['headings'],
                        "outputs/pdf_a_with_bookmarks.pdf"
                    )

                    success_b = self.components['bookmark_generator'].add_bookmarks_to_pdf(
                        pdf_paths['pdf_b'],
                        structure_b['headings'],
                        "outputs/pdf_b_with_bookmarks.pdf"
                    )

                    if success_a and success_b:
                        st.success("しおり付きPDFの生成が完了しました！")
                    else:
                        st.warning("一部のPDFでしおり生成に失敗しました。")

                except Exception as e:
                    st.error(f"しおり生成中にエラーが発生しました: {str(e)}")

    def _cleanup_temp_files(self, pdf_paths: Dict[str, str]):
        """Clean up temporary files."""
        for path in pdf_paths.values():
            try:
                if os.path.exists(path):
                    os.unlink(path)
            except Exception as e:
                logger.warning(f"Failed to cleanup {path}: {e}")

    def _display_numbered_chapters_tree(self, numbered_chapters: List[Dict[str, Any]], key_suffix: str = ""):
        """Display numbered chapters structure in a tree format."""
        if not numbered_chapters:
            st.write("番号付き章が検出されませんでした。")
            return

        st.write(f"**検出された章構造 ({len(numbered_chapters)}個):**")

        # Organize chapters by hierarchy
        main_chapters = {}  # chapter_number -> list of sections
        subsections = {}    # "chapter-section" -> list of subsections

        for chapter in numbered_chapters:
            chapter_num = chapter.get('chapter_number', 1)
            section_num = chapter.get('section_number')
            subsection_num = chapter.get('subsection_number')

            if subsection_num is not None:
                # Subsection
                parent_key = f"{chapter_num}-{section_num}"
                if parent_key not in subsections:
                    subsections[parent_key] = []
                subsections[parent_key].append(chapter)
            elif section_num is not None:
                # Section
                if chapter_num not in main_chapters:
                    main_chapters[chapter_num] = []
                main_chapters[chapter_num].append(chapter)
            else:
                # Main chapter
                if chapter_num not in main_chapters:
                    main_chapters[chapter_num] = []
                # Insert at beginning as main chapter
                main_chapters[chapter_num].insert(0, chapter)

        # Display hierarchically
        for chapter_num in sorted(main_chapters.keys()):
            chapters = main_chapters[chapter_num]

            for chapter in chapters:
                level = chapter.get('level', 1)
                numbering = chapter.get('numbering', '')
                title = chapter.get('title', '')
                page = chapter.get('page', '')

                if chapter.get('section_number') is None:
                    # Main chapter
                    icon = "📖" if level == 1 else "📄"
                    st.write(f"{icon} **{numbering} {title}** (p.{page})")

                    # Show subsections if any
                    section_key = f"{chapter_num}-{chapter.get('section_number', 1)}"
                    if section_key in subsections:
                        for subsection in sorted(subsections[section_key], key=lambda x: x.get('subsection_number', 0)):
                            sub_numbering = subsection.get('numbering', '')
                            sub_title = subsection.get('title', '')
                            sub_page = subsection.get('page', '')
                            st.write(f"    📝 {sub_numbering} {sub_title} (p.{sub_page})")

                else:
                    # Section
                    st.write(f"  📋 **{numbering} {title}** (p.{page})")

        # Summary
        chapter_count = len([c for c in numbered_chapters if c.get('section_number') is None])
        section_count = len([c for c in numbered_chapters if c.get('section_number') is not None and c.get('subsection_number') is None])
        subsection_count = len([c for c in numbered_chapters if c.get('subsection_number') is not None])

        st.caption(f"📊 章: {chapter_count}, 節: {section_count}, 項: {subsection_count}")

    def _display_comprehensive_toc_tree(self, toc_entries: List[Dict[str, Any]], key_suffix: str = ""):
        """Display comprehensive TOC from bookmarks/extracted TOC with integrated sections."""
        if not toc_entries:
            st.write("総目次情報が見つかりませんでした。")
            return

        # Count different types of entries
        chapter_count = len([e for e in toc_entries if e.get('level', 1) == 1])
        section_count = len([e for e in toc_entries if e.get('source') == 'text_header'])
        total_count = len(toc_entries)

        st.write(f"**総目次構造 (章{chapter_count}, 節{section_count}, 計{total_count}項目):**")

        current_chapter = None
        sections_displayed = 0

        # Display hierarchically with chapter-section grouping
        for entry in toc_entries:
            level = entry.get('level', 1)
            title = entry.get('title', '')
            page = entry.get('page', '')
            source = entry.get('source', 'unknown')
            confidence = entry.get('confidence', 0.0)

            if level == 1:  # Chapter level
                # Display chapter
                current_chapter = entry
                icon = "📘" if source == 'bookmark' else "📖"
                source_indicator = " 🔖" if source == 'bookmark' else (" 📝" if source == 'text' else "")

                st.write(f"{icon} **{title}**{source_indicator} (p.{page})")

                # Show confidence for non-bookmark sources
                if source != 'bookmark' and confidence < 1.0:
                    st.caption(f"    信頼度: {confidence:.1%}")

            elif level == 2 and source == 'text_header':  # Section from text headers
                # Display section under chapter
                parent_chapter = entry.get('parent_chapter', '')
                section_info = entry.get('section_info', {})

                # Enhanced section display
                st.write(f"    📄 **{title}** (p.{page}) 📋")
                st.caption(f"      テキスト見出しから抽出 (信頼度: {confidence:.1%})")

                sections_displayed += 1

            else:  # Other levels
                # Create indentation based on level
                indent = "  " * (level - 1)

                # Choose icon based on level and source
                if level == 2:
                    icon = "📝" if source == 'bookmark' else "📄"
                else:
                    icon = "📌" if source == 'bookmark' else "📋"

                # Add source indicator
                source_indicator = ""
                if source == 'bookmark':
                    source_indicator = " 🔖"
                elif source == 'text':
                    source_indicator = " 📝"
                elif source == 'text_header':
                    source_indicator = " 📋"

                # Display entry
                st.write(f"{indent}{icon} **{title}**{source_indicator} (p.{page})")

                # Show confidence for non-bookmark sources
                if source != 'bookmark' and confidence < 1.0:
                    st.caption(f"{indent}    信頼度: {confidence:.1%}")

        # Enhanced summary statistics
        bookmark_count = len([e for e in toc_entries if e.get('source') == 'bookmark'])
        text_count = len([e for e in toc_entries if e.get('source') == 'text'])
        text_header_count = len([e for e in toc_entries if e.get('source') == 'text_header'])

        level_counts = {}
        for entry in toc_entries:
            level = entry.get('level', 1)
            level_counts[level] = level_counts.get(level, 0) + 1

        st.caption("📊 **統計情報:**")
        if bookmark_count > 0:
            st.caption(f"   🔖 ブックマーク: {bookmark_count}項目")
        if text_count > 0:
            st.caption(f"   📝 テキスト目次: {text_count}項目")
        if text_header_count > 0:
            st.caption(f"   📋 テキスト見出し: {text_header_count}項目")

        # Show chapter-section mapping
        chapter_with_sections = len([e for e in toc_entries if e.get('level', 1) == 1])
        if text_header_count > 0:
            st.caption(f"   📚 章・節統合表示: {chapter_with_sections}章に{text_header_count}節を紐づけ")

        level_summary = ", ".join([f"レベル{level}: {count}" for level, count in sorted(level_counts.items())])
        st.caption(f"   📈 階層別: {level_summary}")


def main():
    """Main entry point."""
    app = PDFHikakuApp()
    app.run()

if __name__ == "__main__":
    main()