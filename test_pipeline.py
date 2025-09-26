#!/usr/bin/env python
"""Test script for the complete document processing pipeline."""

import sys
sys.path.append('/workspaces/pdfhikaku')

from pdfhikaku import DocumentProcessor, DocumentComparator

def test_document_pipeline():
    """Test the complete document processing and comparison pipeline."""

    print("🔄 ドキュメント処理パイプラインテスト開始")
    print("=" * 60)

    # Create sample PDF pages (simulating extracted text)
    sample_pages_a = [
        {
            "page": 1,
            "text": """
            1章 概要
            1-1. プロジェクト概要
            このプロジェクトは...
            """,
            "blocks": [],
            "headings": [
                {"text": "1章 概要", "level": 1, "font_size": 18},
                {"text": "1-1. プロジェクト概要", "level": 2, "font_size": 14}
            ]
        },
        {
            "page": 2,
            "text": """
            2章 システム設計
            2-1. アーキテクチャ
            システムの構成は...
            """,
            "blocks": [],
            "headings": [
                {"text": "2章 システム設計", "level": 1, "font_size": 18},
                {"text": "2-1. アーキテクチャ", "level": 2, "font_size": 14}
            ]
        }
    ]

    sample_pages_b = [
        {
            "page": 1,
            "text": """
            1章 概要
            1-1. プロジェクト概要
            このプロジェクトについて...
            """,
            "blocks": [],
            "headings": [
                {"text": "1章 概要", "level": 1, "font_size": 18},
                {"text": "1-1. プロジェクト概要", "level": 2, "font_size": 14}
            ]
        },
        {
            "page": 2,
            "text": """
            2章 システム設計
            2-1. アーキテクチャ
            システムの設計は...
            """,
            "blocks": [],
            "headings": [
                {"text": "2章 システム設計", "level": 1, "font_size": 18},
                {"text": "2-1. アーキテクチャ", "level": 2, "font_size": 14}
            ]
        }
    ]

    # Initialize processor and comparator
    processor = DocumentProcessor(use_ocr=False)
    comparator = DocumentComparator(similarity_threshold=0.6, granularity="sentence")

    # Simulate document processing (normally would extract from PDF)
    print("📄 Step 1-4: ドキュメント処理中...")

    # Manually create document structures for testing
    doc_a = simulate_document_processing(processor, sample_pages_a, "Document A")
    doc_b = simulate_document_processing(processor, sample_pages_b, "Document B")

    # Step 5-6: Compare documents
    print("🔍 Step 5-6: ドキュメント比較中...")
    comparison = comparator.compare_documents(doc_a, doc_b)

    # Display results
    print("\n📊 比較結果:")
    print(f"  - 全体一致率: {comparison['overall_similarity']:.2f}%")
    print(f"  - セクション対応数: {len(comparison['section_mappings'])}")
    print(f"  - 比較結果数: {len(comparison['comparison_results'])}")

    print("\n📋 セクション対応:")
    for mapping in comparison['section_mappings']:
        print(f"  - '{mapping['A_title']}' ↔ '{mapping['B_title']}' (類似度: {mapping['score']:.2f})")

    print("\n📈 各セクションの類似度:")
    for result in comparison['comparison_results']:
        print(f"  - {result['mapping']['A_title']}: {result['similarity']:.2f}%")

    print("\n✅ パイプラインテスト完了!")
    return True

def simulate_document_processing(processor, pages, doc_name):
    """Simulate document processing for testing."""
    print(f"  Processing {doc_name}...")

    # Step 2: Detect headings
    headings = processor.heading_detector.detect_headings_from_pages(pages)
    print(f"    - 検出された見出し: {len(headings)}個")

    # Step 3: Generate sections
    sections_by_chapter = processor.section_generator.generate_sections(pages, headings)
    print(f"    - 生成された章: {len(sections_by_chapter)}章")

    # Step 4: Generate TOC
    toc = processor.toc_generator.generate_toc(sections_by_chapter)
    print(f"    - 目次エントリ: {len(toc)}個")

    return {
        'pages': pages,
        'headings': headings,
        'sections_by_chapter': sections_by_chapter,
        'toc': toc
    }

if __name__ == "__main__":
    try:
        success = test_document_pipeline()
        if success:
            print("\n🎉 すべてのテストに合格しました！")
        else:
            print("\n❌ テストに失敗しました")
    except Exception as e:
        print(f"\n❌ テストでエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()