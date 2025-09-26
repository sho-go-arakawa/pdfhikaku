#!/usr/bin/env python
"""Test script for section extraction and organization by chapter."""

import sys
sys.path.append('/workspaces/pdfhikaku')

from pdfhikaku import OutlineDetector

def test_section_extraction_integration():
    """Test section extraction and organization by chapter."""

    # Create sample PDF pages with section headers
    sample_pages = [
        {
            "page": 1,
            "text": """
            1章 概要
            1-1. プロジェクト概要
            1-2. 目標と目的
            本文がここに続きます...
            """,
        },
        {
            "page": 2,
            "text": """
            2章 システム設計
            2-1. アーキテクチャ
            2-2. データベース設計
            2-3. API設計
            詳細な設計情報...
            """,
        },
        {
            "page": 3,
            "text": """
            2-4. セキュリティ設計
            3章 実装
            3-1. フロントエンド実装
            3-2. バックエンド実装
            3-2-1. 認証機能
            3-2-2. データ処理
            実装の詳細...
            """,
        },
        {
            "page": 4,
            "text": """
            4章 テスト
            4-1. 単体テスト
            4.2. 統合テスト
            4.3. システムテスト
            """,
        }
    ]

    # Initialize detector
    detector = OutlineDetector()

    print("PDFページから節を抽出して章番号別に整理するテスト...")

    # Extract sections organized by chapter
    sections_by_chapter = detector.extract_sections_from_pages(sample_pages)

    print(f"\n抽出された章数: {len(sections_by_chapter)}")

    for chapter_num, sections in sections_by_chapter.items():
        print(f"\n=== {chapter_num}章 ({len(sections)}節) ===")
        for section in sections:
            if 'subsection' in section:
                print(f"  {section['section']}-{section['subsection']}. {section['title']} (ページ {section['page']})")
            else:
                print(f"  {section['section']}. {section['title']} (ページ {section['page']})")

    # Summary statistics
    total_sections = sum(len(sections) for sections in sections_by_chapter.values())
    print(f"\n📊 統計情報:")
    print(f"  - 総章数: {len(sections_by_chapter)}")
    print(f"  - 総節数: {total_sections}")
    print(f"  - 処理ページ数: {len(sample_pages)}")

    # Verify correct organization
    assert 1 in sections_by_chapter, "1章が見つかりません"
    assert 2 in sections_by_chapter, "2章が見つかりません"
    assert 3 in sections_by_chapter, "3章が見つかりません"
    assert 4 in sections_by_chapter, "4章が見つかりません"

    # Check chapter 2 has the most sections (should have 4 sections: 2-1, 2-2, 2-3, 2-4)
    assert len(sections_by_chapter[2]) == 4, f"2章の節数が正しくありません (期待値: 4, 実際: {len(sections_by_chapter[2])})"

    # Check chapter 3 has subsections
    chapter_3_sections = sections_by_chapter[3]
    subsection_found = any('subsection' in section for section in chapter_3_sections)
    assert subsection_found, "3章にサブセクションが見つかりません"

    print("\n✅ すべての検証に合格しました！")
    return True

if __name__ == "__main__":
    try:
        success = test_section_extraction_integration()
        if success:
            print("\n🎉 セクション抽出と章別整理のテストが完了しました！")
        else:
            print("\n❌ テストに失敗しました")
    except Exception as e:
        print(f"\n❌ テストでエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()