#!/usr/bin/env python
"""Debug pattern matching for section headers."""

import re

def test_patterns():
    """Test section header patterns against sample data."""

    section_patterns = [
        r'^(\d+)[-\s](\d+)[-\s](\d+)[\.\s]*(.+)',  # 2-3-1. タイトル (sub-section)
        r'^(\d+)\.(\d+)\.(\d+)[\.\s]*(.+)',        # 2.3.1. タイトル (sub-section)
        r'^(\d+)[-\s](\d+)[\.\s]*(.+)',      # 2-3. タイトル, 2-3 タイトル
        r'^(\d+)\.(\d+)[\.\s]*(.+)',         # 2.3. タイトル, 2.3 タイトル
    ]

    test_lines = [
        "1-1. プロジェクト概要",
        "2-1. アーキテクチャ",
        "3-2-1. 認証機能",
        "3-2-2. データ処理",
        "4.2. 統合テスト"
    ]

    print("パターンマッチングテスト:")

    for line in test_lines:
        print(f"\nテストライン: '{line}'")

        for i, pattern in enumerate(section_patterns):
            match = re.match(pattern, line)
            if match:
                groups = match.groups()
                print(f"  パターン {i+1} マッチ: {groups}")

                if len(groups) == 3:  # X-Y.Title format
                    chapter, section, title = groups
                    print(f"    -> 章: {chapter}, 節: {section}, タイトル: {title}")
                elif len(groups) == 4:  # X-Y-Z.Title format (sub-section)
                    chapter, section, subsection, title = groups
                    print(f"    -> 章: {chapter}, 節: {section}, サブ節: {subsection}, タイトル: {title}")
                break
        else:
            print("  マッチなし")

if __name__ == "__main__":
    test_patterns()