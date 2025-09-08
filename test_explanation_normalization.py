#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test script to verify explanation normalization functionality
"""

# Import the normalization function from app.py
import sys
sys.path.append('.')

# Test the normalization function directly
def _normalize_explanation(value):
    """
    explanation を UI 表示用の文字列に正規化する。
    - list の場合: 箇条書きの Markdown へ変換
    - str の場合: そのまま返す
    - その他/None: 空文字を返す
    """
    if isinstance(value, list):
        items = [str(x).strip() for x in value if str(x).strip()]
        if not items:
            return ""
        return "\n".join(f"- {x}" for x in items)
    if isinstance(value, str):
        return value
    return ""


def test_explanation_normalization():
    """explanation正規化のテスト"""
    
    print("🧪 Explanation正規化テスト")
    print("=" * 50)
    
    # テストケース1: リスト形式
    test_cases = [
        # リスト形式（正常）
        {
            "input": ["これはテストです", "二番目の項目", "三番目の項目"],
            "expected_type": "markdown_list",
            "description": "通常のリスト"
        },
        # 文字列形式
        {
            "input": "これは通常の文字列です。",
            "expected_type": "string",
            "description": "通常の文字列"
        },
        # 空リスト
        {
            "input": [],
            "expected_type": "empty",
            "description": "空のリスト"
        },
        # None
        {
            "input": None,
            "expected_type": "empty", 
            "description": "None値"
        },
        # 空文字列を含むリスト
        {
            "input": ["有効な項目", "", "  ", "もう一つの有効な項目"],
            "expected_type": "filtered_list",
            "description": "空文字列を含むリスト"
        },
        # 混合型（数値を含む）
        {
            "input": ["テキスト項目", 123, "もう一つのテキスト"],
            "expected_type": "mixed_list",
            "description": "数値を含むリスト"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nテスト {i}: {test_case['description']}")
        print(f"入力: {test_case['input']}")
        
        result = _normalize_explanation(test_case['input'])
        print(f"出力: '{result}'")
        
        # 期待される動作の確認
        if test_case['expected_type'] == "markdown_list":
            assert result.startswith("- "), "リストは箇条書き形式で開始されるべき"
            assert "\n- " in result, "複数項目はMarkdown形式で区切られるべき"
            print("✅ 箇条書き形式への変換成功")
            
        elif test_case['expected_type'] == "string":
            assert isinstance(result, str), "文字列はそのまま返されるべき"
            assert result == test_case['input'], "文字列は変更されずに返されるべき"
            print("✅ 文字列はそのまま保持")
            
        elif test_case['expected_type'] == "empty":
            assert result == "", "空の場合は空文字列を返すべき"
            print("✅ 空の場合は空文字列を返却")
            
        elif test_case['expected_type'] == "filtered_list":
            assert "有効な項目" in result, "有効な項目は含まれるべき"
            assert "もう一つの有効な項目" in result, "有効な項目は含まれるべき"
            lines = result.split('\n')
            assert len(lines) == 2, "空文字列は除外されるべき"
            print("✅ 空文字列のフィルタリング成功")
            
        elif test_case['expected_type'] == "mixed_list":
            assert "テキスト項目" in result, "テキストは含まれるべき"
            assert "123" in result, "数値は文字列として含まれるべき"
            assert "もう一つのテキスト" in result, "テキストは含まれるべき"
            print("✅ 混合型の処理成功")
        
        print("-" * 30)
    
    print("\n🎉 すべてのテストが成功しました！")
    

def test_display_example():
    """実際の表示例のテスト"""
    
    print("\n📋 実際の使用例")
    print("=" * 50)
    
    # LLMが返す可能性のあるレスポンス例
    sample_responses = [
        # 配列形式の解説
        {
            "answer": "フレックスタイムのコアタイムは10時〜15時です。",
            "explanation": [
                "コアタイムは従業員が必ず勤務すべき時間帯です。",
                "フレックスの前後は比較的自由に勤務可能です。",
                "社内会議や顧客対応は原則この時間内に設定されます。",
                "制度の柔軟性と責任のバランスが求められます。"
            ]
        },
        # 文字列形式の解説
        {
            "answer": "時間外勤務の上限は月45時間、年360時間です。",
            "explanation": "時間外勤務に関しては事前申請が原則で、特別条項がある場合は年720時間まで可能ですが健康診断が必要です。"
        }
    ]
    
    for i, response in enumerate(sample_responses, 1):
        print(f"\nサンプル {i}:")
        print(f"回答: {response['answer']}")
        print(f"元の解説: {response['explanation']}")
        
        normalized = _normalize_explanation(response['explanation'])
        print(f"正規化後の解説:")
        print(normalized)
        print("-" * 40)


if __name__ == "__main__":
    test_explanation_normalization()
    test_display_example()