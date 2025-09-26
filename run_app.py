#!/usr/bin/env python
"""
PDF比較ツール (PDFHikaku) - 起動スクリプト

使用方法:
1. streamlit run run_app.py
2. ブラウザで http://localhost:8501 にアクセス  
3. PDF A と PDF B をアップロードして比較開始

機能:
- 章ベースのテキスト比較
- 段落・文・語単位での詳細差分表示
- Unicode正規化対応
- 総合分析レポート
"""

if __name__ == "__main__":
    from pdfhikaku import main
    main()
