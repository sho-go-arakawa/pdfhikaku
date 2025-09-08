#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test script to verify enhanced RAG functionality with structured JSON responses
"""

import os
import json
from utils.pinecone_io import PineconeManager
import openai


class EnhancedRAGTester:
    def __init__(self):
        self.client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.embedding_model = "text-embedding-3-small"
        self.chat_model = "gpt-3.5-turbo"
        self.pinecone_manager = PineconeManager()
        
        self.system_prompt = """あなたは企業内規定に関するQAアシスタントです。与えられた文脈（retrieved_context）の情報のみを根拠として、質問に対する回答を生成してください。

以下のルールを厳守して、**必ずJSONのみ**を返してください。

【出力構造】

{
  "answer": "質問に対する簡潔な結論（1～3文）",
  "explanation": "背景や注意点を含む解説（5～8文、専門用語はやさしく）",
  "highlights": [
    {
      "source_id": "company_policies",
      "chunk_index": 12,
      "span": [128, 215],
      "quote": "従業員は...あらかじめ申請が必要です。"
    }
  ],
  "sources": [
    {
      "source_id": "company_policies",
      "chunk_index": 12,
      "confidence": "High"
    }
  ]
}

【ルール】

1. answer は事実ベースで簡潔にまとめてください。
2. explanation は要点・背景・注意点を含め、読み手が理解しやすい構成にしてください（箇条書き可）。
3. highlights は原文の重要な部分をそのまま短く引用し、チャンク内の文字範囲 [start, end] を指定してください（最大5件）。
4. sources は回答に使ったチャンクをユニークにまとめ、confidence を "High" / "Med" / "Low" のいずれかで評価してください。
5. 不明な場合は "不明です" と答え、追加で必要な情報を1〜2点提示してください。
6. 推測する場合は "推測: ..." と明示してください。
7. JSON 以外の文字列（説明文や注釈）は絶対に出力しないでください。"""
    
    def create_query_embedding(self, query: str):
        """クエリを埋め込みベクトルに変換"""
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=query
        )
        return response.data[0].embedding
    
    def search_relevant_chunks(self, query: str, top_k: int = 3):
        """関連するチャンクを検索"""
        query_vector = self.create_query_embedding(query)
        return self.pinecone_manager.search_similar(query_vector, top_k=top_k)
    
    def generate_enhanced_response(self, query: str, context_chunks):
        """構造化されたJSONレスポンスを生成"""
        # retrieved_context を構築
        retrieved_context = []
        for i, chunk in enumerate(context_chunks):
            retrieved_context.append({
                "source_id": "company_policies",
                "chunk_index": chunk["metadata"]["chunk_index"],
                "text": chunk["text"],
                "char_start": chunk["metadata"]["char_start"],
                "char_end": chunk["metadata"]["char_end"]
            })
        
        # ユーザーメッセージを構築
        user_message = {
            "question": query,
            "retrieved_context": retrieved_context
        }
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": json.dumps(user_message, ensure_ascii=False, indent=2)}
        ]
        
        response = self.client.chat.completions.create(
            model=self.chat_model,
            messages=messages,
            temperature=0.1,
            max_tokens=800
        )
        
        # JSONレスポンスをパース
        try:
            response_text = response.choices[0].message.content.strip()
            # JSONマーカーがある場合は除去
            if response_text.startswith('```json'):
                response_text = response_text[7:-3].strip()
            elif response_text.startswith('```'):
                response_text = response_text[3:-3].strip()
            
            parsed_response = json.loads(response_text)
            return parsed_response
        except json.JSONDecodeError as e:
            return {
                "answer": f"JSONパースエラー: {str(e)}",
                "explanation": f"原文: {response_text[:200]}...",
                "highlights": [],
                "sources": []
            }


def test_enhanced_rag():
    """拡張RAGをテスト"""
    
    print("🚀 拡張RAGシステムテスト")
    print("=" * 50)
    
    tester = EnhancedRAGTester()
    
    test_queries = [
        "フレックスタイムのコアタイムは何時ですか？",
        "時間外勤務の上限時間について教えてください",
        "有給休暇の申請はいつまでに行う必要がありますか？"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. 質問: {query}")
        print("-" * 30)
        
        try:
            # 関連チャンクを検索
            context_chunks = tester.search_relevant_chunks(query, top_k=3)
            print(f"検索結果: {len(context_chunks)}件")
            
            # 拡張応答を生成
            response = tester.generate_enhanced_response(query, context_chunks)
            
            # 結果を表示
            print(f"\n📝 回答: {response.get('answer', 'N/A')}")
            print(f"\n💡 解説: {response.get('explanation', 'N/A')}")
            
            # ハイライト表示
            highlights = response.get('highlights', [])
            if highlights:
                print(f"\n🔍 引用ハイライト ({len(highlights)}件):")
                for j, highlight in enumerate(highlights, 1):
                    print(f"  {j}. チャンク {highlight.get('chunk_index', 'N/A')}")
                    print(f"     引用: \"{highlight.get('quote', 'N/A')}\"")
                    print(f"     位置: {highlight.get('span', 'N/A')}")
            
            # 出典表示
            sources = response.get('sources', [])
            if sources:
                print(f"\n📚 出典 ({len(sources)}件):")
                for source in sources:
                    confidence_emoji = {
                        "High": "🟢", 
                        "Med": "🟡", 
                        "Low": "🔴"
                    }.get(source.get('confidence', 'Med'), "⚪")
                    
                    print(f"  {confidence_emoji} チャンク {source.get('chunk_index', 'N/A')} - 信頼度: {source.get('confidence', 'Med')}")
            
        except Exception as e:
            print(f"❌ エラー: {e}")
        
        print("\n" + "=" * 50)


if __name__ == "__main__":
    test_enhanced_rag()