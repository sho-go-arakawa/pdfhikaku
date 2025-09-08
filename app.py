import os
import streamlit as st
from typing import List, Dict, Any
import openai
import json
from utils.pinecone_io import PineconeManager


# 追加: explanation 正規化ヘルパー
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


class RAGChatbot:
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
    
    def create_query_embedding(self, query: str) -> List[float]:
        """クエリを埋め込みベクトルに変換"""
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=query
        )
        return response.data[0].embedding
    
    def search_relevant_chunks(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """関連するチャンクを検索"""
        query_vector = self.create_query_embedding(query)
        return self.pinecone_manager.search_similar(query_vector, top_k=top_k)
    
    def generate_response(self, query: str, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """コンテキストに基づいて構造化された回答を生成"""
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
        
        # JSON レスポンスをパース
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
            # JSONパースに失敗した場合はフォールバック
            return {
                "answer": "申し訳ございませんが、回答の生成中にエラーが発生しました。",
                "explanation": f"システムエラー: {str(e)}",
                "highlights": [],
                "sources": []
            }


def main():
    st.set_page_config(
        page_title="社内規定チャットボット",
        page_icon="📘",
        layout="wide"
    )
    
    st.title("📘 社内規定チャットボット")
    st.markdown("社内の規定について質問してください。")
    
    # サイドバーでオプション設定
    with st.sidebar:
        st.header("設定")
        top_k = st.slider("検索結果数", min_value=1, max_value=10, value=5)
        show_context = st.checkbox("コンテキストを表示", value=False)
        show_highlights = st.checkbox("ハイライトを表示", value=True)
        show_sources = st.checkbox("出典を表示", value=True)
        
        st.header("使用方法")
        st.markdown("""
        1. 下のテキストボックスに質問を入力
        2. Enterキーまたは「送信」ボタンをクリック
        3. AI が社内規定に基づいて回答します
        
        **例：**
        - フレックスタイムのコアタイムは何時ですか？
        - 有給休暇の取得方法を教えて
        - リモートワークの規定について
        """)
    
    # チャットボットの初期化
    if "chatbot" not in st.session_state:
        with st.spinner("チャットボットを初期化中..."):
            try:
                st.session_state.chatbot = RAGChatbot()
                st.success("チャットボットの準備が完了しました！")
            except Exception as e:
                st.error(f"初期化エラー: {e}")
                st.stop()
    
    # チャット履歴の初期化
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # チャット履歴の表示
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(message["content"])
            else:
                # アシスタントの応答を構造化して表示
                response_data = message.get("response_data")
                if response_data and isinstance(response_data, dict):
                    # 回答を表示
                    st.markdown(f"**📝 回答:**")
                    st.markdown(response_data.get("answer", ""))
                    
                    # 解説を表示（配列→Markdown文字列に正規化）
                    raw_explanation = response_data.get("explanation", "")
                    normalized_explanation = _normalize_explanation(raw_explanation)
                    if normalized_explanation:
                        st.markdown("**💡 解説:**")
                        st.markdown(normalized_explanation)
                    
                    # ハイライトを表示
                    if show_highlights and response_data.get("highlights"):
                        with st.expander("🔍 引用ハイライト"):
                            for i, highlight in enumerate(response_data.get("highlights", [])):
                                st.markdown(f"**引用 {i+1}:**")
                                st.code(highlight.get("quote", ""), language=None)
                                st.caption(f"チャンク {highlight.get('chunk_index', 'N/A')}, 位置: {highlight.get('span', [])}")
                    
                    # 出典を表示  
                    if show_sources and response_data.get("sources"):
                        with st.expander("📚 出典情報"):
                            for source in response_data.get("sources", []):
                                confidence_color = {
                                    "High": "🟢", 
                                    "Med": "🟡", 
                                    "Low": "🔴"
                                }.get(source.get("confidence", "Med"), "⚪")
                                
                                st.markdown(f"{confidence_color} チャンク {source.get('chunk_index', 'N/A')} - 信頼度: {source.get('confidence', 'Med')}")
                else:
                    # フォールバック: 従来形式
                    st.markdown(message["content"])
                
                # コンテキスト表示（従来通り）
                if show_context and "context" in message:
                    with st.expander("参考にした社内規定の抜粋"):
                        for i, chunk in enumerate(message["context"]):
                            st.markdown(f"**[関連情報 {i+1}] (類似度: {chunk['score']:.3f})**")
                            st.markdown(chunk["text"])
                            st.markdown("---")
    
    # ユーザー入力
    if prompt := st.chat_input("社内規定について質問してください..."):
        # ユーザーメッセージを追加
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # アシスタントの回答を生成
        with st.chat_message("assistant"):
            with st.spinner("回答を生成中..."):
                try:
                    # 関連チャンクを検索
                    context_chunks = st.session_state.chatbot.search_relevant_chunks(
                        prompt, top_k=top_k
                    )
                    
                    if not context_chunks:
                        response_data = {
                            "answer": "申し訳ございませんが、お探しの情報が社内規定に見つかりませんでした。",
                            "explanation": "検索結果に該当する情報が見つかりませんでした。別の表現で質問してみてください。",
                            "highlights": [],
                            "sources": []
                        }
                        context_chunks = []
                    else:
                        # 回答を生成
                        response_data = st.session_state.chatbot.generate_response(
                            prompt, context_chunks
                        )
                    
                    # 構造化された回答を表示
                    st.markdown(f"**📝 回答:**")
                    st.markdown(response_data.get("answer", ""))
                    
                    # 解説を表示（配列→Markdown文字列に正規化）
                    raw_explanation = response_data.get("explanation", "")
                    normalized_explanation = _normalize_explanation(raw_explanation)
                    if normalized_explanation:
                        st.markdown("**💡 解説:**")
                        st.markdown(normalized_explanation)
                    
                    # ハイライトを表示
                    if show_highlights and response_data.get("highlights"):
                        with st.expander("🔍 引用ハイライト"):
                            for i, highlight in enumerate(response_data.get("highlights", [])):
                                st.markdown(f"**引用 {i+1}:**")
                                st.code(highlight.get("quote", ""), language=None)
                                st.caption(f"チャンク {highlight.get('chunk_index', 'N/A')}, 位置: {highlight.get('span', [])}")
                    
                    # 出典を表示  
                    if show_sources and response_data.get("sources"):
                        with st.expander("📚 出典情報"):
                            for source in response_data.get("sources", []):
                                confidence_color = {
                                    "High": "🟢", 
                                    "Med": "🟡", 
                                    "Low": "🔴"
                                }.get(source.get("confidence", "Med"), "⚪")
                                
                                st.markdown(f"{confidence_color} チャンク {source.get('chunk_index', 'N/A')} - 信頼度: {source.get('confidence', 'Med')}")
                    
                    # コンテキスト表示（従来通り）
                    if context_chunks and show_context:
                        with st.expander("参考にした社内規定の抜粋"):
                            for i, chunk in enumerate(context_chunks):
                                st.markdown(f"**[関連情報 {i+1}] (類似度: {chunk['score']:.3f})**")
                                st.markdown(chunk["text"])
                                st.markdown("---")
                    
                    # 保存前に explanation を文字列へ寄せる（任意）
                    response_data["explanation"] = _normalize_explanation(response_data.get("explanation", ""))
                    
                    # アシスタントメッセージを追加
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response_data.get("answer", ""),
                        "response_data": response_data,
                        "context": context_chunks
                    })
                
                except Exception as e:
                    st.error(f"エラーが発生しました: {e}")
                    # エラー時のフォールバック応答
                    error_response = {
                        "answer": "申し訳ございませんが、システムエラーが発生しました。",
                        "explanation": f"エラー詳細: {str(e)}",
                        "highlights": [],
                        "sources": []
                    }
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": error_response["answer"],
                        "response_data": error_response,
                        "context": []
                    })
    
    # フッター
    st.markdown("---")
    st.markdown("💡 このチャットボットは社内規定に基づいて回答します。最新の情報については人事部にご確認ください。")


if __name__ == "__main__":
    main()