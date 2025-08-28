"""
売上データ分析AIチャットボット
CSV → DuckDB → NL→SQL → 集計/可視化
"""

import streamlit as st
import pandas as pd
import duckdb
import altair as alt
from openai import OpenAI
import json
from typing import Optional, Tuple
import os

# ページ設定
st.set_page_config(
    page_title="売上データ分析AIチャットボット",
    page_icon="📊",
    layout="wide"
)

# OpenAI クライアント初期化
@st.cache_resource
def init_openai_client() -> Optional[OpenAI]:
    """OpenAI クライアントを初期化"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("OPENAI_API_KEY 環境変数が設定されていません")
        return None
    return OpenAI(api_key=api_key)

# データ読み込み
@st.cache_data
def load_sales_df(path: str = "data/sample_sales.csv") -> pd.DataFrame:
    """売上データをCSVから読み込み、日付を適切に変換"""
    try:
        df = pd.read_csv(path)
        df['date'] = pd.to_datetime(df['date'])
        return df
    except Exception as e:
        st.error(f"データファイルの読み込みエラー: {e}")
        return pd.DataFrame()

# DuckDB 接続
@st.cache_resource
def get_duckdb_connection(df: pd.DataFrame) -> duckdb.DuckDBPyConnection:
    """DuckDB接続を作成し、salesテーブルを登録"""
    con = duckdb.connect(":memory:")
    if not df.empty:
        con.execute("""
            CREATE TABLE sales AS 
            SELECT 
                CAST(date AS DATE) as date,
                category,
                units,
                unit_price,
                region,
                sales_channel,
                customer_segment,
                revenue
            FROM df
        """)
    return con

# スキーマ情報取得
@st.cache_data
def get_schema(_con: duckdb.DuckDBPyConnection) -> str:
    """DuckDBのスキーマ情報を文字列として取得"""
    try:
        schema_result = _con.execute("DESCRIBE sales").fetchall()
        schema_text = "テーブル: sales\n列情報:\n"
        for row in schema_result:
            schema_text += f"- {row[0]} ({row[1]})\n"
        return schema_text
    except Exception:
        return "テーブル: sales\n列情報: 取得できませんでした"

# SQL安全性チェック
def is_safe_sql(sql: str) -> bool:
    """SQLが安全かチェック（SELECT文のみ許可）"""
    sql_upper = sql.upper().strip()
    
    # SELECT で始まることを確認
    if not sql_upper.startswith("SELECT"):
        return False
    
    # セミコロンがないことを確認
    if ";" in sql:
        return False
    
    # 危険なキーワードがないことを確認
    dangerous_keywords = [
        "ATTACH", "COPY", "CREATE", "INSERT", "UPDATE", "DELETE", 
        "ALTER", "DROP", "EXPORT", "PRAGMA", "TRUNCATE", "REPLACE"
    ]
    
    for keyword in dangerous_keywords:
        if keyword in sql_upper:
            return False
    
    return True

# SQL実行
def safe_execute_sql(con: duckdb.DuckDBPyConnection, sql: str) -> pd.DataFrame:
    """安全にSQLを実行し、結果をDataFrameで返す"""
    if not is_safe_sql(sql):
        raise ValueError("安全でないSQLが検出されました")
    
    result = con.execute(sql).fetchdf()
    if len(result) > 100:
        result = result.head(100)
    return result

# フォールバック回答
def fallback_answer(user_text: str) -> Tuple[pd.DataFrame, str, str, Optional[str], Optional[str], Optional[str]]:
    """LLM失敗時のフォールバック集計"""
    con = st.session_state.duckdb_con
    user_lower = user_text.lower()
    
    # 優先度1: 月×カテゴリ
    if any(word in user_lower for word in ["月", "month"]) and any(word in user_lower for word in ["カテゴリ", "category", "カテゴリー"]):
        sql = """
        SELECT 
            strftime(date, '%Y-%m') as month,
            category,
            SUM(revenue) as total_revenue
        FROM sales 
        GROUP BY strftime(date, '%Y-%m'), category 
        ORDER BY month, category
        LIMIT 100
        """
        df = con.execute(sql).fetchdf()
        return df, "bar", "month", "total_revenue", "category", sql
    
    # 優先度2: チャネル別売上
    elif any(word in user_lower for word in ["チャネル", "channel"]) and any(word in user_lower for word in ["売上", "売り上げ"]):
        sql = """
        SELECT 
            sales_channel,
            SUM(revenue) as total_revenue
        FROM sales 
        GROUP BY sales_channel 
        ORDER BY total_revenue DESC
        LIMIT 100
        """
        df = con.execute(sql).fetchdf()
        return df, "bar", "sales_channel", "total_revenue", None, sql
    
    # 優先度3: 地域別合計
    elif any(word in user_lower for word in ["地域", "region"]) and any(word in user_lower for word in ["合計", "売上", "売り上げ"]):
        sql = """
        SELECT 
            region,
            SUM(revenue) as total_revenue
        FROM sales 
        GROUP BY region 
        ORDER BY total_revenue DESC
        LIMIT 100
        """
        df = con.execute(sql).fetchdf()
        return df, "bar", "region", "total_revenue", None, sql
    
    # デフォルト: 全データの先頭50件
    else:
        sql = "SELECT * FROM sales LIMIT 50"
        df = con.execute(sql).fetchdf()
        return df, "table", None, None, None, sql

# チャート描画
def render_chart(df: pd.DataFrame, chart_type: str, x: Optional[str], y: Optional[str], series: Optional[str]) -> None:
    """指定されたチャートタイプでデータを可視化"""
    if df.empty:
        st.warning("表示するデータがありません")
        return
    
    chart_type = chart_type.lower()
    
    try:
        if chart_type == "bar":
            if x and y:
                # データ型を明示的に指定（特殊文字を含む列名対応）
                x_type = "nominal" if x in df.columns and not pd.api.types.is_numeric_dtype(df[x]) else "ordinal"
                y_type = "quantitative" if y in df.columns and pd.api.types.is_numeric_dtype(df[y]) else "nominal"
                
                if series:
                    chart = alt.Chart(df).mark_bar().encode(
                        x=alt.X(x, title=x, type=x_type),
                        y=alt.Y(y, title=y, type=y_type),
                        color=alt.Color(series, title=series),
                        tooltip=list(df.columns)
                    ).properties(width=600, height=400)
                else:
                    chart = alt.Chart(df).mark_bar().encode(
                        x=alt.X(x, title=x, type=x_type),
                        y=alt.Y(y, title=y, type=y_type),
                        tooltip=list(df.columns)
                    ).properties(width=600, height=400)
                st.altair_chart(chart, use_container_width=True)
            else:
                st.dataframe(df, use_container_width=True)
        
        elif chart_type == "line":
            if x and y:
                x_type = "temporal" if x in df.columns and pd.api.types.is_datetime64_any_dtype(df[x]) else "ordinal"
                y_type = "quantitative" if y in df.columns and pd.api.types.is_numeric_dtype(df[y]) else "nominal"
                
                if series:
                    chart = alt.Chart(df).mark_line().encode(
                        x=alt.X(x, title=x, type=x_type),
                        y=alt.Y(y, title=y, type=y_type),
                        color=alt.Color(series, title=series),
                        tooltip=list(df.columns)
                    ).properties(width=600, height=400)
                else:
                    chart = alt.Chart(df).mark_line().encode(
                        x=alt.X(x, title=x, type=x_type),
                        y=alt.Y(y, title=y, type=y_type),
                        tooltip=list(df.columns)
                    ).properties(width=600, height=400)
                st.altair_chart(chart, use_container_width=True)
            else:
                st.dataframe(df, use_container_width=True)
        
        elif chart_type == "line_point":
            if x and y:
                x_type = "temporal" if x in df.columns and pd.api.types.is_datetime64_any_dtype(df[x]) else "ordinal"
                y_type = "quantitative" if y in df.columns and pd.api.types.is_numeric_dtype(df[y]) else "nominal"
                
                if series:
                    chart = alt.Chart(df).mark_line(point=True).encode(
                        x=alt.X(x, title=x, type=x_type),
                        y=alt.Y(y, title=y, type=y_type),
                        color=alt.Color(series, title=series),
                        tooltip=list(df.columns)
                    ).properties(width=600, height=400)
                else:
                    chart = alt.Chart(df).mark_line(point=True).encode(
                        x=alt.X(x, title=x, type=x_type),
                        y=alt.Y(y, title=y, type=y_type),
                        tooltip=list(df.columns)
                    ).properties(width=600, height=400)
                st.altair_chart(chart, use_container_width=True)
            else:
                st.dataframe(df, use_container_width=True)
        
        elif chart_type == "area":
            if x and y:
                x_type = "temporal" if x in df.columns and pd.api.types.is_datetime64_any_dtype(df[x]) else "ordinal"
                y_type = "quantitative" if y in df.columns and pd.api.types.is_numeric_dtype(df[y]) else "nominal"
                
                if series:
                    chart = alt.Chart(df).mark_area().encode(
                        x=alt.X(x, title=x, type=x_type),
                        y=alt.Y(y, title=y, type=y_type),
                        color=alt.Color(series, title=series),
                        tooltip=list(df.columns)
                    ).properties(width=600, height=400)
                else:
                    chart = alt.Chart(df).mark_area().encode(
                        x=alt.X(x, title=x, type=x_type),
                        y=alt.Y(y, title=y, type=y_type),
                        tooltip=list(df.columns)
                    ).properties(width=600, height=400)
                st.altair_chart(chart, use_container_width=True)
            else:
                st.dataframe(df, use_container_width=True)
        
        elif chart_type == "pie":
            if x and y:
                chart = alt.Chart(df).mark_arc().encode(
                    theta=alt.Theta(y, title=y),
                    color=alt.Color(x, title=x),
                    tooltip=list(df.columns)
                ).properties(width=400, height=400)
                st.altair_chart(chart, use_container_width=True)
            else:
                st.dataframe(df, use_container_width=True)
        
        elif chart_type == "scatter":
            if x and y:
                x_type = "quantitative" if x in df.columns and pd.api.types.is_numeric_dtype(df[x]) else "nominal"
                y_type = "quantitative" if y in df.columns and pd.api.types.is_numeric_dtype(df[y]) else "nominal"
                
                if series:
                    chart = alt.Chart(df).mark_circle(size=60).encode(
                        x=alt.X(x, title=x, type=x_type),
                        y=alt.Y(y, title=y, type=y_type),
                        color=alt.Color(series, title=series),
                        tooltip=list(df.columns)
                    ).properties(width=600, height=400)
                else:
                    chart = alt.Chart(df).mark_circle(size=60).encode(
                        x=alt.X(x, title=x, type=x_type),
                        y=alt.Y(y, title=y, type=y_type),
                        tooltip=list(df.columns)
                    ).properties(width=600, height=400)
                st.altair_chart(chart, use_container_width=True)
            else:
                st.dataframe(df, use_container_width=True)
        
        elif chart_type == "boxplot":
            if x and y:
                chart = alt.Chart(df).mark_boxplot().encode(
                    x=alt.X(x, title=x, type="nominal"),
                    y=alt.Y(y, title=y, type="quantitative"),
                    tooltip=list(df.columns)
                ).properties(width=600, height=400)
                st.altair_chart(chart, use_container_width=True)
            else:
                st.dataframe(df, use_container_width=True)
        
        elif chart_type == "heatmap":
            if x and y:
                try:
                    # デバッグ情報を表示
                    st.write(f"🔍 デバッグ情報: x='{x}', y='{y}', series='{series}'")
                    st.write(f"📊 データの列名: {list(df.columns)}")
                    
                    # 実際の列名を確認して適切にマッピング
                    available_cols = list(df.columns)
                    
                    # x, y, seriesが実際の列名に存在するかチェック
                    actual_x = x if x in available_cols else None
                    actual_y = y if y in available_cols else None
                    actual_series = series if series and series in available_cols else None
                    
                    # 列名の自動推定
                    if not actual_x and len(available_cols) >= 1:
                        actual_x = available_cols[0]
                        st.info(f"X軸の列名を '{actual_x}' に自動設定しました")
                    
                    if not actual_y and len(available_cols) >= 2:
                        actual_y = available_cols[1]
                        st.info(f"Y軸の列名を '{actual_y}' に自動設定しました")
                    
                    if not actual_series and len(available_cols) >= 3:
                        actual_series = available_cols[2]
                        st.info(f"色の値を '{actual_series}' に自動設定しました")
                    
                    if not actual_x or not actual_y:
                        st.error("適切な列が見つかりません。表形式で表示します。")
                        st.dataframe(df, use_container_width=True)
                        return
                    
                    # ヒートマップ適性チェック
                    x_unique = df[actual_x].nunique()
                    y_unique = df[actual_y].nunique()
                    
                    st.write(f"📈 データの多様性: {actual_x}={x_unique}種類, {actual_y}={y_unique}種類")
                    
                    # X軸またはY軸の値が1種類しかない場合は棒グラフにフォールバック
                    if x_unique <= 1 or y_unique <= 1:
                        st.info(f"ヒートマップには適さないデータです（{actual_x}: {x_unique}種類, {actual_y}: {y_unique}種類）。棒グラフで表示します。")
                        if actual_series:
                            chart = alt.Chart(df).mark_bar().encode(
                                x=alt.X(actual_y, title=actual_y, type="nominal"),
                                y=alt.Y(actual_series, title=actual_series, type="quantitative"),
                                tooltip=list(df.columns)
                            ).properties(width=600, height=400)
                        else:
                            # 数値列を自動選択
                            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                            if numeric_cols:
                                value_col = numeric_cols[0]
                                chart = alt.Chart(df).mark_bar().encode(
                                    x=alt.X(actual_y, title=actual_y, type="nominal"),
                                    y=alt.Y(value_col, title=value_col, type="quantitative"),
                                    tooltip=list(df.columns)
                                ).properties(width=600, height=400)
                            else:
                                st.error("数値列が見つかりません")
                                st.dataframe(df, use_container_width=True)
                                return
                        st.altair_chart(chart, use_container_width=True)
                    else:
                        # ヒートマップ用にデータを集計
                        if actual_series:
                            # seriesが指定されている場合、x, y, seriesの組み合わせで集計
                            heatmap_data = df.groupby([actual_x, actual_y])[actual_series].sum().reset_index()
                            color_col = actual_series
                        else:
                            # seriesが指定されていない場合、x, yの組み合わせの出現回数をカウント
                            heatmap_data = df.groupby([actual_x, actual_y]).size().reset_index(name='count')
                            color_col = 'count'
                        
                        st.write(f"📊 ヒートマップ用集計データ:")
                        st.dataframe(heatmap_data.head(), use_container_width=True)
                        
                        chart = alt.Chart(heatmap_data).mark_rect().encode(
                            x=alt.X(actual_x, title=actual_x, type="ordinal"),
                            y=alt.Y(actual_y, title=actual_y, type="ordinal"), 
                            color=alt.Color(color_col, title=color_col, type="quantitative", scale=alt.Scale(scheme='blues')),
                            tooltip=[actual_x, actual_y, color_col]
                        ).properties(width=600, height=400)
                        st.altair_chart(chart, use_container_width=True)
                except Exception as e:
                    st.warning(f"ヒートマップ生成エラー: {e}")
                    st.error(f"詳細エラー: {str(e)}")
                    st.dataframe(df, use_container_width=True)
            else:
                st.dataframe(df, use_container_width=True)
        
        elif chart_type == "errorbar":
            if x and y:
                # エラーバー表示（標準偏差を計算）
                chart = alt.Chart(df).mark_errorbar(extent='stdev').encode(
                    x=alt.X(x, title=x, type="nominal"),
                    y=alt.Y(y, title=y, type="quantitative"),
                    tooltip=list(df.columns)
                ).properties(width=600, height=400)
                st.altair_chart(chart, use_container_width=True)
            else:
                st.dataframe(df, use_container_width=True)
        
        elif chart_type == "rule":
            if x and y:
                # 基準線・回帰線
                chart = alt.Chart(df).mark_rule().encode(
                    x=alt.X(x, title=x),
                    y=alt.Y(y, title=y),
                    tooltip=list(df.columns)
                ).properties(width=600, height=400)
                st.altair_chart(chart, use_container_width=True)
            else:
                st.dataframe(df, use_container_width=True)
        
        elif chart_type == "density":
            if x and y:
                # 密度プロット（transform_densityを使用）
                chart = alt.Chart(df).transform_density(
                    y,
                    as_=[y, 'density']
                ).mark_area().encode(
                    x=alt.X(f'{y}:Q', title=y),
                    y=alt.Y('density:Q', title='Density'),
                    tooltip=['density:Q']
                ).properties(width=600, height=400)
                st.altair_chart(chart, use_container_width=True)
            else:
                st.dataframe(df, use_container_width=True)
        
        else:  # table or unknown
            st.dataframe(df, use_container_width=True)
            
    except Exception as e:
        st.warning(f"チャート描画エラー: {e}")
        st.dataframe(df, use_container_width=True)

# システムプロンプト
def get_system_prompt(schema_text: str) -> str:
    """システムプロンプトを生成"""
    return f"""あなたは売上データ分析のアシスタントです。ユーザーの自然言語質問に対して、DuckDBのSELECT文を生成し、適切な可視化方法を提案してください。

{schema_text}

ルール:
- SELECT文のみ生成（DDL/DML禁止）
- LIMIT 100 を推奨
- 月次集計: strftime(date,'%Y-%m') または date_trunc('month', date) を使用
- **集計列には必ずAS別名を付ける** (例: SUM(revenue) AS total_revenue)
- x/y/seriesパラメータには別名を指定する
- 回答は必ずrun_sqlツールを使用

代表例:
1. 月別・カテゴリ別売上: SELECT strftime(date,'%Y-%m') AS month, category, SUM(revenue) AS total_revenue FROM sales GROUP BY month, category ORDER BY month, category
2. チャネル別売上: SELECT sales_channel, SUM(revenue) AS total_revenue FROM sales GROUP BY sales_channel ORDER BY total_revenue DESC
3. 地域別売上: SELECT region, SUM(revenue) AS total_revenue FROM sales GROUP BY region ORDER BY total_revenue DESC

可視化タイプ: bar, line, line_point, area, pie, scatter, boxplot, heatmap, errorbar, rule, density, table
- bar: 棒グラフ（カテゴリ別比較、x=カテゴリ、y=数値）
- line: 線グラフ（時系列、x=時間、y=数値）
- line_point: 折れ線+ポイント（時系列詳細）
- area: エリアグラフ（トレンド/積み上げ）
- pie: 円グラフ（構成比、x=分類、y=数値必須）
- scatter: 散布図（x,y=数値）
- boxplot: 箱ひげ図（分布、x=カテゴリ、y=数値）
- heatmap: ヒートマップ（2D強度、x,y=カテゴリ、color=値）
- errorbar: エラーバー（平均±CI等）
- rule: 基準線・回帰線
- density: 密度プロット（分布滑らか表示）
- table: 表（デフォルト）

必ず日本語で explanation を提供してください。"""

# ツール定義
def get_tools() -> list:
    """OpenAI関数ツール定義を返す"""
    return [
        {
            "type": "function",
            "function": {
                "name": "run_sql",
                "description": "DuckDBでSQLを実行し、結果を可視化する",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sql": {
                            "type": "string",
                            "description": "実行するSELECT文"
                        },
                        "chart": {
                            "type": "string",
                            "description": "可視化タイプ",
                            "enum": ["bar", "line", "line_point", "area", "pie", "scatter", "boxplot", "heatmap", "errorbar", "rule", "density", "table"]
                        },
                        "x": {
                            "type": "string",
                            "description": "X軸の列名（チャート用）"
                        },
                        "y": {
                            "type": "string", 
                            "description": "Y軸の列名（チャート用）"
                        },
                        "series": {
                            "type": "string",
                            "description": "系列分けの列名（オプション）"
                        },
                        "explanation": {
                            "type": "string",
                            "description": "結果の日本語説明"
                        }
                    },
                    "required": ["sql", "chart", "explanation"]
                }
            }
        }
    ]

# メイン処理
def main():
    st.title("📊 売上データ分析AIチャットボット")
    st.markdown("自然言語で売上データについて質問してください。AIがSQLを生成して分析結果を表示します。")
    
    # データ初期化
    if 'sales_df' not in st.session_state:
        st.session_state.sales_df = load_sales_df()
    
    if 'duckdb_con' not in st.session_state:
        st.session_state.duckdb_con = get_duckdb_connection(st.session_state.sales_df)
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # OpenAIクライアント
    client = init_openai_client()
    if not client:
        st.stop()
    
    df = st.session_state.sales_df
    con = st.session_state.duckdb_con
    
    if df.empty:
        st.error("売上データが読み込まれていません")
        st.stop()
    
    # サイドバー：データ概要
    with st.sidebar:
        st.header("📋 データ概要")
        st.metric("総レコード数", len(df))
        st.metric("期間", f"{df['date'].min().strftime('%Y-%m-%d')} ～ {df['date'].max().strftime('%Y-%m-%d')}")
        st.metric("カテゴリ数", df['category'].nunique())
        st.metric("地域数", df['region'].nunique()) 
        st.metric("販売チャネル数", df['sales_channel'].nunique())
        
        with st.expander("カテゴリ一覧"):
            st.write(df['category'].unique().tolist())
        
        with st.expander("地域一覧"):
            st.write(df['region'].unique().tolist())
        
        with st.expander("販売チャネル一覧"):
            st.write(df['sales_channel'].unique().tolist())
    
    # チャット履歴表示
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.write(msg["content"])
        else:
            with st.chat_message("assistant"):
                st.write(msg["content"])
    
    # ユーザー入力
    if user_input := st.chat_input("例: 月別カテゴリ別の売上を教えて"):
        # ユーザーメッセージ表示・保存
        with st.chat_message("user"):
            st.write(user_input)
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # AI応答処理
        with st.chat_message("assistant"):
            with st.spinner("分析中..."):
                try:
                    # OpenAI API呼び出し
                    schema_text = get_schema(con)
                    system_prompt = get_system_prompt(schema_text)
                    tools = get_tools()
                    
                    messages = [{"role": "system", "content": system_prompt}]
                    for msg in st.session_state.chat_history[-10:]:  # 直近10件のみ
                        if msg["role"] in ["user", "assistant"]:
                            messages.append(msg)
                    
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=messages,
                        tools=tools,
                        tool_choice="auto",
                        temperature=0.1
                    )
                    
                    # ツール呼び出し処理
                    if response.choices[0].message.tool_calls:
                        tool_call = response.choices[0].message.tool_calls[0]
                        if tool_call.function.name == "run_sql":
                            args = json.loads(tool_call.function.arguments)
                            sql = args.get("sql", "")
                            chart_type = args.get("chart", "table")
                            x = args.get("x")
                            y = args.get("y") 
                            series = args.get("series")
                            explanation = args.get("explanation", "分析結果を表示しています。")
                            
                            try:
                                # SQL実行
                                result_df = safe_execute_sql(con, sql)
                                
                                # 結果表示
                                st.write(f"**分析結果**: {explanation}")
                                
                                with st.expander("実行されたSQL", expanded=False):
                                    st.code(sql, language="sql")
                                
                                st.subheader("📊 データ")
                                st.dataframe(result_df, use_container_width=True)
                                
                                st.subheader("📈 可視化")
                                render_chart(result_df, chart_type, x, y, series)
                                
                                # 履歴に保存
                                st.session_state.chat_history.append({
                                    "role": "assistant", 
                                    "content": explanation
                                })
                                
                            except Exception as e:
                                st.error(f"SQL実行エラー: {e}")
                                # フォールバック処理
                                result_df, chart_type, x, y, series, sql = fallback_answer(user_input)
                                explanation = "エラーが発生したため、代替の分析結果を表示しています。"
                                
                                st.write(f"**フォールバック結果**: {explanation}")
                                
                                with st.expander("実行されたSQL", expanded=False):
                                    st.code(sql, language="sql")
                                
                                st.subheader("📊 データ")
                                st.dataframe(result_df, use_container_width=True)
                                
                                st.subheader("📈 可視化")
                                render_chart(result_df, chart_type, x, y, series)
                                
                                st.session_state.chat_history.append({
                                    "role": "assistant",
                                    "content": explanation
                                })
                        else:
                            # 想定外のツール呼び出し
                            raise ValueError("予期しないツール呼び出し")
                    
                    else:
                        # ツール呼び出しなし→フォールバック
                        raise ValueError("ツール呼び出しなし")
                        
                except Exception as e:
                    st.error(f"処理エラー: {e}")
                    # フォールバック処理
                    result_df, chart_type, x, y, series, sql = fallback_answer(user_input)
                    explanation = "エラーが発生したため、代替の分析結果を表示しています。"
                    
                    st.write(f"**フォールバック結果**: {explanation}")
                    
                    with st.expander("実行されたSQL", expanded=False):
                        st.code(sql, language="sql")
                    
                    st.subheader("📊 データ")
                    st.dataframe(result_df, use_container_width=True)
                    
                    st.subheader("📈 可視化")
                    render_chart(result_df, chart_type, x, y, series)
                    
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": explanation
                    })

if __name__ == "__main__":
    main()
