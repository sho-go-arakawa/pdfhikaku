import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import plotly.express as px
import altair as alt

# ==== 可視化：カテゴリ・地域別の横棒グラフ ====
def show_bar_charts(df: pd.DataFrame) -> None:
    """カテゴリ別・地域別の売上構成（横棒グラフ）を表示。"""
    st.subheader("🗂️ カテゴリ別・地域別売上")

    if df.empty:
        st.info("該当データがありません。")
        return

    col1, col2 = st.columns(2)

    with col1:
        cat_sales = df.groupby("category", as_index=False)["revenue"].sum().sort_values("revenue", ascending=False)
        fig_cat = px.bar(
            cat_sales,
            x="revenue",
            y="category",
            orientation="h",
            labels={"revenue": "売上", "category": "カテゴリ"},
            title="カテゴリ別売上",
            text="revenue",
        )
        fig_cat.update_traces(texttemplate="¥%{text:,}", textposition="outside")
        fig_cat.update_layout(yaxis=dict(tickfont=dict(size=12)), height=400)
        st.plotly_chart(fig_cat, use_container_width=True)

    with col2:
        reg_sales = df.groupby("region", as_index=False)["revenue"].sum().sort_values("revenue", ascending=False)
        fig_reg = px.bar(
            reg_sales,
            x="revenue",
            y="region",
            orientation="h",
            labels={"revenue": "売上", "region": "地域"},
            title="地域別売上",
            text="revenue",
        )
        fig_reg.update_traces(texttemplate="¥%{text:,}", textposition="outside")
        fig_reg.update_layout(yaxis=dict(tickfont=dict(size=12)), height=400)
        st.plotly_chart(fig_reg, use_container_width=True)

# ==== 定数設定 ====
PATH_DATA = Path("data/sample_sales.csv")
REVENUE_TOLERANCE = 1  # 円単位で誤差許容

# ==== ページ基本設定 ====
st.set_page_config(page_title="Sales BI Dashboard", layout="wide", page_icon="📊")

# ==== 共通フォーマッタ ====
def fmt_currency(x: int) -> str:
    return f"¥{x:,}"

def fmt_int(x: int) -> str:
    return f"{x:,}個"

# ==== データ読み込み ====
@st.cache_data(ttl=600)
def load_data(path: Path) -> pd.DataFrame:
    """CSVファイルを読み込み、日付型や型変換、前処理を行う。"""
    try:
        df = pd.read_csv(
            path,
            parse_dates=["date"],
            dtype={
                "category": "string",
                "units": "Int64",
                "unit_price": "Int64",
                "region": "string",
                "sales_channel": "string",
                "customer_segment": "string",
                "revenue": "Int64",
            }
        )
        return df
    except FileNotFoundError:
        st.error("❌ データファイルが見つかりません。`data/sample_sales.csv` を配置してください。")
        st.stop()

# ==== 前処理 ====
def preprocess(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    欠損・不正値・売上整合チェックを実施。整合不良データは別に抽出。
    """
    df = df.copy()
    df["calc_revenue"] = df["units"] * df["unit_price"]
    df["revenue_diff"] = (df["revenue"] - df["calc_revenue"]).abs()

    # 整合性チェック（理論売上との乖離）
    df["inconsistent"] = df["revenue_diff"] > REVENUE_TOLERANCE
    inconsistent_df = df[df["inconsistent"]]

    # 負値・欠損チェック
    df = df[df["units"].notna() & df["unit_price"].notna()]
    df = df[(df["units"] >= 0) & (df["unit_price"] >= 0) & (df["revenue"] >= 0)]

    return df, inconsistent_df

# ==== フィルタ適用 ====
def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    """サイドバーからのユーザー入力に基づいてフィルタを適用。"""
    st.sidebar.header("📅 フィルタ")

    min_date, max_date = df["date"].min(), df["date"].max()
    start_date, end_date = st.sidebar.date_input("期間を選択", [min_date, max_date])

    if isinstance(start_date, tuple):
        start_date, end_date = start_date

    category = st.sidebar.multiselect("カテゴリ", options=sorted(df["category"].dropna().unique()))
    region = st.sidebar.multiselect("地域", options=sorted(df["region"].dropna().unique()))
    channel = st.sidebar.multiselect("チャネル", options=sorted(df["sales_channel"].dropna().unique()))
    segment = st.sidebar.multiselect("セグメント", options=sorted(df["customer_segment"].dropna().unique()))

    df_filtered = df[
        (df["date"] >= pd.to_datetime(start_date)) &
        (df["date"] <= pd.to_datetime(end_date))
    ]

    if category:
        df_filtered = df_filtered[df_filtered["category"].isin(category)]
    if region:
        df_filtered = df_filtered[df_filtered["region"].isin(region)]
    if channel:
        df_filtered = df_filtered[df_filtered["sales_channel"].isin(channel)]
    if segment:
        df_filtered = df_filtered[df_filtered["customer_segment"].isin(segment)]

    return df_filtered
# ==== 可視化：時系列トレンドグラフ（Altair） ====
def show_time_series(df: pd.DataFrame) -> None:
    """売上・数量の時系列推移グラフ（ロールアップ切替）を表示。"""
    st.subheader("📈 時系列トレンド（売上・数量）")

    if df.empty:
        st.info("該当データがありません。")
        return

    rollup = st.radio("集計粒度", ["日次", "週次", "月次"], horizontal=True)

    if rollup == "日次":
        df_trend = df.copy()
        df_trend["period"] = df_trend["date"]
    elif rollup == "週次":
        df_trend = df.copy()
        df_trend["period"] = df_trend["date"].dt.to_period("W").apply(lambda r: r.start_time)
    elif rollup == "月次":
        df_trend = df.copy()
        df_trend["period"] = df_trend["date"].dt.to_period("M").dt.to_timestamp()

    df_grouped = df_trend.groupby("period").agg({
        "revenue": "sum",
        "units": "sum"
    }).reset_index()

    base = alt.Chart(df_grouped).encode(
        x=alt.X("period:T", title="日付")
    )

    chart1 = base.mark_line(color="#1f77b4").encode(
        y=alt.Y("revenue:Q", title="売上"),
        tooltip=[alt.Tooltip("period:T"), alt.Tooltip("revenue:Q", format=",d")]
    ).properties(title="売上推移")

    chart2 = base.mark_line(color="#ff7f0e").encode(
        y=alt.Y("units:Q", title="数量"),
        tooltip=[alt.Tooltip("period:T"), alt.Tooltip("units:Q", format=",d")]
    ).properties(title="販売数量推移")

    st.altair_chart(chart1.interactive(), use_container_width=True)
    st.altair_chart(chart2.interactive(), use_container_width=True)

# ==== KPIカード ====
def make_kpis(df: pd.DataFrame) -> None:
    """主要KPI（総売上・数量・平均単価など）を表示。"""
    total_revenue = int(df["revenue"].sum())
    total_units = int(df["units"].sum())
    avg_unit_price = int(total_revenue / total_units) if total_units > 0 else 0
    n_categories = df["category"].nunique()
    n_segments = df["customer_segment"].nunique()

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("総売上", fmt_currency(total_revenue))
    col2.metric("販売数量", fmt_int(total_units))
    col3.metric("平均単価", fmt_currency(avg_unit_price))
    col4.metric("カテゴリ数", f"{n_categories}件")
    col5.metric("セグメント数", f"{n_segments}件")

# ==== 明細テーブル ====
def show_table(df: pd.DataFrame) -> None:
    """フィルタ後の明細データをテーブル表示し、CSVダウンロード提供。"""
    st.subheader("🧾 明細テーブル")

    if df.empty:
        st.info("該当データがありません。")
        return

    df_disp = df[[
        "date", "category", "region", "sales_channel", "customer_segment",
        "units", "unit_price", "revenue"
    ]].copy()

    df_disp["unit_price"] = df_disp["unit_price"].apply(fmt_currency)
    df_disp["revenue"] = df_disp["revenue"].apply(fmt_currency)

    st.dataframe(df_disp, use_container_width=True)

    csv = df.to_csv(index=False)
    st.download_button("CSVダウンロード", data=csv, file_name="filtered_sales.csv", mime="text/csv")

# ==== アプリ本体 ====
def main() -> None:
    """Streamlitダッシュボードアプリのメイン関数。"""
    st.title("📊 Sales BI ダッシュボード")

    df_raw = load_data(PATH_DATA)
    df, inconsistent_df = preprocess(df_raw)

    # 整合性警告表示
    if not inconsistent_df.empty:
        with st.sidebar.expander("⚠️ 整合性チェック"):
            st.warning(f"{len(inconsistent_df)}件のデータに売上整合性の問題があります。")
            st.write(f"影響額合計: {fmt_currency(int(inconsistent_df['revenue_diff'].sum()))}")
            csv = inconsistent_df.to_csv(index=False)
            st.download_button("不整合データをダウンロード", data=csv, file_name="inconsistent_rows.csv", mime="text/csv")

    # フィルタ適用
    df_filtered = apply_filters(df)

    # KPI表示
    make_kpis(df_filtered)

    # チャート表示
    show_time_series(df_filtered)
    show_bar_charts(df_filtered)


    # 明細テーブル
    show_table(df_filtered)

if __name__ == "__main__":
    main()
