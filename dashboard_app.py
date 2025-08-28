import streamlit as st
import pandas as pd
import plotly.express as px

# ---------------------------------------------
# データ読み込み関数
# ---------------------------------------------
@st.cache_data
def load_data(filepath="data/sample_sales.csv"):
    df = pd.read_csv(filepath)
    return df

# ---------------------------------------------
# データ前処理関数
# ---------------------------------------------
def preprocess_data(df):
    df = df.copy()

    # 日付型に変換
    df["date"] = pd.to_datetime(df["date"])

    # 欠損値処理（基本はドロップ）
    df.dropna(inplace=True)

    # 重複行削除
    df.drop_duplicates(inplace=True)

    # revenueが正しいか確認列（内部チェック用）
    df["calc_revenue"] = df["units"] * df["unit_price"]
    df["revenue_check"] = df["calc_revenue"] == df["revenue"]

    return df

# ---------------------------------------------
# KPI表示用関数
# ---------------------------------------------
def display_kpis(df):
    total_revenue = df["revenue"].sum()
    total_units = df["units"].sum()
    category_count = df["category"].nunique()

    col1, col2, col3 = st.columns(3)
    col1.metric("💰 売上合計", f"¥{total_revenue:,.0f}")
    col2.metric("📦 販売数量合計", f"{total_units:,}")
    col3.metric("🗂 商品カテゴリ数", f"{category_count}")

# ---------------------------------------------
# 可視化関数
# ---------------------------------------------
def plot_category_sales(df):
    category_sales = df.groupby("category")["revenue"].sum().reset_index()
    fig = px.bar(category_sales, x="category", y="revenue",
                 title="カテゴリ別売上", labels={"revenue": "売上", "category": "カテゴリ"})
    st.plotly_chart(fig, use_container_width=True)

def plot_daily_sales(df):
    daily_sales = df.groupby("date")["revenue"].sum().reset_index()
    fig = px.line(daily_sales, x="date", y="revenue",
                  title="日別売上推移", labels={"revenue": "売上", "date": "日付"})
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------
# メイン関数
# ---------------------------------------------
def main():
    st.set_page_config(page_title="販売データBIダッシュボード", layout="wide")
    st.title("📊 販売データBIダッシュボード")

    # データ読み込み＆前処理
    df = load_data()
    df = preprocess_data(df)

    # サイドバー：日付フィルター
    st.sidebar.header("🔎 フィルター")
    min_date = df["date"].min()
    max_date = df["date"].max()

    date_range = st.sidebar.date_input("日付範囲を選択", [min_date, max_date], min_value=min_date, max_value=max_date)

    # フィルタリング処理
    if len(date_range) == 2:
        start_date, end_date = date_range
        filtered_df = df[(df["date"] >= pd.to_datetime(start_date)) & (df["date"] <= pd.to_datetime(end_date))]
    else:
        filtered_df = df.copy()

    # KPI表示
    display_kpis(filtered_df)

    # グラフ表示
    st.markdown("---")
    plot_category_sales(filtered_df)

    st.markdown("---")
    plot_daily_sales(filtered_df)

# ---------------------------------------------
# アプリ起動
# ---------------------------------------------
if __name__ == "__main__":
    main()
