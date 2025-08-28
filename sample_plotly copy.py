import streamlit as st
import pandas as pd
import plotly.express as px

# アプリのタイトルと説明
st.title('Plotly基礎')
st.write('Plotlyを使ってインタラクティブなグラフを作成してみましょう！')

# CSVファイルを読み込む
df = pd.read_csv('data/sample_sales.csv')

st.subheader('地域・カテゴリ別の合計売上グラフ')

# 地域とカテゴリごとの合計売上を計算
region_category_revenue = df.groupby(['region', 'category'])['revenue'].sum().reset_index()

# Plotlyで棒グラフを作成（カテゴリごとに色分け）
fig = px.bar(
    region_category_revenue,
    x='region',
    y='revenue',
    color='category',
    title='地域・カテゴリ別の総売上',
    labels={'region': '地域', 'revenue': '総売上 (円)', 'category': 'カテゴリ'},
    barmode='group'  # 'stack' にすると積み上げ棒グラフにもできます
)

# Streamlitにグラフを表示
st.plotly_chart(fig)

st.write('---')
st.write('このグラフはインタラクティブです！カテゴリごとの色分けと、地域別の比較が可能です。')