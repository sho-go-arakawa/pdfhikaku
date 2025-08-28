import streamlit as st
import pandas as pd
import plotly.express as px

# アプリのタイトルと説明
st.title('Plotly基礎')
st.write('Plotlyを使ってインタラクティブなグラフを作成してみましょう！')

df = pd.read_csv('data/sample_sales.csv', parse_dates=['date']) 

st.subheader('カテゴリ別合計売上グラフ')

category_revenue = df.groupby('category')['revenue'].sum().reset_index()

fig = px.bar(
    category_revenue,
    x='category',  
    y='revenue',
    title='商品カテゴリごとの総売上',
    labels={'category': '商品カテゴリ', 'revenue': '総売上 (円)'}
)

st.plotly_chart(fig)

st.write('---')
st.write('このグラフはインタラクティブです！特定のカテゴリにカーソルを合わせると、そのカテゴリの正確な総売上が表示されます。')