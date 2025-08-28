import streamlit as st
import pandas as pd
import plotly.express as px

# アプリのタイトルと説明
st.title('Plotly基礎')
st.write('日毎の売上推移を折れ線グラフで確認しましょう！')

# CSVファイルを読み込む
df = pd.read_csv('data/sample_sales.csv')

# 日付型に変換（必要な場合）
df['date'] = pd.to_datetime(df['date'])

# 日毎の売上合計を集計
daily_revenue = df.groupby('date')['revenue'].sum().reset_index()

st.subheader('日毎の売上推移（折れ線グラフ）')

# 折れ線グラフを作成（赤色）
fig = px.line(
    daily_revenue,
    x='date',
    y='revenue',
    title='日毎の売上推移',
    labels={'date': '日付', 'revenue': '売上 (円)'},
)

# 線の色を赤に変更
fig.update_traces(line=dict(color='red'))

# グラフを表示
st.plotly_chart(fig)

st.write('---')
st.write('この折れ線グラフにより、売上の時系列的な変動が視覚的に把握できます。')
