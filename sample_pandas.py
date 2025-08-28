import streamlit as st
import pandas as pd

st.title('Pandas基礎')
st.write('Pandasを使ってCSVファイルを読み込み、表示してみましょう！')

# CSVファイルを読み込む
try:
    df = pd.read_csv('data/sample_sales.csv')
    st.success('CSVファイルの読み込みに成功しました！')

    # 読み込んだデータの最初の5行を表示する
    st.subheader('データの最初の5行')
    st.dataframe(df.head())

    # 各列の統計情報を確認する
    st.subheader('各列の統計情報')
    st.dataframe(df.describe())

except FileNotFoundError:
    st.error('data/sample_sales.csv が見つかりません。ファイルパスを確認してください。')
except Exception as e:
    st.error(f'データの読み込み中にエラーが発生しました: {e}')

st.write('---')
st.write('これで、Pandasを使ってCSVファイルを読み込み、Streamlitで表示することができましたね！')