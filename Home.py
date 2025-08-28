import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Streamlit BI x Claude Code Starter", layout="wide")

st.title("Streamlit BI x Claude Code Starter")
@st.cache_data
def load_data():
    try:
        orders_df = pd.read_csv("sample_data/orders.csv")
        users_df = pd.read_csv("sample_data/users.csv")
        return orders_df, users_df
    except FileNotFoundError as e:
        st.error(f"データファイルが見つかりません: {e}")
        return pd.DataFrame(), pd.DataFrame()
    except Exception as e:
        st.error(f"データ読み込みエラー: {e}")
        return pd.DataFrame(), pd.DataFrame()

@st.cache_data
def process_monthly_analysis(orders_df):
    try:
        if orders_df.empty:
            return pd.DataFrame()
        
        orders_df['created_at'] = pd.to_datetime(orders_df['created_at'], errors='coerce')
        orders_df = orders_df.dropna(subset=['created_at'])
        orders_df['year_month'] = orders_df['created_at'].dt.to_period('M')
        
        monthly_orders = orders_df.groupby('year_month').size()
        monthly_cancelled = orders_df[orders_df['status'] == 'Cancelled'].groupby('year_month').size()
        cancel_rate = (monthly_cancelled / monthly_orders * 100).fillna(0)
        
        monthly_analysis = pd.DataFrame({
            'month': monthly_orders.index.astype(str),
            'total_orders': monthly_orders.values,
            'cancelled_orders': monthly_cancelled.reindex(monthly_orders.index, fill_value=0).values,
            'cancel_rate': cancel_rate.values
        })
        
        return monthly_analysis
    except Exception as e:
        st.error(f"月別分析処理エラー: {e}")
        return pd.DataFrame()

def create_monthly_charts(monthly_data):
    try:
        if monthly_data.empty:
            st.warning("表示するデータがありません")
            return None, None
        
        fig_orders = px.bar(
            monthly_data, 
            x='month', 
            y='total_orders',
            title='月別オーダー数',
            labels={'month': '月', 'total_orders': 'オーダー数'},
            color_discrete_sequence=['#1f77b4']
        )
        fig_orders.update_layout(xaxis_tickangle=-45)
        
        fig_cancel = px.line(
            monthly_data, 
            x='month', 
            y='cancel_rate',
            title='月別キャンセル率',
            labels={'month': '月', 'cancel_rate': 'キャンセル率 (%)'},
            markers=True,
            color_discrete_sequence=['#ff7f0e']
        )
        fig_cancel.update_layout(xaxis_tickangle=-45)
        fig_cancel.update_yaxis(title="キャンセル率 (%)")
        
        return fig_orders, fig_cancel
    except Exception as e:
        st.error(f"チャート作成エラー: {e}")
        return None, None

orders_df, users_df = load_data()

if not orders_df.empty:
    monthly_data = process_monthly_analysis(orders_df)
    
    if not monthly_data.empty:
        st.header("📊 月別分析ダッシュボード")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            total_orders = monthly_data['total_orders'].sum()
            st.metric("総オーダー数", f"{total_orders:,}")
        with col2:
            total_cancelled = monthly_data['cancelled_orders'].sum()
            st.metric("総キャンセル数", f"{total_cancelled:,}")
        with col3:
            avg_cancel_rate = monthly_data['cancel_rate'].mean()
            st.metric("平均キャンセル率", f"{avg_cancel_rate:.1f}%")
        
        fig_orders, fig_cancel = create_monthly_charts(monthly_data)
        
        if fig_orders and fig_cancel:
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(fig_orders, use_container_width=True)
            with col2:
                st.plotly_chart(fig_cancel, use_container_width=True)
    
    st.header("📋 データサンプル")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Orders Data (Top 10 rows)")
        st.dataframe(orders_df.head(10))
    with col2:
        st.subheader("Users Data (Top 10 rows)")
        st.dataframe(users_df.head(10))
else:
    st.error("データを読み込めませんでした。データファイルを確認してください。")
