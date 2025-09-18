import streamlit as st
from ..services import category_service
from ..models import CategoryCreate, CategoryUpdate


def render_category_management():
    st.header("📁 カテゴリ管理")
    
    tab1, tab2 = st.tabs(["カテゴリ一覧", "新規作成"])
    
    with tab1:
        render_category_list()
    
    with tab2:
        render_category_form()


def render_category_list():
    categories = category_service.get_all_categories()
    
    if not categories:
        st.info("カテゴリがありません。")
        return
    
    st.subheader(f"全 {len(categories)} 件のカテゴリ")
    
    for category in categories:
        with st.container():
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                color_indicator = f"🔵" if not category.color else f"<span style='color: {category.color}'>●</span>"
                st.markdown(f"### {color_indicator} {category.name}", unsafe_allow_html=True)
                if category.description:
                    st.markdown(f"*{category.description}*")
            
            with col2:
                if st.button("✏️ 編集", key=f"edit_cat_{category.id}"):
                    st.session_state.edit_category_id = category.id
                    st.session_state.show_category_edit = True
            
            with col3:
                if st.button("🗑️ 削除", key=f"delete_cat_{category.id}"):
                    if st.session_state.get(f"confirm_delete_cat_{category.id}", False):
                        category_service.delete_category(category.id)
                        st.success("カテゴリを削除しました")
                        st.rerun()
                    else:
                        st.session_state[f"confirm_delete_cat_{category.id}"] = True
                        st.warning("もう一度クリックして削除を確認してください")
            
            st.divider()
    
    if st.session_state.get('show_category_edit', False):
        render_category_edit_form()


def render_category_form():
    st.subheader("新しいカテゴリを作成")
    
    with st.form("category_form"):
        name = st.text_input("カテゴリ名 *", max_chars=100)
        description = st.text_area("説明", height=100)
        color = st.color_picker("カラー", value="#1f77b4")
        
        submitted = st.form_submit_button("作成")
        
        if submitted:
            if not name.strip():
                st.error("カテゴリ名は必須です")
                return
            
            try:
                category_create = CategoryCreate(
                    name=name.strip(),
                    description=description.strip() if description.strip() else None,
                    color=color
                )
                category_service.create_category(category_create)
                st.success("カテゴリを作成しました")
                st.rerun()
            except Exception as e:
                st.error(f"エラーが発生しました: {str(e)}")


def render_category_edit_form():
    category_id = st.session_state.get('edit_category_id')
    if not category_id:
        return
    
    category = category_service.get_category_by_id(category_id)
    if not category:
        st.error("カテゴリが見つかりません")
        return
    
    st.subheader(f"カテゴリ編集: {category.name}")
    
    with st.form("category_edit_form"):
        name = st.text_input("カテゴリ名 *", value=category.name, max_chars=100)
        description = st.text_area(
            "説明", 
            value=category.description if category.description else "",
            height=100
        )
        color = st.color_picker("カラー", value=category.color if category.color else "#1f77b4")
        
        col1, col2 = st.columns(2)
        
        with col1:
            submitted = st.form_submit_button("更新")
        
        with col2:
            cancelled = st.form_submit_button("キャンセル")
        
        if cancelled:
            st.session_state.show_category_edit = False
            if 'edit_category_id' in st.session_state:
                del st.session_state.edit_category_id
            st.rerun()
        
        if submitted:
            if not name.strip():
                st.error("カテゴリ名は必須です")
                return
            
            try:
                category_update = CategoryUpdate(
                    name=name.strip(),
                    description=description.strip() if description.strip() else None,
                    color=color
                )
                category_service.update_category(category_id, category_update)
                st.success("カテゴリを更新しました")
                
                st.session_state.show_category_edit = False
                if 'edit_category_id' in st.session_state:
                    del st.session_state.edit_category_id
                st.rerun()
            except Exception as e:
                st.error(f"エラーが発生しました: {str(e)}")