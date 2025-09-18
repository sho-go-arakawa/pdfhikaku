import streamlit as st
from ..services import tag_service
from ..models import TagCreate, TagUpdate


def render_tag_management():
    st.header("🏷️ タグ管理")
    
    tab1, tab2 = st.tabs(["タグ一覧", "新規作成"])
    
    with tab1:
        render_tag_list()
    
    with tab2:
        render_tag_form()


def render_tag_list():
    tags = tag_service.get_all_tags()
    
    if not tags:
        st.info("タグがありません。")
        return
    
    st.subheader(f"全 {len(tags)} 件のタグ")
    
    cols = st.columns(3)
    for i, tag in enumerate(tags):
        col = cols[i % 3]
        
        with col:
            with st.container():
                color_indicator = f"🏷️" if not tag.color else f"<span style='color: {tag.color}'>🏷️</span>"
                st.markdown(f"{color_indicator} **{tag.name}**", unsafe_allow_html=True)
                
                edit_col, delete_col = st.columns(2)
                
                with edit_col:
                    if st.button("✏️", key=f"edit_tag_{tag.id}", help="編集"):
                        st.session_state.edit_tag_id = tag.id
                        st.session_state.show_tag_edit = True
                
                with delete_col:
                    if st.button("🗑️", key=f"delete_tag_{tag.id}", help="削除"):
                        if st.session_state.get(f"confirm_delete_tag_{tag.id}", False):
                            tag_service.delete_tag(tag.id)
                            st.success("タグを削除しました")
                            st.rerun()
                        else:
                            st.session_state[f"confirm_delete_tag_{tag.id}"] = True
                            st.warning("もう一度クリックして削除を確認")
                
                st.markdown("---")
    
    if st.session_state.get('show_tag_edit', False):
        render_tag_edit_form()


def render_tag_form():
    st.subheader("新しいタグを作成")
    
    with st.form("tag_form"):
        name = st.text_input("タグ名 *", max_chars=50)
        color = st.color_picker("カラー", value="#ff6b6b")
        
        submitted = st.form_submit_button("作成")
        
        if submitted:
            if not name.strip():
                st.error("タグ名は必須です")
                return
            
            try:
                tag_create = TagCreate(
                    name=name.strip(),
                    color=color
                )
                tag_service.create_tag(tag_create)
                st.success("タグを作成しました")
                st.rerun()
            except Exception as e:
                st.error(f"エラーが発生しました: {str(e)}")


def render_tag_edit_form():
    tag_id = st.session_state.get('edit_tag_id')
    if not tag_id:
        return
    
    tag = tag_service.get_tag_by_id(tag_id)
    if not tag:
        st.error("タグが見つかりません")
        return
    
    st.subheader(f"タグ編集: {tag.name}")
    
    with st.form("tag_edit_form"):
        name = st.text_input("タグ名 *", value=tag.name, max_chars=50)
        color = st.color_picker("カラー", value=tag.color if tag.color else "#ff6b6b")
        
        col1, col2 = st.columns(2)
        
        with col1:
            submitted = st.form_submit_button("更新")
        
        with col2:
            cancelled = st.form_submit_button("キャンセル")
        
        if cancelled:
            st.session_state.show_tag_edit = False
            if 'edit_tag_id' in st.session_state:
                del st.session_state.edit_tag_id
            st.rerun()
        
        if submitted:
            if not name.strip():
                st.error("タグ名は必須です")
                return
            
            try:
                tag_update = TagUpdate(
                    name=name.strip(),
                    color=color
                )
                tag_service.update_tag(tag_id, tag_update)
                st.success("タグを更新しました")
                
                st.session_state.show_tag_edit = False
                if 'edit_tag_id' in st.session_state:
                    del st.session_state.edit_tag_id
                st.rerun()
            except Exception as e:
                st.error(f"エラーが発生しました: {str(e)}")