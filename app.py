import streamlit as st
import pandas as pd
from pipeline import PipelineOrchestrator

# Cấu hình trang
st.set_page_config(page_title="DS2 Store Assistant", page_icon="", layout="wide")

st.title("DS2 Store - Text-to-SQL & Recommendation")

# Cache PipelineOrchestrator để không bị khởi tạo lại mỗi lần chạy Streamlit
@st.cache_resource(show_spinner="Đang khởi tạo hệ thống (load models)...")
def load_pipeline():
    # Khởi tạo pipeline
    pipeline = PipelineOrchestrator()
    return pipeline

try:
    pipeline = load_pipeline()
    st.success("Hệ thống đã sẵn sàng!")
except Exception as e:
    st.error(f"Lỗi khi khởi tạo hệ thống: {e}")
    st.stop()

# Khởi tạo state để lưu lịch sử chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Hiển thị lịch sử chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # Hiển thị các thông tin phụ nếu có (chỉ cho assistant)
        if msg["role"] == "assistant":
            if "df" in msg and msg["df"] is not None:
                st.dataframe(msg["df"])
            
            if "sql" in msg and msg["sql"]:
                with st.expander("Xem câu lệnh SQL đã sinh"):
                    st.code(msg["sql"], language="sql")
            
            if "similar_items" in msg and msg["similar_items"]:
                with st.expander(f"Gợi ý sản phẩm ({len(msg['similar_items'])})"):
                    for item in msg["similar_items"]:
                        st.write(f"- **{item.title}** ({item.category_name}) - ${item.price:.2f}")

# Khung nhập câu hỏi
if prompt := st.chat_input("Nhập câu hỏi của bạn (VD: Tìm phim hành động giá dưới 20 đô)"):
    # Thêm câu hỏi của user vào lịch sử và hiển thị
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Hiển thị spinner trong lúc xử lý
    with st.chat_message("assistant"):
        with st.spinner("Đang suy nghĩ..."):
            try:
                # Chạy pipeline
                result = pipeline.run(prompt)
                
                intent = result.get('intent', 'UNKNOWN')
                answer = result.get('answer', 'Không có câu trả lời.')
                df_result = result.get('df_result', None)
                similar_items = result.get('similar_items', [])
                sql_steps = result.get('sql_steps', {})
                
                sql_query = sql_steps.get('sql', '') if sql_steps else ''
                
                # Hiển thị câu trả lời tự nhiên
                st.markdown(answer)
                
                # Hiển thị dataframe
                if df_result is not None and not df_result.empty:
                    st.dataframe(df_result)
                
                # Hiển thị SQL
                if sql_query:
                    with st.expander("Xem câu lệnh SQL đã sinh"):
                        st.code(sql_query, language="sql")
                
                # Hiển thị gợi ý (nếu intent là SEARCH)
                if similar_items:
                    with st.expander(f"Gợi ý sản phẩm ({len(similar_items)})"):
                        for item in similar_items:
                            st.write(f"- **{item.title}** ({item.category_name}) - ${item.price:.2f}")
                
                # Lưu vào lịch sử
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "df": df_result,
                    "sql": sql_query,
                    "similar_items": similar_items
                })
                
            except Exception as e:
                error_msg = f"Đã xảy ra lỗi trong quá trình xử lý: {e}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
