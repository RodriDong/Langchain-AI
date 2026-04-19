import streamlit as st
import re
import os
from qabot import load_llm, create_writing_agent
from memory import WritingMemory
from dotenv import load_dotenv

# Đọc biến môi trường
load_dotenv()

# Cấu hình trang
st.set_page_config(
    page_title="AI Writing Evaluator",
    layout="wide"
)

st.title("AI Writing Evaluator")
st.caption("Hệ thống chấm bài viết tiếng Anh tự động")

# Khởi tạo session state
if "memory" not in st.session_state:
    st.session_state.memory = WritingMemory()
if "result" not in st.session_state:
    st.session_state.result = None
if "last_score" not in st.session_state:
    st.session_state.last_score = None
if "thread_id" not in st.session_state:
    st.session_state.thread_id = "session_1"

memory = st.session_state.memory

# Sidebar - thông tin người dùng và thống kê
with st.sidebar:
    st.header("Thông tin người dùng")

    user_name = st.text_input("Tên của bạn", value="Người dùng")
    user_level = st.selectbox(
        "Trình độ",
        ["Beginner", "Intermediate", "Advanced"]
    )
    user_goals = st.text_area(
        "Mục tiêu",
        value="Cải thiện kỹ năng viết tiếng Anh"
    )

    st.divider()
    st.header("Thống kê")
    st.metric("Tổng số bài đã chấm", len(memory.history))

    if memory.history:
        scores = [h["score"] for h in memory.history if h.get("score")]
        avg = round(sum(scores) / len(scores), 1) if scores else 0
        st.metric("Điểm trung bình", f"{avg}/10")
        st.metric("Điểm bài gần nhất", f"{memory.history[-1].get('score', 'N/A')}/10")

# Khu vực chính
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Bài viết của bạn")
    writing_input = st.text_area(
        "Dán bài viết vào đây:",
        height=300,
        placeholder="Nhập hoặc dán bài viết tiếng Anh của bạn vào đây..."
    )

    evaluate_btn = st.button(
        "Chấm bài",
        type="primary",
        disabled=not writing_input
    )

with col2:
    st.subheader(f"Kết quả chấm bài của: {user_name}")
    # Hiển thị kết quả từ session state
    if st.session_state.result:
        st.markdown(st.session_state.result)
    else:
        st.info("Kết quả sẽ hiển thị ở đây sau khi chấm bài.")

# Lịch sử chấm bài
st.divider()
st.subheader(f"Lịch sử chấm bài của: {user_name}")

if memory.history:
    for i, entry in enumerate(reversed(memory.history[-5:])):
        with st.expander(f"Bài {len(memory.history) - i} - Điểm: {entry.get('score', 'N/A')}/10"):
            st.write("Đoạn văn bản:")
            st.write(entry["writing_snippet"])
            st.write("Nhận xét:")
            st.markdown(entry["feedback_summary"])
else:
    st.info("Chưa có bài nào được chấm.")

# Xử lý khi bấm nút chấm bài
if evaluate_btn:
    with col2:
        with st.spinner("Đang chấm bài..."):
            try:
                user_context = f"Name: {user_name}, Level: {user_level}, Goals: {user_goals}"

                llm = load_llm()
                agent = create_writing_agent(llm, memory, user_context)

                config = {"configurable": {"thread_id": st.session_state.thread_id}}

                response = agent.invoke(
                    {"messages": [{"role": "user", "content": writing_input}]},
                    config=config
                )
                result = response["messages"][-1].content

                # Lưu kết quả vào session state
                st.session_state.result = result

                # Trích xuất điểm tổng
                score_match = re.search(r'Tổng điểm.*?(\d+(?:\.\d+)?)/10', result)
                score = float(score_match.group(1)) if score_match else 5.0
                st.session_state.last_score = score

                # Lưu vào lịch sử
                memory.save_evaluation(writing_input, result, score)

                # Rerun để cập nhật sidebar thống kê
                st.rerun()

            except Exception as e:
                st.error(f"Lỗi: {str(e)}")