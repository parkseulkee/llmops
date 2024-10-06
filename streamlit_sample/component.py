import streamlit as st
import pandas as pd

# Streamlit 앱 기본 설정
st.title("Streamlit 컴포넌트 실습")
st.write("이 앱은 다양한 Streamlit 컴포넌트를 실습한다.")

# 1. 텍스트 입력 및 출력
st.header("1. 텍스트 입력 및 출력")
user_input = st.text_input("텍스트를 입력하세요:", "안녕하세요!")
if st.button("입력 확인"):
    st.write(f"입력한 텍스트: {user_input}")

# 2. 드롭 다운 목록
st.header("2. 드롭 다운 목록")
options = ["옵션 1", "옵션 2", "옵션 3"]
selected_option = st.selectbox("옵션을 선택하세요:", options)
st.write(f"선택된 옵션: {selected_option}")

# 3. 슬라이더
st.header("3. 슬라이더")
slider_value = st.slider("값을 선택하세요:", min_value=0, max_value=100, value=50)
st.write(f"선택된 값: {slider_value}")

# 4. 테이블 출력
st.header("4. 테이블 출력")
data = {
    "이름": ["Alice", "Bob", "Charlie"],
    "점수": [85, 90, 78],
}
df = pd.DataFrame(data)
st.dataframe(df, use_container_width=True)

# 5. 테이블 편집
st.header("5. 테이블 편집")
edited_data = st.data_editor(df, use_container_width=True)
st.markdown("- 편집된 데이터:")
st.dataframe(edited_data, use_container_width=True)
