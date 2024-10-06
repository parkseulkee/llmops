import streamlit as st

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from lib.model_provider import ChatModelManager

def init():
    if "cmm" not in st.session_state:
        st.session_state.cmm = ChatModelManager()
    if "model_list" not in st.session_state:
        st.session_state.model_list = st.session_state.cmm.get_model_list()
    if "model_args" not in st.session_state:
        st.session_state.model_args = None
    if "chat_prompt_template" not in st.session_state:
        st.session_state.chat_prompt_template = None
    if "input_variables" not in st.session_state:
        st.session_state.input_variables = None
    if "chain" not in st.session_state:
        st.session_state.chain = None
    if "chain_input" not in st.session_state:
        st.session_state.chain_input = None

def update_chain():
    require_args = {k: st.session_state.get(k) for k in st.session_state.model_args.keys()}
    model = st.session_state.get("cmm").get_model(
        st.session_state.get("choice_model"), 
        temperature=st.session_state.get("temperature"), 
        max_tokens=st.session_state.get("max_tokens"), 
        **require_args
        )
    st.session_state.chain = st.session_state.chat_prompt_template | model
    st.session_state.chain_input = {input_variable: st.session_state.get(input_variable) for input_variable in st.session_state.input_variables}

def testing():

    init()

    st.title("LLM 테스트 UI")
    st.write("다양한 모델을 테스트하고, Prompt Template과 파라미터를 조정할 수 있는 도구입니다.")

    st.text_area("system_prompt", key="system_prompt", value="당신은 유능한 조수로써 주어진 질문에 답변합니다.")
    st.text_area("user_prompt", key="user_prompt", value="{text}")

    st.selectbox("모델 선택", st.session_state.get("model_list"), key="choice_model")

    if st.session_state.get("choice_model"):
        st.session_state.model_args = st.session_state.cmm.get_model_require_args(st.session_state.get("choice_model"))
        for k, v in st.session_state.model_args.items():
            st.text_input(k, v, key=k)

    st.slider("Temperature", 0.0, 1.0, 0.7, key="temperature")
    st.number_input("Max Tokens", min_value=1, value=256, key="max_tokens")

    st.session_state.chat_prompt_template = ChatPromptTemplate.from_messages([
        ("system", st.session_state.get("system_prompt")), 
        ("user", st.session_state.get("user_prompt"))
        ])
    st.session_state.input_variables = st.session_state.chat_prompt_template.input_variables

    if st.session_state.get("input_variables"):
        for input_variable in st.session_state.get("input_variables"):
            st.text_area(
                input_variable,
                key=input_variable
            )

    if st.button("Run", on_click=update_chain, use_container_width=True):
        with st.chat_message("ai"):
            st.write_stream(st.session_state.chain.stream(st.session_state.chain_input))


testing()