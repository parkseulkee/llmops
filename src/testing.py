import streamlit as st
import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from llmops_lib.model_provider import ChatModelManager
from llmops_lib.prompt_hub import PromptHub

PROMPT_HUB = PromptHub()

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
    if "default_system_prompt" not in st.session_state:
        st.session_state.default_system_prompt = None
    if "default_user_prompt" not in st.session_state:
        st.session_state.default_user_prompt = None
    if "prompt_versions" not in st.session_state:
        st.session_state.prompt_versions = None

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


def get_last_version():
    rows = PROMPT_HUB.get_prompt_versions(st.session_state.get("choiced_prompt_name"))
    df = pd.DataFrame(rows, columns=["id", "timestamp", "system", "user", "changed_details"])

    last_id_row = df.loc[df['id'].idxmax()]
    st.session_state.prompt_versions = df
    st.session_state.default_system_prompt = last_id_row["system"]
    st.session_state.default_user_prompt = last_id_row["user"]


def exist_prompt():
    st.selectbox('prompt', index=None, options=PROMPT_HUB.get_prompt_list(), key="choiced_prompt_name", on_change=get_last_version)
    st.text_input("변경 사항을 입력하세요.", key="changed_details")

    
def new_prompt():
    st.session_state.default_system_prompt = "당신은 유능한 조수로써 주어진 질문에 답변합니다."
    st.session_state.default_user_prompt = "{text}"

    st.text_input("저장할 프롬프트 이름을 입력하세요.", key="new_prommpt_name")

def add_new_prompt():
    if PROMPT_HUB.add_prompt(
        st.session_state.get("new_prommpt_name"),
        st.session_state.get("system_prompt"), 
        st.session_state.get("user_prompt")):
        st.toast('성공적으로 프롬프트가 저장되었습니다!', icon='🎉')
    else:
        st.toast('프롬프트 저장이 실패하였습니다. 다시 시도해주세요.', icon='🙏🏻')
    

def add_new_version():
    if PROMPT_HUB.add_prompt_version(
        st.session_state.get("choiced_prompt_name"),
        st.session_state.get("system_prompt"), 
        st.session_state.get("user_prompt"),
        st.session_state.get("change_details")):
        st.toast('성공적으로 버전이 추가되었습니다!', icon='🎉')
    else:
        st.toast('프롬프트 저장이 실패하였습니다. 다시 시도해주세요.', icon='🙏🏻')
    

def prompt():
    
    init()

    st.title("Prompt Testing")

    st.checkbox("기존 프롬프트를 불러오겠습니까?", value=False, key="is_exist_prompt")

    if st.session_state.get("is_exist_prompt"):
        exist_prompt()
        if st.session_state.get("prompt_versions") is not None:
            st.dataframe(st.session_state.get("prompt_versions"), hide_index=True, use_container_width=True)
    else:
        new_prompt()

    st.text_area("system_prompt", key="system_prompt", value=st.session_state.get("default_system_prompt"))
    st.text_area("user_prompt", key="user_prompt", value=st.session_state.get("default_user_prompt"))

    st.selectbox("모델 선택", st.session_state.get("model_list"), key="choice_model")

    if st.session_state.get("choice_model"):
        st.session_state.model_args = st.session_state.cmm.get_model_require_args(st.session_state.get("choice_model"))
        for k,v in st.session_state.model_args.items():
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
    
    if st.session_state.get("is_exist_prompt"):
        if st.button("Save New Version", use_container_width=True):
            add_new_version()
    else:
        if st.button("Save New Prompt", use_container_width=True):
            add_new_prompt()
        

prompt()