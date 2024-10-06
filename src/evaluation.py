import os
import json
import inspect
import streamlit as st
import pandas as pd
from llmops_lib.model_provider import ChatModelManager
from llmops_lib.dataset_storage import DatasetStorage
from llmops_lib.evaluator import EvaluatorType
from llmops_lib.prompt_hub import PromptHub
from llmops_lib.prompt import Prompt
from llmops_lib.evaluation import Evaluation
from llmops_lib.retriever import RetrieverType, get_retriever_class, format_docs

PROMPT_HUB = PromptHub()
DATASET_STORAGE = DatasetStorage()

def init():
    if "cmm" not in st.session_state:
        st.session_state.cmm = ChatModelManager()
    if "model_list" not in st.session_state:
        st.session_state.model_list = st.session_state.cmm.get_model_list()
    if "model_args" not in st.session_state:
        st.session_state.model_args = None
    if "prompt_list" not in st.session_state:
        st.session_state.prompt_list = PROMPT_HUB.get_prompt_list()
    if "model_args" not in st.session_state:
        st.session_state.model_args = None
    if "dataset_list" not in st.session_state:
        st.session_state.dataset_list = DATASET_STORAGE.list_datasets()
    if "dataset" not in st.session_state:
        st.session_state.dataset = None
    if "dataset_entries" not in st.session_state:
        st.session_state.dataset_entries = None
    if "prompt_template" not in st.session_state:
        st.session_state.prompt_template = None
    if "system_prompt" not in st.session_state:
        st.session_state.system_prompt = None
    if "user_prompt" not in st.session_state:
        st.session_state.user_prompt = None
    if "evaluator_types" not in st.session_state:
        st.session_state.evaluator_types = [evaluator_type.value for evaluator_type in EvaluatorType]
    if "retriever_types" not in st.session_state:
        st.session_state.retriever_types = [retriever_type.value for retriever_type in RetrieverType]
    if "retriever_args" not in st.session_state:
        st.session_state.retriever_args = None

def update_prompt():
    prompt = Prompt(st.session_state.selected_prompt, st.session_state.selected_prompt_version)
    st.session_state.prompt_template = prompt.get_chat_template()
    st.session_state.system_prompt = prompt.system_template
    st.session_state.user_prompt = prompt.user_template

def update_dataset():
    dataset = DATASET_STORAGE.get_dataset(st.session_state.selected_dataset)
    dataset_entries = dataset.get_entries()
    st.session_state.dataset = dataset
    st.session_state.dataset_entries = dataset_entries

def update_model():
    st.session_state.model_args = st.session_state.cmm.get_model_require_args(st.session_state.selected_model)

def update_retriever():
    retriever_create_func = get_retriever_class(RetrieverType(st.session_state.selected_retriever_type))
    signiture = inspect.signature(retriever_create_func)
    parameters = signiture.parameters
    st.session_state.retriever_args = parameters

def evaluate_model():

    init()

    st.title("Evaluation")

    st.selectbox("프롬프트 선택", st.session_state.prompt_list, index=None, key="selected_prompt")
    with st.expander(label="프롬프트 정보"):
        if st.session_state.selected_prompt:
            prompt_versions = PROMPT_HUB.get_prompt_versions(st.session_state.selected_prompt)
            version_ids = [version[0] for version in prompt_versions]
            st.selectbox("프롬프트 버전 선택", version_ids, index=len(version_ids) - 1, on_change=update_prompt, key="selected_prompt_version")
            update_prompt()
            st.code(st.session_state.system_prompt)
            st.code(st.session_state.user_prompt)
        
    st.selectbox("모델 선택", st.session_state.model_list, key="selected_model", on_change=update_model)
    with st.expander("모델 옵션 조정"):
        update_model()
        for k, v in st.session_state.model_args.items():
            st.text_input(k, v, key=k)
        st.slider("Temperature", 0.0, 1.0, 0.7, key="temperature")
        st.number_input("Max Tokens", min_value=1, value=256, key="max_tokens")
    
    st.selectbox("데이터셋 선택", st.session_state.dataset_list, index=None, key="selected_dataset", on_change=update_dataset)
    with st.expander(label="데이터셋"):
        if st.session_state.dataset:
            st.dataframe(st.session_state.dataset_entries, use_container_width=True)
    
    st.selectbox("평가자 선택", st.session_state.evaluator_types, key="selected_evaluator")
    if st.session_state.selected_evaluator == EvaluatorType.EMBEDDING_DISTANCE.value:
        st.text_input("PINECONE_API_KEY", key="PINECONE_API_KEY")
    if st.session_state.selected_evaluator == EvaluatorType.LLM_JUDGE.value:
        st.text_input("ANTHROPIC_API_KEY", key="ANTHROPIC_API_KEY")
    if st.session_state.selected_evaluator == EvaluatorType.RAG.value:
        st.text_input("ANTHROPIC_API_KEY", key="ANTHROPIC_API_KEY", help="평가자 모델의 API 키")
        with st.expander(label="검색기 정의", expanded=True):
            st.selectbox("Retriever Type", st.session_state.retriever_types, key="selected_retriever_type", on_change=update_retriever)
            update_retriever()
            for params in st.session_state.retriever_args.values():
                st.text_input(params.name, value=params.default if params.default is not inspect.Parameter.empty else "", key=params.name)
            
    if st.button("Run Evaluation") \
        and st.session_state.selected_prompt \
        and st.session_state.selected_model \
        and st.session_state.selected_dataset \
        and st.session_state.selected_evaluator:

        require_args = {k: st.session_state.get(k) for k in st.session_state.model_args.keys()}
        model = st.session_state.get("cmm").get_model(
            st.session_state.selected_model, 
            temperature=st.session_state.temperature, 
            max_tokens=st.session_state.max_tokens, 
            **require_args
            )

        retriever = None
        if st.session_state.selected_evaluator == EvaluatorType.RAG.value:
            retriever_args = {p.name: st.session_state.get(p.name) for p in st.session_state.retriever_args.values()}
            retriever = get_retriever_class(RetrieverType(st.session_state.selected_retriever_type))(**retriever_args)
            chain = (
                (lambda x: {'context': format_docs(retriever.invoke(x['question'])), 'question': x['question']})
                | st.session_state.prompt_template
                | model
            )

        environment = {}
        if "PINECONE_API_KEY" in st.session_state:
            environment["PINECONE_API_KEY"] = st.session_state.PINECONE_API_KEY
        if "ANTHROPIC_API_KEY" in st.session_state:
            environment["ANTHROPIC_API_KEY"] = st.session_state.ANTHROPIC_API_KEY   

        evaluation = Evaluation(
            chain=chain, 
            evaluation_type=EvaluatorType(st.session_state.selected_evaluator), 
            dataset_entries=[(d["input_variables"], d["reference_output"]) for d in st.session_state.dataset_entries],
            metadata={
                "prompt": st.session_state.selected_prompt,
                "prompt_version_id": st.session_state.selected_prompt_version,
                "model": st.session_state.selected_model,
                "dataset": st.session_state.selected_dataset
            },
            environment=environment,
            retriever=retriever
        )
        
        with st.status("Evaluation..."):
            results = evaluation.run_evaluation()
            token_usage = evaluation.token_usage
            latency = evaluation.latency
            score = evaluation.score

            st.caption("token_usage:")
            st.dataframe(pd.json_normalize(token_usage), hide_index=True)

            st.caption("latency:")
            st.dataframe(pd.json_normalize(latency), hide_index=True)
            
            st.caption("score:")
            st.write(score)

            st.dataframe(pd.DataFrame(results), )

evaluate_model()
