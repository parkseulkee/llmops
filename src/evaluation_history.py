import streamlit as st
import sqlite3
import pandas as pd
import json

from st_aggrid import AgGrid, GridOptionsBuilder
from st_aggrid.shared import GridUpdateMode

DATABASE_PATH = "/Users/user/study/llmops/llmops.db"

def aggrid_interactive_table(df: pd.DataFrame):
    options = GridOptionsBuilder.from_dataframe(df)

    # 사이드바 생성: 필터링, 컬럼 선택
    options.configure_side_bar()
    # 행 단일 선택 기능 추가
    options.configure_selection("single")

    selection = AgGrid(
        df,
        gridOptions=options.build(),
    )

    return selection

def load_evaluation_history():
    conn = sqlite3.connect(DATABASE_PATH)
    query = "SELECT * FROM evaluations"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def load_evaluation_details(evaluation_id):
    conn = sqlite3.connect(DATABASE_PATH)
    query = "SELECT * FROM evaluation_details WHERE evaluation_id = ?"
    df = pd.read_sql_query(query, conn, params=(evaluation_id,))
    conn.close()
    return df

def evaluation_history():
    st.title("Evaluation History")

    df = load_evaluation_history()
    if df.empty:
        st.write("No evaluation history found.")
    else:
        # 평가 결과 테이블 출력
        selection = aggrid_interactive_table(df=df)

        # 선택한 행에 대한 상세 정보 출력
        if selection:
            selected_rows = selection.get("selected_rows")
            if isinstance(selected_rows, pd.DataFrame):
                # 선택한 행을 JSON 으로 변환
                selected_json_str = selected_rows.to_json(orient="records")
                selected_json = json.loads(selected_json_str)[0]

                # 선택한 평가에 대한 요약 정보 출력
                st.divider()
                for key, value in selected_json.items():
                    if key in ["metadata", "latency", "token_usage"]:
                        st.caption(f"{key}:")
                        st.json(value, expanded=False)
                    else:
                        st.caption(f"{key}: `{value}`")
                st.divider()

                # 선택한 평가의 상세 데이터 엔트리 정보 출력
                evaluation_id = selected_json["id"]
                details_df = load_evaluation_details(evaluation_id)
                st.dataframe(details_df, use_container_width=True, hide_index=True)

evaluation_history()
