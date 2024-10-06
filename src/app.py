import streamlit as st

testing = st.Page("testing.py", title="Testing")
dataset = st.Page("dataset.py", title="Dataset")
evaluation = st.Page("evaluation.py", title="Evaluation")
evaluation_history = st.Page("evaluation_history.py", title="Evaluation History")

pg = st.navigation([testing, dataset, evaluation, evaluation_history])

pg.run()