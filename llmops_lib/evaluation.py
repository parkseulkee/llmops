import os
import sqlite3
import pandas as pd
from typing import Dict, List, Any, Tuple

from langchain_core.runnables.base import Runnable
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_anthropic import ChatAnthropic
from langchain_pinecone import PineconeEmbeddings
from llmops_lib.evaluator import EvaluatorType, create_evaluator
from langchain_core.retrievers import BaseRetriever

default_database_path = os.path.dirname(os.path.dirname(__file__)) + "/llmops.db"

class Evaluation:
    def __init__(self, 
        chain: Runnable, 
        evaluation_type: EvaluatorType, 
        dataset_entries: List[Tuple[Dict[str, str], str]], 
        metadata: Dict[str, Any] = None,
        database: str = default_database_path,
        environment: Dict[str, str] = None,
        retriever: BaseRetriever = None):
        self.database = database
        self.conn = sqlite3.connect(self.database)
        self._initialize_db()

        self.chain = chain
        self.retriever = retriever
        self.evaluation_type = evaluation_type
        self.dataset_entries = dataset_entries
        self.metadata = metadata or {}

        if environment:
            os.environ.update(environment)

        self._set_chain_metadata()
        
        self.results = None
        self.token_usage = None
        self.latency = None
        self.score = None

    def _initialize_db(self):
        cursor = self.conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS evaluations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            evaluation_type TEXT NOT NULL,
            metadata TEXT,
            token_usage TEXT NOT NULL,
            latency TEXT NOT NULL,
            score REAL NOT NULL
        )
        ''')
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS evaluation_details (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
              INTEGER NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            input_variables TEXT NOT NULL,
            output TEXT NOT NULL,
            reference_output TEXT,
            input_token INTEGER,
            output_token INTEGER,
            latency REAL,
            metadata TEXT,
            score REAL NOT NULL,
            FOREIGN KEY (evaluation_id) REFERENCES evaluations(id) ON DELETE CASCADE
        )
        ''')
        self.conn.commit()

    def _set_chain_metadata(self):
        for step in self.chain.steps:
            if isinstance(step, ChatPromptTemplate):
                self.metadata["prompt_template"] = [m.prompt.template for m in step.messages]
            if isinstance(step, BaseChatModel):
                self.metadata["model"] = step.model
                self.metadata["temperature"] = step.temperature

    def run_evaluation(self):
        
        embedding_model = None
        judge_model = None

        if self.evaluation_type == EvaluatorType.EMBEDDING_DISTANCE:
            embedding_model = PineconeEmbeddings(model="multilingual-e5-large")
        if self.evaluation_type == EvaluatorType.LLM_JUDGE:
            judge_model = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0.1, max_tokens=256)
        if self.evaluation_type == EvaluatorType.RAG:
            judge_model = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0.1, max_tokens=8192)

        evaluator = create_evaluator(
            evaluator_type=self.evaluation_type, 
            chain=self.chain, 
            judge_model=judge_model, 
            embedding_model=embedding_model, 
            retriever=self.retriever
        )
        
        results = []

        for input_variables, reference_output in self.dataset_entries:
            result = evaluator.evaluate(input_variables=input_variables, reference_output=reference_output)
            results.append(result)

        self.results = results
        self._calculate_evaluation_metrics()
        self._save_evaluation_results()

        return results

    def _calculate_evaluation_metrics(self):
        token_usages = [result["input_token"] + result["output_token"] for result in self.results]
        latencies = [result["latency"] for result in self.results]
        scores = [result["score"] for result in self.results]

        self.token_usage = {
            "25%": pd.Series(token_usages).quantile(0.25),
            "50%": pd.Series(token_usages).quantile(0.50),
            "75%": pd.Series(token_usages).quantile(0.75),
            "99%": pd.Series(token_usages).quantile(0.99)
        }

        self.latency = {
            "25%": pd.Series(latencies).quantile(0.25),
            "50%": pd.Series(latencies).quantile(0.50),
            "75%": pd.Series(latencies).quantile(0.75),
            "99%": pd.Series(latencies).quantile(0.99)
        }

        self.score = sum(scores) / len(scores) if scores else 0

    def _save_evaluation_results(self):
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO evaluations (evaluation_type, metadata, token_usage, latency, score)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            self.evaluation_type.value, 
            str(self.metadata), 
            str(self.token_usage), 
            str(self.latency), 
            self.score
        ))
        evaluation_id = cursor.lastrowid

        for result in self.results:
            metadata = {k: v for k, v in result.items() if k not in ["input_variables", "output", "reference_output", "input_token", "output_token", "latency", "score"]}

            cursor.execute('''
                INSERT INTO evaluation_details (evaluation_id, input_variables, output, reference_output, input_token, output_token, latency, metadata, score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                evaluation_id,
                str(result["input_variables"]),
                result["output"],
                result["reference_output"],
                result["input_token"],
                result["output_token"],
                result["latency"],
                str(metadata),
                result["score"]
            ))

        self.conn.commit()
