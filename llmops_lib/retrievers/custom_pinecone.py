from __future__ import annotations

from enum import Enum
from langchain.schema import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from typing import Any, List


    
class CustomPineconeRetriever(BaseRetriever):

    client: Any
    index_name: str
    namespace: str
    embedding_model: str
    reranker_model: str
    top_k: int
    top_n: int

    @classmethod
    def create(cls, pinecone_api_key: str, 
        index_name: str, 
        namespace: str, 
        embedding_model: str = "multilingual-e5-large", 
        reranker_model: str = "bge-reranker-v2-m3",
        top_k: int = 10,
        top_n: int = 3
    ) -> CustomPineconeRetriever:
        from pinecone import Pinecone
        client = Pinecone(api_key=pinecone_api_key)
        return cls(client=client, 
                   index_name=index_name, 
                   namespace=namespace, 
                   embedding_model=embedding_model,
                   reranker_model=reranker_model,
                   top_k=top_k,
                   top_n=top_n)

    def _embed_query(self, query: str):
        """ Pinecone 내장 임베딩 모델을 사용하여 쿼리 벡터화 """
        result = self.client.inference.embed(
            model=self.embedding_model,
            inputs=[query],
            parameters={"input_type": "query"}
        )
        return result[0].values  # 벡터 값 반환

    def _retrieve_documents(self, query_vector):
        """ Pinecone에서 벡터 유사도 기반으로 top_k개 문서 검색 """
        results = self.client.Index(self.index_name).query(
            namespace=self.namespace,
            vector=query_vector,
            top_k=self.top_k,
            include_metadata=True
        )

        return [{"id": match["id"], "text": match["metadata"]["text"]} for match in results["matches"]]

    def _rerank_documents(self, query, documents):
        """ Reranker를 사용하여 문서를 재정렬하고 최적의 top_n 선택 """
        result = self.client.inference.rerank(
            model=self.reranker_model,
            query=query,
            documents=documents,
            top_n=self.top_n,
            return_documents=True,
            parameters={"truncate": "END"}
        )
        return result.data

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:

        query_vector = self._embed_query(query)  # 입력 쿼리 벡터화
        retrieved_docs = self._retrieve_documents(query_vector)  # 1차 검색
        reranked_docs = self._rerank_documents(query, retrieved_docs)  # Reranker 적용

        # 최종 문서 반환
        return [Document(page_content=doc.document.text, metadata={"id": doc.document.id}) for doc in reranked_docs]

