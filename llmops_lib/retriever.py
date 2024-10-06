from enum import Enum

class RetrieverType(Enum):
    CUSTOM_PINECONE_RETRIEVER = "CustomPineconeRetriever"

def get_retriever_class(retriever_type: RetrieverType):
    if retriever_type == RetrieverType.CUSTOM_PINECONE_RETRIEVER:
        from llmops_lib.retrievers import CustomPineconeRetriever
        return CustomPineconeRetriever.create
    else:
        raise ValueError(f"Unsupported retriever type: {retriever_type}")

def format_docs(docs):
    return '\n\n'.join([d.page_content for d in docs])