from setuptools import setup, find_packages

setup(
    name="llmops_lib",
    version="0.1.0",
    description="Llmops package",
    packages=find_packages(),
    install_requires = [
    "anthropic==0.39.0",
    "faiss-cpu==1.10.0",
    "ipykernel==6.29.5",
    "langchain==0.3.15",
    "langchain-anthropic==0.3.0",
    "langchain-ollama==0.2.0",
    "langchain-openai==0.3.2",
    "langchain-pinecone==0.2.2",
    "ollama==0.4.2",
    "pinecone==5.4.2",
    "ragas==0.2.13",
    "streamlit==1.43.2",
    "streamlit-aggrid==1.1.1"
],
)