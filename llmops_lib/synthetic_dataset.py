from langchain_community.document_loaders import PyPDFLoader
from ragas.testset import TestsetGenerator
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_anthropic import ChatAnthropic
from langchain_pinecone import PineconeEmbeddings
from ragas.testset.persona import Persona
from ragas.testset.synthesizers.single_hop.specific import SingleHopSpecificQuerySynthesizer

class PDFSyntheticDatasetGenerator:
    def __init__(self, pdf_path, api_key, pinecone_api_key, personas=None, distribution=None):
        # PDF 문서 파싱
        self.loader = PyPDFLoader(pdf_path)
        self.docs = self.loader.load()
        # LLM, Embedding 정의
        self.generator_llm = LangchainLLMWrapper(ChatAnthropic(
            model="claude-3-5-sonnet-20241022", api_key=api_key))
        self.generator_embeddings = LangchainEmbeddingsWrapper(
            PineconeEmbeddings(model="multilingual-e5-large", api_key=pinecone_api_key))
        # 기본 페르소나 정의
        self.personas = personas or [
            Persona(
                name="korean consumer",
                role_description="한국 고객이 질문하고, 한국어로 답변받길 원하는 사람",
            ),
        ]
        # TestsetGenerator 정의
        self.generator = TestsetGenerator(
            llm=self.generator_llm, 
            embedding_model=self.generator_embeddings,
            persona_list=self.personas
        )
        # 테스트셋의 질문 유형 분포 정의
        self.distribution = distribution or [
            (SingleHopSpecificQuerySynthesizer(llm=self.generator_llm), 1.0),
        ]

    def generate_dataset(self, sample_size=3, testset_size=5):
        dataset = self.generator.generate_with_langchain_docs(
            self.docs[:sample_size], # 문서 내 샘플링
            testset_size=testset_size,  # 생성할 테스트셋 크기
            query_distribution=self.distribution # 질문 유형 분포
        )
        return dataset.to_pandas()

