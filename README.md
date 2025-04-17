# 🚀 LLMOps 실습 코드 레포지토리

이 저장소는 📘 [**『LLMOps를 활용한 LLM 엔지니어링』**](https://www.yes24.com/product/goods/145341599) 책의 실습 코드가 담겨 있습니다.  
LLMOps를 기반으로 **대규모 언어 모델을 실제로 어떻게 운영/서비스화할 수 있는지** 배우게 됩니다.

## 🧭 프로젝트 구조

```
📁 dataset/           → 실습에 사용되는 데이터셋  
📁 llmops_lib/        → LLMOps 도구를 위한 라이브러리 코드  
📁 notebook/          → 장별 Jupyter 노트북 실습 코드  
📁 src/               → 주요 애플리케이션 코드 (Streamlit 기반)  
📁 streamlit_sample/  → 간단한 웹앱 샘플 (Streamlit 활용)
```


## ⚡ 시작하기

### 1️⃣ 클론 & 디렉토리 이동

```bash
git clone https://github.com/parkseulkee/llmops.git
cd llmops
```

### 2️⃣ 가상환경 만들기 및 활성화 🐍

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3️⃣ 의존성 설치

```bash
pip install -e .
```

### 4️⃣ Jupyter Notebook 실행

```bash
jupyter notebook
```

## 📘 주피터 커널 등록 (선택)

> 실습용 커널을 따로 등록하고 싶다면:

```bash
python -m ipykernel install --user --name=my_env --display-name "LLMOps Book"
```


## 🔗 참고 링크

- 📖 책 구매: [Yes24 링크](https://www.yes24.com/product/goods/145341599)
- 🐞 이슈 등록: [Issue 탭](../../issues)
