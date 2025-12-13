# MLflow GenAI Tutorial (한국어)

> MLflow를 활용한 RAG/Agent 실험 테스트베드 플랫폼 구축 가이드

## 📌 프로젝트 개요

본 프로젝트는 MLflow의 GenAI 및 LLM 관련 기능을 활용하여 LangGraph 기반 RAG(Retrieval-Augmented Generation) Agent를 개발하고 평가하는 전체 프로세스를 단계별로 학습할 수 있는 튜토리얼입니다.

MLflow는 전통적인 머신러닝 모델뿐만 아니라 LLM 애플리케이션의 전체 생명주기(실험 추적, 평가, 배포, 모니터링)를 관리할 수 있는 통합 플랫폼을 제공합니다.

**⚠️ 본 프로젝트는 현재 진행 중이며, 지속적으로 업데이트되어 추가적인 노트북과 코드가 제공될 예정입니다.**

## 🎯 학습 목표

1. **MLflow Tracking**: RAG Agent 실험 및 하이퍼파라미터 관리
2. **MLflow Tracing**: LLM 호출 및 체인의 상세 추적
3. **MLflow Evaluation**: LLM 출력 품질의 자동/수동 평가
4. **Model Management**: 모델 패키징, 레지스트리, 배포

## 🛠 사용 기술 스택

- **MLflow**: 3.7.0
- **LangChain**: 1.1.3
- **LangGraph**: 1.0.4
- **AWS Bedrock**: Claude Sonnet 4.5, Titan Embeddings
- **Vector Store**: FAISS
- **Python**: 3.13.9

## 📚 튜토리얼 구성

### Step 0: 환경 설정 및 Baseline RAG Agent 구축
**파일**: `00_setup_baseline_rag_agent.ipynb`

- MLflow 설치 및 기본 환경 설정
- LangGraph 기반 RAG Agent 구현 (MLflow 없이)
- 벤치마크용 baseline 성능 측정
- RAG 파이프라인 기본 구조 이해

**주요 학습 포인트**:
- LangGraph State와 Node 개념
- Retriever와 Generator 노드 구현
- FAISS Vector Store 구축
- 성능 측정 기준 설정

### Step 1: MLflow Tracking - 실험 추적 기초
**파일**: `01_mlflow_tracking_basics.ipynb`

- MLflow Tracking의 기본 개념 이해
- Parameters, Metrics, Artifacts 로깅
- MLflow UI에서 실험 결과 확인
- 여러 실험 비교 및 분석

**주요 학습 포인트**:
- `mlflow.start_run()` 사용법
- `log_param()`, `log_metric()`, `log_artifact()` 활용
- Run, Experiment 개념
- 하이퍼파라미터 조합 실험

### Step 2: MLflow Tracing - LLM 호출 추적 (예정)
**파일**: `02_mlflow_tracing_autolog.ipynb` (작업 예정)

- LangChain Autolog로 자동 추적
- Trace 구조 이해 (Span, Parent-Child)
- Jupyter Notebook에서 실시간 Trace 시각화
- Token usage 자동 추적
- 수동 Span 생성 및 커스터마이징

### Step 3: MLflow Evaluation - LLM 출력 품질 평가 (예정)
**파일**: `03_mlflow_evaluation_basics.ipynb` (작업 예정)

- Built-in Scorers 활용 (Faithfulness, Relevance 등)
- LLM-as-a-Judge 평가
- 평가 데이터셋 구성
- Custom Scorers 작성
- Trace 기반 재평가

### Step 4: Model Packaging & Registry (예정)
**파일**: `04_model_packaging_registry.ipynb` (작업 예정)

- LangChain Model Flavor를 이용한 모델 패키징
- PyFunc 래퍼 커스터마이징
- Model Registry를 통한 버전 관리
- 스테이징 및 프로덕션 배포

### Step 5: 프로덕션 모니터링 (예정)
**파일**: `05_production_monitoring.ipynb` (작업 예정)

- 프로덕션 환경에서의 Tracing
- Assessment 기능을 통한 품질 평가
- 이슈 감지 및 분석
- 성능 모니터링 대시보드

## 🚀 시작하기

### 1. 환경 설정

#### Python 가상환경 생성
```bash
# Python 3.13.9 사용 권장
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

#### 패키지 설치
```bash
# UV 사용 (권장)
uv sync

# 또는 pip 사용
pip install -e .
```

### 2. 환경 변수 설정

`.env.example` 파일을 복사하여 `.env` 파일을 생성하고 필요한 값을 입력하세요:

```bash
cp .env.example .env
```

`.env` 파일 예시:
```bash
# AWS Bedrock Configuration
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_REGION=us-east-1
AWS_MODEL_ID=global.anthropic.claude-sonnet-4-5-20250929-v1:0
AWS_EMD_MODEL_ID=amazon.titan-embed-text-v2:0

# MLflow Tracking Configuration
MLFLOW_TRACKING_URI=http://localhost:5000
```

### 3. Jupyter Notebook 실행

```bash
# Jupyter Notebook 시작
jupyter notebook

# 또는 Jupyter Lab 사용
jupyter lab
```

노트북을 순서대로 실행하세요:
1. `00_setup_baseline_rag_agent.ipynb`
2. `01_mlflow_tracking_basics.ipynb`
3. (추가 노트북은 업데이트 예정)

### 4. MLflow UI 실행

실험 결과를 시각화하려면 별도 터미널에서 MLflow UI를 실행하세요:

```bash
mlflow ui --port 5000
```

브라우저에서 `http://localhost:5000`로 접속하여 실험 결과를 확인할 수 있습니다.

## 📖 주요 개념

### RAG Agent 아키텍처

본 튜토리얼에서 구현하는 RAG Agent는 LangGraph의 StateGraph를 기반으로 합니다:

```
START → Retriever Node → Generator Node → END
```

- **Retriever Node**: FAISS Vector Store에서 유사 문서 검색
- **Generator Node**: 검색된 컨텍스트를 기반으로 LLM 답변 생성

### MLflow 핵심 컴포넌트

#### 1. Tracking
- **Experiment**: 관련된 여러 Run을 그룹화
- **Run**: 단일 실험 실행 단위
- **Parameters**: 하이퍼파라미터 (chunk_size, top_k, temperature 등)
- **Metrics**: 성능 지표 (latency, accuracy 등)
- **Artifacts**: 결과물 (답변, 검색 문서, 설정 파일 등)

#### 2. Tracing
- **Span**: 단일 작업 단위 (LLM 호출, 문서 검색 등)
- **Parent-Child 관계**: 중첩된 작업의 계층 구조
- **Attributes**: 각 Span의 메타데이터

#### 3. Evaluation
- **Built-in Scorers**: Faithfulness, Relevance, Answer Correctness 등
- **LLM-as-a-Judge**: LLM을 활용한 자동 평가
- **Custom Scorers**: 도메인 특화 평가 지표

## 📊 실험 관리 예시

```python
import mlflow

# Experiment 설정
mlflow.set_experiment("rag_agent_experiments")

# Run 시작
with mlflow.start_run(run_name="baseline_v1"):
    # Parameters 로깅
    mlflow.log_param("chunk_size", 512)
    mlflow.log_param("top_k", 3)
    mlflow.log_param("llm_model", "claude-sonnet-4-5")

    # RAG Agent 실행
    result = rag_agent.invoke(query)

    # Metrics 로깅
    mlflow.log_metric("overall_time", overall_time)
    mlflow.log_metric("retrieval_time", retrieval_time)

    # Artifacts 로깅
    mlflow.log_text(result['answer'], "output_answer.txt")
    mlflow.log_dict(retrieved_docs, "retrieved_documents.json")
```

## 🔍 평가 지표

### 성능 메트릭
- **Overall Time**: 전체 응답 생성 시간
- **Retrieval Time**: 문서 검색 시간
- **Generation Time**: LLM 답변 생성 시간
- **Token Usage**: 입력/출력 토큰 수

### 품질 메트릭 (Step 3 이후 추가 예정)
- **Faithfulness**: 답변이 검색된 컨텍스트에 충실한지
- **Relevance**: 답변이 질문과 관련성이 있는지
- **Answer Correctness**: 정답과의 일치도
- **Answer Similarity**: 의미적 유사도

## 📁 프로젝트 구조

```
mlflow-genai-tutorial/
├── 00_setup_baseline_rag_agent.ipynb     # Step 0: Baseline RAG Agent
├── 01_mlflow_tracking_basics.ipynb       # Step 1: MLflow Tracking
├── 02_mlflow_tracing_autolog.ipynb       # Step 2: Tracing (작업 예정)
├── 03_mlflow_evaluation_basics.ipynb     # Step 3: Evaluation (작업 예정)
├── 04_model_packaging_registry.ipynb     # Step 4: Packaging (작업 예정)
├── 05_production_monitoring.ipynb        # Step 5: Monitoring (작업 예정)
├── main.py                                # 간단한 테스트 스크립트
├── mlruns/                                # MLflow 실험 데이터 저장소
├── .env.example                           # 환경 변수 템플릿
├── pyproject.toml                         # 프로젝트 의존성
├── uv.lock                                # UV 락 파일
├── CLAUDE.md                              # Claude Code 가이드
└── README.md                              # 본 문서
```

## 🎓 학습 경로

### 초급
1. Step 0: RAG Agent 기본 구조 이해
2. Step 1: MLflow Tracking으로 실험 관리

### 중급
3. Step 2: Tracing으로 LLM 호출 추적
4. Step 3: Evaluation으로 품질 평가

### 고급
5. Step 4: Model Packaging 및 Registry
6. Step 5: 프로덕션 모니터링

## 💡 활용 사례

### 1. 하이퍼파라미터 최적화
다양한 chunk_size, top_k, temperature 조합을 실험하여 최적의 설정 발견

### 2. 프롬프트 엔지니어링
여러 프롬프트 템플릿을 비교하고 성능 차이 분석

### 3. 모델 비교
다양한 LLM 모델(Claude, GPT 등)의 성능 비교

### 4. 프로덕션 배포
최적화된 RAG Agent를 패키징하여 프로덕션 환경에 배포

## 🤝 기여 방법

본 프로젝트는 지속적으로 개선되고 있습니다. 기여를 환영합니다!

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 🔗 참고 자료

### 공식 문서
- [MLflow Official Documentation](https://www.mlflow.org/docs/latest/)
- [MLflow LLM Tracking](https://www.mlflow.org/docs/latest/llms/)
- [LangChain Documentation](https://python.langchain.com/docs/)
- [LangGraph Tutorial](https://langchain-ai.github.io/langgraph/)

### MLflow GitHub
- [MLflow Repository](https://github.com/mlflow/mlflow)
- [MLflow Examples](https://github.com/mlflow/mlflow/tree/master/examples)

### AWS Bedrock
- [AWS Bedrock Documentation](https://docs.aws.amazon.com/bedrock/)
- [LangChain AWS Integration](https://python.langchain.com/docs/integrations/platforms/aws/)

## 📮 문의

프로젝트와 관련된 질문이나 제안사항이 있으시면 Issue를 등록해주세요.

---

**Last Updated**: 2025-12-13
**Status**: 🚧 In Progress - 지속적으로 업데이트 중입니다.
