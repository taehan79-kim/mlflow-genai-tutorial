# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an MLflow GenAI tutorial repository demonstrating how to build, track, and evaluate RAG (Retrieval-Augmented Generation) agents using LangGraph, LangChain, and MLflow. The project is tutorial-based with interactive Jupyter notebooks that progressively introduce MLflow concepts.

## Environment Setup

### Python Environment
- **Python Version**: 3.13.9 (specified in `.python-version`)
- **Package Manager**: UV (uses `pyproject.toml` and `uv.lock`)
- **Virtual Environment**: `.venv` directory

### Required Environment Variables
Create a `.env` file based on `.env.example`:
- `AWS_ACCESS_KEY_ID`: AWS credentials for Bedrock
- `AWS_SECRET_ACCESS_KEY`: AWS credentials for Bedrock
- `AWS_REGION`: AWS region (e.g., us-east-1)
- `AWS_MODEL_ID`: Bedrock LLM model ID (e.g., `global.anthropic.claude-sonnet-4-5-20250929-v1:0`)
- `AWS_EMD_MODEL_ID`: Bedrock embedding model ID (e.g., `amazon.titan-embed-text-v2:0`)
- `MLFLOW_TRACKING_URI`: MLflow tracking server URI (default: `http://localhost:5000`)

### Installation Commands
```bash
# Install dependencies using UV
uv sync

# Or using pip
pip install -e .
```

## Development Workflow

### Running Jupyter Notebooks
The main workflow is through numbered Jupyter notebooks in sequence:
```bash
# Start Jupyter
jupyter notebook

# Or use Jupyter Lab
jupyter lab
```

Notebooks should be run in order:
1. `00_setup_baseline_rag_agent.ipynb` - Baseline RAG agent without MLflow
2. `01_mlflow_tracking_basics.ipynb` - MLflow tracking fundamentals

### MLflow UI
```bash
# Start MLflow UI to view experiments
mlflow ui --port 5000

# Or specify tracking URI explicitly
mlflow ui --backend-store-uri ./mlruns --port 5000
```

Access at `http://localhost:5000`

### Running Scripts
```bash
# Basic test script
python main.py
```

## Architecture

### RAG Agent Pipeline (LangGraph-based)

The RAG agent is built using LangGraph's StateGraph with two main nodes:

1. **Retriever Node**: Queries the FAISS vector store for similar documents
2. **Generator Node**: Uses retrieved context to generate answers via LLM

**State Flow**: `START → Retriever → Generator → END`

**State Schema** (`RAGState` TypedDict):
- `query`: User question
- `retrieved_documents`: List of Document objects from vector store
- `context`: Concatenated text from retrieved documents
- `answer`: Final LLM-generated response
- `metadata`: Performance metrics (retrieval_time, generation_time, etc.)

### Key Components

**Vector Store Setup**:
- Uses FAISS for similarity search
- Documents chunked via `RecursiveCharacterTextSplitter`
- Default chunk_size: 512, chunk_overlap: 50
- Embeddings via AWS Bedrock (Titan Embed model)

**LLM Configuration**:
- Uses AWS Bedrock ChatBedrock client
- Model specified via `AWS_MODEL_ID` environment variable
- Temperature and other parameters configurable per experiment

### MLflow Integration Pattern

**Tracking Structure**:
- Experiment: `rag_agent_experiments` (groups related runs)
- Run: Individual execution with specific hyperparameters

**What to Log**:
- **Parameters**: `chunk_size`, `chunk_overlap`, `top_k`, `llm_model`, `temperature`, `embedding_model`, `num_chunks`
- **Metrics**: `overall_time`, `retrieval_time`, `generation_time`, `num_retrieved_docs`, `answer_length`
- **Artifacts**: `output_answer.txt`, `retrieved_documents.json`, `run_summary.json`

**Typical MLflow Run Pattern**:
```python
with mlflow.start_run(run_name="experiment_name"):
    # Log parameters
    mlflow.log_param("chunk_size", chunk_size)

    # Execute RAG pipeline
    result = rag_agent.invoke(initial_state)

    # Log metrics
    mlflow.log_metric("overall_time", overall_time)

    # Log artifacts
    mlflow.log_text(result['answer'], "output_answer.txt")
    mlflow.log_dict(retrieved_docs_data, "retrieved_documents.json")
```

## Common Patterns

### Creating Vector Stores
The `create_vector_store()` function pattern:
- Takes `chunk_size` and `chunk_overlap` as parameters
- Returns tuple of (vector_store, num_chunks)
- Uses sample documents defined in the notebook

### Creating RAG Agents
The `create_rag_agent()` function pattern:
- Takes `vector_store`, `top_k`, `llm_model`, `temperature`
- Returns compiled LangGraph StateGraph
- Nodes defined as closures to access parameters

### Batch Experiments
Common pattern for comparing multiple configurations:
```python
experiment_configs = [
    {"name": "config1", "chunk_size": 256, "top_k": 3, ...},
    {"name": "config2", "chunk_size": 512, "top_k": 5, ...},
]

for config in experiment_configs:
    with mlflow.start_run(run_name=config['name']):
        # Run experiment and log results
```

## Important Notes

### AWS Bedrock Usage
This tutorial uses AWS Bedrock (not OpenAI) for:
- **LLM**: Claude Sonnet 4.5 via `ChatBedrock`
- **Embeddings**: Amazon Titan Embed via `BedrockEmbeddings`

When modifying code, ensure AWS credentials are configured and the specified models are available in your AWS region.

### MLflow Storage
- Default tracking URI: `./mlruns` (local filesystem)
- Experiments and runs stored in this directory
- Artifacts saved alongside run metadata

### Notebook State
Notebooks are designed to be run sequentially. Each notebook builds on concepts from the previous one. If you skip notebooks, you may need to manually run setup cells from earlier notebooks.

### Performance Metrics
The tutorial focuses on these key metrics for RAG evaluation:
- **Retrieval time**: Time to find similar documents
- **Generation time**: LLM inference time
- **Overall time**: End-to-end latency
- **Answer length**: Number of characters in response

Consider these when making performance optimizations or comparing experiments.
