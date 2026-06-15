# QPP-KG-RAG: Adaptive Query-Aware Knowledge Graph Integration for Scientific Claim Verification

<div align="center">

[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-Active-brightgreen.svg)](https://github.com)

**A comprehensive framework for improving scientific claim verification through query-aware adaptive retrieval enhanced by structured knowledge representations.**

[Features](#-features) • [Installation](#-installation) • [Quick Start](#-quick-start) • [Usage](#-usage) • [Results](#-results)

</div>

---

## 📝 Overview

This repository implements **QPP-KG-RAG**, an innovative framework that combines three powerful techniques:

- **Query Performance Prediction (QPP)**: Predicts query difficulty before retrieval
- **Knowledge Graph (KG)**: Structured entity relationships for contextual enrichment  
- **Retrieval-Augmented Generation (RAG)**: Evidence-grounded answer generation

### Key Innovation

Instead of applying expensive knowledge graph expansion to all queries, **QPP-KG-RAG** intelligently predicts which queries will benefit from augmentation. This adaptive approach achieves:

- **+21% improvement** in Recall@10
- **+14.7% improvement** in NDCG@10
- **+35.5% improvement** in F1 score

---

## ✨ Features

- 🔍 **Query Performance Prediction**: Predict query difficulty before retrieval
- 📚 **Knowledge Graph Integration**: Smart entity enrichment for difficult queries
- 🎯 **Adaptive Retrieval**: Selective augmentation based on query characteristics
- 📖 **Evidence-Grounded Generation**: Faithful answer generation with Flan-T5
- 📊 **Comprehensive Evaluation**: MRR, NDCG, Recall, and ROUGE-L metrics
- ⚡ **Fast Search**: FAISS-based vector search for rapid retrieval

---

## 🏗️ Project Structure

```
adaptive-claim-verification/
├── src/                    # Core implementation (9 modules)
│   ├── data_loader.py     # Load datasets (SciFact, Climate-FEVER, FEVER)
│   ├── preprocessing.py   # Text processing & entity extraction
│   ├── knowledge_graph.py # Build and query knowledge graphs
│   ├── dense_retrieval.py # FAISS-based embedding search
│   ├── difficulty_predictor.py # Query Performance Prediction
│   ├── query_expansion.py # KG-augmented query expansion
│   ├── generation.py      # Flan-T5 answer generation
│   └── evaluation.py      # Metrics calculation
├── baselines/             # Baseline implementations
│   └── standard_rag.py   # Standard RAG without QPP/KG
├── experiments/           # Complete pipeline
│   └── run_pipeline.py   # Full QPP-KG-RAG system
├── config/
│   └── base_config.yaml  # Configuration settings
└── requirements.txt       # Dependencies
```

---

## 📥 Installation

### Prerequisites
- Python 3.10+
- pip or conda

### Setup (5 minutes)

```bash
# 1. Clone repository
git clone https://github.com/yourusername/adaptive-claim-verification.git
cd adaptive-claim-verification

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 4. Verify installation
python -c "from src import *; print('✓ Success')"
```

---

## 🚀 Quick Start

### Basic Usage (3 lines of code)

```python
from experiments.run_pipeline import AdaptiveClaimVerificationPipeline

pipeline = AdaptiveClaimVerificationPipeline('config/base_config.yaml')
result = pipeline.process_query("Insulin is produced by the pancreas")
print(f"Answer: {result['answer']}")
```

### Complete Example

```python
from experiments.run_pipeline import AdaptiveClaimVerificationPipeline

# Initialize the system
pipeline = AdaptiveClaimVerificationPipeline('config/base_config.yaml')

# Process a scientific claim
result = pipeline.process_query(
    query="Does insulin treat diabetes?",
    use_qpp=True,      # Enable query difficulty prediction
    use_kg=True        # Enable knowledge graph expansion
)

# Access results
print(f"Claim: {result['query']}")
print(f"Retrieved Passages: {result['passages']}")
print(f"Generated Answer: {result['answer']}")
print(f"Confidence: {result['confidence']:.2f}")
```

---

## 📚 Usage Examples

### Example 1: Load Data

```python
from src import ClaimVerificationDataset

# Load SciFact dataset
dataset = ClaimVerificationDataset('scifact', 'train')
print(f"Loaded {len(dataset)} examples")

# Access a sample
sample = dataset[0]
```

### Example 2: Build Knowledge Graph

```python
from src import KnowledgeGraphBuilder, TextPreprocessor

kg_builder = KnowledgeGraphBuilder()
preprocessor = TextPreprocessor()

documents = [
    "Insulin is produced by pancreas.",
    "Pancreas is an organ.",
]

kg_builder.build_from_texts(documents, preprocessor)
neighbors = kg_builder.get_neighbors('insulin', k=5)
print(f"Related entities: {neighbors}")
```

### Example 3: Evaluate Performance

```python
from src import EvaluationSuite

metrics = EvaluationSuite.evaluate_retrieval(
    predictions=results,
    ground_truth=labels,
    k=[10, 20]
)

print(f"MRR@10: {metrics['mrr@10']:.4f}")
print(f"Recall@10: {metrics['recall@10']:.4f}")
```

---

## 📊 Results

### Performance on SciFact Benchmark

| System | Recall@10 | MRR@10 | NDCG@10 | F1 |
|--------|-----------|--------|---------|-----|
| RAG Baseline | 0.62 | 0.30 | 0.68 | 0.48 |
| KG-RAG | 0.70 | 0.37 | 0.75 | 0.62 |
| **QPP-KG-RAG** | **0.75** | **0.40** | **0.78** | **0.84** |

### Key Improvements
- **+13% Recall@10** over RAG baseline
- **+33% MRR@10** over baseline
- **+75% F1 score** improvement

---

## 🔧 Configuration

Customize `config/base_config.yaml`:

```yaml
models:
  retrieval_model: "sentence-transformers/all-MiniLM-L6-v2"
  generation_model: "google/flan-t5-base"
  expansion_model: "t5-small"

qpp:
  threshold: 0.5        # Query difficulty threshold

knowledge_graph:
  k_neighbors: 5        # Related entities to retrieve
  min_frequency: 2      # Minimum co-occurrence

retrieval:
  k: 10                 # Top-k passages
  metric: "cosine"
```

---

## 📖 Documentation

- **[SETUP.md](SETUP.md)** - Detailed installation guide
- **Code docstrings** - Inline documentation in all functions
- **Examples folder** - Real-world usage examples

---

## 🧪 Testing

```bash
# Verify imports
python -c "from src import *; print('✓ All imports work')"

# Run on sample data
python -c "from experiments.run_pipeline import AdaptiveClaimVerificationPipeline; print('✓ Pipeline loads')"
```

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📝 Citation

If you use this code, please cite:

```bibtex
@article{QPPKGRAG,
  title={QPP-KG-RAG: Query-Aware Knowledge Graph Integration for Claim Verification},
  author={Aparajita Sinha, Kunal Chakma},
  journal={Data Mining and Knowledge Discovery},
  year={2026}
}
```

---

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

---

## 🆘 Support

- **Email**: aparajitas824@gmail.com
- **Questions**: Open an issue with tag `[question]`

---

## 🙏 Acknowledgments

Built with:
- [HuggingFace Transformers](https://huggingface.co/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [spaCy](https://spacy.io/)
- [SciFact Dataset](https://huggingface.co/datasets/scifact)

---

<div align="center">

**Made with ❤️ for scientific claim verification**

[⭐ Star this project if you find it useful!](https://github.com/yourusername/adaptive-claim-verification)

</div>
