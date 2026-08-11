# Contributing to Keep Calm

> Think twice. Send once.

## Prerequisites

- Python 3.10 or later
- A virtual environment (recommended)

## Setup

```bash
# Clone and enter the project
git clone <repo-url> keep-calm
cd keep-calm

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

# Install in editable mode with dev dependencies
pip install -e ".[dev]"
```

## Project Structure

```
keep-calm/
├── data/                   # Datasets, models, and results
│   ├── *.jsonl             # Annotated datasets (tracked in git)
│   ├── splits/             # Train/val/test splits
│   ├── models/             # Trained model artifacts (not tracked)
│   └── results/            # Evaluation metrics (not tracked)
├── docs/                   # Documentation
├── notebooks/              # Jupyter notebooks for exploration
├── scripts/                # Data collection, annotation, and training scripts
├── src/keep_calm/          # Library source code
│   ├── analyzer.py         # Main inference engine
│   ├── cli.py              # CLI entry point
│   ├── models/             # Model definitions
│   ├── schemas/            # Pydantic models and enums
│   └── tasks/              # Task-specific logic (risk, tone, intent)
├── ARCHITECTURE.md         # Full project architecture and roadmap
├── CONTRIBUTING.md         # This file
└── pyproject.toml          # Project configuration and dependencies
```

## Development

### Running the CLI

```bash
# After pip install -e ., the CLI is available:
keep-calm "Your message here"

# Or use the shell scripts (load models once):
./keep-calm.sh "Your message here"      # Single analysis
./keep-calm-repl.sh                     # Interactive REPL mode
```

### Linting and Type Checking

```bash
ruff check src/ scripts/
ruff format --check src/ scripts/
mypy src/
```

### Running Tests

```bash
pytest
pytest --cov=keep_calm   # With coverage
```

### Installing Training Dependencies

```bash
pip install -e ".[dev,training]"
```

## Data Pipeline

### Dataset Format

All datasets use JSONL format (one JSON object per line). See the annotation schema in
[ARCHITECTURE.md §5](ARCHITECTURE.md#5-dataset-strategy) for the full field specification.

### Key Scripts

| Script | Purpose |
|---|---|
| `scripts/scrape_reddit.py` | Scrape Reddit comments for annotation |
| `scripts/scrape_github.py` | Scrape GitHub PR comments |
| `scripts/scrape_youtube.py` | Scrape YouTube comments |
| `scripts/auto_annotate.py` | Automated annotation using LLM assistance |
| `scripts/gen_sarcasm.py` | Generate sarcasm examples |
| `scripts/generate_idioms.py` | Generate idiom-based adversarial examples |
| `scripts/generate_adversarial.py` | Generate adversarial test cases |
| `scripts/prepare_data.py` | Prepare data for training (format, split) |
| `scripts/train_baseline.py` | Train classical NLP baseline (TF-IDF + XGBoost) |
| `scripts/train_transformer.py` | Train transformer models |
| `scripts/train_and_save.py` | Train and export models for inference |
| `scripts/benchmark_baseline.py` | Benchmark model performance |

## Model Inference Architecture

The analyzer loads three independent single-task models:

1. **Risk model**: Regression head predicting a continuous 0-1 risk score
2. **Tone model**: Multi-label classification across 5 tones
3. **Intent model**: Multi-class classification across 4 intents

Each model uses `distilbert-base-multilingual-cased` as the encoder backbone.
Output is assembled into a structured `AnalysisResult` via post-processing (thresholds,
risk level discretization, explanation generation).

## Communication Style

- Code style: Follow [Ruff](https://docs.astral.sh/ruff/) rules configured in `pyproject.toml`
- Line length: 100 characters
- Python version: 3.10+ (use `from __future__ import annotations` for modern type hints)
- Docstrings: concise, describing the "why" not the "what"

## Where to Start

1. Read [ARCHITECTURE.md](ARCHITECTURE.md) for the full project vision
2. Read [Annotation Guidelines](docs/annotation_guidelines.md) to understand the labeling schema
3. Look at `src/keep_calm/analyzer.py` for the core inference logic
4. Look at `src/keep_calm/cli.py` for the CLI interface
5. Check `scripts/` for data pipeline and training workflows

## Questions?

Open an issue or discussion on the repository.
