# NanoGPT

NanoGPT is a minimal, educational implementation of a transformer-based language model, inspired by ChatGPT. This project demonstrates how to train a basic transformer from scratch to replicate conversational AI behavior on your own data.

## Features

- Simple, readable codebase for learning and experimentation
- Implements the core transformer architecture (multi-head self-attention, feedforward, etc.)
- Trains on plain text data to generate ChatGPT-like responses
- Easily extensible for research and tinkering

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/nanogpt.git
cd nanogpt
```

### 2. Set Up Your Environment (with UV)

This project uses [UV](https://github.com/astral-sh/uv) for dependency management and virtual environments.

#### Using the Makefile

The Makefile provides convenient targets for setup:

- `make uv` – Installs UV if not already present
- `make uvlock` – Syncs project dependencies and creates a `uv.lock` file
- `make dotenv` – Initializes a `.env` file from `.env.template`
- `make venv` – Sets up the Python virtual environment and installs dependencies
- `make environment` – Runs all the above steps in order

**Quickstart:**

```bash
make environment
source .venv/bin/activate
```

### 3. Prepare Your Data

- Place your training data (plain text, one conversation per line) in a file, e.g., `data/train.txt`.
- You can use any text dataset, but conversational data works best for ChatGPT-like behavior.

### 4. Train the Model

```bash
python src/train.py --data_path data/train.txt --epochs 10 --batch_size 32
```

- Adjust hyperparameters as needed.
- Training progress and checkpoints will be saved in the `checkpoints/` directory.

### 5. Generate Text

After training, you can generate text with:

```bash
python src/generate.py --checkpoint checkpoints/latest.pt --prompt "Hello, how are you?"
```

## Project Structure

```
nanogpt/
  src/
    train.py         # Training script
    model.py         # Transformer model definition
    generate.py      # Text generation script
    utils.py         # Helper functions
  data/              # Place your training data here
  checkpoints/       # Model checkpoints
  README.md
  pyproject.toml
  Makefile
  uv.lock
```

## Customization

- **Model size:** Tweak the transformer's depth, width, and attention heads in `model.py`.
- **Data:** Train on your own conversations or any text corpus.
- **Sampling:** Adjust temperature and top-k in `generate.py` for more/less creative outputs.

## Acknowledgements

- Inspired by [Andrej Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT) and OpenAI's GPT models.
- For educational and research purposes.

## License

MIT License