# deep-q-network

Applying Deep Q-Network to play a game.

This repository contains the Snake reinforcement learning project built with PyTorch and Pygame.

## Requirements

- Python 3.11 or newer

## Quick Start

From the `snake-ai-pytorch` directory, create and sync the environment, then run the training.

**With uv:**

```bash
cd snake-ai-pytorch
uv sync
uv run python agent.py
```

**With pip:**

```bash
cd snake-ai-pytorch
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python agent.py
```

For detailed setup notes, run options, and training time expectations, see [snake-ai-pytorch/README.md](snake-ai-pytorch/README.md).
