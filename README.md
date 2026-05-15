# deep-q-network

Applying Deep Q-Network to play a game.

This repository contains the Snake reinforcement learning project as a git submodule in [snake-ai-pytorch](snake-ai-pytorch).

Requirements:

- Python 3.11 or newer
- uv installed on your system

Use the submodule directory for environment setup, dependency installation, and running the code:

```bash
cd snake-ai-pytorch
uv sync
uv run python agent.py
```

For the full setup notes and alternate run commands, see [snake-ai-pytorch/README.md](snake-ai-pytorch/README.md).

## Fallback Without The Submodule

If the git submodule does not work for you, use a zip archive of the `snake-ai-pytorch` folder instead.

Unpack the archive into this repository so the project still lives at `snake-ai-pytorch/`, then run the same commands from that directory:

```bash
unzip snake-ai-pytorch.zip
cd snake-ai-pytorch
uv sync
uv run python agent.py
```

If the zip file already contains the folder, you only need to extract it and move into the directory before running `uv sync`.
