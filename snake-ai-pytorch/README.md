# Teach AI To Play Snake! Reinforcement Learning With PyTorch and Pygame

In this Python Reinforcement Learning Tutorial series we teach an AI to play Snake! We build everything from scratch using Pygame and PyTorch. The tutorial consists of 4 parts:

You can find all tutorials on my channel: [Playlist](https://www.youtube.com/playlist?list=PLqnslRFeH2UrDh7vUmJ60YrmWd64mTTKV)

- Part 1: I'll show you the project and teach you some basics about Reinforcement Learning and Deep Q Learning.
- Part 2: Learn how to setup the environment and implement the Snake game.
- Part 3: Implement the agent that controls the game.
- Part 4: Implement the neural network to predict the moves and train it.

## Environment Setup With uv

From the `snake-ai-pytorch` directory, create and sync a virtual environment with uv:

```bash
uv sync
```

If you do not already have uv installed, install it first from the official instructions, then run `uv sync` again.

To use the environment in your shell, activate it with:

```bash
source .venv/bin/activate
```

Run the training script with:

```bash
uv run python agent.py
```

If `model/model.pth` exists, training resumes from the full checkpoint automatically. It stores the model, optimizer, episode count, replay memory, and score history. Delete the file if you want to start from scratch. The best model weights are saved separately to `model/best_model.pth` when a new record is reached. Checkpoints are written atomically, with a backup at `model/model.pth.bak` if you need to restore a previous save.

To run a trained model instead of training:

```bash
uv run python agent.py --play
```

To run a specific checkpoint or weight file:

```bash
uv run python agent.py --play --model-path model/best_model.pth
```

Run the human-play version with:

```bash
uv run python snake_game_human.py
```

If pygame installation fails on Linux, install the SDL development packages provided by your distribution and rerun `uv sync`.

## Environment Setup With pip

Alternatively, you can use pip and a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the training script with:

```bash
python agent.py
```

If `model/model.pth` exists, training resumes from the full checkpoint automatically. It stores the model, optimizer, episode count, replay memory, and score history. Delete the file if you want to start from scratch. The best model weights are saved separately to `model/best_model.pth` when a new record is reached. Checkpoints are written atomically, with a backup at `model/model.pth.bak` if you need to restore a previous save.

To run a trained model instead of training:

```bash
python agent.py --play
```

To run a specific checkpoint or weight file:

```bash
python agent.py --play --model-path model/best_model.pth
```

## Training Metrics Log

Training metrics are written to `model/training_metrics.csv` when training exits (including Ctrl+C).

To plot the metrics without additional dependencies:

```bash
uv run python plot_metrics.py
```

To save the plot as a PNG instead of opening a window:

```bash
uv run python plot_metrics.py --save-png training_metrics.png
```

Run the human-play version with:

```bash
python snake_game_human.py
```
