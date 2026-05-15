import argparse
import torch
import random
import numpy as np
import os
import csv
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001
CHECKPOINT_EVERY = 50


class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        self.model_path = os.path.join("model", "model.pth")
        self.best_model_path = os.path.join("model", "best_model.pth")
        self.metrics_path = os.path.join("model", "training_metrics.csv")

    def save_checkpoint(self, record, total_score, plot_scores, plot_mean_scores):
        model_folder_path = "./model"
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        tmp_path = f"{self.model_path}.tmp"
        backup_path = f"{self.model_path}.bak"

        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.trainer.optimizer.state_dict(),
                "n_games": self.n_games,
                "record": record,
                "total_score": total_score,
                "plot_scores": plot_scores,
                "plot_mean_scores": plot_mean_scores,
                "memory": list(self.memory),
            },
            tmp_path,
        )

        if os.path.exists(self.model_path):
            os.replace(self.model_path, backup_path)
        os.replace(tmp_path, self.model_path)

    def save_best_model(self):
        model_folder_path = "./model"
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        torch.save(self.model.state_dict(), self.best_model_path)

    def write_metrics(self, plot_scores, plot_mean_scores):
        model_folder_path = "./model"
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        with open(self.metrics_path, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "game",
                    "score",
                    "mean_score",
                    "record",
                    "epsilon",
                ]
            )
            record = 0
            for idx, score in enumerate(plot_scores, start=1):
                record = max(record, score)
                mean_score = plot_mean_scores[idx - 1]
                epsilon = 80 - idx
                writer.writerow(
                    [
                        idx,
                        score,
                        mean_score,
                        record,
                        epsilon,
                    ]
                )

    def load_checkpoint(self):
        if not os.path.exists(self.model_path):
            return None

        try:
            checkpoint = torch.load(self.model_path, map_location=torch.device("cpu"))
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.n_games = checkpoint.get("n_games", 0)
            self.memory = deque(checkpoint.get("memory", []), maxlen=MAX_MEMORY)
            return checkpoint
        except (KeyError, RuntimeError, ValueError, TypeError) as exc:
            print(f"Checkpoint load failed: {exc}")
            return None

    def load_model_weights(self, weights_path):
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Model file not found: {weights_path}")

        state = torch.load(weights_path, map_location=torch.device("cpu"))
        if isinstance(state, dict) and "model_state_dict" in state:
            self.model.load_state_dict(state["model_state_dict"])
        else:
            self.model.load_state_dict(state)

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r))
            or (dir_l and game.is_collision(point_l))
            or (dir_u and game.is_collision(point_u))
            or (dir_d and game.is_collision(point_d)),
            # Danger right
            (dir_u and game.is_collision(point_r))
            or (dir_d and game.is_collision(point_l))
            or (dir_l and game.is_collision(point_u))
            or (dir_r and game.is_collision(point_d)),
            # Danger left
            (dir_d and game.is_collision(point_r))
            or (dir_u and game.is_collision(point_l))
            or (dir_r and game.is_collision(point_u))
            or (dir_l and game.is_collision(point_d)),
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            # Food location
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y,  # food down
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append(
            (state, action, reward, next_state, done)
        )  # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        # for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move

    def get_action_eval(self, state):
        state0 = torch.tensor(state, dtype=torch.float)
        with torch.no_grad():
            prediction = self.model(state0)
        move = torch.argmax(prediction).item()
        final_move = [0, 0, 0]
        final_move[move] = 1
        return final_move


def train():
    agent = Agent()
    checkpoint = agent.load_checkpoint()

    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    if checkpoint is not None:
        plot_scores = checkpoint.get("plot_scores", [])
        plot_mean_scores = checkpoint.get("plot_mean_scores", [])
        total_score = checkpoint.get("total_score", sum(plot_scores))
        record = checkpoint.get("record", max(plot_scores) if plot_scores else 0)
        print(f"Resumed checkpoint from {agent.model_path}")

    game = SnakeGameAI()
    try:
        while True:
            # get old state
            state_old = agent.get_state(game)

            # get move
            final_move = agent.get_action(state_old)

            # perform move and get new state
            reward, done, score = game.play_step(final_move)
            state_new = agent.get_state(game)

            # train short memory
            agent.train_short_memory(state_old, final_move, reward, state_new, done)

            # remember
            agent.remember(state_old, final_move, reward, state_new, done)

            if done:
                # train long memory, plot result
                game.reset()
                agent.n_games += 1
                agent.train_long_memory()

                new_record = False
                if score > record:
                    record = score
                    new_record = True
                    agent.save_best_model()

                print("Game", agent.n_games, "Score", score, "Record:", record)

                plot_scores.append(score)
                total_score += score
                mean_score = total_score / agent.n_games
                plot_mean_scores.append(mean_score)
                if new_record or agent.n_games % CHECKPOINT_EVERY == 0:
                    agent.save_checkpoint(
                        record, total_score, plot_scores, plot_mean_scores
                    )
                plot(plot_scores, plot_mean_scores)
    except KeyboardInterrupt:
        print("Training interrupted. Saving checkpoint and metrics...")
    finally:
        if agent.n_games > 0:
            agent.save_checkpoint(record, total_score, plot_scores, plot_mean_scores)
            agent.write_metrics(plot_scores, plot_mean_scores)


def run_trained(model_path=None):
    agent = Agent()
    if model_path is None:
        if os.path.exists(agent.best_model_path):
            model_path = agent.best_model_path
        else:
            model_path = agent.model_path

    agent.load_model_weights(model_path)
    game = SnakeGameAI()
    while True:
        state = agent.get_state(game)
        final_move = agent.get_action_eval(state)
        _, done, score = game.play_step(final_move)
        if done:
            print("Final Score", score)
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or run the Snake agent")
    parser.add_argument(
        "--play",
        action="store_true",
        help="Run a trained model instead of training",
    )
    parser.add_argument(
        "--model-path",
        help="Path to a model file (best_model.pth or model.pth)",
    )
    args = parser.parse_args()

    if args.play:
        run_trained(args.model_path)
    else:
        train()
