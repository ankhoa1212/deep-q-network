import argparse
import csv
import os
import matplotlib.pyplot as plt


def load_metrics(csv_path):
    games = []
    scores = []
    mean_scores = []
    records = []

    with open(csv_path, "r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            games.append(int(row["game"]))
            scores.append(float(row["score"]))
            mean_scores.append(float(row["mean_score"]))
            records.append(float(row["record"]))

    return games, scores, mean_scores, records


def main():
    parser = argparse.ArgumentParser(
        description="Plot training metrics from model/training_metrics.csv"
    )
    parser.add_argument(
        "--save-png",
        metavar="PATH",
        help="Save the plot to a PNG file instead of showing it",
    )
    args = parser.parse_args()

    csv_path = os.path.join("model", "training_metrics.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            "training_metrics.csv not found. Run training to generate it."
        )

    games, scores, mean_scores, records = load_metrics(csv_path)

    plt.plot(games, scores, label="score")
    plt.plot(games, mean_scores, label="mean_score")
    plt.plot(games, records, label="record")
    plt.xlabel("game")
    plt.ylabel("score")
    plt.legend()
    plt.title("Training Metrics")
    if args.save_png:
        plt.savefig(args.save_png, dpi=150, bbox_inches="tight")
    else:
        plt.show()


if __name__ == "__main__":
    main()
