from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

THIS_FOLDER = Path(__file__).resolve().parent
OUTPUT_FOLDER = THIS_FOLDER / "dissertation_outputs"

TASK_TYPES = {
    "Fixed-Text": "fixed_normal_across_sessions",
    "Semi-Fixed Text": "semi_fixed_normal_across_sessions",
    "Free-Text": "free_normal_across_sessions",
    "Rushed": "fixed_normal_to_rushed",
    "Distracted": "fixed_normal_to_distracted",
}
SESSIONS = [3, 4, 5]


def plot_task_type(task_name, experiment_name):
    path = OUTPUT_FOLDER / "experiments" / experiment_name / "experiment_results.csv"
    results = pd.read_csv(path)
    results.columns = results.columns.str.strip()  # guards against stray spaces in headers
    count_columns = ["final_test_session", "genuine_accepts", "false_rejections",
                      "false_accepts", "impostor_rejections"]
    for col in count_columns:
        results[col] = pd.to_numeric(results[col], errors="coerce")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax, session in zip(axes, SESSIONS):
        row = results[results["final_test_session"] == session].iloc[0]
        matrix = np.array([
            [row["genuine_accepts"], row["false_rejections"]],
            [row["false_accepts"], row["impostor_rejections"]],
        ])
        ax.imshow(matrix, cmap="Blues")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Accept", "Reject"])
        ax.set_yticklabels(["Genuine", "Impostor"])
        ax.set_title(f"Session {session}", fontsize=12, fontweight="bold")
        ax.xaxis.set_label_position("top")
        ax.xaxis.tick_top()
        max_val = matrix.max()
        for i in range(2):
            for j in range(2):
                val = matrix[i, j]
                colour = "white" if val > max_val * 0.5 else "black"
                ax.text(j, i, f"{val:,.0f}", ha="center", va="center", fontsize=13, color=colour)

    fig.suptitle(f"{task_name} Confusion Matrices", fontsize=15, fontweight="bold", y=1.05)
    plt.tight_layout()

    safe_name = task_name.lower().replace(" ", "_").replace("-", "_")
    out_path = OUTPUT_FOLDER / "figures" / f"confusion_{safe_name}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    for task_name, experiment_name in TASK_TYPES.items():
        plot_task_type(task_name, experiment_name)