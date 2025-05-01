import json
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def extract_eval_value(state, metric):
    return [(log["epoch"], log[metric]) for log in state["log_history"] if metric in log]


def plot_all_metrics(metrics, base_state, adapt_state):
    sns.set_theme(style="whitegrid")
    colors = sns.color_palette("Set2", 2)

    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    axs = axs.flatten()

    for i, metric in enumerate(metrics):
        epochs_base, base_score = zip(*extract_eval_value(base_state, metric))
        epochs_adapt, adapt_score = zip(*extract_eval_value(adapt_state, metric))

        axs[i].plot(epochs_base, base_score, label="Base", color=colors[0], marker='o')
        axs[i].plot(epochs_adapt, adapt_score, label="Adapt", color=colors[1], marker='o')
        axs[i].set_title(metric)
        axs[i].set_xlabel("Epoch")
        axs[i].set_ylabel(metric)
        axs[i].legend()

    plt.tight_layout()
    plt.savefig("output/plots/all_metrics_vs_epoch.jpg", dpi=300)
    # plt.show()


def plot_all_metric_comparisons(log_df):
    sns.set_theme(style="whitegrid")
    colors = sns.color_palette("Set2", 2)

    metrics = [
        "Loss", "Accuracy", "F1", "MCC",
        "Precision", "Recall", "TrainingTime", "InferenceTime"
    ]

    fig, axs = plt.subplots(4, 2, figsize=(14, 16))
    axs = axs.flatten()

    for i, metric in enumerate(metrics):
        values = [log_df[metric][0], log_df[metric][1]]
        bars = axs[i].bar(["Base", "Adapt"], values, color=colors)
        axs[i].set_title("Adapt vs Base " + metric)
        axs[i].set_ylabel(metric)

        # # Add value labels on each bar
        # for bar in bars:
        #     height = bar.get_height()
        #     axs[i].text(bar.get_x() + bar.get_width()/2.0, height - 0.1, f"{height:.3f}", 
        #                 ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig("output/plots/adapt_vs_base_all_metrics.jpg", dpi=300)
    # plt.show()



if __name__ == "__main__":
    # search for the JSON files in the specified directories
    # base_path = glob.glob('output/without_adapt/checkpoint-*/trainer_state.json')
    # adapt_path = glob.glob('output/with_adapt/checkpoint-*/trainer_state.json')
    base_path = glob.glob('/Users/vedantmahangade/Projects/domain_adaptive_llm/output/results/without_adapt.json')
    adapt_path = glob.glob('/Users/vedantmahangade/Projects/domain_adaptive_llm/output/results/with_adapt.json')

    # Load both JSON files
    with open(base_path[0]) as f:
        base_state = json.load(f)

    with open(adapt_path[0]) as f:
        adapt_state = json.load(f)

    metrics = [
        "eval_loss",
        "eval_f1",
        "eval_matthews_correlation",
        "eval_accuracy",
        "eval_precision",
        "eval_recall"
    ]

    plot_all_metrics(metrics, base_state, adapt_state)


    log = pd.read_csv('output/results/log.csv')
    plot_all_metric_comparisons(log)