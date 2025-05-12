import numpy as np
import matplotlib.pyplot as plt

metric2name = {
    "accuracy": "Average of correct guessings",
    "nll": "Listener Cross-entropy",
    "igain": "Information Gain w.r.t the Prior",
}

model2name = {
    "crsa": "CRSA",
    "rsa": "RSA",
    "literal": "Literal",
    "prior": "Prior",
}


def compute_metric(probs, target_cat, metric):
    if metric == "accuracy":
        return (probs.argmax() == target_cat).astype(float)
    elif metric == "nll":
        return -np.log(probs[target_cat])
    else:
        raise ValueError(f"Metric {metric} not supported")


def plot_turns(df, models, metrics, output_dir):

    fig, ax = plt.subplots(1, len(metrics), figsize=(12, 6))
    if len(metrics) == 1:
        ax = np.array([ax])
    for i, metric in enumerate(metrics):
        df_metric = df.groupby(["model", "turn"]).agg({metric: "mean"}).reset_index()
        for c, model in enumerate(models):
            model_df = df_metric[df_metric["model"] == model].sort_values("turn")
            ax[i].plot(model_df["turn"], model_df[metric], label=model2name[model], linestyle="--", linewidth=2, color=f"C{c}")
            ax[i].set_ylabel(metric2name[metric])
            ax[i].set_xlabel("Turn")
            ax[i].grid(True)
            ax[i].set_xticks(model_df["turn"].astype(int))
    ax[-1].legend(loc="lower center", bbox_to_anchor=(-0.1, -0.2), fontsize=12, ncol=4)
    plt.savefig(output_dir / f"scores.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)