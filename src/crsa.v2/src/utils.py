import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt

INF = 1e10
ZERO = 1e-10


metric2name = {
    "accuracy": "Average of correct guessings",
    "nll": "Listener Cross-entropy"
}

def model2name(model_name):
    if model_name == "crsa":
        return "CRSA",
    elif model_name == "memoryless_rsa":
        return "RSA on each turn (no history)"
    elif model_name == "memoryless_literal":
        return "Literal model on each turn"
    elif model_name == "prior_model":
        return "Random (prior)"
    elif model_name.startswith("llm_"):
        return f"LLM {model_name[4:]}"
    elif model_name.startswith("llmrsa_"):
        return f"LLM-RSA {model_name[7:]}"
    else:
        raise ValueError(f"Model {model_name} not recognized.")

def save_yaml(data, file_path):
    with open(file_path, "w") as f:
        yaml.safe_dump(data, f, default_flow_style=True)


def read_config_file(config_file):
    """
    Read the configuration file and return the configuration dictionary
    """
    with open(f"configs/{config_file}.yaml") as f:
        config = yaml.safe_load(f)

    if config is None:
        config = {}
            
    return config


def is_list_of_strings(lst):
    """
    Check if the input is a list of strings
    """

    if not isinstance(lst, list):
        return False

    if not all(isinstance(x, str) for x in lst):
        return False
    
    return True

def is_numeric_ndarray(obj):
    return isinstance(obj, np.ndarray) and (obj.dtype.kind == 'f' or obj.dtype.kind == 'i') and np.isnan(obj).sum() == 0

def is_list_of_numbers(lst):
    return isinstance(lst, list) and all(isinstance(x, float) or isinstance(x, int) for x in lst)

def is_positive_number(obj):
    return isinstance(obj, float) or isinstance(obj, int) and obj > 0

def is_positive_integer(obj):
    return isinstance(obj, int) and obj > 0

def is_list_of_list_of_numbers(lst):
    return isinstance(lst, list) and all(is_list_of_numbers(x) for x in lst)


def compute_metric(probs, y, metric):
    if metric == "accuracy":
        return (probs.argmax(axis=1) == y).astype(float)
    elif metric == "nll":
        return -np.log(probs[np.arange(probs.shape[0]),y])
    else:
        raise ValueError(f"Metric {metric} not supported")


def compute_metrics(category_dist, y, categories, metrics):
    y_vec = categories.index(y) * np.ones(category_dist.shape[0],dtype=int)
    results = {}
    for metric in metrics:
        results[metric] = compute_metric(category_dist, y_vec, metric)
    return results


def plot_results(df, models, alpha, max_depth, tolerance, metrics, output_dir):
    
    fig, ax = plt.subplots(1, len(metrics), figsize=(12, 6))
    if len(metrics) == 1:
        ax = np.array([ax])
    for i, metric in enumerate(metrics):
        df_metric = df.groupby(["model","turn"]).mean().reset_index()
        for c, model in enumerate(models):
            model_df = df_metric[df_metric["model"] == model].sort_values("turn")
            ax[i].plot(model_df["turn"], model_df[metric], label=model2name(model), linestyle="--", linewidth=2, color=f"C{c}")
            # ax[i].errorbar(
            #     model_df["turn"], model_df["mean"], yerr=model_df["std"], 
            #     fmt="o", capsize=5, capthick=2, elinewidth=2, markersize=5, color=f"C{c}",
            # )
            ax[i].set_ylabel(metric2name[metric])
            ax[i].set_xlabel("Turn")
            ax[i].grid(True)
            ax[i].set_xticks(model_df["turn"].astype(int))
    fig.suptitle(f"Model results over Turns for alpha={alpha}, max_depth={max_depth}, tolerance={tolerance}")
    ax[-1].legend(loc="lower center", bbox_to_anchor=(-0.1, -0.2), fontsize=12, ncol=4)
    plt.savefig(output_dir / f"scores_alpha={alpha}.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)