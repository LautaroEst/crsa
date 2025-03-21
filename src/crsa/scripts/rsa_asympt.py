
from pathlib import Path
import shutil
import sys
import numpy as np
import matplotlib.pyplot as plt
import yaml

from ..src.utils import read_config_file
from ..src.rsa import RSA

def main(config, root_results_dir, alphas_list, depth):
    entropies_list = []
    values_list = []
    gains_list = []
    last_speakers = []

    for new_alpha in alphas_list:
        new_alpha = float(new_alpha)
        depth = int(depth)
        results_dir = root_results_dir / f"alpha={new_alpha}" / f"max_depth={depth}_tolerance=None"
        history = np.load(results_dir / "history.npy", allow_pickle=True)
        gain_history = np.load(results_dir / "gain.npy", allow_pickle=True).item()
        rsa = RSA(config["meanings"], config["utterances"], config["lexicon"], config["prior"], config["cost"], new_alpha, depth, tolerance=None)

        last_speakers.append(history[-1]["speaker"].values)

        entropies_list.append(gain_history["cond_entropy"])
        values_list.append(gain_history["listener_value"])
        gains_list.append(gain_history["gain"])

    import pdb; pdb.set_trace()

    fig, axs = plt.subplots(len(alphas_list)+1, 4, figsize=(6*4, 4*(len(alphas_list)+1)))

    fig_bis, axs_bis = plt.subplots(1, 1+len(alphas_list), figsize=((1+len(alphas_list))*4, 4))

    fig_CI, axs_CI = plt.subplots(1, len(alphas_list), figsize=(len(alphas_list)*4, 3))

    axs[0,0].remove()
    im = axs[0,1].imshow(np.round(rsa.lexicon,3), cmap='viridis', interpolation='nearest')
    cbar = fig.colorbar(im, ax=axs[0,1])
    for (j, k), value in np.ndenumerate(np.round(rsa.lexicon,3)):
            axs[0,1].text(k, j, f'{value:.2f}', ha='center', va='center', color='black')

    axs_bis[0].imshow(np.round(rsa.lexicon,3), cmap='viridis', interpolation='nearest')
    axs_bis[0].set_title(f"Initial Lexicon")
    for (j, k), value in np.ndenumerate(np.round(rsa.lexicon,3)):
            axs_bis[0].text(k, j, f'{value:.2f}', ha='center', va='center', color='black')

    axs[0,1].set_title("Initial Lexicon")
    axs[0,2].remove()
    axs[0,3].remove()

    axs[1,0].set_title("Conditional entropy")
    axs[1,1].set_title("Listener value")
    axs[1,2].set_title("Gain function")
    axs[1,3].set_title("Speaker")        

    for i in range(len(alphas_list)):
        axs[1+i,0].set_ylabel(f"Alpha={alphas_list[i]}")
        axs[1+i,0].plot(entropies_list[i])

        axs_bis[i+1].imshow(np.round(last_speakers[i],3), cmap='viridis', interpolation='nearest')
        axs_bis[i+1].set_title(f"Final Speaker for Alpha={alphas_list[i]}")
        for (j, k), value in np.ndenumerate(np.round(last_speakers[i],3)):
            axs_bis[i+1].text(k, j, f'{value:.2f}', ha='center', va='center', color='black')

        axs[1+i,1].plot(values_list[i])

        axs[1+i,2].plot(gains_list[i])

        im = axs[1+i,3].imshow(np.round(last_speakers[i],3), cmap='viridis', interpolation='nearest')
        cbar = fig.colorbar(im, ax=axs[1+i,3])
        for (j, k), value in np.ndenumerate(np.round(last_speakers[i],3)):
            axs[i+1,3].text(k, j, f'{value:.2f}', ha='center', va='center', color='black')

    plt.savefig(root_results_dir / "asymptotic_analysis.png")


def setup():

    # Read configuration file
    if len(sys.argv) < 2:
        raise ValueError("Please provide a configuration file")
    config_file = sys.argv[1]
    config = read_config_file(f"{config_file}")

    # Check if output directory already exists
    output_dir = Path("outputs") / config_file
    if not output_dir.exists():
        raise ValueError(f"Output directory {output_dir} does not exist. Please run RSA first.")

    return config, output_dir

if __name__ == '__main__':
    config, output_dir = setup()
    main(config, output_dir, [0.5, 1, 2], depth=15)    