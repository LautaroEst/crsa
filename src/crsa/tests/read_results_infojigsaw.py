
import pandas as pd


path = "run_infojigsaw.target_pos_2.0"
# path = "run_infojigsaw.clicked_pos"
# path = "run_infojigsaw.target_pos"

def main():
    results = pd.read_csv(f"outputs/{path}/alpha=2.0/max_depth=inf_tolerance=0.001/seed=1234/results.csv", header=0, index_col=None)
    results = results.groupby(["model","sample_id"]).apply(lambda x: x.sort_values("turn").iloc[-1]["nll"]).reset_index()
    print(results.groupby("model")[0].agg(
        median="median",
        q1=lambda x: x.quantile(0.25),
        q3=lambda x: x.quantile(0.75),
    ))


if __name__ == "__main__":
    main()