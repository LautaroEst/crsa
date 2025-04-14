import pandas as pd
from pathlib import Path

mapping = {
    "cyan": "blue",  # used in game_properties files
    "blue": "blue",
    "Chartreuse": "green",  # used in game_properties files
    "green": "green",
    "yellow": "yellow",
    "diamond": "diamond",
    "rect": "square",  # used in game_properties files
    "square": "square",
    "circle": "circle",
    "right": "right",
    "left": "left",
    "middle": "middle",
    "top": "top",
    "bottom": "bottom",
    "not": "not",
    "no": "no",
    "yes": "yes",
}


def load_ground_truth(file):
    with open(file, "r") as f:
        rounds = []
        new_msg = []
        is_first = True
        for line in f:
            if is_first:
                path, rnd = line[:-1].split(" ")
                is_first = False
                continue
            if line.startswith("2017-"):
                rounds.append({
                    "path": path.split(".csv")[0],
                    "round": int(rnd),
                    "messages": new_msg
                })
                path, rnd = line[:-1].split(" ")
                new_msg = []
            else:
                new_msg.append(line[:-1])
        rounds.append({
            "path": path.split(".csv")[0],
            "round": int(rnd),
            "messages": new_msg
        })

    return pd.DataFrame(rounds)


def load_valid_rounds(root_dir):
    # Check if the clickedObj, game_properties, and message directories exist and have the same files
    clicked_paths = sorted([p.stem for p in (root_dir / "clickedObj").glob("*.csv") if not p.stem.startswith(".")])
    game_paths = sorted([p.stem for p in (root_dir / "game_properties").glob("*.csv") if not p.stem.startswith(".")])
    mess_paths = sorted([p.stem for p in (root_dir / "message").glob("*.csv") if not p.stem.startswith(".")])
    assert clicked_paths == game_paths == mess_paths

    # Check for valid paths and rounds
    valid_rounds = []
    empty_paths = 0
    all_words = {}
    tedGoodUnique = 0
    tedGood = 0
    ted = 0
    for path in clicked_paths:

        clicks = pd.read_csv(root_dir / f"clickedObj/{path}.csv", header=0, index_col=None)
        clicks.columns = clicks.columns.str.strip()
        msgs = pd.read_csv(root_dir / f"message/{path}.csv", header=0, index_col=None)
        msgs.columns = msgs.columns.str.strip()
        msgs = msgs.sort_values(by=["roundNum", "time"])
        props = pd.read_csv(root_dir / f"game_properties/{path}.csv", header=0, index_col=None)
        props.columns = props.columns.str.strip()

        # Discard .csv files that are empty
        if clicks.empty or msgs.empty or props.empty:
            empty_paths += 1
            continue

        for r in clicks["roundNum"].unique():
            if path.startswith("2017-9-22-19-4-11-90_2939-9aa") and r == 1:
                import pdb; pdb.set_trace()
            round_msgs = msgs.loc[msgs["roundNum"] == r,"contents"]
            corrected_content = []
            keep_round = True
            for s in round_msgs:
                s = s.lower().strip()
                tokens = s.lower().split(" ")
                newMsg = ""
                for t in tokens:
                    t = t.strip().lower()
                    ct = mapping.get(t.lower(), None)
                    if not t in all_words:
                        all_words[t] = 1
                        if ct is not None:
                            tedGoodUnique += 1
                    else:
                        x = all_words[t]
                        all_words[t] = x + 1
                    if ct is not None:
                        tedGood += 1
                    if ct is not None:
                        newMsg += " " + ct
                    ted += 1
                            
                newMsg = newMsg.lower().strip()
                tokens = newMsg.split(" ")
                # you cannot have no or yes in a message with size 2! or having not in a message with size 1
                for t in tokens:
                    t = t.strip().lower()
                    ct = mapping.get(t, None)
                    if (len(tokens) == 2 and (ct == "no" or ct == "yes")) or (len(tokens) == 1 and (ct == "not")):
                        newMsg = ""
                        break
                
                tokens = newMsg.split(" ")
                if len(tokens) > 1:
                    if tokens[0] < tokens[1]:
                        newMsg = tokens[0] + " " + tokens[1]
                    elif tokens[0] > tokens[1]:
                        newMsg = tokens[1] + " " + tokens[0]
                    else:
                        newMsg = tokens[0]
                if len(newMsg.strip()) > 0:
                    corrected_content.append(newMsg.lower().strip())
                else:
                    corrected_content.append("nonesense")
                    keep_round = False
                    break
            if keep_round:
                valid_rounds.append({
                    "path": path,
                    "round": r-1,
                    "messages": corrected_content
                })
    
    print(f"Discarded {empty_paths} empty paths.")
    print(f"Valid paths: {len(valid_rounds)}")

    df = pd.DataFrame(valid_rounds)

    return df


                


def main():
    df_truth = load_ground_truth("../pip/corpus_statistics.log").sort_values(by=["path", "round"]).reset_index(drop=True)
    df_truth["messages"] = df_truth["messages"].apply(lambda x: x[:-1])

    root_dir = Path("data") / "twoEnglishWords_bold"
    df = load_valid_rounds(root_dir)

    extra_rows = []
    for i, row in df.iterrows():
        # check if df_truth has a row matching this row
        if not ((df_truth["path"] == row["path"]) & (df_truth["round"] == row["round"]) & (df_truth["messages"].str.join("<EOM>") == "<EOM>".join(row["messages"]))).any():
            extra_rows.append(row)
    df_extra = pd.DataFrame(extra_rows)

    print("Extra rows in df:")
    print(df_extra)

    missing_rows = []
    for i, row in df_truth.iterrows():
        # check if df has a row matching this row
        if not ((df["path"] == row["path"]) & (df["round"] == row["round"]) & (df["messages"].str.join("<EOM>") == "<EOM>".join(row["messages"]))).any():
            missing_rows.append(row)
    df_missing = pd.DataFrame(missing_rows)
    
    print("Missing rows in df:")
    print(df_missing)
        


if __name__ == "__main__":
    main()