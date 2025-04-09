
import pandas as pd
from pathlib import Path

def consecutive(series):
    return any(series.iloc[i] == series.iloc[i+1] for i in range(len(series)-1))

def normalize(msg):
    # Normalize the message by removing punctuation and converting to lowercase
    # msg = msg.replace("'", "").replace('"', "").replace(",", "").replace(".", "").replace("!", "").replace("?", "")
    return msg.lower().strip()

def check_valid_paths(root_dir: Path):

    # Check if the clickedObj, game_properties, and message directories exist and have the same files
    clicked_paths = sorted([p.stem for p in (root_dir / "clickedObj").glob("*.csv")])
    game_paths = sorted([p.stem for p in (root_dir / "game_properties").glob("*.csv")])
    mess_paths = sorted([p.stem for p in (root_dir / "message").glob("*.csv")])
    assert clicked_paths == game_paths == mess_paths

    # Check for valid paths and rounds
    valid_paths = []
    valid_rounds = {}
    empty_paths = consecutives_msgs = round_info_incomplete = repeated_gameid = 0
    unique_gameids = []
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
    
        # Check that there is only one gameid in each file and that they are the same
        assert clicks["gameid"].nunique() == 1 
        assert msgs["gameid"].nunique() == 1 
        assert props["gameid"].nunique() == 1 
        assert clicks["gameid"].values[0] == msgs["gameid"].values[0] == props["gameid"].values[0]
        
        # Check that the gameid is in the path and correct it to avoid repetitions
        gameid = clicks["gameid"].values[0]
        assert path.split("_")[1].startswith(gameid)
        gameid = path.split("_")[1][:7]
        if gameid not in unique_gameids:
            unique_gameids.append(gameid)
        else:
            repeated_gameid += 1
            continue

        # Check for valid rounds within path
        has_consecutive = msgs.sort_values(["roundNum","time"]).loc[:,["roundNum","sender"]].groupby("roundNum").agg(consecutive)
        valid_path_rounds = []
        for r in range(1, 11):

            # Check that the roundNum is in all three files
            if r not in clicks["roundNum"].values:
                round_info_incomplete += 1
                continue
            if r not in msgs["roundNum"].values:
                round_info_incomplete += 1
                continue
            if r-1 not in props["roundNum"].values:
                round_info_incomplete += 1
                continue
            
            # Check that messages does not contain two consecutive identical speakers in the same round, e.g. ["playerNum", "playerNum", "playerChar"]
            if has_consecutive.loc[r,"sender"]:
                consecutives_msgs += 1
                continue

            # keep round as valid
            valid_path_rounds.append(r)

        if len(valid_path_rounds) == 0:
            empty_paths += 1
            continue

        # Save valid paths
        valid_rounds[path] = valid_path_rounds
        valid_paths.append(path)

    # Check that the clicker is not the one that ends the conversation
    clickers = clicks.set_index("roundNum").loc[valid_rounds[path],"clicker"].values
    msgs_enders = msgs.loc[msgs.groupby("roundNum").idxmax()["time"].values,["roundNum","sender"]].set_index("roundNum").loc[valid_rounds[path],"sender"].values
    assert all(clickers != msgs_enders), path

    # Check that the gameid is not repeated
    assert repeated_gameid == 0

    print(f"Found {empty_paths} empty paths")
    print(f"Found {consecutives_msgs} rounds with consecutive messages")
    print(f"Found {round_info_incomplete} rounds with incomplete information")
    print(f"Total number of valid paths: {len(valid_paths)}")
    print(f"Total number of valid rounds: {sum(len(v) for v in valid_rounds.values())}")

    return valid_paths, valid_rounds

def load_data(paths, valid_rounds, root_dir: Path):
    data = []
    for path in paths:
        clicks = pd.read_csv(root_dir / f"clickedObj/{path}.csv", header=0, index_col=None)
        clicks.columns = clicks.columns.str.strip()
        msgs = pd.read_csv(root_dir / f"message/{path}.csv", header=0, index_col=None)
        msgs.columns = msgs.columns.str.strip()
        msgs = msgs.sort_values(by=["roundNum", "time"])
        props = pd.read_csv(root_dir / f"game_properties/{path}.csv", header=0, index_col=None)
        props.columns = props.columns.str.strip()
        props["roundNum"] = props["roundNum"] + 1
        for r in range(1, 11):
            if r not in valid_rounds[path]:
                continue
            round_props = props[props["roundNum"] == r]
            round_msgs = msgs[msgs["roundNum"] == r].sort_values("time")
            board_dims = (round_props["pos_x"].max() + 1, round_props["pos_y"].max() + 1)
            target_pos = tuple(round_props.loc[round_props["goal"] == 1, ["pos_x", "pos_y"]].values[0])
            clicked_pos = tuple(clicks.loc[clicks["roundNum"] == r, ["pos_x", "pos_y"]].values[0])
            msgs_content = "<EOM>".join(normalize(m) for m in round_msgs["contents"])
            data.append({
                "gameid": round_props["gameid"].values[0],
                "round": r,
                "agent_A": round_msgs["sender"].values[0],
                "agent_B": ({"playerNum", "playerChar"} - {round_msgs["sender"].values[0]}).pop(),
                "target_pos": target_pos,
                "clicked_pos": clicked_pos,
                "board_dims": board_dims,
                "board_shapes": round_props["shape"].values,
                "board_colors": round_props["color"].values,
                "board_chars": round_props["char"].values,
                "board_numbers": round_props["num"].values,
                "messages": msgs_content,
            })
    df = pd.DataFrame(data).set_index(["gameid","round"])
    return df


def main():

    # Output directory
    output_dir = Path("outputs") / "preprocessed_data"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "infojigsaw.csv"

    # Check paths
    root_dir = Path("data") / "twoEnglishWords_bold"
    paths, valid_rounds = check_valid_paths(root_dir)
    
    # Load data and save to CSV
    df = load_data(paths, valid_rounds, root_dir)
    import pdb; pdb.set_trace()
    df.to_csv(output_path, index=True, header=True)


if __name__ == "__main__":
    main()