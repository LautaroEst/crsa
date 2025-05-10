
import pandas as pd

class InfojigsawDataset:

    data_path = "data/twoEnglishWords_bold/processed.csv"

    def __init__(self):
        self.data = pd.read_csv(self.data_path, header=0, index_col=None)
        self.data["find"] = self.data.apply(self._get_target_name, axis=1)
        self._unique_scenarios = self.data.loc[:,["find","board_dims","board_shapes","board_colors","board_chars","board_numbers"]].drop_duplicates().reset_index(drop=True)

        meanings_letter_props = self._unique_scenarios.loc[:,["find","board_dims","board_shapes","board_colors","board_chars"]].drop_duplicates().reset_index(drop=True).to_records(index=False)
        self._meanings_letter_id_2_props = {f"scenario_{idx}_letter": row for idx, row in enumerate(meanings_letter_props)}
        meanings_number_props = self._unique_scenarios.loc[:,["find","board_dims","board_shapes","board_colors","board_numbers"]].drop_duplicates().reset_index(drop=True).to_records(index=False)
        self._meanings_number_id_2_props = {f"scenario_{idx}_number": row for idx, row in enumerate(meanings_number_props)}

    def __len__(self):
        return len(self.data)
    
    def _get_target_name(self, row):
        chars = row["board_chars"].split(",")
        nums = row["board_numbers"].split(",")
        target_i, target_j = row["target_pos"].split(",")
        rows, columns = row["board_dims"].split(",")
        idx = int(rows) * int(target_j) + int(target_i)
        target = chars[idx] + nums[idx]
        return target
    
    def iter_samples(self):
        for i, row in self.data.iterrows():

            for idx, props in self._meanings_letter_id_2_props.items():
                if row["find"] == props["find"] and \
                row["board_dims"] == props["board_dims"] and \
                row["board_shapes"] == props["board_shapes"] and \
                row["board_colors"] == props["board_colors"] and \
                row["board_chars"] == props["board_chars"]:
                    meaning_letter = idx
                    break

            for idx, props in self._meanings_number_id_2_props.items():
                if row["find"] == props["find"] and \
                row["board_dims"] == props["board_dims"] and \
                row["board_shapes"] == props["board_shapes"] and \
                row["board_colors"] == props["board_colors"] and \
                row["board_numbers"] == props["board_numbers"]:
                    meaning_number = idx
                    break

            y = row["clicked_pos"]
            starter_speaker = row["starter_player"]
            meanings_A = meaning_letter if starter_speaker == "playerChar" else meaning_number
            meanings_B = meaning_number if starter_speaker == "playerChar" else meaning_letter

            yield i, meanings_A, meanings_B, y, row["original_messages"].split("<EOM>")

    @property
    def world(self):
        return {}
