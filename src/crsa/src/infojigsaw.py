
import numpy as np
import pandas as pd


mapping = {
    "left": "position_x",
    "right": "position_x",
    "middle": "position_xy",
    "top": "position_y",
    "bottom": "position_y",
    "square": "shape",
    "circle": "shape",
    "rect": "shape",
    "diamond": "shape",
    "blue": "color",
    "green": "color",
    "yellow": "color",
    "not": "not",
}

def position_to_mask(position, board_dims):
    if position == "left" and board_dims == "2,3":
        return np.array([1, 1, 0, 0, 0, 0])
    elif position == "left" and board_dims == "3,2":
        return np.array([1, 1, 1, 0, 0, 0])
    elif position == "middle" and board_dims == "2,3":
        return np.array([0, 0, 1, 1, 0, 0])
    elif position == "middle" and board_dims == "3,2":
        return np.array([0, 1, 0, 1, 0, 1])
    elif position == "right" and board_dims == "2,3":
        return np.array([0, 0, 0, 0, 1, 1])
    elif position == "right" and board_dims == "3,2":
        return np.array([0, 0, 0, 1, 1, 1])
    elif position == "top" and board_dims == "2,3":
        return np.array([1, 0, 1, 0, 1, 0])
    elif position == "top" and board_dims == "3,2":
        return np.array([1, 0, 0, 1, 0, 0])
    elif position == "bottom" and board_dims == "2,3":
        return np.array([0, 1, 0, 1, 0, 1])
    elif position == "bottom" and board_dims == "3,2":
        return np.array([0, 0, 1, 0, 0, 1])
    else:
        raise ValueError(f"Unknown position: {position} for board_dims: {board_dims}")
    
def shape_to_mask(shape, shapes):
    return np.array([int(s == shape) for s in shapes.split(",")])

def color_to_mask(color, colors):
    return np.array([int(c == color) for c in colors.split(",")])

class InfoJigsawDataset:

    data_path = "data/twoEnglishWords_bold/processed.csv"

    def __init__(self, model="target_pos"):
        if model not in ["target_pos", "clicked_pos"]:
            raise ValueError(f"Model represents what the samples should model. Must be 'target_pos' or 'clicked_pos'.")
        self.model = model
        self.data, self.utterance_counts = self._load_data()
        self.data["find"] = self.data.apply(self._get_target_name, axis=1)
        self._unique_scenarios = self.data.loc[:,["find","board_dims","board_shapes","board_colors","board_chars","board_numbers"]].drop_duplicates().reset_index(drop=True)

        meanings_letter_props = self._unique_scenarios.loc[:,["find","board_dims","board_shapes","board_colors","board_chars"]].drop_duplicates().reset_index(drop=True).to_records(index=False)
        self._meanings_letter_id_2_props = {f"scenario_{idx}_letter": row for idx, row in enumerate(meanings_letter_props)}
        meanings_number_props = self._unique_scenarios.loc[:,["find","board_dims","board_shapes","board_colors","board_numbers"]].drop_duplicates().reset_index(drop=True).to_records(index=False)
        self._meanings_number_id_2_props = {f"scenario_{idx}_number": row for idx, row in enumerate(meanings_number_props)}

    def _load_data(self):
        data = pd.read_csv(self.data_path, header=0, index_col=None)
        
        # replace "yes" and "no" with the absolute utterance
        for idx in data.index:
            corrected_messages = data.loc[idx, "corrected_messages"].split("<EOM>")
            new_corrected_messages = []
            for i, m in enumerate(corrected_messages):
                if m != "yes" and m != "no":
                    new_corrected_messages.append(m)
                elif i == 0 or corrected_messages[i-1] not in ["yes", "no"]:
                    new_corrected_messages = ["nonesense"]
                    break
                elif m == "yes":
                    new_corrected_messages.append(corrected_messages[i-1])
                elif m == "no":
                    if "not" in corrected_messages[i-1]:
                        last_corrected = " ".join([m for m in corrected_messages[i-1].split(" ") if m != "not"])
                        new_corrected_messages.append(last_corrected)
                    else:
                        new_corrected_messages.append(corrected_messages[i-1] + " not")

            data.loc[idx, "corrected_messages"] = "<EOM>".join(new_corrected_messages)

        # discard nonesense messages as in the pip paper
        data = data[data["corrected_messages"] != "nonesense"].reset_index(drop=True)

            # if "yes" in corrected_messages:
            #     new_corrected_messages = []
            #     for i, m in enumerate(corrected_messages):
            #         if m != "yes":
            #             new_corrected_messages.append(m)
            #         else:
            #             new_corrected_messages.append(corrected_messages[i-1])
            #     corrected_messages = new_corrected_messages
            # if "no" in corrected_messages:
            #     new_corrected_messages = []
            #     for i, m in enumerate(corrected_messages):
            #         if m != "no":
            #             new_corrected_messages.append(m)
            #         elif "not" in corrected_messages[i-1]:
            #             last_corrected = " ".join([m for m in corrected_messages[i-1].split(" ") if m != "not"])
            #             new_corrected_messages.append(last_corrected)
            #         else:
            #             new_corrected_messages.append(corrected_messages[i-1] + " not")
            #     print(data.loc[idx, "original_messages"], corrected_messages, new_corrected_messages)
            #     corrected_messages = new_corrected_messages
            
            # data.loc[idx, "corrected_messages"] = "<EOM>".join(corrected_messages)

        # discard again nonesense messages
        # data = data[data["corrected_messages"] != "nonesense"].reset_index(drop=True)

        # read the utterance vocabulary
        utterances = {}
        for idx in data.index:
            corrected_messages = data.loc[idx, "corrected_messages"].split("<EOM>")
            for m in corrected_messages:
                if m not in utterances:
                    utterances[m] = 1
                else:
                    utterances[m] += 1
        utterances = sorted(utterances.items(), key=lambda x: x[1], reverse=True)
        # import pdb; pdb.set_trace()

        return data, utterances
    
    @property
    def lexicon_letter(self):
        lexicon = np.zeros((len(self.utterance_counts), len(self.meanings_letter)))
        for i, (utt, _) in enumerate(self.utterance_counts):
            for j, m in enumerate(self.meanings_letter):
                props = self._meanings_letter_id_2_props[m]
                letter_mask = np.array([int(c == props["find"][0]) for c in props["board_chars"].split(",")])
                mask = self.board_compatible_with_utt(utt, props)
                lexicon[i,j] = np.sum(letter_mask * mask) > 0
        return lexicon

    @property
    def lexicon_number(self):
        lexicon = np.zeros((len(self.utterance_counts), len(self.meanings_number)))
        for i, (utt, _) in enumerate(self.utterance_counts):
            for j, m in enumerate(self.meanings_number):
                props = self._meanings_number_id_2_props[m]
                number_mask = np.array([int(c == props["find"][0]) for c in props["board_numbers"].split(",")])
                mask = self.board_compatible_with_utt(utt, props)
                lexicon[i,j] = np.sum(number_mask * mask) > 0
        return lexicon
    
    def board_compatible_with_utt(self, utterance, props):

        utterances = utterance.split(" ")
        
        if len(utterances) == 1:
            if "position" in mapping[utterances[0]]:
                position_mask = position_to_mask(utterances[0], props["board_dims"])
                return position_mask
            elif mapping[utterances[0]] == "shape":
                shape_mask = shape_to_mask(utterances[0], props["board_shapes"])
                return shape_mask
            elif mapping[utterances[0]] == "color":
                color_mask = color_to_mask(utterances[0], props["board_colors"])
                return color_mask
            else:
                raise ValueError(f"Unknown utterance: {utterances[0]}")
            
        elif len(utterances) == 2:
            if (mapping[utterances[0]] == "position_x" and mapping[utterances[1]] == "position_y") or \
                (mapping[utterances[0]] == "position_y" and mapping[utterances[1]] == "position_x") or \
                (utterance[0] == "middle" and props["board_dims"] == "2,3" and utterance[1] == "position_x") or \
                (utterance[0] == "middle" and props["board_dims"] == "3,2" and utterance[1] == "position_y") or \
                (utterance[1] == "middle" and props["board_dims"] == "2,3" and utterance[0] == "position_x") or \
                (utterance[1] == "middle" and props["board_dims"] == "3,2" and utterance[0] == "position_y"):
                position_mask_1 = position_to_mask(utterances[0], props["board_dims"])
                position_mask_2 = position_to_mask(utterances[1], props["board_dims"])
                return position_mask_1 * position_mask_2
            elif "position" in mapping[utterances[0]] and "position" in mapping[utterances[1]]:
                position_mask_1 = position_to_mask(utterances[0], props["board_dims"])
                position_mask_2 = position_to_mask(utterances[1], props["board_dims"])
                return np.logical_or(position_mask_1, position_mask_2)
            elif (mapping[utterances[0]] == "shape" and "position" in mapping[utterances[1]]):
                shape_mask = shape_to_mask(utterances[0], props["board_shapes"])
                position_mask = position_to_mask(utterances[1], props["board_dims"])
                return shape_mask * position_mask
            elif ("position" in mapping[utterances[0]] and mapping[utterances[1]] == "shape"):
                shape_mask = shape_to_mask(utterances[1], props["board_shapes"])
                position_mask = position_to_mask(utterances[0], props["board_dims"])
                return shape_mask * position_mask
            elif (mapping[utterances[0]] == "color" and "position" in mapping[utterances[1]]):
                color_mask = color_to_mask(utterances[0], props["board_colors"])
                position_mask = position_to_mask(utterances[1], props["board_dims"])
                return color_mask * position_mask
            elif ("position" in mapping[utterances[0]] and mapping[utterances[1]] == "color"):
                color_mask = color_to_mask(utterances[1], props["board_colors"])
                position_mask = position_to_mask(utterances[0], props["board_dims"])
                return color_mask * position_mask
            elif (mapping[utterances[0]] == "shape" and mapping[utterances[1]] == "color"):
                shape_mask = shape_to_mask(utterances[0], props["board_shapes"])
                color_mask = color_to_mask(utterances[1], props["board_colors"])
                return shape_mask * color_mask
            elif (mapping[utterances[0]] == "color" and mapping[utterances[1]] == "shape"):
                shape_mask = shape_to_mask(utterances[1], props["board_shapes"])
                color_mask = color_to_mask(utterances[0], props["board_colors"])
                return shape_mask * color_mask
            elif (mapping[utterances[0]] == "shape" and mapping[utterances[1]] == "shape"):
                shape_mask_1 = shape_to_mask(utterances[0], props["board_shapes"])
                shape_mask_2 = shape_to_mask(utterances[1], props["board_shapes"])
                return np.logical_or(shape_mask_1, shape_mask_2)
            elif (mapping[utterances[0]] == "color" and mapping[utterances[1]] == "color"):
                color_mask_1 = color_to_mask(utterances[0], props["board_colors"])
                color_mask_2 = color_to_mask(utterances[1], props["board_colors"])
                return np.logical_or(color_mask_1, color_mask_2)
            elif "not" in utterances:
                mask = self.board_compatible_with_utt(" ".join([u for u in utterances if u != "not"]), props)
                return 1 - mask
            else:
                raise ValueError(f"Unknown utterance: {utterances}")
        
        elif len(utterances) == 3 and "not" in utterances:
            mask = self.board_compatible_with_utt(" ".join([u for u in utterances if u != "not"]), props)
            return 1 - mask

        else:
            raise ValueError(f"Unknown utterance: {utterance}")

    def _get_target_name(self, row):
        chars = row["board_chars"].split(",")
        nums = row["board_numbers"].split(",")
        target_i, target_j = row["target_pos"].split(",")
        rows, columns = row["board_dims"].split(",")
        idx = int(rows) * int(target_j) + int(target_i)
        target = chars[idx] + nums[idx]
        return target
    
    @property
    def meanings_letter(self):
        return [f"scenario_{idx}_letter" for idx in range(len(self._meanings_letter_id_2_props))]
        
    @property
    def meanings_number(self):
        return [f"scenario_{idx}_number" for idx in range(len(self._meanings_number_id_2_props))]
    
    @property
    def categories(self):
        return [f"{i},{j}" for i in range(3) for j in range(3)]
    
    @property
    def world(self):
        return {
            "meanings_letter": self.meanings_letter,
            "meanings_number": self.meanings_number,
            "categories": self.categories,
            "utterances": [utt for utt, _ in self.utterance_counts],
            "lexicon_letter": self.lexicon_letter,
            "lexicon_number": self.lexicon_number,
            "prior": self.prior,
        }
        
    @property
    def prior(self):
        prior = np.zeros((len(self.meanings_letter), len(self.meanings_number), 9))
        for l, m_letter in enumerate(self.meanings_letter):
            for n, m_number in enumerate(self.meanings_number):
                for i in range(3):
                    for j in range(3):
                        l_props = self._meanings_letter_id_2_props[m_letter]
                        n_props = self._meanings_number_id_2_props[m_number]
                        if (l_props["find"] != n_props["find"]) or \
                        (l_props["board_dims"] != n_props["board_dims"]) or \
                        (l_props["board_shapes"] != n_props["board_shapes"]) or \
                        (l_props["board_colors"] != n_props["board_colors"]):
                            continue
                        part = self.data.loc[
                            (self.data["find"] == l_props["find"]) &
                            (self.data["board_dims"] == l_props["board_dims"]) &
                            (self.data["board_shapes"] == l_props["board_shapes"]) &
                            (self.data["board_colors"] == l_props["board_colors"]) &
                            (self.data["board_chars"] == l_props["board_chars"]) &
                            (self.data["board_numbers"] == n_props["board_numbers"]) &
                            (self.data[f"{self.model}"] == f"{i},{j}"),:
                        ]
                        prior[l, n, i * 3 + j] = len(part)
        prior = prior / np.sum(prior)
        return prior
                        
        
    
    def samples(self):
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

            y = row[f"{self.model}"]
            starter_speaker = row["starter_player"]

            yield i, (meaning_letter, meaning_number, y, starter_speaker, row["corrected_messages"].split("<EOM>"))

    
    def __len__(self):
        return len(self.data)

def read_dataset(dataset_name):
    
    if dataset_name == "infojigsaw":
        dataset = InfoJigsawDataset()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return dataset


if __name__ == "__main__":
    dataset = read_dataset("infojigsaw")
    dataset.world["lexicon_letter"]
    import pdb; pdb.set_trace()
    print(dataset)