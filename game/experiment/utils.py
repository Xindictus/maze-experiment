import csv
import os
import pickle
from datetime import date

import numpy as np


def load_csv(file_path):
    data = []
    with open(file_path, newline="") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            data.append(row)
    return data


def save_pickle(participant_name, mode, game_mode, data, is_baseline=False):
    name_of_file = f"{participant_name}_{mode}_{date.today()}.pickle"
    additional_part = 0
    folder_path = os.path.join(
        "results", "baseline" if is_baseline else mode, game_mode
    )

    os.makedirs(folder_path, exist_ok=True)

    while os.path.isfile(os.path.join(folder_path, name_of_file)):
        name_of_file = f"{participant_name}_{'baseline' if is_baseline else mode}_{date.today()}_{additional_part}.pickle"
        additional_part += 1

    with open(os.path.join(folder_path, name_of_file), "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def normalize_state(observation):
    def norm_feature(feature, min_val, max_val):
        return 2 * ((feature - min_val) / (max_val - min_val)) - 1

    ranges = [
        (-2, 2),
        (-2, 2),
        (-4, 4),
        (-4, 4),
        (-30, 30),
        (-30, 30),
        (-1.9, 1.9),
        (-1.9, 1.9),
    ]
    norm_obs = [
        norm_feature(observation[i], *ranges[i])
        for i in range(len(observation))
    ]

    return np.clip(norm_obs, -1.3, 1.3)
