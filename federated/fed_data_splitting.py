# federated/fed_data_splitting.py

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from typing import Dict, List

# federated/fed_data_splitting.py

def get_user_groups(dataset: Dataset, num_users: int) -> Dict[int, List[int]]:
    if num_users > 4:
        raise ValueError("For Omniglot / miniImageNet, max users=4")

    df = dataset.df
    unique_classes = df['class_id'].unique()
    np.random.shuffle(unique_classes)

    user_groups = {uid: [] for uid in range(num_users)}
    chunk_size = len(unique_classes) // num_users
    remainder = len(unique_classes) % num_users

    idx = 0
    for uid in range(num_users):
        plus_one = 1 if uid < remainder else 0
        classes_for_user = unique_classes[idx : idx + chunk_size + plus_one]
        idx += chunk_size + plus_one

        user_indices = df[df['class_id'].isin(classes_for_user)]['id'].tolist()
        user_groups[uid] = user_indices

    return user_groups
