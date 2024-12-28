# federated/fed_data_splitting.py
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, List

def get_user_groups(dataset: Dataset, num_users: int) -> Dict[int, List[int]]:
    # Additional check
    if num_users > 4:
        raise ValueError("For Omniglot and miniImageNet, the maximum number of users is 4!")

    # The rest of your class-based splitting, example:
    if not hasattr(dataset, 'df'):
        raise ValueError("Dataset must have a 'df' attribute (pandas DataFrame).")

    df = dataset.df
    unique_classes = df['class_id'].unique()
    np.random.shuffle(unique_classes)

    user_groups = {uid: [] for uid in range(num_users)}
    num_classes = len(unique_classes)

    # basic chunking
    chunk_size = num_classes // num_users
    remainder = num_classes % num_users

    idx = 0
    for uid in range(num_users):
        # each user gets chunk_size classes, plus 1 if remainder
        plus_one = 1 if uid < remainder else 0
        classes_for_user = unique_classes[idx: idx + chunk_size + plus_one]
        idx += chunk_size + plus_one

        user_indices = df[df['class_id'].isin(classes_for_user)]['id'].tolist()
        user_groups[uid] = user_indices

    return user_groups
