# federated/fed_data_splitting.py

import numpy as np
from torch.utils.data import Dataset
from typing import Dict, List

def get_user_groups(dataset: Dataset, num_users: int) -> Dict[int, List[int]]:
    """
    Splits the dataset into user groups while ensuring each user has enough data
    to construct NShot tasks.

    # Arguments
        dataset: Dataset instance with a 'df' attribute (pandas DataFrame).
        num_users: Number of users to split the data across.

    # Returns
        user_groups: Dictionary mapping user_id -> list of valid dataset indices.
    """
    if not hasattr(dataset, 'df') or 'id' not in dataset.df.columns:
        raise ValueError("Dataset must have a DataFrame with an 'id' column.")

    df = dataset.df

    # Shuffle classes randomly for user splits
    unique_classes = df['class_id'].unique()
    np.random.shuffle(unique_classes)

    user_groups = {user_id: [] for user_id in range(num_users)}
    num_classes = len(unique_classes)
    classes_per_user = num_classes // num_users
    remainder = num_classes % num_users

    idx = 0
    for user_id in range(num_users):
        num_user_classes = classes_per_user + (1 if user_id < remainder else 0)
        user_classes = unique_classes[idx : idx + num_user_classes]
        idx += num_user_classes

        user_indices = df[df['class_id'].isin(user_classes)]['id'].tolist()

        # Ensure users have enough data for k-way, n-shot, q-query tasks
        if len(user_indices) < 10:  # Minimum for 1-way, 1-shot, and 1-query
            raise ValueError(
                f"User {user_id} assigned too few samples ({len(user_indices)}) to construct NShot tasks."
            )

        user_groups[user_id] = user_indices

    return user_groups
