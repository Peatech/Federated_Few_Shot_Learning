# federated/fed_data_splitting.py

import numpy as np
from torch.utils.data import Dataset
from typing import Dict, List

def get_user_groups(dataset: Dataset, num_users: int) -> Dict[int, List[int]]:
    """
    Class-based splitting logic for users with additional validation
    to ensure valid indices are assigned to each user.

    # Arguments
        dataset: Dataset instance with a 'df' attribute (pandas DataFrame).
        num_users: Number of users to split the data across.

    # Returns
        user_groups: Dictionary mapping user_id -> list of valid dataset indices.
    """
    # Ensure the dataset has a valid DataFrame
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
        # Assign classes to users
        num_user_classes = classes_per_user + (1 if user_id < remainder else 0)
        user_classes = unique_classes[idx : idx + num_user_classes]
        idx += num_user_classes

        # Collect indices for all images of these classes
        user_indices = df[df['class_id'].isin(user_classes)]['id'].tolist()
        user_groups[user_id] = user_indices

        # Validate indices are within dataset length
        for idx in user_indices:
            if idx >= len(dataset):
                raise IndexError(f"Index {idx} out of range for dataset length {len(dataset)}.")

    return user_groups
