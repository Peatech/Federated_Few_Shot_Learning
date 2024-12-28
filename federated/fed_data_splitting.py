# federated/fed_data_splitting.py

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from typing import Dict, List

def get_user_groups(dataset: Dataset, num_users: int) -> Dict[int, List[int]]:
    """
    Class-based splitting for Omniglot/miniImageNet so each user 
    has entire classes. This ensures no user has out-of-range 
    indices or insufficient images for n-shot.

    # Arguments
        dataset: An instance of OmniglotDataset or MiniImageNet,
                 each with a df containing 'id' and 'class_id'.
        num_users: Number of users to allocate classes to (<= 4).
    
    # Returns
        user_groups: dict -> user_id : list of valid dataset 'id' indices
    """
    # 1) Validate user count
    if num_users > 4:
        raise ValueError("For Omniglot/miniImageNet, max number of users is 4!")

    # 2) Check that dataset has .df with 'id' and 'class_id'
    if not hasattr(dataset, 'df'):
        raise ValueError("Dataset must have a 'df' attribute (pandas DataFrame).")
    df = dataset.df
    required_cols = {'id', 'class_id'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"df must contain columns {required_cols}")

    # 3) Collect classes & shuffle
    classes = df['class_id'].unique()
    np.random.shuffle(classes)
    num_classes = len(classes)

    # 4) Split classes among users
    user_groups = {uid: [] for uid in range(num_users)}
    base_chunk_size = num_classes // num_users
    remainder = num_classes % num_users

    idx = 0
    for user_id in range(num_users):
        # chunk_size + remainder
        plus_one = 1 if user_id < remainder else 0
        chunk_size = base_chunk_size + plus_one

        # get these classes
        subset_classes = classes[idx : idx + chunk_size]
        idx += chunk_size

        # gather all 'id' indices for these classes
        user_df = df[df['class_id'].isin(subset_classes)]
        # convert to python list
        user_ids = user_df['id'].tolist()

        # Validate they are in-bounds for the dataset length
        # Only if you plan to do Subset(dataset, user_ids)
        # ensure they are < len(dataset). But typically 'id' 
        # is the correct index from the df, so we assume it's fine.
        for x in user_ids:
            if x < 0 or x >= len(dataset):
                raise IndexError(f"User {user_id} has out-of-range index {x} for dataset size {len(dataset)}.")

        user_groups[user_id] = user_ids

    # If there’s still leftover classes (shouldn’t happen unless chunk rounding),
    # you can optionally append them to the last user.

    return user_groups
