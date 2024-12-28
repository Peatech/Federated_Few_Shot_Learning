# federated/fed_data_splitting.py

import numpy as np
from torch.utils.data import Dataset
from typing import Dict, List

def get_user_groups(dataset: Dataset, num_users: int, **kwargs) -> Dict[int, List[int]]:
    """
    Class-based splitting logic:
    
    We group all images of certain classes to the same user, ensuring that each 
    user has full coverage of those classes (rather than partial images).

    # Arguments
        dataset: An instance of OmniglotDataset or MiniImageNet. Must have:
                 - df: a pandas.DataFrame with 'class_id' and 'id'
        num_users: Number of users to split among

    # Returns
        user_groups: A dict mapping user_id -> list of dataset indices
                     (i.e. values for dataset's 'id' column).
    """

    # 1) Extract all classes from dataset.df
    if not hasattr(dataset, 'df'):
        raise ValueError("Dataset must have a 'df' attribute (pandas.DataFrame).")

    df = dataset.df
    if 'class_id' not in df.columns:
        raise ValueError("DataFrame must contain a 'class_id' column for class-based splitting.")

    # Unique classes in the entire dataset
    unique_classes = df['class_id'].unique()
    np.random.shuffle(unique_classes)

    # 2) Chunk the classes among the users
    user_groups = {uid: [] for uid in range(num_users)}

    # How many classes per user (minimum)
    base_chunk_size = len(unique_classes) // num_users
    remainder = len(unique_classes) % num_users

    idx = 0
    for uid in range(num_users):
        # If there's a remainder, let that user have 1 extra class
        chunk_size = base_chunk_size + (1 if uid < remainder else 0)
        subset_classes = unique_classes[idx: idx + chunk_size]
        idx += chunk_size

        # 3) Gather all 'id' indices from those classes
        #    For each class in subset_classes, pick all rows from df
        df_subset = df[df['class_id'].isin(subset_classes)]
        # user_groups[uid] is a list of dataset indices (the 'id' column)
        user_groups[uid] = df_subset['id'].tolist()

    return user_groups
