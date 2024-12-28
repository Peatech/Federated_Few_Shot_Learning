# federated/fed_data_splitting.py

import numpy as np

def get_user_groups(dataset: Dataset, num_users: int, iid: bool = True) -> Dict[int, List[int]]:
    """Create user groups for Federated Learning.

    Args:
        dataset: Dataset to split.
        num_users: Number of users to split the dataset into.
        iid: Whether to create IID or Non-IID splits.

    Returns:
        user_groups: Dictionary mapping user_id to indices.
    """
    num_items = len(dataset) // num_users
    indices = np.arange(len(dataset))

    if iid:
        np.random.shuffle(indices)
        user_groups = {i: list(indices[i * num_items:(i + 1) * num_items]) for i in range(num_users)}
    else:
        labels = dataset.df['class_id'].values if hasattr(dataset, 'df') else None
        if labels is None:
            raise ValueError("Dataset must have a 'df' attribute with 'class_id' for non-IID splitting.")
        
        sorted_indices = np.argsort(labels)
        user_groups = {}
        shards_per_user = len(dataset) // (num_users * 2)
        for i in range(num_users):
            shard_indices = np.concatenate([
                sorted_indices[j * shards_per_user:(j + 1) * shards_per_user]
                for j in range(i * 2, (i + 1) * 2)
            ])
            user_groups[i] = shard_indices.tolist()

    return user_groups


def _iid_split(dataset, num_users):
    """Simple random equal-chunk IID splitting."""
    all_indices = np.arange(len(dataset))
    np.random.shuffle(all_indices)

    num_items = len(dataset) // num_users
    user_groups = {}

    start = 0
    for uid in range(num_users):
        if uid != num_users - 1:
            user_groups[uid] = all_indices[start : start + num_items]
            start += num_items
        else:
            # Last user gets the remaining items
            user_groups[uid] = all_indices[start:]

    return user_groups


def _label_based_shard_split(dataset, num_users):
    """
    Non-IID approach: Partition data by label shards.
    Each user gets shards of sorted label indices, 
    so data is concentrated in fewer classes per user.

    # Implementation details:
    # - Define #shards as num_users * shards_per_user (e.g., 2)
    # - Sort the entire dataset by 'class_id'
    # - Assign consecutive shards to each user
    """
    df = dataset.df
    labels = df['class_id'].values  # array of shape [len(dataset)]
    all_idxs = np.arange(len(df))

    # Sort indices by class_id
    sorted_idxs = all_idxs[np.argsort(labels)]

    shards_per_user = 2  # Number of shards per user
    total_shards = num_users * shards_per_user
    shard_size = len(dataset) // total_shards

    # Create list of shard IDs
    idx_shard = list(range(total_shards))

    user_groups = {uid: np.array([], dtype=int) for uid in range(num_users)}

    for uid in range(num_users):
        # pick shards_per_user shards for this user
        chosen = set(np.random.choice(idx_shard, shards_per_user, replace=False))
        # remove them from the shard pool
        idx_shard = list(set(idx_shard) - chosen)

        # for each shard, assign those indices
        for shard_id in chosen:
            start = shard_id * shard_size
            end = (shard_id + 1) * shard_size
            user_groups[uid] = np.concatenate(
                [user_groups[uid], sorted_idxs[start:end]],
                axis=0
            )

    return user_groups
