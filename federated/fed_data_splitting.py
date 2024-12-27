##############################################################
# federated/fed_data_splitting.py
# Provides a get_user_groups() function for IID or non-IID splits
##############################################################
import numpy as np

def get_user_groups(dataset, num_users, iid=True):
    """
    Splits the dataset among `num_users` clients. 
    If iid=True => random equal-chunk splits (IID).
    If iid=False => label-based shard approach using dataset.df['class_id'].

    # Arguments
        dataset: A Dataset object with a .df DataFrame containing at least 'class_id' column
        num_users: How many clients to split among
        iid: Boolean indicating whether to do IID or non-IID splits
    # Returns
        user_groups: dict -> {user_id: array of dataset indices}
    """
    if iid:
        return _iid_split(dataset, num_users)
    else:
        return _label_based_shard_split(dataset, num_users)


def _iid_split(dataset, num_users):
    """Simple random equal-chunk IID splitting."""
    all_indices = np.arange(len(dataset))
    np.random.shuffle(all_indices)

    num_items = len(dataset) // num_users
    user_groups = {}

    start = 0
    for uid in range(num_users):
        user_groups[uid] = all_indices[start : start + num_items]
        start += num_items

    return user_groups


def _label_based_shard_split(dataset, num_users):
    """
    Non-IID approach: We partition data by label shards.
    This uses dataset.df['class_id'] for label-based splits.
    Each user gets shards of sorted label indices, 
    so data is concentrated in fewer classes per user.

    # Implementation details:
    # - Suppose we want 2 shards per user => total_shards = num_users * 2
    # - Sort the entire dataset by label => 'class_id'
    # - Assign consecutive shards to each user
    """
    df = dataset.df
    labels = df['class_id'].values  # array of shape [len(dataset)]
    all_idxs = np.arange(len(df))

    # Sort indices by class_id
    sorted_idxs = all_idxs[np.argsort(labels)]

    # We'll define #shards as num_users * 2 (for example)
    # You can also make this dynamic or pass as an argument
    shards_per_user = 2
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
