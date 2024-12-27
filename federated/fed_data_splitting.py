###############################################
# federated/fed_data_splitting.py
# Provides get_user_groups(...) with IID or non-IID.
###############################################
import numpy as np

def get_user_groups_iid(dataset, num_users):
    """Simple IID splitting: each user gets an equal chunk."""
    num_items = len(dataset) // num_users
    all_idxs = np.arange(len(dataset))
    np.random.shuffle(all_idxs)

    user_groups = {}
    start = 0
    for user_id in range(num_users):
        user_groups[user_id] = all_idxs[start:start + num_items]
        start += num_items

    return user_groups

def get_user_groups_noniid(dataset, num_users):
    """
    Example non-IID approach.
    For Omniglot or miniImageNet, we might do something
    similar to your old sampling strategy: cluster by class_id,
    shard, etc.
    We'll do a simple approach: each user gets examples from fewer classes.
    """

    # Extract labels from dataset
    # For example, OmniglotDataset and MiniImageNet store label in df['class_id']
    df = dataset.df
    labels = df['class_id'].values
    all_idxs = np.arange(len(df))

    # Sort by label
    idxs_labels = np.vstack((all_idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1,:].argsort()]
    sorted_idxs = idxs_labels[0,:]

    # Let's say we create 'num_shards = num_users * 2' shards, each shard has data from ~1-2 classes
    num_shards = num_users * 2
    shard_size = len(dataset) // num_shards
    idx_shard = [i for i in range(num_shards)]
    user_groups = {i: np.array([], dtype=int) for i in range(num_users)}

    for user_id in range(num_users):
        # pick 2 shards for each user, for instance
        num_shards_per_user = 2
        rand_set = set(np.random.choice(idx_shard, num_shards_per_user, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)

        for r in rand_set:
            user_groups[user_id] = np.concatenate(
                (user_groups[user_id], 
                 sorted_idxs[r*shard_size : (r+1)*shard_size]),
                axis=0
            )
    return user_groups

def get_user_groups(dataset, num_users, iid=True):
    """Unified entry point: returns user_groups (dict) for IID or non-IID."""
    if iid:
        return get_user_groups_iid(dataset, num_users)
    else:
        return get_user_groups_noniid(dataset, num_users)
