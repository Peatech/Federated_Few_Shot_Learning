# tests/test_federated.py

import unittest
import torch
import os

from few_shot.datasets import DummyDataset
from federated.fed_data_splitting import get_user_groups
from federated.fed_trainer import _local_update_few_shot
from few_shot.models import MatchingNetwork


class TestFedDataSplitting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Instead of 2 classes x 5 samples, let's do 3 classes x 10 samples => total 30
        cls.dataset = DummyDataset(samples_per_class=10, n_classes=3, n_features=2)

    def test_get_user_groups(self):
        # We'll request 2 users
        num_users = 2
        user_groups = get_user_groups(self.dataset, num_users=num_users)

        # check we have 2 keys
        self.assertEqual(len(user_groups), num_users, "Should have num_users groups.")

        # check total coverage
        total_indices = sum(len(v) for v in user_groups.values())
        self.assertEqual(
            total_indices, len(self.dataset),
            "All dataset samples must be allocated to some user."
        )

        # ensure no out-of-bounds
        for uid, idxs in user_groups.items():
            for idx in idxs:
                self.assertTrue(
                    0 <= idx < len(self.dataset),
                    f"Index {idx} out of range for dataset size {len(self.dataset)}"
                )

        # ensure pure class-based splitting => no overlapping classes
        classes_user0 = set(
            self.dataset.df.iloc[user_groups[0]].class_id.unique()
        )
        classes_user1 = set()
        for u in range(1, num_users):
            classes_user1 |= set(
                self.dataset.df.iloc[user_groups[u]].class_id.unique()
            )
        intersect = classes_user0.intersection(classes_user1)
        self.assertTrue(
            len(intersect) == 0,
            "No user should share a class if it's truly class-based splitting."
        )


class TestFedTrainer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Also do 3 classes x 10 => 30 total
        cls.dataset = DummyDataset(samples_per_class=10, n_classes=3, n_features=2)

        # Build a small MatchingNetwork
        cls.model = MatchingNetwork(
            n=1, k=2, q=2, fce=False,
            num_input_channels=1,  # dummy
            lstm_layers=1,
            lstm_input_size=64,  # dummy
            unrolling_steps=1,
            device=torch.device('cpu')
        )

    def test_local_update_few_shot(self):
        # We'll do a class-based split for 2 users
        # Suppose user 0 => class 0, user 1 => classes 1,2
        # We'll do it manually for demonstration
        user0_classes = [0]
        user1_classes = [1, 2]

        df = self.dataset.df
        user0_indices = df[df['class_id'].isin(user0_classes)]['id'].tolist()
        user1_indices = df[df['class_id'].isin(user1_classes)]['id'].tolist()

        user_groups = {0: user0_indices, 1: user1_indices}

        # Now let's pick user 0 for local update
        user_idxs = user_groups[0]  # class0 => 10 samples

        # Enough for n=1, q=2, k=2? Actually we only have 1 class (class 0) => can't do k=2
        # => We'll adjust to k=1 to pass the test
        # or we can combine classes 0,1 -> but let's reduce k=1 for demonstration

        from torch import nn
        from torch.utils.data import Subset

        class Args:
            lr = 0.01
            momentum = 0.9
            n_train = 1  # n=1
            k_train = 1  # changed from 2 to 1, so we can do tasks from 1 class
            q_train = 2  # q=2
            fce = False
            distance = 'cosine'
            local_ep = 1

        args = Args()

        from few_shot.core import NShotTaskSampler
        # We'll create a user_subset
        user_subset = Subset(self.dataset, user_idxs)
        # Now local update
        updated_w, local_loss = _local_update_few_shot(
            args, self.model, self.dataset, user_idxs, device=torch.device('cpu')
        )

        self.assertIsInstance(updated_w, dict, "updated_w should be a state_dict (dict).")
        self.assertIsInstance(local_loss, float, "local_loss should be a float.")
        self.assertGreaterEqual(local_loss, 0.0, "Loss should be non-negative.")
