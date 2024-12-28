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
        # Create a small dataset: 2 classes, 5 samples each => 10 total
        cls.dataset = DummyDataset(samples_per_class=5, n_classes=2, n_features=2)

    def test_get_user_groups(self):
        # We'll request 2 users
        num_users = 2
        user_groups = get_user_groups(self.dataset, num_users=num_users)

        # Check we have 2 keys
        self.assertEqual(len(user_groups), num_users, "Should have num_users groups.")

        # check total coverage
        total_indices = sum(len(v) for v in user_groups.values())
        self.assertEqual(total_indices, len(self.dataset), 
                         "All dataset samples must be allocated to some user.")
        
        # ensure no out-of-bounds
        for uid, idxs in user_groups.items():
            for idx in idxs:
                self.assertTrue(0 <= idx < len(self.dataset), 
                                f"Index {idx} is out of range for dataset size {len(self.dataset)}")

        # ensure class-based splitting => each user gets entire classes
        # with 2 classes, we expect 1 class per user if chunk is perfect
        # Just a basic check that user doesn't have partial class
        classes_user0 = set(self.dataset.df.iloc[idxs].class_id.unique())
        classes_user1 = set()
        for u in range(1, num_users):
            classes_user1 |= set(self.dataset.df.iloc[user_groups[u]].class_id.unique())
        intersect = classes_user0.intersection(classes_user1)
        self.assertTrue(len(intersect) == 0, 
                        "No user should share a class if it's truly class-based splitting.")


class TestFedTrainer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Minimal dataset with 2 classes, 10 samples total
        cls.dataset = DummyDataset(samples_per_class=5, n_classes=2, n_features=2)

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
        # We'll just test local update on 1 user
        # 1) get user groups
        user_groups = {0: list(range(5)), 1: list(range(5, 10))}
        
        # 2) pick user 0
        user_id = 0
        user_idxs = user_groups[user_id]

        # 3) call _local_update_few_shot
        from torch import nn
        from torch.utils.data import Subset

        # minimal args simulation
        class Args:
            lr = 0.01
            momentum = 0.9
            n_train = 1
            k_train = 2
            q_train = 2
            fce = False
            distance = 'cosine'
            local_ep = 1
        args = Args()

        device = torch.device('cpu')
        updated_w, local_loss = _local_update_few_shot(args, self.model, self.dataset, user_idxs, device)

        self.assertIsInstance(updated_w, dict, "updated_w should be a state_dict (dict).")
        self.assertIsInstance(local_loss, float, "local_loss should be a float.")
        self.assertGreaterEqual(local_loss, 0.0, "Loss should be non-negative.")


if __name__ == '__main__':
    unittest.main()
