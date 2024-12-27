import os


PATH = os.path.dirname(os.path.realpath(__file__))

DATA_PATH = "/kaggle/working/Federated_Few_Shot_Learning/Omniglot"

EPSILON = 1e-8

if DATA_PATH is None:
    raise Exception('Configure your data folder location in config.py before continuing!')
