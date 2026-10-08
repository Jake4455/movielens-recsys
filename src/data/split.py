import numpy as np
import pandas as pd

from src.data import schema


def split_per_user(ratings, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"split ratios must sum to 1.0, got {total}")
    frame = ratings.sort_values(
        [schema.USER, schema.TIMESTAMP, schema.MOVIE], kind="mergesort"
    ).reset_index(drop=True)
    group = frame.groupby(schema.USER, sort=False)
    n = group[schema.MOVIE].transform("size").to_numpy()
    position = group.cumcount().to_numpy()
    n_val = np.floor(n * val_ratio).astype(np.int64)
    n_test = np.floor(n * test_ratio).astype(np.int64)
    n_train = n - n_val - n_test
    labels = np.where(
        position < n_train,
        schema.TRAIN,
        np.where(position < n_train + n_val, schema.VAL, schema.TEST),
    )
    frame[schema.SPLIT] = labels
    return frame


def frames_by_split(frame):
    return {
        name: frame[frame[schema.SPLIT] == name].reset_index(drop=True)
        for name in schema.SPLITS
    }
