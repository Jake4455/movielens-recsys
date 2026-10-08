import numpy as np
import pandas as pd

from src.data import schema
from src.utils.config import get


def read_ratings(path, cfg=None):
    frame = pd.read_csv(
        path,
        dtype=schema.RATINGS_DTYPES,
        usecols=schema.RATINGS_COLUMNS,
    )
    max_users = get(cfg, "data.dev.max_users") if cfg is not None else None
    seed = get(cfg, "seed", 20260907) if cfg is not None else 20260907
    if max_users:
        users = np.sort(frame[schema.USER].unique())
        rng = np.random.default_rng(int(seed))
        size = min(int(max_users), users.size)
        keep = rng.choice(users, size=size, replace=False)
        frame = frame[frame[schema.USER].isin(keep)].reset_index(drop=True)
    return frame


def read_movies(path):
    return pd.read_csv(path, dtype={schema.MOVIE: "int64"})


def read_tags(path, cfg=None):
    frame = pd.read_csv(
        path,
        dtype={
            schema.USER: "int64",
            schema.MOVIE: "int64",
            schema.TAG: "string",
            schema.TIMESTAMP: "int64",
        },
    )
    max_users = get(cfg, "data.dev.max_users") if cfg is not None else None
    if max_users:
        users = np.sort(frame[schema.USER].unique())
        rng = np.random.default_rng(int(get(cfg, "seed", 20260907)))
        size = min(int(max_users), users.size)
        keep = rng.choice(users, size=size, replace=False)
        frame = frame[frame[schema.USER].isin(keep)].reset_index(drop=True)
    return frame
