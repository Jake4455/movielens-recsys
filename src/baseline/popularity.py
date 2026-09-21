import numpy as np

from src.data import schema


def popularity_scores(train, candidates, positive_only=False, threshold=4.0):
    frame = train
    if positive_only:
        frame = frame[frame[schema.RATING] >= threshold]
    counts = frame.groupby(schema.MOVIE).size()
    return (
        candidates[schema.MOVIE]
        .map(counts)
        .fillna(0.0)
        .to_numpy(dtype=np.float64)
    )
