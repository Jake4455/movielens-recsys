from pathlib import Path

import numpy as np
import pandas as pd

from src.data import schema

SECONDS_PER_DAY = 86_400


class ActivityContext:
    def __init__(self, user_daily, movie_daily):
        (
            self.user_index,
            self.user_keys,
            self.user_cumsum,
            self.user_min_day,
            self.user_n_days,
        ) = self._prepare(user_daily, schema.USER)
        (
            self.movie_index,
            self.movie_keys,
            self.movie_cumsum,
            self.movie_min_day,
            self.movie_n_days,
        ) = self._prepare(movie_daily, schema.MOVIE)

    @staticmethod
    def _prepare(frame, id_column):
        day = (frame["window_start"].astype("int64") // 1_000_000_000) // SECONDS_PER_DAY
        ids = frame[id_column].to_numpy()
        index = pd.Index(np.sort(np.unique(ids)))
        codes = index.get_indexer(ids)
        order = np.lexsort((day, codes))
        codes = codes[order]
        day = day[order]
        counts = frame["count"].to_numpy(dtype=np.float64)[order]
        min_day = int(day.min())
        n_days = int(day.max() - min_day) + 1
        keys = codes.astype(np.int64) * n_days + (day - min_day)
        cumsum = np.concatenate([[0.0], np.cumsum(counts)])
        return index, keys, cumsum, min_day, n_days

    @staticmethod
    def feature_columns(mode="absolute", user_windows=(7, 30, 90), movie_windows=(7, 30)):
        if mode == "ratio":
            return [
                "user_activity_ratio_7_30",
                "user_activity_ratio_30_90",
                "movie_recent_ratio_7_30",
            ]
        return [f"user_activity_{int(w)}d" for w in user_windows] + [
            f"movie_recent_{int(w)}d" for w in movie_windows
        ]

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        user_daily = pd.read_parquet(directory / "user_daily")
        movie_daily = pd.read_parquet(directory / "movie_daily")
        return cls(user_daily, movie_daily)

    @staticmethod
    def _window_sums(frame, id_column, index, keys, cumsum, min_day, n_days, windows, prefix):
        target_day = frame[schema.TARGET_TS].to_numpy() // SECONDS_PER_DAY
        codes = index.get_indexer(frame[id_column].to_numpy())
        known = codes >= 0
        output = {}
        for window in windows:
            values = np.full(len(frame), np.nan, dtype=np.float32)
            hi_offsets = np.clip(target_day[known] - min_day, 0, n_days)
            lo_offsets = np.clip(
                target_day[known] - int(window) - min_day, 0, n_days
            )
            hi_keys = codes[known].astype(np.int64) * n_days + hi_offsets
            lo_keys = codes[known].astype(np.int64) * n_days + lo_offsets
            hi = np.searchsorted(keys, hi_keys)
            lo = np.searchsorted(keys, lo_keys)
            values[known] = (cumsum[hi] - cumsum[lo]).astype(np.float32)
            output[f"{prefix}_{int(window)}d"] = values
        return output

    def compute(
        self,
        frame,
        user_windows=(7, 30, 90),
        movie_windows=(7, 30),
        mode="absolute",
    ):
        columns = {}
        columns.update(
            self._window_sums(
                frame,
                schema.USER,
                self.user_index,
                self.user_keys,
                self.user_cumsum,
                self.user_min_day,
                self.user_n_days,
                user_windows,
                "user_activity",
            )
        )
        columns.update(
            self._window_sums(
                frame,
                schema.MOVIE,
                self.movie_index,
                self.movie_keys,
                self.movie_cumsum,
                self.movie_min_day,
                self.movie_n_days,
                movie_windows,
                "movie_recent",
            )
        )
        if mode == "ratio":
            user_7 = columns["user_activity_7d"]
            user_30 = columns["user_activity_30d"]
            user_90 = columns["user_activity_90d"]
            movie_7 = columns["movie_recent_7d"]
            movie_30 = columns["movie_recent_30d"]
            columns = {
                "user_activity_ratio_7_30": (user_7 + 1.0) / (user_30 + 1.0),
                "user_activity_ratio_30_90": (user_30 + 1.0) / (user_90 + 1.0),
                "movie_recent_ratio_7_30": (movie_7 + 1.0) / (movie_30 + 1.0),
            }
        return pd.DataFrame(columns, index=frame.index)
