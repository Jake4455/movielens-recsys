from pathlib import Path

import numpy as np
import pandas as pd

from src.data import schema

CHUNK = 250_000


class FeatureContext:
    def __init__(
        self,
        splits,
        movies,
        als_model_dir=None,
        threshold=4.0,
        use_bayes_rating=True,
        use_weighted_genre=True,
        weighted_genre_threshold=3.5,
        bayes_quantile=0.9,
    ):
        self.threshold = float(threshold)
        self.use_bayes_rating = bool(use_bayes_rating)
        self.use_weighted_genre = bool(use_weighted_genre)
        self.weighted_genre_threshold = float(weighted_genre_threshold)
        self.bayes_quantile = float(bayes_quantile)
        self.movie_ids = np.sort(movies[schema.MOVIE].unique())
        self.movie_index = pd.Index(self.movie_ids)
        n_movies = self.movie_ids.size

        self.genres = sorted(
            {
                genre
                for value in movies[schema.GENRES].fillna("")
                for genre in str(value).split("|")
                if genre and genre != "(no genres listed)"
            }
        )
        movie_codes = self.movie_index.get_indexer(movies[schema.MOVIE])
        genre_matrix = np.zeros((len(movies), len(self.genres)), dtype=np.float32)
        genre_series = movies[schema.GENRES].fillna("")
        for position, genre in enumerate(self.genres):
            genre_matrix[:, position] = genre_series.str.contains(
                genre, regex=False
            ).to_numpy(dtype=np.float32)
        self.item_genre = np.zeros((n_movies, len(self.genres)), dtype=np.float32)
        self.item_genre[movie_codes] = genre_matrix

        year = (
            movies[schema.TITLE]
            .astype(str)
            .str.extract(r"\((\d{4})\)\s*$")[0]
            .astype(np.float32)
        )
        self.item_year = np.full(n_movies, np.nan, dtype=np.float32)
        self.item_year[movie_codes] = year.to_numpy(dtype=np.float32)

        train = splits[schema.TRAIN]
        self.user_ids = np.sort(train[schema.USER].unique())
        self.user_index = pd.Index(self.user_ids)
        n_users = self.user_ids.size
        user_codes = self.user_index.get_indexer(train[schema.USER])
        movie_codes = self.movie_index.get_indexer(train[schema.MOVIE])
        ratings = train[schema.RATING].to_numpy(dtype=np.float32)
        positive_mask = ratings >= self.threshold
        self.global_mean = float(ratings.mean())

        self.movie_count = np.bincount(movie_codes, minlength=n_movies).astype(np.float32)
        self.movie_pos_count = np.bincount(
            movie_codes[positive_mask], minlength=n_movies
        ).astype(np.float32)
        self.movie_rating_sum = np.bincount(
            movie_codes, weights=ratings, minlength=n_movies
        ).astype(np.float32)
        self.user_count = np.bincount(user_codes, minlength=n_users).astype(np.float32)
        self.user_pos_count = np.bincount(
            user_codes[positive_mask], minlength=n_users
        ).astype(np.float32)
        self.user_rating_sum = np.bincount(
            user_codes, weights=ratings, minlength=n_users
        ).astype(np.float32)

        user_genre = np.zeros((n_users, len(self.genres)), dtype=np.float32)
        for position in range(len(self.genres)):
            user_genre[:, position] = np.bincount(
                user_codes[positive_mask],
                weights=self.item_genre[movie_codes[positive_mask], position],
                minlength=n_users,
            )
        self.user_genre_sum = user_genre
        self.user_genre = user_genre / np.maximum(self.user_pos_count, 1.0)[:, None]

        self.bayes_m = 0.0
        if self.use_bayes_rating:
            observed_counts = self.movie_count[self.movie_count > 0]
            self.bayes_m = float(
                np.quantile(observed_counts, self.bayes_quantile)
            ) if observed_counts.size else 0.0

        self.user_genre_weighted = None
        if self.use_weighted_genre:
            liked_mask = ratings >= self.weighted_genre_threshold
            weights = np.maximum(ratings - 2.5, 0.1).astype(np.float32)
            liked_users = user_codes[liked_mask]
            liked_movies = movie_codes[liked_mask]
            liked_weights = weights[liked_mask]
            weight_sum = np.bincount(
                liked_users, weights=liked_weights, minlength=n_users
            )
            weighted_genre = np.zeros_like(user_genre)
            for position in range(len(self.genres)):
                weighted_genre[:, position] = np.bincount(
                    liked_users,
                    weights=self.item_genre[liked_movies, position] * liked_weights,
                    minlength=n_users,
                )
            self.user_genre_weighted_sum = weighted_genre
            self.user_genre_weight_sum = weight_sum
            self.user_genre_weighted = weighted_genre / np.maximum(
                weight_sum, 1e-6
            )[:, None]

        item_year_pos = np.nan_to_num(
            self.item_year[movie_codes[positive_mask]], nan=0.0
        )
        year_sum = np.bincount(
            user_codes[positive_mask], weights=item_year_pos, minlength=n_users
        )
        year_count = np.bincount(
            user_codes[positive_mask],
            weights=(~np.isnan(self.item_year[movie_codes[positive_mask]])).astype(
                np.float32
            ),
            minlength=n_users,
        )
        self.user_year_sum = year_sum
        self.user_year_count = year_count
        self.user_mean_year = np.where(
            year_count > 0, year_sum / np.maximum(year_count, 1.0), np.nan
        ).astype(np.float32)

        valid = (user_codes >= 0) & (movie_codes >= 0)
        pair_keys = (
            user_codes[valid].astype(np.int64) * n_movies + movie_codes[valid]
        )
        pair_ratings = ratings[valid]
        order = np.argsort(pair_keys, kind="stable")
        self._pair_keys = pair_keys[order]
        self._pair_ratings = pair_ratings[order]

        self.als_user_index = None
        self.als_item_index = None
        self.als_user_factors = None
        self.als_item_factors = None
        if als_model_dir is not None:
            self._load_als(Path(als_model_dir))

    def _load_als(self, model_dir):
        user_path = model_dir / "userFactors"
        item_path = model_dir / "itemFactors"
        if not user_path.exists() or not item_path.exists():
            return
        user_factors = pd.read_parquet(user_path)
        item_factors = pd.read_parquet(item_path)
        self.als_user_index = pd.Index(user_factors["id"].to_numpy())
        self.als_item_index = pd.Index(item_factors["id"].to_numpy())
        self.als_user_factors = np.vstack(
            user_factors["features"].to_numpy()
        ).astype(np.float32)
        self.als_item_factors = np.vstack(
            item_factors["features"].to_numpy()
        ).astype(np.float32)

    def als_score(self, user_ids, movie_ids):
        size = len(user_ids)
        result = np.zeros(size, dtype=np.float32)
        if self.als_user_factors is None:
            return result
        user_rows = self.als_user_index.get_indexer(user_ids)
        item_rows = self.als_item_index.get_indexer(movie_ids)
        valid = np.flatnonzero((user_rows >= 0) & (item_rows >= 0))
        for start in range(0, valid.size, CHUNK):
            chunk = valid[start : start + CHUNK]
            result[chunk] = np.einsum(
                "ij,ij->i",
                self.als_user_factors[user_rows[chunk]],
                self.als_item_factors[item_rows[chunk]],
            )
        return result

    def _pair_rating(self, keys):
        if self._pair_keys.size == 0:
            return np.full(keys.size, np.nan, dtype=np.float32)
        position = np.searchsorted(self._pair_keys, keys)
        position = np.clip(position, 0, self._pair_keys.size - 1)
        found = self._pair_keys[position] == keys
        values = np.where(found, self._pair_ratings[position], np.nan)
        return values.astype(np.float32)

    def compute(self, frame, als_scores=None, adjust_train=False):
        user_ids = frame[schema.USER].to_numpy()
        movie_ids = frame[schema.MOVIE].to_numpy()
        user_codes = self.user_index.get_indexer(user_ids)
        movie_codes = self.movie_index.get_indexer(movie_ids)
        known_user = user_codes >= 0
        known_movie = movie_codes >= 0
        safe_user = np.where(known_user, user_codes, 0)
        safe_movie = np.where(known_movie, movie_codes, 0)
        n_movies = self.movie_ids.size

        movie_count = np.where(known_movie, self.movie_count[safe_movie], np.nan)
        movie_pos_count = np.where(known_movie, self.movie_pos_count[safe_movie], np.nan)
        movie_rating_sum = np.where(
            known_movie, self.movie_rating_sum[safe_movie], np.nan
        )
        user_count = np.where(known_user, self.user_count[safe_user], np.nan)
        user_pos_count = np.where(known_user, self.user_pos_count[safe_user], np.nan)
        user_rating_sum = np.where(known_user, self.user_rating_sum[safe_user], np.nan)

        positive = None
        liked = None
        own_rating = None
        if adjust_train:
            own_rating = self._pair_rating(
                user_codes.astype(np.int64) * n_movies + movie_codes.astype(np.int64)
            )
            in_train = ~np.isnan(own_rating)
            own_rating = np.nan_to_num(own_rating, nan=0.0)
            positive = in_train & (own_rating >= self.threshold) & known_user & known_movie
            liked = (
                in_train
                & (own_rating >= self.weighted_genre_threshold)
                & known_user
                & known_movie
            )
            movie_count = movie_count - in_train.astype(np.float32)
            movie_rating_sum = movie_rating_sum - own_rating
            movie_pos_count = movie_pos_count - (
                in_train & (own_rating >= self.threshold)
            ).astype(np.float32)
            user_count = user_count - in_train.astype(np.float32)
            user_rating_sum = user_rating_sum - own_rating
            user_pos_count = user_pos_count - (
                in_train & (own_rating >= self.threshold)
            ).astype(np.float32)

        if als_scores is None:
            als_scores = self.als_score(user_ids, movie_ids)

        genre_affinity = np.full(len(frame), np.nan, dtype=np.float32)
        rows = np.flatnonzero(known_user & known_movie)
        genre_affinity[rows] = np.einsum(
            "ij,ij->i",
            self.user_genre[user_codes[rows]],
            self.item_genre[movie_codes[rows]],
        )

        year_diff = np.full(len(frame), np.nan, dtype=np.float32)
        year_diff[rows] = np.abs(
            self.item_year[movie_codes[rows]] - self.user_mean_year[user_codes[rows]]
        )

        if positive is not None and positive.any():
            adjusted = np.flatnonzero(positive)
            adjusted_users = user_codes[adjusted]
            adjusted_movies = movie_codes[adjusted]
            adjusted_genres = self.item_genre[adjusted_movies]
            genre_numerator = self.user_genre_sum[adjusted_users] - adjusted_genres
            genre_denominator = np.maximum(
                self.user_pos_count[adjusted_users] - 1.0, 1.0
            )
            genre_affinity[adjusted] = (
                np.einsum("ij,ij->i", genre_numerator, adjusted_genres)
                / genre_denominator
            )

            adjusted_years = self.item_year[adjusted_movies]
            year_valid = ~np.isnan(adjusted_years)
            year_sum = self.user_year_sum[adjusted_users] - np.nan_to_num(
                adjusted_years, nan=0.0
            )
            year_count = self.user_year_count[adjusted_users] - year_valid.astype(
                np.float32
            )
            year_mean = np.where(
                year_count > 0, year_sum / np.maximum(year_count, 1.0), np.nan
            )
            year_diff[adjusted[year_valid]] = np.abs(
                adjusted_years[year_valid] - year_mean[year_valid]
            )

        columns = {
            "als_score": als_scores.astype(np.float32),
            "movie_count": movie_count.astype(np.float32),
            "movie_pos_count": movie_pos_count.astype(np.float32),
            "movie_mean_rating": np.where(
                movie_count > 0, movie_rating_sum / np.maximum(movie_count, 1.0), np.nan
            ).astype(np.float32),
            "movie_positive_ratio": np.where(
                movie_count > 0,
                movie_pos_count / np.maximum(movie_count, 1.0),
                np.nan,
            ).astype(np.float32),
            "user_count": user_count.astype(np.float32),
            "user_pos_count": user_pos_count.astype(np.float32),
            "user_mean_rating": np.where(
                user_count > 0, user_rating_sum / np.maximum(user_count, 1.0), np.nan
            ).astype(np.float32),
            "genre_affinity": genre_affinity,
            "year_diff": year_diff,
        }

        if self.use_bayes_rating:
            with np.errstate(invalid="ignore", divide="ignore"):
                observed_mean = np.where(
                    movie_count > 0,
                    movie_rating_sum / np.maximum(movie_count, 1.0),
                    np.nan,
                )
                weight = np.maximum(movie_count, 0.0)
                smoothed = (weight / (weight + self.bayes_m)) * observed_mean + (
                    self.bayes_m / (weight + self.bayes_m)
                ) * self.global_mean
            smoothed = np.where(
                np.isnan(movie_count),
                np.nan,
                np.where(weight > 0, smoothed, self.global_mean),
            )
            columns["movie_bayes_rating"] = smoothed.astype(np.float32)

        if self.user_genre_weighted is not None:
            weighted_affinity = np.full(len(frame), np.nan, dtype=np.float32)
            if rows.size:
                weighted_affinity[rows] = np.einsum(
                    "ij,ij->i",
                    self.user_genre_weighted[user_codes[rows]],
                    self.item_genre[movie_codes[rows]],
                )
            if positive is not None:
                if liked.any():
                    adjusted = np.flatnonzero(liked)
                    adjusted_users = user_codes[adjusted]
                    adjusted_movies = movie_codes[adjusted]
                    adjusted_genres = self.item_genre[adjusted_movies]
                    self_weight = np.maximum(
                        own_rating[adjusted] - 2.5, 0.1
                    ).astype(np.float32)
                    numerator = (
                        self.user_genre_weighted_sum[adjusted_users]
                        - adjusted_genres * self_weight[:, None]
                    )
                    denominator = np.maximum(
                        self.user_genre_weight_sum[adjusted_users] - self_weight,
                        1e-6,
                    )
                    weighted_affinity[adjusted] = (
                        np.einsum("ij,ij->i", numerator, adjusted_genres)
                        / denominator
                    )
            columns["genre_affinity_weighted"] = weighted_affinity

        return pd.DataFrame(columns, index=frame.index)
