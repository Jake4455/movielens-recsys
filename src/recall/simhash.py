import time

import numpy as np


class SimHashIndex:
    def __init__(self, incidence, n_bits=128, band_size=16, seed=20260907):
        self.n_bits = int(n_bits)
        self.band_size = int(band_size)
        if self.n_bits % self.band_size != 0:
            raise ValueError("n_bits must be divisible by band_size")
        self.n_bands = self.n_bits // self.band_size
        n_items = incidence.shape[1]
        rng = np.random.default_rng(int(seed))
        projection = rng.choice(
            np.asarray([-1.0, 1.0], dtype=np.float32),
            size=(incidence.shape[0], self.n_bits),
        )
        scores = incidence.T @ projection
        self.signatures = np.asarray(scores) >= 0
        keys = np.zeros((self.n_bands, n_items), dtype=np.int64)
        for band in range(self.n_bands):
            block = self.signatures[:, band * self.band_size : (band + 1) * self.band_size]
            value = np.zeros(n_items, dtype=np.int64)
            for offset in range(self.band_size):
                value = (value << 1) | block[:, offset].astype(np.int64)
            keys[band] = value
        self.keys = keys
        self.buckets = {}
        for band in range(self.n_bands):
            band_keys = keys[band]
            order = np.argsort(band_keys, kind="stable")
            sorted_keys = band_keys[order]
            boundaries = np.flatnonzero(
                np.diff(sorted_keys, prepend=sorted_keys[0] - 1)
            )
            for position, start in enumerate(boundaries):
                stop = boundaries[position + 1] if position + 1 < boundaries.size else sorted_keys.size
                members = order[start:stop]
                if members.size > 1:
                    self.buckets[(band, int(sorted_keys[start]))] = members

    def candidates(self, code):
        collected = []
        for band in range(self.n_bands):
            members = self.buckets.get((band, int(self.keys[band, code])))
            if members is not None:
                collected.append(members)
        if not collected:
            return np.empty(0, dtype=np.int64)
        merged = np.unique(np.concatenate(collected))
        return merged[merged != code]

    def query(self, code, transposed, k=50):
        start = time.perf_counter()
        candidate_codes = self.candidates(code)
        if candidate_codes.size:
            query_vector = transposed[code]
            candidates = transposed[candidate_codes]
            scores = np.asarray((candidates @ query_vector.T).todense()).ravel()
            order = np.argsort(-scores)[:k]
            result = candidate_codes[order]
        else:
            result = np.empty(0, dtype=np.int64)
        latency = time.perf_counter() - start
        return result, latency, candidate_codes.size
