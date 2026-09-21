import time

import numpy as np
import pandas as pd
from scipy import sparse

from src.data import schema


def build_incidence(splits, threshold=4.0):
    train = splits[schema.TRAIN]
    positives = train[train[schema.RATING] >= threshold]
    item_ids = np.sort(train[schema.MOVIE].unique())
    user_ids = np.sort(train[schema.USER].unique())
    item_index = pd.Index(item_ids)
    user_index = pd.Index(user_ids)
    user_codes = user_index.get_indexer(positives[schema.USER])
    item_codes = item_index.get_indexer(positives[schema.MOVIE])
    incidence = sparse.csr_matrix(
        (np.ones(len(user_codes), dtype=np.float32), (user_codes, item_codes)),
        shape=(len(user_ids), len(item_ids)),
    )
    return incidence, item_ids, user_ids


def exact_topk(incidence, top_k=50, block_size=500):
    n_items = incidence.shape[1]
    counts = np.asarray(incidence.sum(axis=0)).ravel()
    incidence_csc = incidence.tocsc()
    rows = []
    cols = []
    vals = []
    for start in range(0, n_items, block_size):
        stop = min(start + block_size, n_items)
        block = incidence_csc[:, start:stop]
        cooccur = (block.T @ incidence).tocsr()
        local_rows = np.arange(stop - start)
        self_columns = start + local_rows
        self_similarity = sparse.csr_matrix(
            (counts[start:stop], (local_rows, self_columns)),
            shape=cooccur.shape,
        )
        cooccur = (cooccur - self_similarity).tocsr()
        cooccur.eliminate_zeros()
        dense = np.asarray(cooccur.todense(), dtype=np.float32)
        block_counts = np.maximum(counts[start:stop], 1.0).astype(np.float32)
        all_counts = np.maximum(counts, 1.0).astype(np.float32)
        dense = dense / (np.sqrt(block_counts)[:, None] * np.sqrt(all_counts)[None, :])
        kth = min(top_k, max(dense.shape[1] - 1, 1))
        indices = np.argpartition(-dense, kth, axis=1)[:, :kth]
        values = np.take_along_axis(dense, indices, axis=1)
        mask = values > 0
        row_ids, col_ids = np.nonzero(mask)
        rows.append(row_ids + start)
        cols.append(indices[row_ids, col_ids])
        vals.append(values[row_ids, col_ids])
    topk = sparse.csr_matrix(
        (
            np.concatenate(vals),
            (np.concatenate(rows), np.concatenate(cols)),
        ),
        shape=(n_items, n_items),
    )
    topk.sum_duplicates()
    return topk


def brute_force_queries(incidence, query_codes, k=50):
    transposed = incidence.T.tocsr()
    latencies = []
    results = []
    for code in query_codes:
        start = time.perf_counter()
        sims = np.asarray((transposed[code] @ incidence).todense()).ravel()
        sims[code] = -1.0
        order = np.argsort(-sims)[:k]
        latencies.append(time.perf_counter() - start)
        results.append(order)
    return float(np.mean(latencies)), results


def topk_rows(topk, codes, k=None):
    out = []
    for code in codes:
        start, stop = topk.indptr[code], topk.indptr[code + 1]
        cols = topk.indices[start:stop]
        vals = topk.data[start:stop]
        order = np.argsort(-vals)
        if k is not None:
            order = order[:k]
        out.append(cols[order])
    return out
