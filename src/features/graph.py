import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from src.data import schema

ARTIFACT_FILE = "graph_artifacts.npz"
META_FILE = "graph_meta.json"


class GraphContext:
    def __init__(
        self,
        user_ids,
        item_ids,
        pagerank,
        graph_item_ids,
        global_to_local,
        topk,
        labels,
        profile_indptr,
        profile_indices,
        profile_data,
        n_communities,
        user_positive,
        graph_item_count,
        meta,
    ):
        self.user_ids = user_ids
        self.item_ids = item_ids
        self.user_index = pd.Index(user_ids)
        self.item_index = pd.Index(item_ids)
        self.pagerank = pagerank
        self.graph_item_ids = graph_item_ids
        self.graph_item_index = pd.Index(graph_item_ids)
        self.global_to_local = global_to_local
        self.topk = topk
        self.topk_t_csr = topk.T.tocsr()
        self.labels = labels
        self.profile_indptr = profile_indptr
        self.profile_indices = profile_indices
        self.profile_data = profile_data
        self.n_communities = int(n_communities)
        self.user_positive = user_positive
        self.graph_item_count = graph_item_count
        self._count_scale = (1.0 / np.sqrt(np.maximum(graph_item_count, 1.0))).astype(
            np.float32
        )
        self.meta = meta
        self._profile_keys = self._build_profile_keys()

    def _build_profile_keys(self):
        counts = np.diff(self.profile_indptr)
        users = np.repeat(np.arange(len(counts), dtype=np.int64), counts)
        keys = users * self.n_communities + self.profile_indices
        return keys.astype(np.int64)

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            directory / ARTIFACT_FILE,
            user_ids=self.user_ids,
            item_ids=self.item_ids,
            pagerank=self.pagerank,
            graph_item_ids=self.graph_item_ids,
            global_to_local=self.global_to_local,
            topk_data=self.topk.data,
            topk_indices=self.topk.indices,
            topk_indptr=self.topk.indptr,
            topk_shape=np.asarray(self.topk.shape),
            labels=self.labels,
            profile_indptr=self.profile_indptr,
            profile_indices=self.profile_indices,
            profile_data=self.profile_data,
            n_communities=np.asarray(self.n_communities),
            user_positive=self.user_positive,
            graph_item_count=self.graph_item_count,
        )
        with open(directory / META_FILE, "w", encoding="utf-8") as handle:
            json.dump(self.meta, handle, ensure_ascii=False, indent=2)
        return directory

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        data = np.load(directory / ARTIFACT_FILE)
        with open(directory / META_FILE, "r", encoding="utf-8") as handle:
            meta = json.load(handle)
        topk = sparse.csr_matrix(
            (data["topk_data"], data["topk_indices"], data["topk_indptr"]),
            shape=tuple(data["topk_shape"]),
        )
        return cls(
            user_ids=data["user_ids"],
            item_ids=data["item_ids"],
            pagerank=data["pagerank"],
            graph_item_ids=data["graph_item_ids"],
            global_to_local=data["global_to_local"],
            topk=topk,
            labels=data["labels"],
            profile_indptr=data["profile_indptr"],
            profile_indices=data["profile_indices"],
            profile_data=data["profile_data"],
            n_communities=int(data["n_communities"]),
            user_positive=data["user_positive"],
            graph_item_count=data["graph_item_count"],
            meta=meta,
        )

    def compute(self, frame, adjust_train=False):
        user_codes = self.user_index.get_indexer(frame[schema.USER].to_numpy())
        item_codes = self.item_index.get_indexer(frame[schema.MOVIE].to_numpy())
        known = (user_codes >= 0) & (item_codes >= 0)

        pagerank = np.full(len(frame), np.nan, dtype=np.float32)
        pagerank[known] = self.pagerank[item_codes[known]]

        local = np.full(len(frame), -1, dtype=np.int64)
        local[known] = self.global_to_local[item_codes[known]]
        in_graph = local >= 0

        walk = np.full(len(frame), np.nan, dtype=np.float32)
        community = np.full(len(frame), np.nan, dtype=np.float32)

        rows = np.flatnonzero(in_graph)
        chunk = 200_000
        for start in range(0, rows.size, chunk):
            selection = rows[start : start + chunk]
            chunk_users = user_codes[selection]
            chunk_items = local[selection]
            incidence = self._incidence[chunk_users]
            incoming = self.topk_t_csr[chunk_items]
            overlap = incidence.multiply(incoming)
            chunk_walk = np.asarray(overlap.sum(axis=1)).ravel().astype(np.float32)

            query_keys = chunk_users.astype(np.int64) * self.n_communities + self.labels[
                chunk_items
            ].astype(np.int64)
            position = np.searchsorted(self._profile_keys, query_keys)
            position = np.clip(position, 0, max(self._profile_keys.size - 1, 0))
            found = self._profile_keys[position] == query_keys
            values = np.where(found, self.profile_data[position], 0.0)
            chunk_community = values.astype(np.float32)

            if adjust_train:
                selector = sparse.csr_matrix(
                    (
                        np.ones(selection.size, dtype=np.float32),
                        (np.arange(selection.size), chunk_items),
                    ),
                    shape=(selection.size, self._incidence.shape[1]),
                )
                owned = (
                    np.asarray(incidence.multiply(selector).sum(axis=1)).ravel() > 0
                )
                if owned.any():
                    incoming_binary = (incoming > 0).astype(np.float32)
                    scaled_incidence = incidence.multiply(self._count_scale[None, :])
                    own_partial = np.asarray(
                        scaled_incidence.multiply(incoming_binary).sum(axis=1)
                    ).ravel().astype(np.float32)
                    own_walk = own_partial * self._count_scale[chunk_items]
                    chunk_walk[owned] = np.maximum(
                        chunk_walk[owned] - own_walk[owned], 0.0
                    )
                    liked = self.user_positive[chunk_users[owned]].astype(np.float32)
                    counts = chunk_community[owned] * liked
                    denominator = np.maximum(liked - 1.0, 1.0)
                    chunk_community[owned] = np.where(
                        liked > 1.0, (counts - 1.0) / denominator, 0.0
                    )

            walk[selection] = chunk_walk
            community[selection] = chunk_community

        outside = known & ~in_graph
        walk[outside] = 0.0
        community[outside] = 0.0
        with np.errstate(invalid="ignore"):
            pagerank = np.log1p(np.maximum(pagerank, 0.0))
            walk = np.log1p(np.maximum(walk, 0.0))

        return pd.DataFrame(
            {
                "graph_pagerank": pagerank,
                "graph_walk2": walk,
                "graph_community_affinity": community,
            },
            index=frame.index,
        )


def _detect_communities(
    symmetric, method="louvain", lpa_iters=5, seed=20260907, resolution=1.0
):
    n_graph = symmetric.shape[0]
    if method == "louvain":
        try:
            import igraph as ig
        except ImportError:
            method = "lpa"
        else:
            coo = sparse.triu(symmetric, k=1).tocoo()
            graph = ig.Graph(
                n=n_graph,
                edges=list(zip(coo.row.tolist(), coo.col.tolist())),
                directed=False,
            )
            graph.es["weight"] = coo.data.tolist()
            try:
                membership = graph.community_multilevel(
                    weights="weight", return_levels=False, resolution=float(resolution)
                )
            except TypeError:
                membership = graph.community_multilevel(
                    weights="weight", return_levels=False
                )
            labels = np.asarray(membership.membership, dtype=np.int64)
            _, labels = np.unique(labels, return_inverse=True)
            return labels
    labels = np.arange(n_graph)
    for _ in range(lpa_iters):
        updated = labels.copy()
        for node in range(n_graph):
            start, stop = symmetric.indptr[node], symmetric.indptr[node + 1]
            if start == stop:
                continue
            neighbor_labels = updated[symmetric.indices[start:stop]]
            weights = symmetric.data[start:stop]
            values, inverse = np.unique(neighbor_labels, return_inverse=True)
            totals = np.bincount(inverse, weights=weights, minlength=values.size)
            updated[node] = values[np.argmax(totals)]
        labels = updated
    _, labels = np.unique(labels, return_inverse=True)
    return labels


def build_graph_context(
    splits,
    threshold=4.0,
    top_k=50,
    max_items=20000,
    pagerank_iters=20,
    damping=0.85,
    lpa_iters=5,
    block_size=500,
    community_method="louvain",
    community_resolution=1.0,
    seed=20260907,
):
    train = splits[schema.TRAIN]
    positives = train[train[schema.RATING] >= threshold]
    item_ids = np.sort(train[schema.MOVIE].unique())
    user_ids = np.sort(train[schema.USER].unique())
    item_index = pd.Index(item_ids)
    user_index = pd.Index(user_ids)
    user_codes = user_index.get_indexer(positives[schema.USER])
    item_codes = item_index.get_indexer(positives[schema.MOVIE])
    n_users = len(user_ids)
    n_items = len(item_ids)

    incidence = sparse.csr_matrix(
        (np.ones(len(user_codes), dtype=np.float32), (user_codes, item_codes)),
        shape=(n_users, n_items),
    )
    item_count = np.asarray(incidence.sum(axis=0)).ravel()

    adjacency = sparse.bmat(
        [[None, incidence], [incidence.T, None]], format="csr"
    )
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    inv_degree = np.where(degree > 0, 1.0 / np.maximum(degree, 1.0), 0.0).astype(
        np.float32
    )
    n_nodes = n_users + n_items
    rank = np.full(n_nodes, 1.0 / n_nodes, dtype=np.float32)
    for _ in range(pagerank_iters):
        rank = (1.0 - damping) / n_nodes + damping * (
            adjacency @ (rank * inv_degree)
        )
    pagerank = rank[n_users:]
    pagerank = (pagerank / pagerank.sum() * n_items).astype(np.float32)

    graph_items = np.sort(np.argsort(-item_count)[: max_items])
    graph_item_ids = item_ids[graph_items]
    global_to_local = np.full(n_items, -1, dtype=np.int64)
    global_to_local[graph_items] = np.arange(graph_items.size)
    n_graph = graph_items.size

    incidence_csc = incidence.tocsc()
    graph_counts = np.maximum(item_count[graph_items], 1.0).astype(np.float32)
    topk_rows = []
    topk_cols = []
    topk_vals = []
    for start in range(0, n_graph, block_size):
        stop = min(start + block_size, n_graph)
        block = incidence_csc[:, graph_items[start:stop]]
        cooccur = (block.T @ incidence[:, graph_items]).tocsr()
        local_rows = np.arange(stop - start)
        self_columns = start + local_rows
        self_similarity = sparse.csr_matrix(
            (item_count[graph_items[start:stop]], (local_rows, self_columns)),
            shape=cooccur.shape,
        )
        cooccur = (cooccur - self_similarity).tocsr()
        cooccur.eliminate_zeros()
        dense = np.asarray(cooccur.todense(), dtype=np.float32)
        block_counts = graph_counts[start:stop]
        denominator = np.sqrt(block_counts)[:, None] * np.sqrt(graph_counts)[None, :]
        dense = dense / denominator
        kth = min(top_k, max(dense.shape[1] - 1, 1))
        indices = np.argpartition(-dense, kth, axis=1)[:, :kth]
        values = np.take_along_axis(dense, indices, axis=1)
        mask = values > 0
        row_ids, col_ids = np.nonzero(mask)
        topk_rows.append(row_ids + start)
        topk_cols.append(indices[row_ids, col_ids])
        topk_vals.append(values[row_ids, col_ids])
    topk = sparse.csr_matrix(
        (
            np.concatenate(topk_vals),
            (np.concatenate(topk_rows), np.concatenate(topk_cols)),
        ),
        shape=(n_graph, n_graph),
    )
    topk.sum_duplicates()

    symmetric = (topk + topk.T).astype(np.float32).tocsr()
    symmetric.setdiag(0.0)
    symmetric.eliminate_zeros()
    labels = _detect_communities(
        symmetric,
        method=community_method,
        lpa_iters=lpa_iters,
        seed=seed,
        resolution=community_resolution,
    )
    n_communities = int(labels.max()) + 1

    valid = global_to_local[item_codes] >= 0
    community_counts = sparse.csr_matrix(
        (
            np.ones(int(valid.sum()), dtype=np.float32),
            (
                user_codes[valid],
                labels[global_to_local[item_codes[valid]]],
            ),
        ),
        shape=(n_users, n_communities),
    )
    user_positive = np.asarray(incidence.sum(axis=1)).ravel()
    profile = (
        sparse.diags(1.0 / np.maximum(user_positive, 1.0)) @ community_counts
    ).tocsr()

    coverage = float(item_count[graph_items].sum() / max(item_count.sum(), 1.0))
    meta = {
        "seed": int(seed),
        "positive_threshold": float(threshold),
        "n_users": int(n_users),
        "n_items": int(n_items),
        "max_items": int(max_items),
        "n_graph_items": int(n_graph),
        "top_k": int(top_k),
        "pagerank_iters": int(pagerank_iters),
        "damping": float(damping),
        "lpa_iters": int(lpa_iters),
        "community_method": str(community_method),
        "community_resolution": float(community_resolution),
        "n_communities": int(n_communities),
        "graph_item_positive_coverage": coverage,
        "topk_nnz": int(topk.nnz),
    }

    context = GraphContext(
        user_ids=user_ids,
        item_ids=item_ids,
        pagerank=pagerank,
        graph_item_ids=graph_item_ids,
        global_to_local=global_to_local,
        topk=topk,
        labels=labels,
        profile_indptr=profile.indptr,
        profile_indices=profile.indices,
        profile_data=profile.data,
        n_communities=n_communities,
        user_positive=user_positive.astype(np.float32),
        graph_item_count=graph_counts.astype(np.float32),
        meta=meta,
    )
    context._incidence = incidence[:, graph_items].tocsr()
    return context


def compute_incidence(context, splits, threshold=4.0):
    train = splits[schema.TRAIN]
    positives = train[train[schema.RATING] >= threshold]
    user_codes = context.user_index.get_indexer(positives[schema.USER])
    item_codes = context.item_index.get_indexer(positives[schema.MOVIE])
    return sparse.csr_matrix(
        (np.ones(len(user_codes), dtype=np.float32), (user_codes, item_codes)),
        shape=(len(context.user_ids), len(context.item_ids)),
    )


def load_or_build(splits, directory, force=False, **kwargs):
    directory = Path(directory)
    if not force and (directory / ARTIFACT_FILE).exists():
        context = GraphContext.load(directory)
        full_incidence = compute_incidence(
            context, splits, threshold=kwargs.get("threshold", 4.0)
        )
        graph_positions = np.flatnonzero(context.global_to_local >= 0)
        context._incidence = full_incidence[:, graph_positions].tocsr()
        return context, False
    context = build_graph_context(splits, **kwargs)
    context.save(directory)
    return context, True
