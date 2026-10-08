"""物品-物品精确余弦相似度特征（任务六：把召回模块的输出接入排序模型）。

设计要点
--------
1. **只用训练集**：相似度矩阵与"用户正反馈集合"都由 `train` 分片构建，与项目中
   其他统计量（movie_count / user_count / 图工件）的口径一致，不含任何 val/test 信息。
2. **训练行做 leave-one-out**：训练组的正样本本身就在用户正反馈集合里，若不剔除会得到
   `sim = 1.0` 的自我相似度（与 basic.py / graph.py 的 `adjust_train` 同类问题）。
3. **保持稀疏**：71,411 × 71,411 的稠密矩阵需要约 20 GB，这里全程使用 scipy.sparse
   逐块计算，峰值仅为单个 500 × n_items 的稠密块（约 143 MB）。
4. 特征语义：候选电影与该用户训练期正反馈电影之间的**最大余弦相似度**；
   无正反馈历史或候选不在训练目录内时取 0（与该项目的图特征"图外补 0"一致）。
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from src.data import schema
from src.recall.itemcf import build_incidence, exact_topk

FEATURE_NAME = "itemcf_max_sim"


class ItemCFContext:
    def __init__(self, similarity, incidence, item_ids, user_ids, top_k):
        self.similarity = similarity.tocsr()
        self.incidence = incidence.tocsr()
        self.item_index = pd.Index(np.asarray(item_ids))
        self.user_index = pd.Index(np.asarray(user_ids))
        self.item_counts = np.asarray(self.incidence.sum(axis=0)).ravel().astype(np.float64)
        self.top_k = int(top_k)

    # ------------------------------------------------------------------ build
    @classmethod
    def build(cls, splits, threshold=4.0, top_k=50, block_size=500, verbose=True):
        started = time.perf_counter()
        incidence, item_ids, user_ids = build_incidence(splits, threshold=threshold)
        if verbose:
            print(
                f"  incidence: {incidence.shape[0]:,} users x {incidence.shape[1]:,} items, "
                f"{incidence.nnz:,} positives ({time.perf_counter() - started:.1f}s)",
                flush=True,
            )
        topk = exact_topk(incidence, top_k=top_k, block_size=block_size)
        # 对称化：top-K 是按行截断的，取逐元素最大值可同时利用两个方向的邻居
        similarity = topk.maximum(topk.T).tocsr()
        similarity.eliminate_zeros()
        if verbose:
            print(
                f"  similarity: {similarity.shape}, {similarity.nnz:,} nnz, "
                f"top_k={top_k} ({time.perf_counter() - started:.1f}s total)",
                flush=True,
            )
        return cls(similarity, incidence, item_ids, user_ids, top_k)

    # ------------------------------------------------------------------- io
    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            sim_data=self.similarity.data,
            sim_indices=self.similarity.indices,
            sim_indptr=self.similarity.indptr,
            sim_shape=self.similarity.shape,
            inc_data=self.incidence.data,
            inc_indices=self.incidence.indices,
            inc_indptr=self.incidence.indptr,
            inc_shape=self.incidence.shape,
            item_ids=self.item_index.to_numpy(),
            user_ids=self.user_index.to_numpy(),
            top_k=np.asarray([self.top_k]),
        )
        return path

    @classmethod
    def load(cls, path):
        blob = np.load(Path(path), allow_pickle=False)
        similarity = sparse.csr_matrix(
            (blob["sim_data"], blob["sim_indices"], blob["sim_indptr"]), shape=tuple(blob["sim_shape"])
        )
        incidence = sparse.csr_matrix(
            (blob["inc_data"], blob["inc_indices"], blob["inc_indptr"]), shape=tuple(blob["inc_shape"])
        )
        return cls(similarity, incidence, blob["item_ids"], blob["user_ids"], int(blob["top_k"][0]))

    @classmethod
    def load_or_build(cls, splits, directory, threshold=4.0, top_k=50, block_size=500,
                      force=False, verbose=True):
        directory = Path(directory)
        path = directory / f"itemcf_top{int(top_k)}.npz"
        meta_path = directory / f"itemcf_top{int(top_k)}_meta.json"
        if path.exists() and not force:
            if verbose:
                print(f"  loading cached similarity: {path}", flush=True)
            return cls.load(path), False
        context = cls.build(splits, threshold=threshold, top_k=top_k,
                            block_size=block_size, verbose=verbose)
        context.save(path)
        meta_path.write_text(
            json.dumps(
                {
                    "n_items": int(context.similarity.shape[0]),
                    "n_users": int(context.incidence.shape[0]),
                    "n_positive_pairs": int(context.incidence.nnz),
                    "similarity_nnz": int(context.similarity.nnz),
                    "top_k": int(top_k),
                    "block_size": int(block_size),
                    "positive_threshold": float(threshold),
                    "source": "train split positives only (leave-one-out applied per training row)",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return context, True

    # ---------------------------------------------------------------- compute
    def compute(self, frame, adjust_train=False, leave_one_user_out=True):
        """返回单列 DataFrame：候选电影与该用户训练期正反馈的最大余弦相似度。

        ``leave_one_user_out``（默认开启）会从相似度统计中扣除**该用户自己**的贡献：
        否则训练组的正样本与用户其它正反馈物品天然共现，余弦相似度被自身边抬高
        （实测会把 best_iter 压到 1，与 `graph_walk2` 的历史失败同因）。校正公式：

            cooc'[i,j] = cooc[i,j] - 1{u 同时评过 i 与 j}
            c_i' = c_i - 1{u 评过 i}，c_j' = c_j - 1{u 评过 j}
            sim'[i,j] = cooc'[i,j] / sqrt(c_i' · c_j')

        其中 cooc = sim · sqrt(c_i · c_j) 由已存的归一化相似度反推。
        """
        user_codes = self.user_index.get_indexer(frame[schema.USER].to_numpy())
        item_codes = self.item_index.get_indexer(frame[schema.MOVIE].to_numpy())
        values = np.zeros(len(frame), dtype=np.float32)

        order = np.argsort(user_codes, kind="stable")
        codes_sorted = user_codes[order]
        boundaries = np.flatnonzero(
            np.r_[True, codes_sorted[1:] != codes_sorted[:-1]]
        )
        boundaries = np.r_[boundaries, len(codes_sorted)]

        for index in range(len(boundaries) - 1):
            rows = order[boundaries[index] : boundaries[index + 1]]
            user_code = int(codes_sorted[boundaries[index]])
            if user_code < 0:
                continue
            start, stop = self.incidence.indptr[user_code], self.incidence.indptr[user_code + 1]
            positives = np.sort(self.incidence.indices[start:stop])
            if positives.size == 0:
                continue
            candidate_codes = item_codes[rows]
            known = candidate_codes >= 0
            if not known.any():
                continue
            known_rows = rows[known]
            known_codes = candidate_codes[known]
            # 行 = 用户的正反馈物品，列 = 该用户这批判候选物品
            block = self.similarity[positives][:, known_codes].toarray().astype(np.float64)

            if leave_one_user_out:
                counts_i = self.item_counts[known_codes]
                counts_j = self.item_counts[positives]
                cooc = block * np.sqrt(counts_j)[:, None] * np.sqrt(counts_i)[None, :]
                # u 同时评过候选 i 与历史 j 时，共同现计数各减 1
                slot = np.clip(np.searchsorted(positives, known_codes), 0, positives.size - 1)
                in_history = positives[slot] == known_codes
                cooc = cooc - in_history[None, :]
                counts_j = counts_j - 1.0
                counts_i = counts_i - in_history.astype(np.float64)
                denominator = np.sqrt(np.maximum(counts_j, 1.0))[:, None] * np.sqrt(
                    np.maximum(counts_i, 1.0)
                )[None, :]
                block = np.clip(cooc, 0.0, None) / denominator
            else:
                slot = np.clip(np.searchsorted(positives, known_codes), 0, positives.size - 1)
                in_history = positives[slot] == known_codes

            if adjust_train or leave_one_user_out:
                # 自身相似度（i == j）不参与：训练正样本本身就是用户的历史物品
                columns = np.flatnonzero(in_history)
                if columns.size:
                    block[slot[columns], columns] = -np.inf
            best = block.max(axis=0)
            best[~np.isfinite(best)] = 0.0  # 除自身外无其它正反馈可比 → 0
            values[known_rows] = best.astype(np.float32)

        return pd.DataFrame({FEATURE_NAME: values}, index=frame.index)
