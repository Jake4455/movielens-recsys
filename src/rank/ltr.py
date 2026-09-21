import lightgbm as lgb
import numpy as np

from src.utils.config import get


def build_params(cfg):
    params = dict(get(cfg, "ltr.params", {}) or {})
    params.setdefault("objective", "lambdarank")
    params.setdefault("metric", "ndcg")
    params.setdefault("ndcg_eval_at", [10])
    params.setdefault("lambdarank_truncation_level", 10)
    params["seed"] = int(get(cfg, "seed", 20260907))
    params["verbose"] = -1
    return params


def train_ranker(cfg, features_train, labels_train, groups_train, features_val, labels_val, groups_val):
    params = build_params(cfg)
    feature_names = list(features_train.columns)
    train_set = lgb.Dataset(
        features_train,
        label=labels_train,
        group=groups_train,
        feature_name=feature_names,
        free_raw_data=True,
    )
    valid_set = lgb.Dataset(
        features_val,
        label=labels_val,
        group=groups_val,
        feature_name=feature_names,
        reference=train_set,
        free_raw_data=True,
    )
    callbacks = [
        lgb.early_stopping(int(get(cfg, "ltr.early_stopping", 50)), verbose=False),
        lgb.log_evaluation(int(get(cfg, "ltr.log_every", 50))),
    ]
    model = lgb.train(
        params,
        train_set,
        num_boost_round=int(get(cfg, "ltr.num_boost_round", 300)),
        valid_sets=[valid_set],
        valid_names=["val"],
        callbacks=callbacks,
    )
    return model


def predict_scores(model, features, chunk=2_000_000):
    scores = np.empty(len(features), dtype=np.float64)
    for start in range(0, len(features), int(chunk)):
        stop = start + int(chunk)
        scores[start:stop] = model.predict(features.iloc[start:stop])
    return scores
