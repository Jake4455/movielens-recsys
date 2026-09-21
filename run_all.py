import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.stages import STAGES
from src.utils.config import ensure_dirs, get, load_config, path_of
from src.utils.logging_utils import setup_logger
from src.utils.seeds import set_all_seeds

DEFAULT_STAGES = [
    "validate",
    "split",
    "stats",
    "candidates",
    "baseline",
    "als",
    "graph",
    "stream",
    "lsh",
    "ltr",
]


def parse_args():
    parser = argparse.ArgumentParser(description="MovieLens 32M recommender pipeline")
    parser.add_argument("--config", default="conf/final.yaml")
    parser.add_argument("--stages", default=",".join(DEFAULT_STAGES))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    ensure_dirs(
        path_of(cfg, "paths.outputs_dir"),
        path_of(cfg, "paths.processed_dir"),
        path_of(cfg, "paths.reports_dir"),
        path_of(cfg, "paths.logs_dir"),
    )
    logger = setup_logger("run_all", path_of(cfg, "paths.logs_dir"))
    seed = set_all_seeds(get(cfg, "seed", 20260907))
    logger.info("config=%s seed=%d", cfg["_config_path"], seed)
    requested = [name.strip() for name in args.stages.split(",") if name.strip()]
    unknown = [name for name in requested if name not in STAGES]
    if unknown:
        raise SystemExit(f"unknown stages: {unknown}, available: {sorted(STAGES)}")
    for name in requested:
        start = time.perf_counter()
        logger.info("stage start: %s", name)
        STAGES[name](cfg, force=args.force)
        logger.info("stage done: %s in %.2fs", name, time.perf_counter() - start)
    logger.info("pipeline finished")


if __name__ == "__main__":
    main()
