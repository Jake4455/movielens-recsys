import os
import sys
from pathlib import Path

from src.utils.config import get


def _configure_hadoop_home(cfg):
    if os.name != "nt":
        return
    hadoop_home = get(cfg, "spark.hadoop_home")
    if not hadoop_home:
        return
    hadoop_path = Path(hadoop_home)
    if not hadoop_path.is_absolute():
        hadoop_path = Path(cfg["_root"]) / hadoop_path
    if not (hadoop_path / "bin" / "winutils.exe").exists():
        return
    os.environ["HADOOP_HOME"] = str(hadoop_path)
    os.environ["hadoop.home.dir"] = str(hadoop_path)
    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    bin_path = str(hadoop_path / "bin")
    if bin_path not in path_entries:
        os.environ["PATH"] = bin_path + os.pathsep + os.environ.get("PATH", "")


def get_spark(cfg, app_name="movielens-recsys", master=None):
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")
    _configure_hadoop_home(cfg)
    from pyspark.sql import SparkSession

    builder = (
        SparkSession.builder.appName(app_name)
        .master(master or get(cfg, "spark.master", "local[8]"))
        .config("spark.driver.memory", get(cfg, "spark.driver_memory", "8g"))
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config(
            "spark.sql.shuffle.partitions",
            str(get(cfg, "spark.shuffle_partitions", 64)),
        )
        .config("spark.ui.enabled", "false")
        .config("spark.sql.adaptive.enabled", "true")
    )
    local_dir = get(cfg, "spark.local_dir")
    if local_dir:
        local_path = Path(local_dir)
        if not local_path.is_absolute():
            local_path = Path(cfg["_root"]) / local_path
        local_path.mkdir(parents=True, exist_ok=True)
        builder = builder.config("spark.local.dir", str(local_path))
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel(get(cfg, "spark.log_level", "WARN"))
    return spark
