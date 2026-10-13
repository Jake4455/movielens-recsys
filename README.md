# MovieLens 32M 分布式增量电影推荐系统

课程项目：基于 MovieLens 32M，在教师协议（每位测试用户 1 个正样本 + 100 个固定负样本，种子 `20260907`）下优化 **NDCG@10**。

**最终结果**（2026-10-08）：**R13_ltr_itemcf_noes**（17 特征 = F1 协同 + F2 内容 + F3 流式 + F4 图 + **ItemCF 召回相似度**；固定 300 轮）test NDCG@10 = **0.8728**，相对热门基线（0.7540）**+15.75%**；用户配对 95% CI 下界 **+0.1177**（p<0.001）；σ_seed = **0.000363**（5 套候选 lift 1.1561–1.1575，**5/5 ≥ 1.1**）。

> **两项关键发现（2026-10-08）**
> 1. **候选集负采样口径缺陷（已修）**：`src/data/candidates.py` 的负采样此前在 `np.unique` 升序结果上直接截断，实际保留每次抽样中**最小**的 100 个 code（约 51% 目录不可达、负样本平均热度为均匀抽样的 2.2 倍），导致基线被低估为 0.6154、lift 被夸大为 +23.56%。修复后基线 0.7540、lift +13.88%（当时口径）。修复前证据留在 `outputs_dev/pre_fix_evidence/`。
> 2. **早停缺陷（已定位，影响面已界定）**：LightGBM 内部验证指标按行序打破并列，而训练组把正样本放在每组首行；当新特征在前几轮产生大量并列分数时，内部 NDCG 被系统性抬高，早停会锁死在第 1 轮。同一组特征开早停得 0.8296、关早停得 0.8728（差 0.043）。**拆分实验 R14（16 特征 + 固定 300 轮 = 0.8588）证明该缺陷只在高并列场景发作**：R10 与 R14 仅差 +0.000130，其余早停臂未被拖累。详见 `docs/消融总表.md` §3 表注 9。

---

## 1. 环境准备

| 项 | 值 |
|---|---|
| OS | Windows 10/11 |
| Python | 3.11（conda env `mlrecsys`） |
| 依赖 | PySpark 4.0.0、LightGBM 4.7.0、pandas 2.2.3、numpy 1.26.4、scipy、python-igraph、matplotlib、pyarrow、PyYAML、pytest |
| JDK | 21 LTS |

```bash
conda create -n mlrecsys python=3.11 -y
conda activate mlrecsys
pip install -r requirements.txt
```

**Windows Spark 依赖**：`tools/hadoop/bin/winutils.exe` 与 `hadoop.dll` 已随仓库提供（来源见 §7 合规声明）。`src/utils/spark.py` 会自动设置 `HADOOP_HOME`、`hadoop.home.dir`、`PATH` 与 `SPARK_LOCAL_IP=127.0.0.1`。

### 1.2 在 WSL(Ubuntu 26.04) 中运行（2026-10-12 验证）

原则：**代码留 `/mnt/d`（git 工作区，Windows 可编辑），重数据与 Spark 落盘放 ext4**。本机实测 9p（`/mnt/d`）顺序读 286 MB/s vs ext4 2.6 GB/s；建 2000 个小文件慢约 **120×**——Spark 的 shuffle/spill 正是海量小文件，留在 9p 会被拖死。

| 项 | 值 |
|---|---|
| JDK | OpenJDK **21.0.12.1**（经 `wsl.exe -u root` + apt 安装，无需交互密码） |
| 环境 | conda env `mlrecsys` @ `/opt/work/envs/mlrecsys`（python 3.11.17，依赖与 Windows 版逐项同版本） |
| 原始数据 | `/opt/work/data/movielens-recsys/ml-32m`（MD5 与 `checksums.txt` 全部一致） |
| 派生数据 | `/opt/work/data/movielens-recsys/processed`（重新生成，不复用 Windows 产物） |
| Spark 落盘 | `/opt/work/tmp/spark-mlrecsys`（ext4） |
| 运行产物 | `/opt/work/data/movielens-recsys/outputs_wsl`（**不覆盖仓库 `outputs/`**） |
| 配置 | `conf/wsl.yaml`（= `conf/final.yaml`，仅改 5 个 `paths.*` + `spark.local_dir` + `driver_memory 8g→6g`）、`conf/wsl_itemcf_noes.yaml`（R13 臂同口径） |

```bash
source /opt/work/miniforge3/etc/profile.d/conda.sh && conda activate mlrecsys
bash /mnt/d/Ubuntu/tools/install-mlrecsys-wsl.sh    # 幂等：JDK + conda 环境 + 依赖 + 数据落位
bash /mnt/d/Ubuntu/tools/run-mlrecsys-wsl.sh conf/wsl.yaml \
     validate,split,stats,candidates,baseline,als,graph,stream,lsh,ltr wsl-full
bash /mnt/d/Ubuntu/tools/verify-mlrecsys-wsl.sh     # 只读自检 + 与 Windows 基准逐 run 对账
```

**复算结果**：`R00_pop`、`R03_als`、`R07_ltr_content`、`R13_ltr_itemcf_noes` 四个 run 的 `ndcg_at_k / hr_at_k / recall_at_k / mrr_at_k / mean_rank / val_ndcg_at_k / best_iteration / n_features` **全部逐位相同**（27 项比对，delta = 0.00e+00）。R13 = **0.8727866454 / lift 1.1575**，与提交值一致。

| 阶段 | Windows | WSL | 说明 |
|---|---|---|---|
| candidates / baseline | 21.2 s / 25.4 s | 20.3 s / 10.5 s | 数据在 ext4 |
| als | 351.5 s | 420.1 s | driver 6g（WSL 内存所限）略慢 |
| stream | 218.2 s | 110.4 s | 快 2× |
| ltr（R07 12 特征 / R13 17 特征） | ~194 s / ~564 s | 200.6 s / 462.3 s | 早停轮数同为 225 |

三条 WSL 特有约束：① `spark.local_dir` 必须在 ext4 上；② `driver_memory` 降到 6g，`ltr` 阶段峰值会顶到 ~11.9 GB（11.9 GiB 上限，靠 swap 兜底）；③ 后台长任务必须用 DSH 托管后台任务，`setsid nohup` 起的进程会在工具调用结束时被回收。


## 2. 数据准备

从 GroupLens 下载 `ml-32m.zip` 并解压到 `data/ml-32m/`，目录应包含：

```
data/ml-32m/{ratings.csv, movies.csv, tags.csv, links.csv, README.txt, checksums.txt}
```

用 `checksums.txt` 校验 MD5（ratings.csv = `cf12b74f9ad4b94a011f079e26d4270a` 等）。**原始数据不提交**，只提交脚本。

## 3. 一键复现

```bash
conda activate mlrecsys
python run_all.py --config conf/final.yaml
```

默认依次执行：`validate → split → stats → candidates → baseline → als → graph → stream → lsh → ltr`（每步产物落盘，重复运行自动跳过；`--force` 强制重算；`--stages` 指定子集）。

各实验臂复现命令：

| 目标 | 命令 |
|---|---|
| 协议层 + 热门基线（R00） | `python run_all.py --config conf/final.yaml --stages validate,split,stats,candidates,baseline` |
| Spark ALS（R03） | `python run_all.py --config conf/final.yaml --stages als` |
| 图特征（R08） | `python run_all.py --config conf/graph.yaml --stages graph,ltr --force` |
| 流式 F3（R10） | `python run_all.py --config conf/stream.yaml --stages stream,ltr --force` |
| **召回特征 + 固定轮数（R13，提交臂）** | `python run_all.py --config conf/itemcf_noes.yaml --stages stream,ltr --force` |
| 任务二 精确 vs LSH | `python run_all.py --config conf/final.yaml --stages lsh --force` |
| 任务三 分布式统计 + 扩展性 | `python run_all.py --config conf/stream.yaml --stages spark_features,scalability --force` |
| 难负样本消融（R09） | `python run_all.py --config conf/hardneg.yaml --stages ltr --force` |
| σ_seed 稳健性 | `python run_all.py --config conf/stream.yaml --stages seed --force` |
| **补强消融 9 臂（R15–R22/R26）** | `python -u /mnt/d/Ubuntu/tools/run-mlrecsys-wsl-batch.py`（WSL 批跑，读 `tools/wsl-arms.json`；配置由 `gen-wsl-confs.py` 生成，见 §1.2） |
| 单元测试 | `python -m pytest tests -q` |

## 4. 目录结构

```
conf/                配置：final（R07）/ f1（R06）/ graph（R08）/ stream（R10）/ hardneg（R09）/ realneg（R11）/ itemcf_noes（R13）/ wsl*（WSL 运行版：仅换路径与落盘位置）/ dev（冒烟）
src/
  data/              校验、80/10/10 时间划分、1+100 候选集、描述统计
  recall/            ItemCF 精确余弦、SimHash LSH 索引
  features/          basic（10+2 特征）、graph（PageRank/Louvain/随机游走）、distributed（Spark 统计）
  stream/            时间戳回放、Structured Streaming 窗口聚合、时序安全特征
  rank/              ALS 训练与打分、训练组构造、LightGBM LambdaRank
  eval/              NDCG@10、配对 bootstrap、leaderboard、种子稳健性
  serving/           Top-10 输出与元数据
  stages.py          13 个流水线阶段
run_all.py            一键入口
requirements.txt      Python 依赖（pip install -r requirements.txt）
tests/                协议层单元测试
outputs/
  leaderboard.csv     指标总表
  metrics/            各 run 指标 JSON + 逐用户 NDCG + 特征重要性
  top10/              各 run 的 Top-10 提交文件
  reports/            数据质量/统计/LSH/σ_seed/Spark 扩展性报告
data/processed/       train/val/test、candidates、图工件、流式聚合（均由脚本生成）
tools/hadoop/         Windows Spark 所需 winutils（第三方二进制）
```

## 5. 关键配置

| 项 | 默认 | 说明 |
|---|---|---|
| `seed` | 20260907 | 全局种子，影响候选、ALS、LightGBM |
| `data.positive_threshold` | 4.0 | 正反馈定义 |
| `data.candidates.num_negatives` | 100 | 每用户固定负样本数 |
| `als.rank / max_iter` | 64 / 10 | Spark MLlib ALS |
| `features.*` | bayes_rating / weighted_genre / graph / stream | 特征开关（用于消融） |
| `ltr.run_id / params` | R07_ltr_content / 300 轮 | LightGBM LambdaRank |
| `ltr.positive_row` | `first` | 正样本在训练组内的行序。**默认值禁止改动**；`last` 仅用于 R20 早停缺陷根因反事实实验（LightGBM 内部指标按行序打破并列，正样本置首会在高并列场景虚高内部 NDCG） |
| `itemcf_sim.columns` | `[max_sim]` | ItemCF 召回特征口径：`max_sim`（默认，历史行为）/ `top3_mean`（Top-3 邻居均值）/ `has_neighbor`（是否存在相似度>0 的邻居，0/1）；可多选 |
| `eval.bootstrap` | B=1000 | 配对检验 |
| `scalability.parallelism` | [2,4,6,8] | 扩展性实验并行度 |

## 6. 结果速查

| run | 模型 | test NDCG@10 | lift |
|---|---|---|---|
| R00_pop | 热门基线 | 0.7540 | 1.000 |
| R03_als | Spark ALS 直接排序 | 0.2794 | 0.371 |
| R06_ltr_f1 | LightGBM（10 特征） | 0.8205 | +8.82% |
| R07_ltr_content | +贝叶斯/加权画像（12） | 0.8202 | +8.78% |
| R08_ltr_graph | +PageRank（13） | 0.8213 | +8.93% |
| R10_ltr_stream | +F3 比值（16 特征） | 0.8587 | +13.88% |
| R14_ltr_stream_noes | 同 R10 + 固定 300 轮（拆分实验） | 0.8588 | +13.89% |
| R09_hardneg | 流行难负（16 特征，负结果） | 0.8518 | +12.97% |
| R11_ltr_realneg | 真实负反馈难负（16 特征，负结果） | 0.8224 | +9.06% |
| R12_ltr_itemcf | +ItemCF 召回相似度（17 特征，**早停失效**） | 0.8296 | +10.02% |
| **R13_ltr_itemcf_noes** | **同上特征 + 固定 300 轮（当前最优、提交臂）** | **0.8728** | **+15.75%** |

**2026-10-12 补强臂（9 个，完整分析见 `docs/消融总表.md` §6）**：

| run | 内容 | test NDCG@10 | 关键结论 |
|---|---|---|---|
| R17_ltr_itemcf_nostream | R13 − F3 时序三件套（LOO） | 0.8497 | F3 真实边际 **−0.0231**（链式口径曾给 +0.0373，高估 1.6×） |
| R18_ltr_itemcf_nograph | R13 − PageRank（LOO） | 0.8726 | 有 ItemCF 后 PageRank 边际 **−0.0002**（几乎归零） |
| R19_ltr_itemcf_nocontent | R13 − 内容画像（LOO） | 0.8726 | −0.0002，与链式口径一致 |
| **R20_ltr_itemcf_poslast** | **正样本置末行 + 早停开启** | **0.8726** | **根因实证：best_iter 1→261，test 0.8296→0.8726**；R20 ≈ R13 |
| R15/R16 · R21/R22 · R26 | 轮数补格 · leaves 31/127 · ItemCF top_k 20 | 0.8204/0.8213 · 0.8726/0.8724 · 0.8711 | 轮数因子在低特征数下 ≤1.1e-4；叶子数不敏感（±4e-4）；邻居 20→50 值 +0.0017 |

**2026-10-13 E 组：ItemCF 召回特征的聚合口径（3 臂，详见 `docs/消融总表.md` §6.6）**

| run | 内容 | test NDCG@10 | 关键结论 |
|---|---|---|---|
| **R29_ltr_itemcf_top3mean** | **R13 + Top-3 邻居均值（18 特征）** | **0.8749** | **显著优于 R13（+0.00216，p=0.001）——当前最高** |
| R28_ltr_itemcf_binary | R13 + 二值命中（18） | 0.8726 | 不显著（−0.0002）：二值是 `max_sim` 的确定性函数 |
| R30_ltr_itemcf_binary_only | 二值命中替换 `max_sim`（17） | 0.8653 | −0.0075：相似度强度是有效信息，不能只保留"命中与否" |

> `latency_ms` / `throughput_rps` 在 `outputs/leaderboard.csv` 中为**管线代理值**（该 run 阶段总耗时 / 测试用户数）；服务层**推理延迟**与吞吐写在 `outputs/top10/<run>_meta.json`（`latency_scope: inference_only`），R13 为 0.2504 ms/用户、40.3 万行/秒。

详见 `docs/消融总表.md`、`docs/项目报告.md`。

## 7. 合规声明

- **不提交原始数据**，仅提交下载/校验/处理脚本；数据使用遵循 GroupLens 条款。
- `tools/hadoop/bin/winutils.exe` 来源 `github.com/cdarlint/winutils`（hadoop-3.3.6），SHA256 = `496A591EB1E67DF2A620F710D529BA6DDFE1C19149E6647CC4E320BB0EFD8553`。
- 引用：PySpark / LightGBM / SciPy / python-igraph / pandas / numpy / matplotlib；移植特征来源 Kaggle `hybrid-movie-recommendation-with-movielens-32m`（已在报告标注）。

## 8. 常见问题

| 现象 | 处理 |
|---|---|
| Spark 报 `HADOOP_HOME unset` | 确认 `tools/hadoop/bin/winutils.exe` 存在；`get_spark` 会自动配置（WSL/Linux 下不需要 winutils，代码按 `os.name` 跳过） |
| Python worker 连接超时 | 已固定 `SPARK_LOCAL_IP=127.0.0.1`、`spark.driver.host`；勿改本机名 |
| 内存不足 | 用 `data.dev.max_users` 降采样冒烟；最终指标需全量。WSL 侧把 `spark.driver_memory` 降到 6g 并把 `.wslconfig` 的 `memory` 提到 12GB |
| WSL 里后台任务跑一会儿就消失 | DSH 的 bash 工具在调用结束时会回收整棵进程树，`setsid nohup` 也救不回来；长任务用 DSH 托管后台任务 |
| 想跳过已完成阶段 | 默认命中缓存；需要重算加 `--force` |
