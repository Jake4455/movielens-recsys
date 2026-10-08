# 资源使用实测报告

> 对应任务书 §二「系统还应报告固定运行环境下的数据处理吞吐量、推荐延迟和资源使用情况」。
> 采样时间：2026-10-08 19:21:47；采样方式：PowerShell 每 2 秒记录 `java`+`python` 进程 WorkingSet 之和与 CPU 累计秒数。
> 生成脚本：`outputs_dev/fix_run/make_resource_report.py`（负载全部写入 `outputs_dev/`，不触碰交付产物）。

## 1. 运行环境

| 项 | 值 |
|---|---|
| CPU | Intel i5-12500H（4P+8E / 12 核 16 线程） |
| 内存 | 16 GB |
| GPU | NVIDIA RTX 3050 Ti Laptop 4 GB（本项目不使用） |
| JDK | 21 LTS；Python 3.11.16（conda `mlrecsys`） |
| Spark | `local[8]`，driver 8g，shuffle partitions 64 |

## 2. 负载 A：Spark ALS（600 万评分样本，rank=64，**iter=10（生产配置）**）

| 指标 | 实测 |
|---|---|
| 训练样本行数 | 5,999,264 |
| fit 耗时 | 61.3 s |
| fit 吞吐 | 97,826 ratings/s |
| 打分对数 | 18,915,078 |
| 打分耗时 / 吞吐 | 2.9 s / 6,487,795 pairs/s |
| 该阶段墙钟（含 Spark 启停） | 73.6 s |

> 注：本探针使用生产配置 `als.max_iter=10`；`outputs/reports/scalability.md` 的并行度对比固定 `iter=5`，
> 因此那里 local[8] 的 31.8 s 不能与本表的 61.3 s 直接比较（迭代次数不同）。

## 3. 负载 B：LightGBM LambdaRank（**10 万用户 × 101 行 = 1010 万行真实特征，生产规模**）

| 指标 | 实测 |
|---|---|
| 训练行数 × 特征数 | 10,100,000 × 16 |
| 训练耗时 | 68.2 s |
| 训练吞吐 | 148,060 行/s |
| best_iteration | 283 |
| 特征矩阵原始大小（float32） | 616 MB |

## 4. 进程内存与 CPU（采样峰值）

| 指标 | 实测 |
|---|---|
| `java`+`python` 进程 WorkingSet 之和峰值 | **9.27 GB** |
| 采样窗口内累计 CPU 时间 | 561 s（窗口 250 s，含 16 线程并行） |
| 采样点数（每 2s 一个） | 126 |

## 5. 磁盘占用（实测）

| 目录 | 大小 | 说明 |
|---|---|---|
| `data/ml-32m` | 911.4 MB | 原始数据（不提交） |
| `data/processed` | 826.5 MB | 清洗层 + 划分 + 候选集 + 图/流/Spark 工件 |
| `outputs` | 844.7 MB | 模型、ALS 分数、Top-10、指标、报告 |
| `outputs_dev` | 17.6 MB | 冒烟运行、修复前证据、本轮重跑日志 |
| `tools` | 0.2 MB | winutils.exe / hadoop.dll |
| `.git` | 0.7 MB | 版本库 |

## 6. 结论

1. **内存**：峰值约 **9.27 GB**，16 GB 物理内存在「Spark 与 LightGBM 不同时运行」的前提下有充足余量（NFR-2 满足）。
2. **磁盘**：全流程产物约 2.5 GB（含约 0.9 GB 原始数据），可整体清理后由脚本重建。
3. **吞吐**：ALS 与 LightGBM 的实测吞吐见 §2/§3；数据处理吞吐（候选集 21s / 窗口聚合 70s / Spark 统计 33s）见 `docs/阶段性总结.md` §6.9。
4. **延迟**：服务层推理延迟见 `outputs/top10/*_meta.json`（R10 = 0.121 ms/用户）。
5. **瓶颈**：全流程的墙钟瓶颈是 Spark ALS fit（352s，全量 2577 万评分）与 LightGBM 训练（约 200–330s/臂）；两者都受 4P+8E 混合核与本地模式 driver 开销限制。

