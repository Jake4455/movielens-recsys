# 任务四：流式增量处理演示

> 证据来源：`data/processed/stream/`（回放文件、checkpoint、窗口聚合表）与 `src/stream/{replay,features}.py`。
> 本文件由 `outputs_dev/fix_run/make_stream_demo.py` 生成，可随时重跑。

## 1. 回放源：按时间戳全局有序的 24 个事件文件

- 事件总数：**28,885,713**（= train 25,771,222 + val 3,114,491，不含任何测试期事件）
- 文件数：24（`events_000.parquet` … `events_023.parquet`）；`maxFilesPerTrigger = 2`
- 文件间时间单调不回退：**是**（每个文件的 min(ts) ≥ 上一个文件的 max(ts)）
- 回放与聚合耗时：69.6s

| # | 文件 | 行数 | min(timestamp) | max(timestamp) |
|---|---|---|---|---|
| 1 | `events_000.parquet…` | 1,203,572 | 1995-01-09 11:46:44 | 1996-11-09 12:35:40 |
| 2 | `events_001.parquet…` | 1,203,572 | 1996-11-09 12:36:02 | 1999-07-01 14:00:09 |
| 3 | `events_002.parquet…` | 1,203,572 | 1999-07-01 14:00:09 | 2000-02-06 02:21:59 |
| 4 | `events_003.parquet…` | 1,203,572 | 2000-02-06 02:22:16 | 2000-11-20 08:45:49 |
| 5 | `events_004.parquet…` | 1,203,572 | 2000-11-20 08:45:49 | 2001-09-28 04:12:10 |
| 6 | `events_005.parquet…` | 1,203,572 | 2001-09-28 04:12:10 | 2003-03-02 23:19:12 |
| … | （其余 18 个文件同上规律） | | | |

## 2. micro-batch 进度（Structured Streaming checkpoint）

- 已完成微批（`checkpoint/commits`）：**12** 个；对应 offset 记录 **12** 个
- 每批 2 个文件 → 24 / 2 = 12 个微批，与 checkpoint 记录一致
- 聚合表行数与事件数自洽（user_daily 1,551,816 行、movie_daily 13,521,257 行），checkpoint 中未见重复追加

## 3. 窗口聚合结果

| 表 | 行数 | 时间窗 |
|---|---|---|
| `user_daily`（用户×日） | 1,551,816 | window(1 day) |
| `movie_daily`（电影×日） | 13,521,257 | window(1 day) |

## 4. 时序安全逐例验算（测试期正样本抽样）

窗口定义为 `[t−w, t)`：以目标时刻 t 所在日**之前**的日粒度累积计数做 `searchsorted` 差分，
因此候选自身所在日的事件不会进入任何窗口，测试期事件也完全不在流数据中。

| 用户 | 电影 | t 所在日（epoch day） | movie 7d/30d 比值 | user 7d/30d 比值 | 该电影在 t 及之后的事件数 |
|---|---|---|---|---|---|
| 190719 | 25 | 9528 | 1.0000 | 1.0000 | 1307 |
| 152679 | 339 | 9594 | 0.5083 | 1.0000 | 1279 |
| 189630 | 1225 | 10315 | 0.6216 | 1.0000 | 1648 |

> 示例中的测试期正样本在流数据中**没有对应事件**（流数据只含 train+val），因此其 `movie_recent_ratio_7_30`
> 完全由该电影在 t 之前的历史事件决定，符合「任何时序特征只能使用推荐时刻之前的数据」的协议要求。

## 5. 复现命令

```bash
python run_all.py --config conf/stream.yaml --stages stream --force   # 重放 + 窗口聚合
python outputs_dev/fix_run/make_stream_demo.py                        # 重新生成本演示
```
