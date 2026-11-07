# 4号交付：Error Analysis & Uncertainty

本包已完成你提供的 **baseline** 的误差与不确定性分析。分析期为 2025 年 12 月至 2026 年 5 月，两条赛道各 10,069 套房屋；两条赛道使用同一批验证房屋，共 20,138 条预测记录，不能当作 20,138 套独立房屋。

这些不是队友后续改进模型或隐藏测试集的结果。最终模型更新后，应使用新模型的逐月验证预测重新运行代码，并更新报告数字。

## 先看什么

1. `report/error_analysis_brief.pdf`：3 页英文支持材料，适合发给队友讨论；不是完整参赛报告，不能直接替代有 4 页上限的团队报告。
2. `report/分析说明_中文.md`：结论、解释、区间方法和限制。
3. `report/report_section_en.md`：可整合进团队报告的英文段落。
4. `report/video_script_en.md`：约 1 分钟的英文讲稿。
5. `figures/`：3 张图，均提供 PNG 和可编辑的 SVG。
6. `tables/`：整体、房型、月份、价格、社区、行政区指标，以及逐行误差、极端案例和区间回测。

## 关键发现

| 指标 | Off-market | List-aware |
|---|---:|---:|
| MdAPE | 20.84% | 7.26% |
| MAPE | 27.43% | 8.09% |
| 有符号误差中位数 | +8.07% | +5.68% |
| 绝对百分比误差第 90 百分位 | 57.99% | 15.89% |
| 预测高于成交价的比例 | 60.43% | 76.03% |
| 名义 90% 区间实际覆盖率* | 90.11% | 89.36% |

*区间仅评估每条赛道满足历史校准条件的 8,045 套；点预测指标使用全部 10,069 套。Off-market 区间宽度中位数为点预测的 114.23%，Detached 覆盖率仅 80.28%，所以整体覆盖率不能说明所有房型都可靠。

## 运行

推荐单独建立 Python 3.12 虚拟环境。在本目录执行：

```bash
python -m pip install -r requirements.txt
python analyze_errors.py
python verify_analysis.py
```

所有核心输入已随包提供，因此默认分析无需联网。`verify_analysis.py` 针对随包 baseline，检查原报告分数、分组总数、区间公式与时间顺序，并验证修改 5 月真实价格不会影响任何区间边界。`tables/verification.json` 保存已通过的检查。

用队友的新模型结果重跑，写入新目录：

```bash
python analyze_errors.py --predictions path/to/new_validation_predictions.csv --metadata inputs/validation_metadata.csv --out new_model_analysis
```

预测 CSV 必须含 `ml_num,track,month,house_type_name,sold_price,predicted_price`；两条赛道必须分别命名为 `off_market`、`list_aware`，每个房屋 ID 在每条赛道只能出现一次。代码依赖官方按成交月份验证的结构。若验证房屋发生变化，使用完整官方 `listings.parquet` 或包含这些房屋的新版 metadata。

**重跑会更新 CSV、JSON 和图表；本包的中英文文字、讲稿及 PDF 是针对 baseline 审阅的静态内容，不会自动更新。** 新模型的训练时间合规性仍由 1 号和 3 号检查；预测 CSV 本身不能证明模型训练没有泄漏。合并 metadata 中的成交价、日期和 list_price 仅用于分析，不是新增训练特征。

## 可选复核

完整官方数据未打包。若要重建极端错误案例的 10 套可比成交，先从官方数据集取得 `listings.parquet`，执行：

```bash
python investigate_baseline_comparables.py --listings path/to/listings.parquet
python build_supporting_report.py
```

第二条命令重建 baseline 支持 PDF，需 reportlab。辅助代码已经抽查 24 套随机房屋及 3 个最严重案例，27 条 Off-market 预测均与官方 10 邻居算法一致；List-aware 全部 10,069 条均等于挂牌价。这不是完整重新训练或全量 Off-market 重跑。

## 和队友衔接

- 给 1/2 号：Detached 和极端价格案例值得优先改进；检验面积、时间与社区匹配，以及 list-aware 挂牌价偏差的校准。
- 给 3 号：本包不改变点预测，也没有挑选“最优模型”。收到最终模型逐月预测后，用同样分析规则重新比较；调过参数的验证期不能再宣称是全新的独立测试。
- 4 号写作：优先保留房型差异、低价高估/高价低估、区间总体与分组覆盖不一致这三条故事线。社区排名只作探索性证据，不直接解释为社区造成误差。
- 最终合稿：用 `report_section_en.md` 作为素材；全队报告需使用官方模板，并满足手册中的篇幅、视频与链接要求。

## 来源

- 用户提供的 `baseline_results.zip`；原始预测与 JSON 报告分别保存在 `inputs/`。
- [官方数据集](https://huggingface.co/datasets/Baliyang/SSU_data_dash)：只读取 `listings.parquet`，仅打包验证房屋的 metadata；未使用隐藏测试答案。
- [官方 Starter Code](https://github.com/erwinzhang7/SSU-Data-Dash)：用于理解基准算法、验证月份与指标。
- 用户提供的案例 PDF 与两张分工图，用于理解交付背景。
- [Conformal prediction 方法参考](https://arxiv.org/abs/2107.07511)：这里只借用有限样本分位数构造；时序漂移和逐月模型变化下不声称严格的 90% 覆盖保证。
- Launch Video 已确认是 Google Drive 视频，但读取失败，未观看、转录，也未以视频内容支撑结论。

分析于 2026-11-07 完成。输入哈希和计算设定见 `tables/audit.json` 与 `inputs/source_manifest.json`。
