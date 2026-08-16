# 候选队列测试数据

以下 CSV 仅用于验证 TrialScopeAI 的导入、规则执行、缺失提示和边界判断，不包含真实患者信息，也不会被系统自动载入。每个文件均包含 500 条虚构候选记录，并通过固定随机种子生成，可重复构建。

| 文件 | 用途 |
|---|---|
| `cohort_01_complete_mixed.csv` | 字段完整，包含不同年龄、性别和风险状态，用于查看混合判断结果 |
| `cohort_02_low_risk.csv` | 以低风险、完整记录为主，用于验证较顺畅的评估路径 |
| `cohort_03_missing_fields.csv` | 故意只保留少量字段，用于验证“信息不足”和缺失字段统计 |
| `cohort_04_boundary_values.csv` | 包含年龄、时间窗、QTc、肺功能等边界值，用于检查阈值相等及临界情况 |

所有 `patient_id` 均为虚构测试编号。上传前仍应以当前研究在“队列评估”页面生成的字段模板为准。

如需重新生成：

```powershell
python scripts/generate_test_cohorts.py
```
