# Streamlit 部署与复核清单

当前公开地址：https://trialscopeai.streamlit.app/

## 部署前

- [ ] 在 DeepSeek 控制台撤销曾粘贴到对话中的旧 Key；
- [ ] 生成新的 Key；
- [x] GitHub `main` 分支包含 `app.py` 和依赖文件；
- [x] 仓库中不存在 `.streamlit/secrets.toml`、`.env` 或真实密钥；
- [x] 本地 65 项自动化测试全部通过。

## Community Cloud

1. 打开 https://share.streamlit.io/ 并使用 GitHub 登录；
2. 点击 `Create app`；
3. Repository：`liziyaaa/TrialScopeAI`；
4. Branch：`main`；
5. Main file path：`app.py`；
6. Python version：`3.12`；
7. 在 Advanced settings → Secrets 填写：

```toml
DEEPSEEK_API_KEY = "新生成的Key"
DEEPSEEK_MODEL = "deepseek-v4-flash"
ENABLE_LIVE_LLM = true

ENABLE_FEISHU_SYNC = true
FEISHU_APP_ID = "your-feishu-app-id"
FEISHU_APP_SECRET = "your-feishu-app-secret"
FEISHU_BITABLE_APP_TOKEN = "your-base-token"
FEISHU_CRITERIA_TABLE_ID = "your-criteria-table-id"
FEISHU_SNAPSHOT_TABLE_ID = "your-snapshot-table-id"
FEISHU_VALIDATION_TABLE_ID = "your-validation-table-id"
FEISHU_BITABLE_URL = "https://your-tenant.feishu.cn/base/your-base-token"
```

## 部署后

- [x] 中英文首页正常打开，语言切换后排版无溢出；
- [x] 通过 NCT02347774 获取 ClinicalTrials.gov 公开记录；
- [x] DeepSeek 实时解析成功，生成 25 条待审核候选规则；
- [x] 方案标准首次同步至飞书，新增 25 条审核记录；
- [x] 飞书“待审核”视图按试验编号显示完整记录；
- [x] 导入 500 条固定种子合成队列并完成规则仿真；
- [x] 研究历史、审核回读、边际影响和情景分析页面可打开；
- [x] Streamlit 日志未输出 Key、PDF 正文或候选者明细；
- [x] 应用地址已写入 README 和参赛报告；
- [ ] 提交前再次确认 Streamlit Secrets 中使用的是有效的新 Key；
- [ ] 如需临时控制费用，将 `ENABLE_LIVE_LLM` 改为 `false`。
