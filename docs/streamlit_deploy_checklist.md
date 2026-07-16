# Streamlit 部署清单

## 部署前

- [ ] 在 DeepSeek 控制台撤销曾粘贴到对话中的旧 Key；
- [ ] 生成新的 Key；
- [ ] 确认 GitHub `main` 分支包含 `app.py` 和 `requirements.txt`；
- [ ] 确认仓库中不存在 `.streamlit/secrets.toml`、`.env` 或真实密钥；
- [ ] GitHub 测试或本地 `python -m pytest` 全部通过。

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
```

## 部署后

- [ ] 首页正常打开；
- [ ] “下载演示 PDF”正常；
- [ ] 上传该 PDF 后能定位入排标准；
- [ ] 缓存标准可载入；
- [ ] 500 人预筛和所有图表正常；
- [ ] 用一小段文本测试一次 DeepSeek 实时解析；
- [ ] 检查 Streamlit 日志没有打印 Key 或上传文本；
- [ ] 将应用 URL 写入报名材料；
- [ ] 如需控制费用，将 `ENABLE_LIVE_LLM` 改为 `false`。
