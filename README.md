# TrialScopeAI

> **2026 AI先锋未来人才大赛 · 健康元药业企业命题**<br>
> 临床试验方案约束仿真与招募可行性协同决策平台

我们希望回答一个具体问题：**一份复杂的临床试验方案进入多中心执行前，能否先看清哪些入排标准最影响招募、哪些患者信息容易缺失，以及潜在参与人群是否发生偏移？**

TrialScopeAI 将方案中的自然语言入排标准转为可审核规则，在合成候选人群中进行方案约束仿真，比较每项条件对候选规模、数据完整性与人群构成的影响。它不是把“符合/不符合”做得更快，而是帮助医学、统计和运营人员在方案执行前看清约束与权衡。

**[在线系统](https://trialscopeai.streamlit.app/)** ｜ **[项目补充材料 PDF](output/pdf/TrialScopeAI_项目补充材料.pdf)** ｜ **[预置研究 GOLDEN-4](https://clinicaltrials.gov/study/NCT02347774)**

> 在线系统默认载入经过审核的 GOLDEN-4 研究，**无需 API Key 即可完成标准审核、500 人候选人群仿真、单项标准边际影响与多目标情景权衡**。界面支持中文与英文，两种语言分别按阅读习惯组织信息，并非逐句替换。

![TrialScopeAI 项目工作台](docs/images/trialscope-overview.png)

## 为什么选择这个问题

临床试验招募并不只是“患者数量不足”。方案中的年龄、肺功能、合并症、用药史、检查阈值与时间窗通常以长篇自然语言呈现，人工逐条解释耗时，不同研究中心也可能产生执行差异。过严、难执行或依赖缺失字段的条件，还会缩小候选人群并影响研究人群的代表性。

因此，我们没有尝试覆盖临床试验全流程，而是聚焦健康元临床开发中一个**可量化、可获得数据、可由学生团队完成方法验证**的环节：

1. 把入排标准从“只能阅读的文本”转成“可以审核的规则”；
2. 在不使用真实患者数据的情况下模拟执行，提前定位主要筛减条件；
3. 从预防医学视角观察缺失数据、选择偏倚与人群代表性风险；
4. 为医学、统计和运营团队提供可追溯的讨论依据，而不是替他们作决定。

## 我们做了什么

```mermaid
flowchart LR
    A["NCT 编号 / 标准文本 / 文字型 PDF"] --> B["定位并确认入排标准原文"]
    B --> C["大模型提取字段、阈值与时间窗"]
    C --> D["同步至飞书多维表格"]
    D --> E["医学人员协同审核与留痕"]
    E --> K["读取差异并人工确认"]
    K --> L["确定性规则引擎"]
    F["500 名固定种子合成候选者"] --> L
    L --> G["候选人群约束仿真"]
    G --> H["逐候选者证据链"]
    G --> I["单项标准反事实影响"]
    G --> J["候选规模、代表性、数据负担"]
    I --> K["多目标情景权衡"]
    J --> K
```

预置研究采用公开 COPD Ⅲ期试验 [GOLDEN-4（NCT02347774）](https://clinicaltrials.gov/study/NCT02347774)。该方案同时包含年龄、吸烟史、肺功能、用药、合并症和时间窗，适合验证从方案导入到招募评估的完整闭环。

## 评委 2 分钟体验路径

打开 **[在线系统](https://trialscopeai.streamlit.app/)** 后，可按以下顺序使用：

1. **项目概览**：了解五步业务流程、当前研究、验证证据和数据边界；
2. **02 标准审核**：浏览 27 条结构化标准，核对原文、字段、阈值、单位、时间窗和执行方式；
3. **03 协作确认**：查看飞书协作流程、待审核数据和无配置时的审核模板；
4. **04 约束仿真**：点击“运行方案约束仿真”，查看固定合成人群中的计算结果与逐项证据；
5. **05 决策沙盘**：先看哪项标准造成最大的候选池边际变化，再比较候选规模、人群代表性和数据采集负担之间的情景权衡；
6. **验证证据与边界**：区分已完成的工程验证、待采集的业务证据和当前不能外推的结论。

若希望体验输入能力，可在 **01 方案导入** 中选择 NCT 编号、粘贴标准原文、上传文字型 PDF，或重新载入内置演示。

## 当前已经实现

| 能力 | 当前实现 | 评委可验证的结果 |
|---|---|---|
| 方案导入 | NCT 编号、粘贴文本、文字型 PDF、内置案例 | 原文确认后才进入解析，不对扫描件生成猜测结果 |
| 标准结构化 | 提取字段、运算符、阈值、单位、时间窗、适用条件与原文来源 | GOLDEN-4 提供 27 条人工审核标准 |
| 医学审核 | 可逐条修改结构化结果，主观标准标为人工确认 | 保存后才进入候选人群仿真 |
| 飞书协同 | 标准双向审核，并保存情景快照与真实验证记录 | 只同步结构化标准和汇总数据，不上传 PDF 正文或候选者明细 |
| 确定性仿真 | 输出“模拟符合 / 不符合 / 信息不足 / 人工复核” | 每个结论保留候选者值、标准值、原因和原文 |
| 单项标准边际影响 | 逐条暂不执行可计算标准并重跑固定队列 | 回答“移除这一约束后，候选规模与人群构成怎样变化” |
| 多目标情景权衡 | 同时比较候选规模、代表性差异和数据未决负担 | 呈现取舍，不自动建议放宽方案 |
| 结果导出 | 结构化标准 JSON/CSV、仿真结果 CSV、中英文评估摘要 Markdown | 结果可继续用于汇报与复核 |
| 离线模式 | 无网络或无 API Key 时使用审核后的缓存案例 | 核心评审路径不依赖外部服务稳定性 |

## 几个关键设计取舍

### 1. 模型理解，规则执行，人最终审核

我们没有让大模型直接判断患者能否入组。大模型只负责把自然语言转成结构化候选规则；Pydantic 负责结构校验；医学人员负责确认；最终仿真由确定性规则引擎执行。这让结果可以复核、复现和追责。

### 2. 每个结果都有证据链

系统不只输出“符合/不符合”，还展示：标准编号、方案原文、患者值、标准值、执行结果和判断原因。缺失字段输出“信息不足”，主观标准输出“人工复核”，不会被静默处理成排除。

### 3. 把预防医学能力放进临床开发场景

除了候选人数，我们还关注数据完整性、选择偏倚和人群代表性。通过比较候选队列与潜在参与人群的年龄、性别和疾病程度，帮助团队讨论某些标准可能带来的人群构成变化。

### 4. 首版不依赖难以获得的医院数据

当前使用 ClinicalTrials.gov 公开方案、人工审核规则与合成候选者，能够在合规、低成本条件下验证产品闭环。未来只有在企业授权、伦理审批和数据治理到位后，才考虑接入去标识化历史筛选统计。

## 当前验证证据

| 资产或检查 | 当前结果 | 说明 |
|---|---:|---|
| GOLDEN-4 人工审核规则 | 27 条 | 覆盖类型、字段、运算符、阈值、单位、时间窗和原文来源 |
| 合成 COPD 候选者 | 500 名 | 固定随机种子 `20260716`，结果可复现 |
| 独立边界病例 | 50 个 | 覆盖阈值相等、缺失值、单位、时间窗、主观标准和多重失败 |
| 自动化测试 | 55 项通过 | 覆盖规则、PDF、NCT、模型响应、飞书同步、审核差异、边际影响、情景权衡和 Streamlit 完整路径 |
| 真实患者记录 | 0 条 | 当前不采集、不处理个人医疗信息 |

结构化提取 F1 ≥ 0.85、患者匹配准确率 ≥ 90% 是下一阶段验收目标，**不是我们已经实现的企业效果**。真实效率提升还需要在健康元医学、统计与运营人员参与的试点中测量。

![TrialScopeAI 招募可行性评估](docs/images/trialscope-analysis.png)

## 这个系统能帮助讨论什么

TrialScopeAI 希望在方案讨论和招募准备阶段，为健康元团队提供三个更早出现的信号：

- **方案可执行性**：哪些标准最严格、最主观或最依赖难获得字段；
- **招募准备优先级**：哪些筛减环节值得优先准备数据、检查流程与中心培训；
- **人群代表性风险**：条件调整前后，潜在参与人群构成可能如何变化。

它不是自动修改方案的工具，而是把原本分散在方案文本、人工经验和筛选记录中的信息，整理成可量化、可追溯的讨论材料。

其中，“单项标准边际影响”是一次一项的反事实计算，而不是对方案的修改建议；“多目标情景权衡”把规模、代表性和数据负担放在同一个视图中，也不自动替企业选择更优方案。两者都需要医学、统计与伦理人员结合临床意义复核。

## 我们的分工

| 成员 | 学校与专业 | 在项目中的工作 |
|---|---|---|
| 朱春兰 | 复旦大学，预防医学，本科在读（2023-2028） | 标准医学含义、流行病学逻辑、数据质量、人群代表性和研究边界 |
| 李朝元 | 成都理工大学，测控技术与仪器，本科在读（2023-2027） | 数据导入、结构化解析、规则引擎、合成数据、可视化、测试和部署 |

我们的专业背景不同，但正好对应这个问题的两部分：先判断哪些医学和人群问题值得分析，再把分析过程做成能够复现和检查的产品。联系方式已经在报名系统中填写，不在公开仓库重复展示。

## 医疗与数据安全边界

- 不诊断、不自动入组，不替代研究者、统计人员或伦理委员会；
- 大模型只进行语义提取，不直接决定候选者资格；
- PDF 在当前会话内存中处理，不保存上传文件或正文；
- 首版不支持扫描 PDF OCR，无法提取有效文本时明确提示；
- 所有情景结果均为合成模拟，不构成临床试验方案修改建议；
- 任何真实数据接入都必须经过合法授权、去标识化、医学验证和伦理审批。

## 技术实现与复现

<details>
<summary><strong>本地运行</strong></summary>

要求 Python 3.12。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

默认即可运行完整 GOLDEN-4 缓存案例。

</details>

<details>
<summary><strong>可选：配置飞书协同审核</strong></summary>

在 Streamlit Secrets 中配置飞书自建应用和指定多维表格。应用密钥不得提交到仓库。

```toml
ENABLE_FEISHU_SYNC = true
FEISHU_APP_ID = "your-feishu-app-id"
FEISHU_APP_SECRET = "your-feishu-app-secret"
FEISHU_BITABLE_APP_TOKEN = "your-base-token"
FEISHU_CRITERIA_TABLE_ID = "your-criteria-table-id"
FEISHU_SNAPSHOT_TABLE_ID = "your-snapshot-table-id"
FEISHU_VALIDATION_TABLE_ID = "your-validation-table-id"
FEISHU_BITABLE_URL = "https://your-tenant.feishu.cn/base/your-base-token"
```

同步只发送结构化标准、情景汇总和主动填写的验证记录，不发送 PDF 正文或患者级数据。系统写入字段与审核字段分离；读取审核结果后，必须在应用中确认差异才会更新当前规则。

</details>

<details>
<summary><strong>可选：配置 DeepSeek 实时结构化</strong></summary>

复制 `.streamlit/secrets.example.toml` 为不进入版本控制的 `.streamlit/secrets.toml`：

```toml
DEEPSEEK_API_KEY = "your-new-key"
DEEPSEEK_MODEL = "deepseek-v4-flash"
ENABLE_LIVE_LLM = true
```

实时解析包含会话限额、进程小时限额和文本哈希缓存，可随时关闭并保留缓存演示。

</details>

<details>
<summary><strong>运行测试</strong></summary>

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest
```

GitHub Actions 会在 `main` 更新和 Pull Request 上使用 Python 3.12 运行同一套测试。

</details>

主要模块：

- `app.py`：单入口 Streamlit 工作台；
- `src/trial_sources.py`：NCT、文本和内存 PDF 导入；
- `src/llm_parser.py`：DeepSeek JSON 解析、校验、缓存与限额；
- `src/feishu.py`：飞书应用鉴权、标准审核回读、方案快照与验证记录同步；
- `src/rules.py`：确定性规则、结果优先级与证据链；
- `src/analytics.py`：漏斗、筛减原因、代表性、单项边际影响和多目标情景权衡；
- `src/synthetic.py`：可复现合成队列与边界病例；
- `data/`：公开试验摘要、金标准规则与合成数据；
- `tests/`：离线自动化验收。

## 公开来源

- [ClinicalTrials.gov Data API](https://clinicaltrials.gov/data-api/about-api)
- [GOLDEN-4 registry record](https://clinicaltrials.gov/study/NCT02347774)
- [FDA: Enhancing the Diversity of Clinical Trial Populations](https://www.fda.gov/regulatory-information/search-fda-guidance-documents/enhancing-diversity-clinical-trial-populations-eligibility-criteria-enrollment-practices-and-trial)
- [DeepSeek API](https://api-docs.deepseek.com/)
- [DeepSeek JSON Output](https://api-docs.deepseek.com/guides/json_mode/)
- [飞书多维表格 OpenAPI](https://open.feishu.cn/document/server-docs/docs/bitable-v1/bitable-overview)

## 免责声明

TrialScopeAI 当前版本面向临床试验方案评估与招募可行性协作，不是医疗器械。任何真实患者筛选、方案修改或运营决策，均需由有资质的专业人员在完成数据治理、医学验证和伦理审批后进行。
