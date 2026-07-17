"""Build the evidence-focused competition attachment for TrialScopeAI."""

from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    Flowable,
    HRFlowable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from build_submission_pdf import (
    BLUE,
    BLUE_DARK,
    BLUE_SOFT,
    CANVAS,
    GREEN,
    INK,
    LINE,
    MUTED,
    ROOT,
    WHITE,
    bullet_list,
    callout,
    decorate_page,
    grid_table,
    make_styles,
    p,
    page_title,
    register_fonts,
    screenshot_block,
)


OUTPUT = ROOT / "output" / "pdf" / "TrialScopeAI_项目补充材料.pdf"


class HorizontalBars(Flowable):
    """Small report-style horizontal bar chart with fixed labels and values."""

    def __init__(self, rows: list[tuple[str, int]], width: float, height: float, color=BLUE):
        super().__init__()
        self.rows = rows
        self.width = width
        self.height = height
        self.color = color

    def draw(self) -> None:
        canvas = self.canv
        max_value = max(value for _, value in self.rows) or 1
        label_width = 53 * mm
        value_width = 12 * mm
        chart_width = self.width - label_width - value_width - 4 * mm
        row_height = self.height / len(self.rows)
        bar_height = min(7 * mm, row_height * .48)
        canvas.setFont("MSYH", 7.4)
        for index, (label, value) in enumerate(self.rows):
            center_y = self.height - row_height * (index + .5)
            canvas.setFillColor(MUTED)
            canvas.drawRightString(label_width - 3 * mm, center_y - 2.2, label)
            canvas.setFillColor(colors.HexColor("#E8EDF1"))
            canvas.rect(label_width, center_y - bar_height / 2, chart_width, bar_height, fill=1, stroke=0)
            canvas.setFillColor(self.color)
            canvas.rect(
                label_width,
                center_y - bar_height / 2,
                chart_width * value / max_value,
                bar_height,
                fill=1,
                stroke=0,
            )
            canvas.setFillColor(INK)
            canvas.drawRightString(self.width, center_y - 2.2, str(value))


def report_box(title: str, body: str, styles: dict[str, ParagraphStyle]) -> Table:
    title_style = ParagraphStyle(
        "ReportBoxTitle",
        parent=styles["h3"],
        fontSize=10,
        leading=14,
        textColor=BLUE_DARK,
        spaceAfter=3,
    )
    body_style = ParagraphStyle(
        "ReportBoxBody",
        parent=styles["small"],
        fontSize=7.8,
        leading=12.5,
        textColor=INK,
    )
    table = Table(
        [[[p(title, title_style), p(body, body_style)]]],
        colWidths=[55 * mm],
        rowHeights=[31 * mm],
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), CANVAS),
                ("BOX", (0, 0), (-1, -1), .7, LINE),
                ("LINEABOVE", (0, 0), (-1, 0), 2, BLUE),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    return table


def build_pdf() -> None:
    register_fonts()
    styles = make_styles()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        rightMargin=20 * mm,
        leftMargin=20 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title="TrialScopeAI 项目补充材料",
        author="朱春兰、李朝元",
        subject="健康元企业命题项目证据与验证记录",
    )
    story: list = []

    # Cover
    story.append(Spacer(1, 18 * mm))
    cover = Table(
        [
            [p("2026 AI先锋未来人才大赛", styles["kicker"])],
            [p("TrialScopeAI", styles["h1"])],
            [p("项目补充材料", styles["h1"])],
            [p("临床试验入排标准结构化与招募可行性评估", styles["body"])],
        ],
        colWidths=[174 * mm],
        rowHeights=[13 * mm, 28 * mm, 24 * mm, 20 * mm],
    )
    cover.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), CANVAS),
                ("LINEBEFORE", (0, 0), (0, -1), 5, BLUE),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 14 * mm),
                ("RIGHTPADDING", (0, 0), (-1, -1), 14 * mm),
                ("TOPPADDING", (0, 0), (-1, -1), 2 * mm),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2 * mm),
            ]
        )
    )
    story.append(cover)
    story.append(Spacer(1, 15 * mm))
    story.append(
        p(
            "本材料用于说明项目问题界定、原型实现、数据资产、验证记录和落地条件。材料中的人数与比例均来自公开方案和合成候选者，不代表真实临床试验结果。",
            styles["body"],
        )
    )
    story.append(Spacer(1, 10 * mm))
    cover_meta = [
        ["所选企业", "健康元药业"],
        ["参赛项目", "TrialScopeAI"],
        ["项目成员", "朱春兰（复旦大学） / 李朝元（成都理工大学）"],
        ["在线演示", "https://trialscopeai.streamlit.app/"],
        ["代码与材料", "https://github.com/liziyaaa/TrialScopeAI"],
        ["材料版本", "1.0 / 2026年7月"],
    ]
    story.append(grid_table(cover_meta, [38 * mm, 136 * mm], header=False))
    story.append(Spacer(1, 12 * mm))
    story.append(
        callout(
            "阅读建议：第2页为项目摘要；第5页列出数据与演示案例；第7页给出合成队列结果；第8页记录测试和限制。",
            styles,
        )
    )
    story.append(PageBreak())

    # Summary
    story += page_title("01", "项目摘要", "用一页说明我们处理什么问题、服务谁、已经完成到什么程度。", styles)
    summary_rows = [
        ["项目问题", "试验方案中的自然语言入排标准难以快速转成可审核、可执行且可追溯的筛选逻辑。"],
        ["主要使用者", "药企临床开发、医学、统计和运营团队。"],
        ["当前方案", "方案导入、标准结构化、人工审核、合成队列预筛、招募可行性评估和条件情景比较。"],
        ["主要输出", "四类候选者状态、逐条证据链、招募漏斗、主要筛减项、数据缺口和代表性变化。"],
        ["当前数据", "GOLDEN-4 公开方案、27条审核规则、500名合成候选者、50个边界病例。"],
        ["使用边界", "不诊断、不自动入组、不使用真实患者数据，不直接建议修改试验方案。"],
    ]
    story.append(grid_table(summary_rows, [36 * mm, 138 * mm], header=False))
    story.append(Spacer(1, 9 * mm))
    story.append(p("已经完成的证据", styles["h2"]))
    boxes = Table(
        [[
            report_box("可运行原型", "Streamlit 在线部署；无 API Key 仍可完成主演示。", styles),
            report_box("结构化标准", "27条 GOLDEN-4 标准均保留原文与来源。", styles),
            report_box("可复现数据", "500名固定种子合成候选者，不含个人信息。", styles),
        ], [
            report_box("规则验证", "50个边界病例覆盖阈值、缺失、时间窗和主观标准。", styles),
            report_box("自动化测试", "40项测试通过，覆盖输入、规则、缓存与完整演示路径。", styles),
            report_box("评审材料", "在线演示、GitHub 仓库、提交正文和本补充附件。", styles),
        ]],
        colWidths=[58 * mm, 58 * mm, 58 * mm],
        rowHeights=[34 * mm, 34 * mm],
    )
    boxes.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 1.5), ("RIGHTPADDING", (0, 0), (-1, -1), 1.5)]))
    story.append(boxes)
    story.append(Spacer(1, 9 * mm))
    story.append(
        callout(
            "我们的重点不是让模型替代医学判断，而是把方案解释、规则执行和最终审核分开，使每一步都能检查。",
            styles,
        )
    )
    story.append(PageBreak())

    # Problem framing
    story += page_title("02", "问题界定与数据可获得性", "项目从可验证的招募准备环节切入，不覆盖诊疗和真实患者入组。", styles)
    story.append(p("我们具体处理的四类困难", styles["h2"]))
    problem_rows = [
        ["方案解释", "标准包含阈值、单位、时间窗、组合逻辑和主观判断；不同中心可能出现执行差异。"],
        ["字段准备", "部分标准依赖检查、病史或近期事件；如果字段缺失，应先补充信息，而不是直接排除。"],
        ["招募瓶颈", "总人数不足不能解释问题，需要进一步定位候选池在哪一组条件上损失最多。"],
        ["人群构成", "年龄、性别、疾病程度和共病条件会共同影响潜在参与人群的代表性。"],
    ]
    story.append(grid_table(problem_rows, [36 * mm, 138 * mm], header=False))
    story.append(Spacer(1, 9 * mm))
    story.append(p("为什么当前阶段可以开展", styles["h2"]))
    access_rows = [
        ["阶段", "所需数据", "获得方式", "结论"],
        ["原型", "试验注册信息和方案标准", "ClinicalTrials.gov 公开来源", "已获得"],
        ["规则验证", "人工审核标准与边界病例", "团队依据方案构造", "已完成"],
        ["招募模拟", "候选者字段", "固定随机种子合成数据", "已完成"],
        ["企业试点", "去标识化历史筛选统计", "企业授权、伦理与数据治理", "尚未接入"],
    ]
    story.append(grid_table(access_rows, [27 * mm, 54 * mm, 58 * mm, 35 * mm]))
    story.append(Spacer(1, 9 * mm))
    story.append(p("预防医学在项目中的作用", styles["h2"]))
    story += bullet_list(
        [
            "检查入排标准是否可能引入选择偏倚，并明确哪些结论只能在合成数据范围内解释。",
            "关注缺失数据、测量条件、时间窗和人群代表性，而不只计算候选人数。",
            "把临床安全、数据治理和伦理限制写入产品流程，避免把原型结果包装成真实疗效或招募率。",
        ],
        styles,
    )
    story.append(PageBreak())

    # Method
    story += page_title("03", "处理流程与判定逻辑", "语义提取负责整理文本，确定性规则负责执行，人员负责最终审核。", styles)
    flow_rows = [
        ["步骤", "输入", "处理", "输出"],
        ["01 方案导入", "NCT / 文本 / PDF", "定位并展示标准原文", "待确认文本"],
        ["02 标准结构化", "已确认原文", "提取字段、条件、阈值、单位、时间窗", "待审核规则"],
        ["03 医学审核", "待审核规则", "逐条确认、修订并标记主观项", "已审核规则"],
        ["04 模拟预筛", "规则 + 合成候选者", "按固定优先级执行", "四类状态 + 证据链"],
        ["05 招募评估", "预筛结果", "汇总筛减、缺失和人群构成", "漏斗、瓶颈和情景比较"],
    ]
    story.append(grid_table(flow_rows, [30 * mm, 43 * mm, 65 * mm, 36 * mm]))
    story.append(Spacer(1, 9 * mm))
    story.append(p("候选者状态的固定优先级", styles["h2"]))
    priority_rows = [
        ["优先级", "条件", "输出"],
        ["1", "任一明确标准失败", "不符合"],
        ["2", "无失败，但缺少执行所需字段", "信息不足"],
        ["3", "无失败、无缺失，但存在主观标准", "人工复核"],
        ["4", "其余可执行标准均通过", "模拟符合"],
    ]
    story.append(grid_table(priority_rows, [24 * mm, 108 * mm, 42 * mm]))
    story.append(Spacer(1, 9 * mm))
    story.append(p("三项设计选择", styles["h2"]))
    choices = [
        ["选择", "原因"],
        ["模型不直接判定入组", "临床标准可能主观、缺失或依赖上下文，最终资格必须由研究人员确认。"],
        ["每条结果保留方案原文", "便于医学复核、定位误解和解释中心间差异。"],
        ["扫描 PDF 不做猜测", "无法获得可靠文本时明确提示，比生成不确定规则更安全。"],
    ]
    story.append(grid_table(choices, [52 * mm, 122 * mm]))
    story.append(PageBreak())

    # Data and case
    story += page_title("04", "数据资产与主演示案例", "当前原型只使用公开方案和明确标注的合成候选者。", styles)
    story.append(p("GOLDEN-4（NCT02347774）", styles["h2"]))
    story.append(
        p(
            "主演示案例来自 ClinicalTrials.gov 公共记录。该 COPD Ⅲ期试验同时包含年龄、吸烟史、肺功能、近期急性加重、感染、用药和合并症标准，可以覆盖首版规则引擎的主要类型。",
            styles["body"],
        )
    )
    story.append(Spacer(1, 5 * mm))
    assets = [
        ["资产", "数量", "用途", "来源"],
        ["方案标准原文", "1 份", "输入与原文追溯", "ClinicalTrials.gov"],
        ["人工审核规则", "27 条", "结构化金标准与预筛执行", "依据公开方案整理"],
        ["合成候选者", "500 名", "漏斗、筛减项和代表性演示", "固定随机种子生成"],
        ["边界病例", "50 个", "验证阈值、缺失、时间窗与优先级", "团队独立构造"],
        ["真实患者记录", "0 条", "当前不处理个人医疗信息", "不采集"],
    ]
    story.append(grid_table(assets, [43 * mm, 25 * mm, 67 * mm, 39 * mm]))
    story.append(Spacer(1, 8 * mm))
    story.append(p("规则构成", styles["h2"]))
    criteria_rows = [
        ["类别", "数量", "示例"],
        ["入组标准", "10", "年龄、COPD 诊断、吸烟包年、FEV1、FEV1/FVC"],
        ["排除标准", "17", "近期急性加重、感染、氧疗、其他呼吸疾病、近期试验药物"],
        ["需要人工判断", "5", "知情同意、访视依从性、严重共病风险等"],
    ]
    story.append(grid_table(criteria_rows, [45 * mm, 25 * mm, 104 * mm]))
    story.append(Spacer(1, 8 * mm))
    story.append(p("合成候选者字段范围", styles["h2"]))
    story += bullet_list(
        [
            "人口学：年龄、性别；",
            "疾病与肺功能：COPD 诊断、FEV1 预计值百分比、FEV1 容量、FEV1/FVC；",
            "暴露与治疗：吸烟包年、氧疗时长、近期全身激素和试验药物；",
            "时间窗与排除因素：急性加重、呼吸道感染、恶性肿瘤、青光眼、药物超敏等。",
        ],
        styles,
    )
    story.append(callout("合成数据用于验证系统行为，不代表真实 COPD 人群分布，也不能据此推断实际招募率。", styles, tone="warning"))
    story.append(PageBreak())

    # Product evidence
    story += page_title("05", "原型界面与操作路径", "操作入口按任务组织，评审时可以沿 01 至 04 顺序直接体验。", styles)
    story.append(
        screenshot_block(
            ROOT / "docs" / "images" / "trialscope-overview.png",
            "项目概览：显示当前研究、27条标准、500名合成候选者以及四步任务入口。",
            styles,
            174 * mm,
            118 * mm,
        )
    )
    story.append(Spacer(1, 7 * mm))
    interface_rows = [
        ["页面", "主要操作", "检查点"],
        ["01 方案导入", "选择 NCT、文本、PDF 或内置案例", "确认原文后才进入结构化"],
        ["02 标准审核", "修改字段、阈值、单位、时间窗和执行方式", "主观项保留人工确认"],
        ["03 模拟预筛", "运行规则并点击候选者结果", "查看逐条患者值、标准值和原文"],
        ["04 招募评估", "查看漏斗、筛减项、缺失和情景比较", "所有结果标明合成模拟"],
    ]
    story.append(grid_table(interface_rows, [36 * mm, 72 * mm, 66 * mm]))
    story.append(PageBreak())

    # Screening and analysis
    story += page_title("06", "预筛结果与证据查看", "本页展示当前合成案例的实际运行结果，不代表真实试验招募表现。", styles)
    left = screenshot_block(
        ROOT / "docs" / "images" / "trialscope-import.png",
        "方案导入：四种来源分开展示，避免把上传、粘贴和公开试验查询混在同一表单。",
        styles,
        84 * mm,
        69 * mm,
    )
    right = screenshot_block(
        ROOT / "docs" / "images" / "trialscope-screening.png",
        "模拟预筛：四类状态、筛选控件和候选者结果表位于同一任务页面。",
        styles,
        84 * mm,
        69 * mm,
    )
    pair = Table([[left, right]], colWidths=[87 * mm, 87 * mm])
    pair.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 3)]))
    story.append(pair)
    story.append(Spacer(1, 8 * mm))
    story.append(p("500 名合成候选者的四类结果", styles["h2"]))
    story.append(
        HorizontalBars(
            [("模拟符合", 127), ("不符合", 339), ("信息不足", 14), ("人工复核", 20)],
            width=174 * mm,
            height=47 * mm,
            color=BLUE,
        )
    )
    story.append(Spacer(1, 6 * mm))
    story.append(
        callout(
            "解释示例：候选者可能先因肺功能阈值明确失败，也可能在无失败时因为近期事件字段缺失而进入“信息不足”。系统不会把缺失和主观判断混同为“不符合”。",
            styles,
        )
    )
    story.append(PageBreak())

    # Findings
    story += page_title("07", "合成案例中的招募瓶颈", "结果用于验证分析方法和界面，不用于评价 GOLDEN-4 的真实招募效率。", styles)
    story.append(
        screenshot_block(
            ROOT / "docs" / "images" / "trialscope-analysis.png",
            "招募评估页：候选者从 500 名逐层筛减；右侧展示主要未通过标准。",
            styles,
            174 * mm,
            116 * mm,
        )
    )
    story.append(Spacer(1, 7 * mm))
    story.append(p("主要未通过标准（合成队列）", styles["h2"]))
    story.append(
        HorizontalBars(
            [
                ("I06  FEV1/FVC", 88),
                ("I04  FEV1预计值百分比", 48),
                ("I03  吸烟包年", 48),
                ("E06  全身激素时间窗", 47),
                ("I02  COPD诊断", 40),
            ],
            width=174 * mm,
            height=43 * mm,
            color=colors.HexColor("#B94A48"),
        )
    )
    story.append(Spacer(1, 6 * mm))
    story.append(
        p(
            "在当前合成分布下，FEV1/FVC 是影响人数最多的单项标准，共涉及 88 名候选者。该结论只说明系统能够定位主要筛减条件；是否具有临床意义，需要结合真实目标人群、方案背景和医学审核判断。",
            styles["body"],
        )
    )
    story.append(PageBreak())

    # Validation
    story += page_title("08", "验证记录、已知限制与评价计划", "我们把“代码能运行”“规则能执行”和“企业效果成立”分开记录。", styles)
    story.append(p("当前验证记录", styles["h2"]))
    validation_rows = [
        ["检查对象", "覆盖范围", "当前结果"],
        ["规则运算符", "数值比较、区间、布尔、枚举、缺失、时间窗", "自动化测试通过"],
        ["边界病例", "阈值相等、单位、缺失、主观项、多重失败", "50个病例通过"],
        ["PDF 输入", "文字型、空白、无标准章节、扫描件、超限", "异常路径有明确提示"],
        ["模型接口", "成功、空响应、无效 JSON、超时、无 Key、限额和缓存", "使用模拟响应验证"],
        ["完整演示", "无网络、无 API Key 的 Streamlit 路径", "缓存案例可运行"],
        ["自动化测试总数", "规则、输入、解析、分析和界面", "40项通过"],
    ]
    story.append(grid_table(validation_rows, [40 * mm, 86 * mm, 48 * mm]))
    story.append(Spacer(1, 8 * mm))
    story.append(p("当前结果与后续目标", styles["h2"]))
    target_rows = [
        ["项目", "当前状态", "下一步"],
        ["GOLDEN-4 规则执行", "已完成离线验证", "由医学人员再次独立复核"],
        ["结构化提取 F1", "尚未形成正式实测结论", "以人工金标准评测，目标 ≥ 0.85"],
        ["患者匹配准确率", "边界病例用于代码验证", "在合规数据上验证，目标 ≥ 90%"],
        ["企业效率提升", "尚未测量", "记录审核用时、修订率和中心间一致性"],
    ]
    story.append(grid_table(target_rows, [48 * mm, 55 * mm, 71 * mm]))
    story.append(Spacer(1, 8 * mm))
    story.append(p("已知限制", styles["h2"]))
    story += bullet_list(
        [
            "当前只有一个 COPD 主案例，不能代表其他适应症的标准结构。",
            "合成队列不代表真实疾病人群，当前比例不能作为流行病学估计。",
            "扫描型 PDF 暂不支持 OCR；复杂表格和跨页逻辑仍需人工确认。",
            "模型结构化目标需要正式金标准评测后才能对外报告。",
        ],
        styles,
    )
    story.append(PageBreak())

    # Feasibility and references
    story += page_title("09", "落地条件、团队分工与参考资料", "下一阶段需要企业医学、统计、运营和数据治理人员共同参与。", styles)
    story.append(p("建议的企业验证路径", styles["h2"]))
    roadmap = [
        ["阶段", "主要工作", "所需条件"],
        ["1. 方案复核", "选择1至2个代表性方案，审核字段字典和规则", "医学与统计人员参与"],
        ["2. 历史回放", "使用去标识化筛选统计比较人工结果与系统结果", "授权、伦理与数据治理"],
        ["3. 过程测量", "记录审核用时、规则修订率、缺失字段和中心一致性", "明确评价指标"],
        ["4. 扩展验证", "增加适应症、中心和方案版本", "完成前述验证后再扩展"],
    ]
    story.append(grid_table(roadmap, [32 * mm, 88 * mm, 54 * mm]))
    story.append(Spacer(1, 8 * mm))
    story.append(p("团队分工", styles["h2"]))
    team_rows = [
        ["成员", "学校与专业", "项目分工"],
        [
            p("朱春兰", styles["body"]),
            p("复旦大学 · 预防医学<br/>本科在读（2023-2028）", styles["small"]),
            p("医学标准、流行病学逻辑、数据质量、人群代表性和研究边界", styles["small"]),
        ],
        [
            p("李朝元", styles["body"]),
            p("成都理工大学 · 测控技术与仪器<br/>本科在读（2023-2027）", styles["small"]),
            p("数据导入、结构化解析、规则引擎、合成数据、可视化、测试和部署", styles["small"]),
        ],
    ]
    story.append(grid_table(team_rows, [27 * mm, 67 * mm, 80 * mm]))
    story.append(Spacer(1, 3 * mm))
    story.append(p("联系方式已在报名系统中填写，本附件不重复列出。", styles["small"]))
    story.append(Spacer(1, 8 * mm))
    story.append(p("资料与入口", styles["h2"]))
    links = [
        ["在线演示", p('<link href="https://trialscopeai.streamlit.app/" color="#0B5CAB">trialscopeai.streamlit.app</link>', styles["link"])],
        ["GitHub 仓库", p('<link href="https://github.com/liziyaaa/TrialScopeAI" color="#0B5CAB">github.com/liziyaaa/TrialScopeAI</link>', styles["link"])],
        ["GOLDEN-4", p('<link href="https://clinicaltrials.gov/study/NCT02347774" color="#0B5CAB">clinicaltrials.gov / NCT02347774</link>', styles["link"])],
        ["FDA 代表性指南", p('<link href="https://www.fda.gov/regulatory-information/search-fda-guidance-documents/enhancing-diversity-clinical-trial-populations-eligibility-criteria-enrollment-practices-and-trial" color="#0B5CAB">FDA - Enhancing Diversity in Clinical Trial Populations</link>', styles["link"])],
        ["自动化筛选研究", p('<link href="https://pmc.ncbi.nlm.nih.gov/articles/PMC11141802/" color="#0B5CAB">PMC11141802</link>', styles["link"])],
    ]
    story.append(grid_table(links, [42 * mm, 132 * mm], header=False, font_size=7.5))
    story.append(Spacer(1, 8 * mm))
    story.append(
        callout(
            "材料结论：TrialScopeAI 已完成可运行原型和离线验证基础，适合进入由企业专业人员参与的小范围方案复核；尚不具备直接用于真实患者筛选的条件。",
            styles,
        )
    )

    doc.build(story, onFirstPage=decorate_page, onLaterPages=decorate_page)
    print(f"Created: {OUTPUT}")


if __name__ == "__main__":
    build_pdf()
