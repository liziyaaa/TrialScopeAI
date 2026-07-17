"""Build the polished TrialScopeAI competition submission PDF."""

from __future__ import annotations

import re
from pathlib import Path

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    HRFlowable,
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "pdf" / "TrialScopeAI_大赛提交材料.pdf"

BLUE = colors.HexColor("#0B5CAB")
BLUE_DARK = colors.HexColor("#17324D")
BLUE_SOFT = colors.HexColor("#EAF3FB")
TEAL = colors.HexColor("#2F7F78")
GREEN = colors.HexColor("#20744A")
GREEN_SOFT = colors.HexColor("#E7F4EC")
INK = colors.HexColor("#1B2733")
MUTED = colors.HexColor("#5F6F7D")
LINE = colors.HexColor("#D7DEE5")
CANVAS = colors.HexColor("#F6F8FA")
WARNING = colors.HexColor("#8A5A12")
WARNING_SOFT = colors.HexColor("#FBF2DF")
WHITE = colors.white


PART_1 = (
    "临床试验招募并非单纯的“患者不足”问题。方案中的年龄、肺功能、合并症、用药史与时间窗常以长篇"
    "自然语言呈现，人工逐条解释耗时，不同中心还可能产生执行差异。FDA 2025年指南强调，应审视入排"
    "标准对代表性人群参与的影响；相关研究也表明，自然语言处理可降低标准结构化与初筛的人工负担。"
    "因此我们聚焦健康元临床开发中可量化、数据可获得的环节：将入排标准转为可审核规则，并在合成队列"
    "中提前识别招募瓶颈、数据缺口与代表性风险。"
)

PART_2 = (
    "TrialScopeAI 面向健康元临床开发、医学与运营团队，形成“方案导入—标准结构化—人工审核—模拟预筛"
    "—招募评估”的闭环。用户可输入 NCT 编号、粘贴标准或上传文字型 PDF；系统定位入组/排除章节，通过"
    "大模型提取字段、运算符、阈值、单位、时间窗与适用条件，并进行结构校验。医学人员逐条确认后，确定性"
    "规则引擎在合成候选者队列中输出“模拟符合、不符合、信息不足、人工复核”，每个结论均保留患者值、"
    "规则阈值、判定原因和方案原文。分析端展示筛减漏斗、主要排除项、缺失字段、年龄/性别/疾病程度代表性，"
    "并支持年龄、吸烟包年、FEV1 和时间窗等情景比较。创新点不是让模型直接决定入组，而是将语义提取、"
    "可解释规则与预防医学代表性评估分层：模型负责理解，规则负责执行，人负责最终审核。首版使用 GOLDEN-4 "
    "公共方案、27条人工审核规则、500名固定种子合成候选者和50个边界病例，不依赖真实患者数据即可复现"
    "演示；未来在合规授权下可接入去标识化数据，扩展到多适应症与多中心可行性分析。"
)


def char_count(text: str) -> int:
    return len(re.sub(r"\s+", "", text))


def register_fonts() -> None:
    pdfmetrics.registerFont(TTFont("MSYH", r"C:\Windows\Fonts\msyh.ttc"))
    pdfmetrics.registerFont(TTFont("MSYH-Bold", r"C:\Windows\Fonts\msyhbd.ttc"))


def make_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "body": ParagraphStyle(
            "BodyCN",
            parent=base["BodyText"],
            fontName="MSYH",
            fontSize=9.2,
            leading=16,
            textColor=INK,
            spaceAfter=5,
            wordWrap="CJK",
        ),
        "small": ParagraphStyle(
            "SmallCN",
            parent=base["BodyText"],
            fontName="MSYH",
            fontSize=7.7,
            leading=12,
            textColor=MUTED,
            wordWrap="CJK",
        ),
        "tiny": ParagraphStyle(
            "TinyCN",
            parent=base["BodyText"],
            fontName="MSYH",
            fontSize=6.9,
            leading=10,
            textColor=MUTED,
            wordWrap="CJK",
        ),
        "h1": ParagraphStyle(
            "H1CN",
            parent=base["Heading1"],
            fontName="MSYH-Bold",
            fontSize=23,
            leading=31,
            textColor=INK,
            spaceAfter=8,
            wordWrap="CJK",
        ),
        "h2": ParagraphStyle(
            "H2CN",
            parent=base["Heading2"],
            fontName="MSYH-Bold",
            fontSize=15,
            leading=21,
            textColor=INK,
            spaceBefore=4,
            spaceAfter=8,
            wordWrap="CJK",
        ),
        "h3": ParagraphStyle(
            "H3CN",
            parent=base["Heading3"],
            fontName="MSYH-Bold",
            fontSize=10.5,
            leading=15,
            textColor=BLUE_DARK,
            spaceBefore=3,
            spaceAfter=4,
            wordWrap="CJK",
        ),
        "cover_title": ParagraphStyle(
            "CoverTitle",
            parent=base["Title"],
            fontName="MSYH-Bold",
            fontSize=29,
            leading=38,
            alignment=TA_LEFT,
            textColor=WHITE,
            wordWrap="CJK",
        ),
        "cover_subtitle": ParagraphStyle(
            "CoverSubtitle",
            parent=base["BodyText"],
            fontName="MSYH",
            fontSize=13,
            leading=22,
            textColor=colors.HexColor("#DCEBFA"),
            wordWrap="CJK",
        ),
        "kicker": ParagraphStyle(
            "Kicker",
            parent=base["BodyText"],
            fontName="MSYH-Bold",
            fontSize=7.4,
            leading=10,
            textColor=BLUE,
            spaceAfter=4,
            wordWrap="CJK",
        ),
        "quote": ParagraphStyle(
            "QuoteCN",
            parent=base["BodyText"],
            fontName="MSYH",
            fontSize=9,
            leading=16,
            textColor=INK,
            leftIndent=2,
            rightIndent=2,
            wordWrap="CJK",
        ),
        "center": ParagraphStyle(
            "CenterCN",
            parent=base["BodyText"],
            fontName="MSYH",
            fontSize=8.3,
            leading=13,
            alignment=TA_CENTER,
            textColor=INK,
            wordWrap="CJK",
        ),
        "link": ParagraphStyle(
            "LinkCN",
            parent=base["BodyText"],
            fontName="MSYH",
            fontSize=8.2,
            leading=13,
            textColor=BLUE,
            wordWrap="CJK",
        ),
    }


def p(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text, style)


def page_title(number: str, title: str, subtitle: str, styles: dict[str, ParagraphStyle]) -> list:
    return [
        p(f"{number} / TRIALSCOPEAI", styles["kicker"]),
        p(title, styles["h1"]),
        p(subtitle, styles["body"]),
        HRFlowable(width="100%", thickness=.7, color=LINE, spaceBefore=4, spaceAfter=12),
    ]


def callout(text: str, styles: dict[str, ParagraphStyle], *, tone: str = "blue") -> Table:
    if tone == "warning":
        background, border, text_color = WARNING_SOFT, colors.HexColor("#E8D6AF"), WARNING
    elif tone == "green":
        background, border, text_color = GREEN_SOFT, colors.HexColor("#B9DCC7"), GREEN
    else:
        background, border, text_color = BLUE_SOFT, colors.HexColor("#BBD7EF"), BLUE_DARK
    style = ParagraphStyle("Callout", parent=styles["body"], textColor=text_color, fontSize=8.8)
    table = Table([[p(text, style)]], colWidths=[174 * mm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), background),
                ("BOX", (0, 0), (-1, -1), .7, border),
                ("LINEBEFORE", (0, 0), (0, -1), 3, BLUE if tone == "blue" else text_color),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    return table


def bullet_list(items: list[str], styles: dict[str, ParagraphStyle]) -> list[Paragraph]:
    bullet_style = ParagraphStyle(
        "BulletCN",
        parent=styles["body"],
        leftIndent=12,
        firstLineIndent=-8,
        bulletIndent=0,
        spaceAfter=4,
    )
    return [p(f"• {item}", bullet_style) for item in items]


def fit_image(path: Path, max_width: float, max_height: float) -> Image:
    with PILImage.open(path) as source:
        width, height = source.size
    scale = min(max_width / width, max_height / height)
    return Image(str(path), width=width * scale, height=height * scale)


def screenshot_block(path: Path, caption: str, styles: dict[str, ParagraphStyle], width: float, height: float) -> Table:
    image = fit_image(path, width, height)
    table = Table([[image], [p(caption, styles["small"])]], colWidths=[width])
    table.setStyle(
        TableStyle(
            [
                ("BOX", (0, 0), (-1, 0), .7, LINE),
                ("BACKGROUND", (0, 1), (-1, 1), CANVAS),
                ("LEFTPADDING", (0, 0), (-1, 0), 0),
                ("RIGHTPADDING", (0, 0), (-1, 0), 0),
                ("TOPPADDING", (0, 0), (-1, 0), 0),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 0),
                ("LEFTPADDING", (0, 1), (-1, 1), 7),
                ("RIGHTPADDING", (0, 1), (-1, 1), 7),
                ("TOPPADDING", (0, 1), (-1, 1), 5),
                ("BOTTOMPADDING", (0, 1), (-1, 1), 5),
            ]
        )
    )
    return table


def grid_table(
    rows: list[list],
    widths: list[float],
    *,
    header: bool = True,
    font_size: float = 8.1,
) -> Table:
    table = Table(rows, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    commands = [
        ("FONTNAME", (0, 0), (-1, -1), "MSYH"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("TEXTCOLOR", (0, 0), (-1, -1), INK),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), .55, LINE),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]
    if header:
        commands += [
            ("BACKGROUND", (0, 0), (-1, 0), BLUE_DARK),
            ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
            ("FONTNAME", (0, 0), (-1, 0), "MSYH-Bold"),
        ]
    for index in range(1 if header else 0, len(rows)):
        if index % 2 == 0:
            commands.append(("BACKGROUND", (0, index), (-1, index), CANVAS))
    table.setStyle(TableStyle(commands))
    return table


def decorate_page(canvas, doc) -> None:
    canvas.saveState()
    width, height = A4
    canvas.setFillColor(BLUE)
    canvas.rect(0, height - 4 * mm, width, 4 * mm, fill=1, stroke=0)
    canvas.setStrokeColor(LINE)
    canvas.setLineWidth(.5)
    canvas.line(20 * mm, 13 * mm, width - 20 * mm, 13 * mm)
    canvas.setFont("MSYH", 7)
    canvas.setFillColor(MUTED)
    canvas.drawString(20 * mm, 8 * mm, "TrialScopeAI · 2026 AI先锋未来人才大赛 · 健康元命题")
    canvas.drawRightString(width - 20 * mm, 8 * mm, f"{doc.page}")
    canvas.restoreState()


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
        title="TrialScopeAI 大赛提交材料",
        author="循证智筛",
        subject="健康元临床试验招募可行性评估命题",
    )

    story: list = []

    # Cover
    cover_content = [
        p("2026 AI先锋未来人才大赛 · 健康元企业命题", styles["cover_subtitle"]),
        Spacer(1, 16 * mm),
        p("TrialScopeAI", styles["cover_title"]),
        p("临床试验入排标准结构化与招募可行性评估", styles["cover_subtitle"]),
        Spacer(1, 20 * mm),
        p("从方案原文到可审核规则，从模拟预筛到人群代表性分析。", styles["cover_subtitle"]),
    ]
    cover_table = Table([[cover_content]], colWidths=[174 * mm], rowHeights=[126 * mm])
    cover_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), BLUE_DARK),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 16 * mm),
                ("RIGHTPADDING", (0, 0), (-1, -1), 16 * mm),
                ("TOPPADDING", (0, 0), (-1, -1), 12 * mm),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 12 * mm),
            ]
        )
    )
    story += [Spacer(1, 16 * mm), cover_table, Spacer(1, 12 * mm)]
    meta_rows = [
        [p("推荐队伍名称", styles["small"]), p("循证智筛", styles["h3"]), p("所选企业", styles["small"]), p("健康元药业", styles["h3"])],
        [p("团队成员", styles["small"]), p("待填写", styles["body"]), p("学校", styles["small"]), p("待填写", styles["body"])],
        [p("在线演示", styles["small"]), p('<link href="https://trialscopeai.streamlit.app/" color="#0B5CAB">trialscopeai.streamlit.app</link>', styles["link"]), p("代码仓库", styles["small"]), p('<link href="https://github.com/liziyaaa/TrialScopeAI" color="#0B5CAB">github.com/liziyaaa/TrialScopeAI</link>', styles["link"])],
    ]
    story.append(grid_table(meta_rows, [27 * mm, 55 * mm, 24 * mm, 68 * mm], header=False))
    story.append(Spacer(1, 9 * mm))
    story.append(callout("提交提醒：报名截止时间为 2026年7月19日 24:00（北京时间）。请在提交前补齐成员、学校与赛区信息。", styles, tone="warning"))
    story.append(PageBreak())

    # Fill-ready content
    story += page_title("01", "报名表可直接粘贴内容", "正文已按报名表字数要求整理；个人信息仍需团队自行补齐。", styles)
    story.append(p("基本信息", styles["h2"]))
    basic = [
        ["字段", "建议填写"],
        ["队伍名称", "循证智筛"],
        ["企业命题", "健康元药业"],
        ["项目名称", "TrialScopeAI——临床试验入排标准结构化与招募可行性评估助手"],
        ["一句话介绍", "将自然语言入排标准转为可审核规则，在合成候选者中模拟预筛，提前定位招募瓶颈与代表性风险。"],
        ["补充材料链接", "https://github.com/liziyaaa/TrialScopeAI"],
        ["在线演示", "https://trialscopeai.streamlit.app/"],
    ]
    story.append(grid_table(basic, [35 * mm, 139 * mm]))
    story.append(Spacer(1, 7 * mm))
    story.append(p(f"开题报告 Part 1｜命题前置分析与洞察（约 {char_count(PART_1)} 字）", styles["h2"]))
    story.append(callout(PART_1, styles))
    story.append(Spacer(1, 6 * mm))
    story.append(p(f"开题报告 Part 2｜整体解决方案设计（约 {char_count(PART_2)} 字）", styles["h2"]))
    story.append(callout(PART_2, styles, tone="green"))
    story.append(PageBreak())

    # Insight
    story += page_title("02", "命题洞察：问题不只在“找不到患者”", "我们把临床、数据与运营问题拆成可验证的三个层次。", styles)
    insight_rows = [
        [p("临床解释", styles["h3"]), p("方案标准包含阈值、单位、时间窗、组合逻辑与主观判断。人工逐条解释容易形成中心间差异。", styles["body"])],
        [p("数据可获得性", styles["h3"]), p("部分标准依赖检查或病史字段；如果字段缺失，系统应输出“信息不足”，不能静默排除。", styles["body"])],
        [p("人群代表性", styles["h3"]), p("年龄、疾病程度、共病和治疗史等条件会共同改变可参与人群构成，需要在方案阶段观察潜在偏移。", styles["body"])],
    ]
    table = Table(insight_rows, colWidths=[37 * mm, 137 * mm])
    table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), .6, LINE),
        ("BACKGROUND", (0, 0), (0, -1), BLUE_SOFT),
        ("LEFTPADDING", (0, 0), (-1, -1), 9),
        ("RIGHTPADDING", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
    ]))
    story.append(table)
    story.append(Spacer(1, 8 * mm))
    story.append(p("外部证据如何支持选题", styles["h2"]))
    story += bullet_list(
        [
            "FDA 2025年最终指南强调，应通过试验设计和招募实践提高临床试验人群的代表性，并审视入排标准的影响。",
            "公开研究已验证，将自然语言入排标准解析为计算机可执行规则并与候选者数据匹配具有技术可行性。",
            "自动化匹配研究显示，机器可以显著缩短规则执行时间；但提取准确性仍需人工审核，因此首版采用“模型理解、规则执行、人工确认”的分层结构。",
        ],
        styles,
    )
    story.append(Spacer(1, 6 * mm))
    story.append(p("为什么适合预防医学团队", styles["h2"]))
    story.append(
        callout(
            "预防医学的优势不仅是“懂疾病”，还包括流行病学、偏倚控制、数据质量和人群代表性分析。项目把这些能力放在招募可行性环节，而不是与影像诊断或处方决策竞争。",
            styles,
        )
    )
    story.append(Spacer(1, 7 * mm))
    story.append(p("数据能否获得", styles["h2"]))
    data_access = [
        ["阶段", "使用数据", "可获得性"],
        ["当前原型", "ClinicalTrials.gov 公开方案 + 团队构造的合成候选者", "可公开复现，不需要医院授权"],
        ["校内验证", "人工标注标准 + 独立边界病例", "团队可自行完成"],
        ["企业试点", "去标识化历史筛选日志或字段统计", "需企业授权、伦理与数据治理"],
    ]
    story.append(grid_table(data_access, [31 * mm, 78 * mm, 65 * mm]))
    story.append(PageBreak())

    # Architecture
    story += page_title("03", "解决方案：把语义理解与入组判定分开", "TrialScopeAI 不让大模型直接决定患者是否入组。", styles)
    flow_cells = []
    flow_labels = [
        ("01", "方案导入", "NCT / 文本 / PDF"),
        ("02", "语义结构化", "字段 / 阈值 / 时间窗"),
        ("03", "医学审核", "逐条确认 / 修订"),
        ("04", "确定性预筛", "四类结果 / 证据链"),
        ("05", "招募评估", "漏斗 / 代表性 / 情景"),
    ]
    for number, title, hint in flow_labels:
        flow_cells.append(
            [
                p(number, styles["kicker"]),
                p(title, styles["h3"]),
                p(hint, styles["small"]),
            ]
        )
    flow = Table(
        flow_cells,
        colWidths=[18 * mm, 56 * mm, 100 * mm],
        rowHeights=[17 * mm] * 5,
    )
    flow.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), CANVAS),
        ("BOX", (0, 0), (-1, -1), .7, LINE),
        ("INNERGRID", (0, 0), (-1, -1), .4, LINE),
        ("BACKGROUND", (0, 0), (0, -1), BLUE_SOFT),
        ("LINEBEFORE", (0, 0), (0, -1), 4, BLUE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(flow)
    story.append(Spacer(1, 8 * mm))
    story.append(p("核心分层", styles["h2"]))
    layers = [
        ["层", "职责", "安全控制"],
        ["输入层", "定位并展示方案原文", "人工确认后才解析；扫描件不猜测"],
        ["语义层", "提取字段、运算符、阈值、单位、时间窗", "JSON 输出 + 结构校验 + 缓存与限额"],
        ["规则层", "按确定性优先级执行入排标准", "每项判断保留患者值、条件与原文"],
        ["分析层", "展示瓶颈、缺失和代表性变化", "仅做模拟，不直接建议修改方案"],
    ]
    story.append(grid_table(layers, [26 * mm, 74 * mm, 74 * mm]))
    story.append(Spacer(1, 8 * mm))
    story.append(callout("四类输出优先级：任一明确标准失败 → 不符合；无失败但字段缺失 → 信息不足；无失败无缺失但有主观标准 → 人工复核；其余 → 模拟符合。", styles, tone="warning"))
    story.append(PageBreak())

    # UI home
    story += page_title("04", "产品界面：任务驱动，而不是功能堆叠", "每一步都明确告诉用户“现在要做什么、状态如何、点击哪里继续”。", styles)
    story.append(screenshot_block(ROOT / "tmp" / "ui-home.png", "项目概览：四项任务、当前状态与明确操作按钮同屏呈现。", styles, 174 * mm, 125 * mm))
    story.append(Spacer(1, 7 * mm))
    story += bullet_list(
        [
            "侧栏导航改为完整可点击按钮，当前步骤用蓝色边线与底色标识。",
            "首页不再只展示指标，而是提供可执行任务清单；当前主任务只有一个蓝色按钮。",
            "设计语言参考临床研究工作台与成熟企业设计系统，采用直角卡片、清晰层级、低装饰度和可见焦点状态。",
        ],
        styles,
    )
    story.append(PageBreak())

    # UI actions
    story += page_title("05", "产品界面：输入、审核与证据链", "减少隐含操作，把选择、确认和下钻都变成可见动作。", styles)
    left_shot = screenshot_block(ROOT / "tmp" / "ui-import.png", "四种输入来源以可选卡片呈现，选中后只展示对应表单。", styles, 84 * mm, 70 * mm)
    right_shot = screenshot_block(ROOT / "tmp" / "ui-review.png", "审核页把操作说明、可编辑表格与保存动作串成一个任务。", styles, 84 * mm, 70 * mm)
    pair = Table([[left_shot, right_shot]], colWidths=[87 * mm, 87 * mm], hAlign="LEFT")
    pair.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 3)]))
    story.append(pair)
    story.append(Spacer(1, 8 * mm))
    story.append(screenshot_block(ROOT / "tmp" / "ui-screening.png", "预筛页：运行按钮、四类结果、状态筛选与可点击候选者表格。点击任一行即可查看逐条证据。", styles, 174 * mm, 111 * mm))
    story.append(PageBreak())

    # Analysis UI
    story += page_title("06", "招募评估：从“谁不符合”到“为什么招不到”", "把单个候选者判断汇总为筛减路径、主要瓶颈和人群构成变化。", styles)
    story.append(screenshot_block(ROOT / "tmp" / "ui-analysis.png", "GOLDEN-4 合成演示：500 名候选者经年龄、吸烟史、肺功能、近期事件等步骤逐层筛减；FEV1/FVC 为首要瓶颈。", styles, 174 * mm, 125 * mm))
    story.append(Spacer(1, 7 * mm))
    value_rows = [
        ["使用者", "可回答的问题"],
        ["临床开发", "哪些标准最影响候选池？是否存在难执行的时间窗？"],
        ["医学团队", "主观标准和缺失数据集中在哪里？哪些规则需要进一步澄清？"],
        ["运营团队", "哪个筛减环节最值得优先准备数据与中心培训？"],
        ["预防医学/流行病学", "潜在参与人群与候选队列在年龄、性别和疾病程度上有何差异？"],
    ]
    story.append(grid_table(value_rows, [40 * mm, 134 * mm]))
    story.append(PageBreak())

    # Validation
    story += page_title("07", "验证、数据与落地可行性", "所有指标分为“原型已完成”和“后续验收目标”，不混淆。", styles)
    achieved = [
        ["原型已完成", "当前状态", "可复现证据"],
        ["人工审核结构化标准", "27 条", "GOLDEN-4 金标准 JSON"],
        ["合成候选队列", "500 名", "固定随机种子 20260716"],
        ["独立边界病例", "50 个", "阈值相等、单位、缺失、时间窗和主观标准"],
        ["自动化测试", "40 项通过", "离线环境可运行缓存演示"],
        ["真实患者数据", "0 条", "当前无需医院数据与个人信息"],
    ]
    story.append(grid_table(achieved, [58 * mm, 32 * mm, 84 * mm]))
    story.append(Spacer(1, 8 * mm))
    story.append(p("后续验收目标（尚不能表述为已达成企业效果）", styles["h2"]))
    targets = [
        ["指标", "目标", "验证方法"],
        ["结构化提取 F1", "≥ 0.85", "与人工金标准比较类型、字段、运算符、阈值、单位与时间窗"],
        ["患者规则匹配准确率", "≥ 90%", "由医学人员独立复核边界病例与真实去标识化样本"],
        ["标准来源追溯率", "100%", "每条规则均保留方案原文与来源"],
        ["人工审核效率", "企业试点后确定", "记录同一方案人工审核用时、修订率和中心间一致性"],
    ]
    story.append(grid_table(targets, [53 * mm, 30 * mm, 91 * mm]))
    story.append(Spacer(1, 8 * mm))
    story.append(p("落地路径", styles["h2"]))
    story += bullet_list(
        [
            "第一阶段：使用公开方案和合成队列验证交互、规则执行与演示稳定性。",
            "第二阶段：由健康元医学、统计与运营人员审核标准，确定企业场景中的字段字典和验证指标。",
            "第三阶段：在伦理、合同与数据治理到位后接入去标识化历史筛选统计，验证效率和一致性。",
            "第四阶段：扩展到多适应症、多中心和多版本方案比较，并保留审计记录。",
        ],
        styles,
    )
    story.append(PageBreak())

    # Safety, references, checklist
    story += page_title("08", "安全边界、参考资料与提交检查", "最后一页用于评委快速确认方案可信度，也用于团队提交前自查。", styles)
    story.append(p("医疗与数据安全边界", styles["h2"]))
    story += bullet_list(
        [
            "不诊断、不自动入组、不替代研究者、统计人员或伦理委员会。",
            "大模型只负责语义提取；最终判定由可解释规则执行，主观标准进入人工复核。",
            "当前不使用真实患者数据；上传 PDF 仅在会话内存中处理，不写入数据库。",
            "扫描型 PDF 不做 OCR；无法提取有效文本时明确提示，不生成猜测结果。",
            "情景分析均为合成模拟，不构成修改临床试验方案的建议。",
        ],
        styles,
    )
    story.append(Spacer(1, 5 * mm))
    story.append(p("主要参考资料", styles["h2"]))
    references = [
        '<link href="https://clinicaltrials.gov/study/NCT02347774" color="#0B5CAB">ClinicalTrials.gov：GOLDEN-4（NCT02347774）</link>',
        '<link href="https://www.fda.gov/regulatory-information/search-fda-guidance-documents/enhancing-diversity-clinical-trial-populations-eligibility-criteria-enrollment-practices-and-trial" color="#0B5CAB">FDA：Enhancing the Diversity of Clinical Trial Populations（2025 Final Guidance）</link>',
        '<link href="https://pmc.ncbi.nlm.nih.gov/articles/PMC11141802/" color="#0B5CAB">PMC：Automating clinical trial eligibility screening with NLP and synthetic EHR data</link>',
        '<link href="https://pmc.ncbi.nlm.nih.gov/articles/PMC10857751/" color="#0B5CAB">PMC：Automated clinical trial matching in pediatric leukemia</link>',
        '<link href="https://www.veeva.com/cn/products/vault-ctms/" color="#0B5CAB">Veeva Vault CTMS：角色化临床研究工作台参考</link>',
        '<link href="https://carbondesignsystem.com/components/button/usage/" color="#0B5CAB">Carbon Design System：操作层级与按钮设计参考</link>',
        '<link href="https://design-system.service.gov.uk/components/task-list/" color="#0B5CAB">GOV.UK Design System：多步骤任务清单参考</link>',
    ]
    for index, reference in enumerate(references, start=1):
        story.append(p(f"{index}. {reference}", styles["link"]))
    story.append(Spacer(1, 6 * mm))
    story.append(p("提交前 5 分钟检查", styles["h2"]))
    checklist = [
        ["检查项", "状态"],
        ["队伍名称、成员、学校、赛区信息已填写", "□"],
        ["企业命题选择“健康元药业”", "□"],
        ["Part 1 与 Part 2 已按本 PDF 文本粘贴", "□"],
        ["补充材料 PDF 已上传或 GitHub 链接已填写", "□"],
        ["在线演示地址可在无登录状态打开", "□"],
        ["提交时间早于 2026年7月19日 24:00（北京时间）", "□"],
    ]
    story.append(grid_table(checklist, [145 * mm, 29 * mm]))

    doc.build(story, onFirstPage=decorate_page, onLaterPages=decorate_page)
    print(f"Part 1 characters: {char_count(PART_1)}")
    print(f"Part 2 characters: {char_count(PART_2)}")
    print(f"Created: {OUTPUT}")


if __name__ == "__main__":
    build_pdf()
