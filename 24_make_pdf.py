r"""
第二十四课（素材准备）：造一份《XYZ 公司员工手册》PDF
=======================================================
手头没有现成 PDF，先用 fpdf2 造一份：8 章员工手册，每章一页。
内容基于第八课的 knowledge.txt（虚构 XYZ 公司资料）扩写，
之后用 24_pdf_rag.py 解析它做 RAG——内容你熟悉，检索好不好肉眼可判。

为什么中文要用特殊字体：PDF 内置的 Helvetica 等 14 种标准字体只支持
西文，写中文必须嵌一个 TTF 字体（这里用 Windows 自带的黑体 simhei.ttf）。

运行（在 C:\Users\朱天佑\UCYP 目录下）：
    ai-learning\.venv\Scripts\python.exe ai-learning\24_make_pdf.py
"""
from pathlib import Path

from fpdf import FPDF

BASE_DIR = Path(__file__).parent
PDF_PATH = BASE_DIR / "xyz_handbook.pdf"


# ────────────────────────────────────────────
# 8 章手册内容：每章 (标题, 正文段落列表)
# 正文=第八课 knowledge.txt 扩写，每章多几句凑点体量
# ────────────────────────────────────────────
CHAPTERS = [
    ("公司简介", [
        "XYZ 公司成立于 2019 年，创始人李文，总部位于深圳南山区科技园。",
        "公司目前员工约 200 人，主营业务为企业级软件服务。",
        "公司秉持务实创新的文化，扁平化管理，各部门协作顺畅。",
    ]),
    ("考勤与年假", [
        "年假政策：入职满 1 年 5 天，满 3 年 10 天，满 5 年 15 天。",
        "年假当年未休完可顺延至次年 3 月底，之后清零，请提前安排休假。",
        "年假需提前 3 个工作日在 OA 系统提交申请，经直属主管批准后生效。",
    ]),
    ("报销制度", [
        "报销流程：在 OA 系统提交发票，经理审批后财务 3 个工作日内打款。",
        "单笔超过 5000 元需总监额外审批，超过 20000 元需 CFO 审批。",
        "报销发票须为本人实际消费，虚报冒领将按公司制度严肃处理。",
    ]),
    ("员工福利", [
        "每月 300 元餐补，随工资发放。",
        "每年 1500 元健身补贴，凭发票报销。",
        "生日当月发放 500 元礼品卡，子女教育补贴每年 2000 元。",
    ]),
    ("技术栈", [
        "后端使用 Go 语言，前端使用 React 框架。",
        "数据库为 PostgreSQL，部署在 Kubernetes 集群上。",
        "监控体系采用 Prometheus 加 Grafana，日志收集使用 ELK。",
    ]),
    ("技术分享", [
        "每周三下午 4 点举行技术分享会，全员必须参加。",
        "迟到扣 50 元，缺席需提前向主管报备，并补看会议录像。",
        "分享主题不限，鼓励新员工踊跃报名分享。",
    ]),
    ("入职流程", [
        "发放 offer 后 3 个工作日内回复确认。",
        "入职当天携带身份证原件和学历证明到 HR 处登记。",
        "下午领取工牌和电脑，由部门导师带领熟悉环境。",
    ]),
    ("请假流程", [
        "事假在 OA 系统提交申请。",
        "1 天以内主管审批，3 天以内部门经理审批。",
        "超过 3 天需总监审批，请合理安排工作交接。",
    ]),
]


def main():
    pdf = FPDF()
    # 黑体 simhei.ttf 是 Windows 自带的单文件 TTF，支持中文
    pdf.add_font("SimHei", "", r"C:\Windows\Fonts\simhei.ttf")

    # 封面页
    pdf.add_page()
    pdf.set_font("SimHei", size=26)
    pdf.cell(0, 40, "XYZ 公司员工手册", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("SimHei", size=14)
    pdf.cell(0, 10, "版本 1.0    人力资源部", new_x="LMARGIN", new_y="NEXT")

    # 每章一页：大标题 + 段落正文
    for title, paras in CHAPTERS:
        pdf.add_page()
        pdf.set_font("SimHei", size=18)
        pdf.cell(0, 14, title, new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("SimHei", size=12)
        for p in paras:
            # multi_cell：宽度 0=页宽，行高 8mm，超行自动换行
            pdf.multi_cell(0, 8, p)
            pdf.ln(2)  # 段间距

    pdf.output(str(PDF_PATH))
    print(f"已生成：{PDF_PATH}")
    print(f"共 {len(CHAPTERS)} 章 + 1 封面 = {1 + len(CHAPTERS)} 页")


if __name__ == "__main__":
    main()
