r"""
第七课：完整 RAG 问答——检索 + 生成
上一课只做了"检索"：从文档里找出和问题最相关的 top-k 块。
这一课把它和"生成"拼起来：把检索到的块塞进 system prompt，
让模型只能根据这些文档回答，不能自己编。

这就是 RAG 完整流程：
  用户问题 → 检索相关文档块 → 块塞进 prompt → 模型回答（基于文档）

为了看清 RAG 的价值，这里用一段【虚构的公司内部资料】，
模型绝对没见过，只有靠文档才能答对。

运行（在 C:\Users\朱天佑\UCYP 目录下）：
    ai-learning\.venv\Scripts\python.exe ai-learning\06_rag_qa.py
"""
import os
import math
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(
    api_key=os.getenv("ZHIPU_API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4/",
)

# ────────────────────────────────────────────
# 复用第五课的两个函数
# ────────────────────────────────────────────
def get_embedding(text: str) -> list[float]:
    resp = client.embeddings.create(model="embedding-3", input=text)
    return resp.data[0].embedding

def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb)


# ────────────────────────────────────────────
# 知识文档：虚构的 "XYZ 公司" 内部资料（模型从没见过）
# ────────────────────────────────────────────
DOCUMENT = """XYZ 公司成立于 2019 年，创始人叫李文，总部在深圳南山区。
XYZ 公司年假政策：入职满 1 年 5 天，满 3 年 10 天，满 5 年 15 天。
XYZ 公司报销流程：在 OA 系统提交发票，经理审批后财务 3 个工作日内打款。
XYZ 公司福利：每月 300 元餐补，每年 1500 元健身补贴，生日当月发 500 元礼品卡。
XYZ 公司技术栈：后端 Go，前端 React，数据库 PostgreSQL，部署在 Kubernetes。
XYZ 公司每周三下午 4 点有技术分享会，全员必须参加，迟到扣 50 元。"""

chunks = [line.strip() for line in DOCUMENT.strip().split("\n") if line.strip()]
chunk_vectors = [get_embedding(c) for c in chunks]   # 预计算向量


# ────────────────────────────────────────────
# 第一步：检索函数（封装第六课的 top-k 逻辑）
# ────────────────────────────────────────────
def retrieve(question: str, top_k: int = 2) -> list[tuple[float, int, str]]:
    """返回 [(相似度, 块序号, 块原文), ...] 按相关度从高到低"""
    q_vec = get_embedding(question)
    scored = [(cosine(q_vec, cv), i, chunks[i])
             for i, cv in enumerate(chunk_vectors)]
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]


# ────────────────────────────────────────────
# 第二步：两种回答方式——有 RAG / 无 RAG
# ────────────────────────────────────────────
def ask_with_rag(question: str, top_k: int = 2) -> tuple[str, list]:
    """完整 RAG：检索 → 拼 prompt → 模型基于文档回答"""
    top = retrieve(question, top_k)
    # 把检索到的块拼成上下文文本
    context = "\n".join(f"片段{i+1}：{text}" for sim, i, text in top)

    messages = [
        {"role": "system",
         "content": f"你是文档助手。只能根据下面给的文档片段回答问题。"
                    f"文档里没有的就说\"我不知道\"，禁止编造。\n\n{context}"},
        {"role": "user", "content": question},
    ]
    resp = client.chat.completions.create(model="glm-4-flash", messages=messages)
    return resp.choices[0].message.content, top


def ask_without_rag(question: str) -> str:
    """无 RAG：直接问模型，它会根据自己训练时见过的内容答"""
    resp = client.chat.completions.create(
        model="glm-4-flash",
        messages=[{"role": "user", "content": question}],
    )
    return resp.choices[0].message.content


# ────────────────────────────────────────────
# 第三步：对比两种方式
# ────────────────────────────────────────────
QUESTIONS = [
    "XYZ 公司年假多少天？",          # 模型没见过，无 RAG 会瞎编或说不知道
    "XYZ 公司报销要多久到账？",       # 同上
]

for q in QUESTIONS:
    print("=" * 56)
    print(f"问题：{q}")
    print("=" * 56)

    # 无 RAG
    print("\n【无 RAG】直接问模型：")
    print("  " + ask_without_rag(q).replace("\n", "\n  "))

    # 有 RAG
    print("\n【有 RAG】检索 + 喂文档：")
    answer, top = ask_with_rag(q)
    print("  检索到的文档块：")
    for sim, i, text in top:
        print(f"    - (相似度 {sim:.3f}) {text}")
    print("  模型回答：")
    print("  " + answer.replace("\n", "\n  "))
    print()

print("=" * 56)
print("这就是完整 RAG！无 RAG 时模型瞎编，有 RAG 时只基于文档回答。")
print("=" * 56)
