r"""
第六课：文档向量检索——RAG 的"检索"那一步
上一课我们把一句话变成向量，算了两两相似度。
这一课往前走一步：把一整段文档切成多块，每块向量化；
用户问一个问题，也向量化，然后找最相关的 top-k 块。

这就是 RAG（检索增强生成）里"R"（Retrieval 检索）的核心：
从一堆文档里，挑出和问题最相关的几段，喂给模型回答。

运行（在 C:\Users\朱天佑\UCYP 目录下）：
    ai-learning\.venv\Scripts\python.exe ai-learning\05_doc_retrieval.py
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


def get_embedding(text: str) -> list[float]:
    """把一句话变成向量（复用第五课）"""
    resp = client.embeddings.create(
        model="embedding-3",
        input=text,
    )
    return resp.data[0].embedding


def cosine(a: list[float], b: list[float]) -> float:
    """余弦相似度（复用第五课）"""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb)


# ────────────────────────────────────────────
# 第一步：准备一段"知识文档"，切成多个小块
# 每行一个 chunk，用换行分隔——真实场景会按句号/段落/固定长度切
# ────────────────────────────────────────────
DOCUMENT = """Python 是一种解释型高级编程语言，由 Guido van Rossum 于 1991 年发布。
Python 使用缩进来划分代码块，不使用花括号，这让代码可读性很高。
Python 的基本数据类型有 int float str bool，复合类型有 list tuple dict set。
Python 用 pip 安装第三方包，命令是 pip install 包名。
Python 常用于 Web 开发 Django Flask，数据分析 pandas，人工智能 PyTorch。
Python 的列表推导式 x乘2 for x in range 10 是构造列表的简洁写法。"""

# 按换行切分成多个 chunk，去掉空行
chunks = [line.strip() for line in DOCUMENT.strip().split("\n") if line.strip()]

print("=" * 56)
print("第一步：文档切成块")
print("=" * 56)
for i, chunk in enumerate(chunks):
    print(f"  chunk {i}: {chunk}")
print(f"共 {len(chunks)} 块")

# ────────────────────────────────────────────
# 第二步：把每块文档向量化（一次性算好缓存，后面复用）
# ────────────────────────────────────────────
print()
print("=" * 56)
print("第二步：每块文档向量化（一次性预计算）")
print("=" * 56)
chunk_vectors = [get_embedding(c) for c in chunks]
print(f"已把 {len(chunks)} 块文档全部变成 {len(chunk_vectors[0])} 维向量")
print("（这些向量可以存进向量数据库，下次直接查，不用重算）")

# ────────────────────────────────────────────
# 第三步：用户提问 → 向量化 → 算相似度 → 取 top-k
# ────────────────────────────────────────────
QUESTIONS = [
    "怎么安装 Python 的第三方库？",       # 应该匹配 chunk 3（pip）
    "Python 都能用来干什么？",             # 应该匹配 chunk 4（应用领域）
    "Python 代码怎么划分代码块？",         # 应该匹配 chunk 1（缩进）
]
TOP_K = 2   # 每个问题取最相关的 2 块

for q in QUESTIONS:
    print()
    print("=" * 56)
    print(f"问题：{q}")
    print("=" * 56)

    # 问题向量化
    q_vec = get_embedding(q)

    # 算问题和每块文档的相似度，组成 (相似度, 序号, 文本) 列表
    scored = []
    for i, c_vec in enumerate(chunk_vectors):
        sim = cosine(q_vec, c_vec)
        scored.append((sim, i, chunks[i]))

    # 按相似度从高到低排序
    scored.sort(key=lambda x: x[0], reverse=True)

    # 打印所有块的得分（让用户看见排名过程）
    print("所有块得分（从高到低）：")
    for rank, (sim, i, text) in enumerate(scored):
        mark = "★" if rank < TOP_K else " "
        print(f"  {mark} {sim:.3f}  chunk{i}: {text}")

    # 取 top-k
    print(f"\n→ 检索结果（top-{TOP_K}）：")
    for rank, (sim, i, text) in enumerate(scored[:TOP_K]):
        print(f"  [{rank+1}] chunk{i} (相似度 {sim:.3f})")
        print(f"      {text}")

print()
print("=" * 56)
print("这就是 RAG 的检索步骤！下一课把检索到的块喂给模型回答问题。")
print("=" * 56)
