r"""
第五课：Embedding——把文字变成向量
大模型再强，也算不了"两句话有多像"——除非把文字变成数字向量。
Embedding 就干这件事：输入一句话，输出一个高维向量（一串浮点数）。
语义相近的句子，向量也相近（余弦相似度高）；语义无关的句子，向量离得远。

本课：用智谱 embedding-3 把几句话变成向量，手算余弦相似度，看"语义相似度"如何被量化。
运行（在 ai-learning 目录下）：
    .venv\Scripts\python.exe 04_embedding.py
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
    """把一句话变成向量。返回的就是一串浮点数，长度=向量维度"""
    resp = client.embeddings.create(
        model="embedding-3",   # 智谱的 Embedding 模型
        input=text,
    )
    return resp.data[0].embedding


def cosine(a: list[float], b: list[float]) -> float:
    """余弦相似度：1=方向完全一致(最像)，0=无关，-1=方向相反。
    用向量点积除以两个模长的乘积。不依赖 numpy，手写更直观。"""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb)


# ────────────────────────────────────────────
# 4 句话：前 3 句都在讲天气（语义相近），第 4 句讲编程（语义无关）
# ────────────────────────────────────────────
sentences = [
    "今天天气真好，阳光明媚",        # 天气 1
    "今日晴朗，温度适宜",            # 天气 2
    "外面下着大雨，记得带伞",        # 天气 3（也是天气话题）
    "我在学习 Python 编程",          # 编程（无关）
]

# 把每句话都变成向量
vectors = [get_embedding(s) for s in sentences]

# 看看向量长什么样
print("=" * 56)
print("第一步：一句话变成的向量长什么样")
print("=" * 56)
print(f"句子：{sentences[0]}")
print(f"向量维度：{len(vectors[0])}")
print(f"前 5 个分量：{[round(x, 4) for x in vectors[0][:5]]}")
print(f"（这只是 {len(vectors[0])} 维里的前 5 个，光看数字没意义，意义在「比较」）")

# ────────────────────────────────────────────
# 算两两之间的余弦相似度
# ────────────────────────────────────────────
print()
print("=" * 56)
print("第二步：两两余弦相似度（越接近 1 越像）")
print("=" * 56)
print(f"{'':<22}", end="")
for j in range(len(sentences)):
    print(f"句{j+1:<6}", end="")
print()

for i in range(len(sentences)):
    label = sentences[i][:20]   # 截短显示
    print(f"{label:<22}", end="")
    for j in range(len(sentences)):
        sim = cosine(vectors[i], vectors[j])
        print(f"{sim:.3f}  ", end="")
    print()

# ────────────────────────────────────────────
# 重点对比：天气句 vs 编程句
# ────────────────────────────────────────────
print()
print("=" * 56)
print('第三步：重点看「语义相近」和「语义无关」的差距')
print("=" * 56)
sim_same = cosine(vectors[0], vectors[1])      # 天气1 vs 天气2
sim_diff = cosine(vectors[0], vectors[3])      # 天气1 vs 编程
print(f"「{sentences[0]}」")
print(f"  vs「{sentences[1]}」相似度 = {sim_same:.3f}  ← 都讲天气，高")
print(f"  vs「{sentences[3]}」相似度 = {sim_diff:.3f}  ← 一个天气一个编程，低")
print()
print(f"差距：{sim_same - sim_diff:.3f}（语义相似度被量化成了可比的数字）")
