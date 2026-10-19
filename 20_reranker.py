r"""
RAG 深水区第一课：Reranker 重排序——提升检索准确率
=====================================================
前 9 课学的 RAG 流程：问题 → 向量检索 top-k → 拼 context → LLM 回答。
但向量检索有个短板：它按"语义相似度"排序，而"语义相似" ≠ "真的回答了问题"。

举例：问"退货要付运费吗"
  向量检索可能返回：
    1. 退货流程（语义相近，但没提运费）        ← 粗排排第一
    2. 退货政策（语义相近，也没提运费）
    3. 退款时间（提到了"运费不退"——这才是真正该排第一的！）  ← 精排应该提上来

Reranker（重排序器）解决方法：两阶段检索
  第一阶段（粗排）：向量检索取 top-5（快，但可能不够精准）
  第二阶段（精排）：用 LLM 逐条打分 0-10 分重排序（慢，但准）
  最终取重排后的 top-3 喂给 LLM 生成回答

这就像考试选拔：
  粗排 = 笔试（快，刷掉明显不相关的，保留可能相关的）
  精排 = 面试（慢，逐条细看，选出真正回答了问题的）

本课用 GLM-4-Flash 当 reranker：
  对每个检索到的块，让 LLM 打 0-10 分（与问题的相关度）
  按分数排序，取 top-3

对比：同一批问题，粗排 top-3 vs 精排 top-3，看排序和回答差异。

运行（在 C:\Users\朱天佑\UCYP 目录下）：
    ai-learning\.venv\Scripts\python.exe ai-learning\20_reranker.py
"""
import os
import re
import chromadb
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(
    api_key=os.getenv("ZHIPU_API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4/",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def get_embedding(text: str) -> list[float]:
    """调用智谱 embedding-3 生成向量（2048 维），复用第五课逻辑"""
    resp = client.embeddings.create(model="embedding-3", input=text)
    return resp.data[0].embedding


# ────────────────────────────────────────────
# 初始化 FAQ 知识库（复用第十三课 Chroma 持久化）
# ────────────────────────────────────────────
db_path = os.path.join(BASE_DIR, ".chroma_faq")
chroma_client = chromadb.PersistentClient(path=db_path)
collection = chroma_client.get_or_create_collection(
    name="faq_customer_service",
    metadata={"hnsw:space": "cosine"},
)

if collection.count() == 0:
    with open(os.path.join(BASE_DIR, "faq_knowledge.txt"), encoding="utf-8") as f:
        chunks = [c.strip() for c in f.read().split("\n\n") if c.strip()]
    collection.add(
        documents=chunks,
        embeddings=[get_embedding(c) for c in chunks],
        ids=[f"faq_{i}" for i in range(len(chunks))],
    )
    print(f"已从 faq_knowledge.txt 读入 {len(chunks)} 条 FAQ")
else:
    print(f"FAQ 知识库就绪（{collection.count()} 条已入库）")


# ════════════════════════════════════════════
# 第一阶段：向量粗排（复用第十三课 retrieve）
# ════════════════════════════════════════════
def vector_retrieve(question: str, top_k: int = 5):
    """
    向量检索 top_k 条 FAQ（粗排）。
    top_k=5 比第十三课的 3 多取一些——给精排留选择空间。
    """
    q_vec = get_embedding(question)
    results = collection.query(query_embeddings=[q_vec], n_results=top_k)
    docs = results["documents"][0]
    dists = results["distances"][0]
    return list(zip(dists, docs))


# ════════════════════════════════════════════
# 第二阶段：LLM 精排（reranker 核心逻辑）
# ════════════════════════════════════════════
def rerank(question: str, retrieved: list) -> list:
    """
    用 LLM 对每条检索结果打分（0-10），按分数重排序。
    输入：question + [(distance, doc), ...]
    输出：[(score, doc), ...] 按分数降序

    原理：向量检索只看"语义相似度"（词向量距离），
    LLM 打分能理解"这条 FAQ 到底有没有回答到点上"——理解力更强。
    """
    scored = []
    for dist, doc in retrieved:
        # 让 LLM 给这条 FAQ 和问题的相关度打分
        messages = [
            {
                "role": "system",
                "content": (
                    "你是检索质量评估器。给下面这条 FAQ 与用户问题的相关度打分（0-10整数）。"
                    "10分=完全回答了问题，5分=部分相关，0分=毫不相关。"
                    "只输出一个数字，不要输出其他任何内容。"
                ),
            },
            {"role": "user", "content": f"用户问题：{question}\n\nFAQ内容：{doc}"},
        ]
        resp = client.chat.completions.create(model="glm-4-flash", messages=messages)
        score_text = resp.choices[0].message.content.strip()
        # 提取第一个数字（模型可能输出 "8" 或 "8分" 或 "评分8"）
        match = re.search(r"\d+", score_text)
        score = int(match.group()) if match else 0
        scored.append((score, doc))

    # 按分数降序排序（分数高的排前面）
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored


# ────────────────────────────────────────────
# RAG 回答（用检索结果生成回答）
# ────────────────────────────────────────────
def rag_answer(question: str, top_docs: list[str]) -> str:
    """用 top_docs 拼成 context，让 LLM 回答"""
    context = "\n".join(f"FAQ{i+1}：{doc}" for i, doc in enumerate(top_docs))
    messages = [
        {
            "role": "system",
            "content": (
                "你是「好物优选」电商平台的智能客服。请根据下面的 FAQ 知识库回答客户问题。"
                "要求：1) 只用知识库里的信息，不要编造；"
                "2) 语气专业、简洁、有温度；"
                "3) 如果知识库里没有相关信息，礼貌告知并建议联系人工客服。\n\n"
                f"{context}"
            ),
        },
        {"role": "user", "content": question},
    ]
    resp = client.chat.completions.create(model="glm-4-flash", messages=messages)
    return resp.choices[0].message.content


# ────────────────────────────────────────────
# 对比演示：粗排 top-3 vs 精排 top-3
# ────────────────────────────────────────────
# 精心挑选的问题——向量检索可能排错，精排应该能纠正
QUESTIONS = [
    "退货要付运费吗",          # 退款时间条目提到了"运费不退"，但语义上退货流程更近
    "新疆运费多少",            # 快递公司条目提到了"新疆"，但运费政策条目才真正答了运费
    "花呗分期手续费多少",      # 支持渠道条目提到了"花呗"，但分期付款条目才答了手续费
]

print()
print("=" * 56)
print("  RAG 深水区第一课：Reranker 重排序")
print("=" * 56)

for q in QUESTIONS:
    print(f"\n{'─' * 56}")
    print(f"问题：{q}")
    print(f"{'─' * 56}")

    # 第一阶段：向量粗排 top-5
    retrieved = vector_retrieve(q, top_k=5)

    # 第二阶段：LLM 精排
    reranked = rerank(q, retrieved)

    # 粗排 top-3（不重排序，直接取前3）
    coarse_top3 = [doc for _, doc in retrieved[:3]]

    # 精排 top-3（重排序后取前3）
    fine_top3 = [doc for _, doc in reranked[:3]]

    # 展示粗排原始顺序
    print("\n【粗排 top-5】（向量检索，按语义相似度排序）：")
    for i, (dist, doc) in enumerate(retrieved, 1):
        mark = " ← 取" if i <= 3 else ""
        print(f"  {i}. [距离={dist:.3f}] {doc[:45]}...{mark}")

    # 展示精排重排后顺序
    print("\n【精排 top-5】（LLM 打分重排序，按相关度排序）：")
    for i, (score, doc) in enumerate(reranked, 1):
        mark = " ← 取" if i <= 3 else ""
        print(f"  {i}. [分数={score:2d}]  {doc[:45]}...{mark}")

    # 检查排序是否变化
    if coarse_top3 != fine_top3:
        print("\n  ⚠ 排序有变化！精排调整了 top-3 的内容")
        # 找出具体变化
        new_in = [d for d in fine_top3 if d not in coarse_top3]
        if new_in:
            print(f"  新进入 top-3：{new_in[0][:40]}...")
        old_out = [d for d in coarse_top3 if d not in fine_top3]
        if old_out:
            print(f"  被挤出 top-3：{old_out[0][:40]}...")
    else:
        print("\n  排序无变化（粗排已经够准）")

    # 对比回答
    print("\n  【粗排回答】（用粗排 top-3）：")
    ans_coarse = rag_answer(q, coarse_top3)
    print(f"  {ans_coarse[:120]}...")

    print("\n  【精排回答】（用精排 top-3）：")
    ans_fine = rag_answer(q, fine_top3)
    print(f"  {ans_fine[:120]}...")

print()
print("=" * 56)
print("总结：")
print("  粗排（向量检索）快但可能不准——语义相似 ≠ 真的回答了问题")
print("  精排（LLM 打分）慢但准——LLM 能理解'这条FAQ到底答没答到点上'")
print("  两阶段检索 = 粗排取宽（top-5）+ 精排取准（top-3）= 又快又准")
print("  代价：精排多 N 次 API 调用（N=粗排的 top_k）")
print("=" * 56)
